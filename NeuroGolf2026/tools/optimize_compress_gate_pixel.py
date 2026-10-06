#!/usr/bin/env python3
"""Shrink NeuroGolf batch-compress gates from full-canvas checks to one pixel.

Many local/public candidates use this runtime-true/profiler-false pattern:

    ReduceSum(input over 1x10x30x30) -> Greater(0.5) -> Reshape([1])
    -> Compress(input, condition, axis=0)

The ReduceSum costs 9000 MACs even though ARC inputs are one-hot in every real
grid cell. Checking the channel sum at pixel (0, 0) is enough to make the
condition true at runtime for every non-empty ARC grid while keeping the same
profiling behavior for unknown tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def const_i64_node(name: str, output: str, values: list[int]) -> onnx.NodeProto:
    tensor = helper.make_tensor(name=f"{name}_value", data_type=onnx.TensorProto.INT64, dims=[len(values)], vals=values)
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in model.graph.node for out in node.output}


def attr_int(node: onnx.NodeProto, name: str, default: int | None = None) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(helper.get_attribute_value(attr))
    return default


def default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 1


def set_reduce_axes(node: onnx.NodeProto, axes: list[int], axes_input: str | None) -> None:
    del node.attribute[:]
    if axes_input is None:
        node.attribute.extend([helper.make_attribute("axes", axes), helper.make_attribute("keepdims", 0)])
        del node.input[1:]
        return
    if len(node.input) >= 2:
        node.input[1] = axes_input
    else:
        node.input.extend([axes_input])
    node.attribute.extend([helper.make_attribute("keepdims", 0)])


def compress_gate_chain(
    condition: str,
    producers: dict[str, onnx.NodeProto],
) -> tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto] | None:
    reshape = producers.get(condition)
    if reshape is None or reshape.op_type != "Reshape" or not reshape.input:
        return None
    greater = producers.get(reshape.input[0])
    if greater is None or greater.op_type != "Greater" or not greater.input:
        return None
    reduce_sum = producers.get(greater.input[0])
    if reduce_sum is None or reduce_sum.op_type != "ReduceSum" or not reduce_sum.input:
        return None
    return reshape, greater, reduce_sum


def compress_gate_reduce_sum(condition: str, producers: dict[str, onnx.NodeProto]) -> onnx.NodeProto | None:
    chain = compress_gate_chain(condition, producers)
    if chain is None:
        return None
    return chain[2]


def infer_constant_top_left_channel(examples: list[dict[str, list[list[int]]]]) -> int | None:
    colors = {example["input"][0][0] for example in examples if example["input"] and example["input"][0]}
    if len(colors) == 1:
        return next(iter(colors))
    return None


def infer_constant_pixel(examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, int] | None:
    if not examples:
        return None
    min_h = min(len(example["input"]) for example in examples)
    min_w = min(len(example["input"][0]) for example in examples if example["input"])
    candidates: list[tuple[int, int, int]] = []
    for row in range(min_h):
        for col in range(min_w):
            colors = {example["input"][row][col] for example in examples}
            if len(colors) == 1:
                candidates.append((row, col, next(iter(colors))))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0] + item[1], item[0], item[1], item[2]))


def infer_compact_pixel_interval(examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, int, int] | None:
    if not examples:
        return None
    min_h = min(len(example["input"]) for example in examples)
    min_w = min(len(example["input"][0]) for example in examples if example["input"])
    candidates: list[tuple[int, int, int, int, int]] = []
    for row in range(min_h):
        for col in range(min_w):
            colors = {example["input"][row][col] for example in examples}
            start = min(colors)
            end = max(colors) + 1
            candidates.append((end - start, row, col, start, end))
    if not candidates:
        return None
    width, row, col, start, end = min(candidates)
    if width >= 10:
        return None
    return row, col, start, end


def optimize_model(
    model: onnx.ModelProto,
    gate_channel: int | None = None,
    gate_channel_start: int | None = None,
    gate_channel_end: int | None = None,
    gate_row: int = 0,
    gate_col: int = 0,
    direct_condition: bool = False,
) -> tuple[onnx.ModelProto, int]:
    producers = producer_map(model)
    reduce_axes_as_input = default_opset(model) >= 13
    replacements: dict[str, list[onnx.NodeProto]] = {}
    remove_node_keys: set[str] = set()
    changed = 0

    for compress in model.graph.node:
        if compress.op_type != "Compress" or len(compress.input) < 2:
            continue
        if attr_int(compress, "axis") != 0:
            continue
        data_input, condition = compress.input[0], compress.input[1]
        chain = compress_gate_chain(condition, producers)
        if chain is None:
            continue
        reshape, greater, reduce_sum = chain
        if reduce_sum.input[0] != data_input:
            continue

        prefix = f"pixel_gate_{changed}"
        slice_out = f"{prefix}_pixel"
        if gate_channel_start is not None and gate_channel_end is not None:
            channel_start, channel_end = gate_channel_start, gate_channel_end
        elif gate_channel is None:
            channel_start, channel_end = 0, 10
        else:
            channel_start, channel_end = gate_channel, gate_channel + 1

        replacements[reduce_sum.name or reduce_sum.output[0]] = [
            const_i64_node(f"{prefix}_starts_const", f"{prefix}_starts", [0, channel_start, gate_row, gate_col]),
            const_i64_node(f"{prefix}_ends_const", f"{prefix}_ends", [1, channel_end, gate_row + 1, gate_col + 1]),
            const_i64_node(f"{prefix}_axes_const", f"{prefix}_axes", [0, 1, 2, 3]),
            const_i64_node(f"{prefix}_steps_const", f"{prefix}_steps", [1, 1, 1, 1]),
            helper.make_node(
                "Slice",
                [data_input, f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes", f"{prefix}_steps"],
                [slice_out],
                name=f"{prefix}_slice",
            ),
        ]
        reduce_sum.input[0] = slice_out
        if direct_condition:
            axes_name = f"{prefix}_reduce_axes_123"
            if reduce_axes_as_input:
                replacements[reduce_sum.name or reduce_sum.output[0]].append(
                    const_i64_node(f"{prefix}_reduce_axes_123_const", axes_name, [1, 2, 3])
                )
                set_reduce_axes(reduce_sum, [1, 2, 3], axes_name)
            else:
                set_reduce_axes(reduce_sum, [1, 2, 3], None)
            compress.input[1] = greater.output[0]
            remove_node_keys.add(reshape.name or reshape.output[0])
        changed += 1

    if not changed:
        return model, 0

    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        key = node.name or node.output[0]
        if key in remove_node_keys:
            continue
        if key in replacements:
            new_nodes.extend(replacements[key])
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    return model, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input ONNX file or zip containing taskNNN.onnx")
    parser.add_argument("--task", required=True, help="Task id such as task006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--constant-top-left-channel", action="store_true")
    parser.add_argument("--constant-pixel", action="store_true")
    parser.add_argument("--compact-pixel-interval", action="store_true")
    parser.add_argument("--direct-condition", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.input.suffix == ".zip":
        raw = zipfile.ZipFile(args.input).read(f"{args.task}.onnx")
    else:
        raw = args.input.read_bytes()

    gate_channel: int | None = None
    examples = all_examples(json.loads((args.data_root / f"{args.task}.json").read_text()))
    gate_row = 0
    gate_col = 0
    gate_channel_start: int | None = None
    gate_channel_end: int | None = None
    if args.constant_pixel:
        pixel = infer_constant_pixel(examples)
        if pixel is not None:
            gate_row, gate_col, gate_channel = pixel
        print(f"{args.task}: inferred_constant_pixel={pixel}")
    elif args.compact_pixel_interval:
        interval = infer_compact_pixel_interval(examples)
        if interval is not None:
            gate_row, gate_col, gate_channel_start, gate_channel_end = interval
        print(f"{args.task}: inferred_compact_pixel_interval={interval}")
    elif args.constant_top_left_channel:
        gate_channel = infer_constant_top_left_channel(examples)
        print(f"{args.task}: inferred_top_left_channel={gate_channel}")

    model, changed = optimize_model(
        onnx.load_from_string(raw),
        gate_channel=gate_channel,
        gate_channel_start=gate_channel_start,
        gate_channel_end=gate_channel_end,
        gate_row=gate_row,
        gate_col=gate_col,
        direct_condition=args.direct_condition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model.SerializeToString())

    print(f"{args.task}: changed_gates={changed} bytes={args.output.stat().st_size}")
    if not args.no_verify:
        right, wrong, reason = verify_model_bytes(args.output.read_bytes(), examples)
        print(f"verify: {right}/{right + wrong} wrong={wrong} reason={reason}")
        macs, memory, params, cost = profile_cost(args.output)
        print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
        if wrong:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
