#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task190.

The task contains same-color 2x2 blocks with diagonal singleton endpoints.
Each singleton endpoint is extended away from the block along its diagonal.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
DIRECTIONS = ("n", "s", "e", "w", "nw", "ne", "sw", "se")


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def shift_spec(direction: str, k: int) -> tuple[list[int], list[int], list[int]]:
    if direction == "n":
        return [0, 0, k, 0], [1, 1, 10, 10], [0, 0, 0, 0, 0, 0, k, 0]
    if direction == "s":
        return [0, 0, 0, 0], [1, 1, 10 - k, 10], [0, 0, k, 0, 0, 0, 0, 0]
    if direction == "e":
        return [0, 0, 0, 0], [1, 1, 10, 10 - k], [0, 0, 0, k, 0, 0, 0, 0]
    if direction == "w":
        return [0, 0, 0, k], [1, 1, 10, 10], [0, 0, 0, 0, 0, 0, 0, k]
    if direction == "ne":
        return [0, 0, k, 0], [1, 1, 10, 10 - k], [0, 0, 0, k, 0, 0, k, 0]
    if direction == "nw":
        return [0, 0, k, k], [1, 1, 10, 10], [0, 0, 0, 0, 0, 0, k, k]
    if direction == "se":
        return [0, 0, 0, 0], [1, 1, 10 - k, 10 - k], [0, 0, k, k, 0, 0, 0, 0]
    if direction == "sw":
        return [0, 0, 0, k], [1, 1, 10 - k, 10], [0, 0, k, 0, 0, 0, 0, k]
    raise ValueError(direction)


def shift_nodes(src: str, dst: str, direction: str, k: int) -> list[onnx.NodeProto]:
    prefix = f"shift_{direction}_{k}"
    return [
        make_slice(src, f"{dst}_part", prefix),
        helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
    ]


def mul_chain(nodes: list[onnx.NodeProto], names: list[str], out: str, prefix: str) -> None:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        nxt = out if idx == len(names) - 1 else f"{prefix}_mul{idx}"
        nodes.append(helper.make_node("Mul", [cur, name], [nxt], name=f"{prefix}_mul_{idx}"))
        cur = nxt


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones10", np.ones((1, 1, 10, 10), dtype=np.float32)),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        *slice_4d("small", [0, 0, 0, 0], [1, 10, 10, 10]),
    ]
    for color in range(1, 10):
        inits.extend(slice_4d(f"in_ch{color}", [0, color, 0, 0], [1, color + 1, 10, 10]))
        inits.extend(slice_4d(f"out_ch{color}", [0, color, 0, 0], [1, color + 1, 10, 10]))
    for direction in DIRECTIONS:
        for k in range(1, 10):
            starts, ends, pads = shift_spec(direction, k)
            inits.extend(slice_4d(f"shift_{direction}_{k}", starts, ends))
            inits.append(const_i64(f"shift_{direction}_{k}_pad", pads))

    nodes: list[onnx.NodeProto] = [make_slice("input", "small", "small")]
    output_channels: list[str] = []
    for color in range(1, 10):
        ch = f"in{color}"
        nodes.append(make_slice("small", ch, f"in_ch{color}"))
        neigh: dict[str, str] = {}
        for direction in DIRECTIONS:
            out = f"c{color}_{direction}"
            nodes.extend(shift_nodes(ch, out, direction, 1))
            neigh[direction] = out
        nodes.extend(
            [
                helper.make_node("Sum", [neigh["n"], neigh["s"], neigh["e"], neigh["w"]], [f"c{color}_card"], name=f"c{color}_card"),
                helper.make_node("Less", [f"c{color}_card", "half"], [f"c{color}_no_card_bool"], name=f"c{color}_no_card_bool"),
                helper.make_node("Cast", [f"c{color}_no_card_bool"], [f"c{color}_no_card"], to=TensorProto.FLOAT, name=f"c{color}_no_card"),
            ]
        )
        seed_pairs = [
            ("ne", "ne"),
            ("se", "se"),
            ("sw", "sw"),
            ("nw", "nw"),
        ]
        ray_terms = [ch]
        for out_dir, neighbor_dir in seed_pairs:
            seed = f"c{color}_seed_{out_dir}"
            mul_chain(nodes, [ch, f"c{color}_no_card", neigh[neighbor_dir]], seed, seed)
            for k in range(1, 10):
                shifted = f"c{color}_{out_dir}_{k}"
                nodes.extend(shift_nodes(seed, shifted, out_dir, k))
                ray_terms.append(shifted)
        nodes.extend(
            [
                helper.make_node("Sum", ray_terms, [f"c{color}_raw"], name=f"c{color}_raw"),
                helper.make_node("Greater", [f"c{color}_raw", "half"], [f"c{color}_bool"], name=f"c{color}_bool"),
                helper.make_node("Cast", [f"c{color}_bool"], [f"out{color}"], to=TensorProto.FLOAT, name=f"out{color}_cast"),
            ]
        )
        output_channels.append(f"out{color}")

    nodes.extend(
        [
            helper.make_node("Sum", output_channels, ["occupied"], name="occupied_colors"),
            helper.make_node("Sub", ["ones10", "occupied"], ["ch0"], name="background"),
        ]
    )
    channels = ["ch0"] + output_channels
    nodes.extend(
        [
            helper.make_node("Concat", channels, ["small_out"], axis=1, name="assemble_10x10"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task190_diagonal_rays",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task190_diagonal_rays",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task190_diagonal_rays")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task190.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task190 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task190.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task190.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
