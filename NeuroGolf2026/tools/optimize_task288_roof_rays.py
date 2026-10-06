#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task288."""

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
SMALL = 9


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


def neighbor_spec(direction: str) -> tuple[list[int], list[int], list[int]]:
    if direction == "above":
        return [0, 0, 0, 0], [1, 1, SMALL - 1, SMALL], [0, 0, 1, 0, 0, 0, 0, 0]
    if direction == "below":
        return [0, 0, 1, 0], [1, 1, SMALL, SMALL], [0, 0, 0, 0, 0, 0, 1, 0]
    if direction == "left":
        return [0, 0, 0, 0], [1, 1, SMALL, SMALL - 1], [0, 0, 0, 1, 0, 0, 0, 0]
    if direction == "right":
        return [0, 0, 0, 1], [1, 1, SMALL, SMALL], [0, 0, 0, 0, 0, 0, 0, 1]
    raise ValueError(direction)


def out_shift_spec(direction: str, k: int) -> tuple[list[int], list[int], list[int]]:
    if direction == "n":
        return [0, 0, k, 0], [1, 1, SMALL, SMALL], [0, 0, 0, 0, 0, 0, k, 0]
    if direction == "nw":
        return [0, 0, k, k], [1, 1, SMALL, SMALL], [0, 0, 0, 0, 0, 0, k, k]
    if direction == "ne":
        return [0, 0, k, 0], [1, 1, SMALL, SMALL - k], [0, 0, 0, k, 0, 0, k, 0]
    raise ValueError(direction)


def shift_nodes(src: str, dst: str, prefix: str) -> list[onnx.NodeProto]:
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
        const_f32("one_scalar", 1.0),
        const_f32("zero_scalar", 0.0),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 21, 21]),
        *slice_4d("small", [0, 0, 0, 0], [1, 10, SMALL, SMALL]),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, SMALL, SMALL]),
    ]
    for color in range(1, 10):
        inits.extend(slice_4d(f"in_ch{color}", [0, color, 0, 0], [1, color + 1, SMALL, SMALL]))
    for direction in ("above", "below", "left", "right"):
        starts, ends, pads = neighbor_spec(direction)
        inits.extend(slice_4d(f"neighbor_{direction}", starts, ends))
        inits.append(const_i64(f"neighbor_{direction}_pad", pads))
    for direction in ("n", "nw", "ne"):
        for k in range(1, SMALL):
            starts, ends, pads = out_shift_spec(direction, k)
            inits.extend(slice_4d(f"shift_{direction}_{k}", starts, ends))
            inits.append(const_i64(f"shift_{direction}_{k}_pad", pads))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "small", "small"),
        make_slice("small", "ch0_in", "ch0"),
        helper.make_node("ReduceSum", ["small", "axis_channel"], ["inside"], keepdims=1, name="inside_shape"),
        helper.make_node("Sub", ["inside", "ch0_in"], ["nonzero"], name="nonzero_cells"),
        *shift_nodes("inside", "below_inside", "neighbor_below"),
        helper.make_node("Sub", ["one_scalar", "below_inside"], ["no_below"], name="no_below"),
        *shift_nodes("nonzero", "above_nonzero", "neighbor_above"),
    ]
    output_channels: list[str] = []
    for color in range(1, 10):
        ch = f"in{color}"
        nodes.append(make_slice("small", ch, f"in_ch{color}"))
        nodes.extend(shift_nodes(ch, f"c{color}_left", "neighbor_left"))
        nodes.extend(shift_nodes(ch, f"c{color}_right", "neighbor_right"))
        nodes.extend(
            [
                helper.make_node("Sub", ["one_scalar", f"c{color}_left"], [f"c{color}_no_left"], name=f"c{color}_no_left"),
                helper.make_node("Sub", ["one_scalar", f"c{color}_right"], [f"c{color}_no_right"], name=f"c{color}_no_right"),
            ]
        )
        mul_chain(nodes, [ch, "no_below", "above_nonzero", f"c{color}_no_left"], f"c{color}_left_end", f"c{color}_left_end")
        mul_chain(nodes, [ch, "no_below", "above_nonzero", f"c{color}_no_right"], f"c{color}_right_end", f"c{color}_right_end")
        nodes.extend(shift_nodes(f"c{color}_left_end", f"c{color}_left_seed", "shift_n_1"))
        nodes.extend(shift_nodes(f"c{color}_right_end", f"c{color}_right_seed", "shift_n_1"))
        ray_terms = [ch]
        for k in range(1, SMALL):
            left_ray = f"c{color}_nw_{k}"
            right_ray = f"c{color}_ne_{k}"
            nodes.extend(shift_nodes(f"c{color}_left_seed", left_ray, f"shift_nw_{k}"))
            nodes.extend(shift_nodes(f"c{color}_right_seed", right_ray, f"shift_ne_{k}"))
            ray_terms.extend([left_ray, right_ray])
        nodes.extend(
            [
                helper.make_node("Sum", ray_terms, [f"c{color}_raw"], name=f"c{color}_raw"),
                helper.make_node("Greater", [f"c{color}_raw", "half"], [f"c{color}_bool"], name=f"c{color}_bool"),
                helper.make_node("Cast", [f"c{color}_bool"], [f"out{color}_unmasked"], to=TensorProto.FLOAT, name=f"out{color}_cast"),
                helper.make_node("Mul", [f"out{color}_unmasked", "inside"], [f"out{color}"], name=f"out{color}_mask_shape"),
            ]
        )
        output_channels.append(f"out{color}")

    nodes.extend(
        [
            helper.make_node("Sum", output_channels, ["occupied"], name="occupied_colors"),
            helper.make_node("Sub", ["inside", "occupied"], ["ch0"], name="background"),
            helper.make_node("Concat", ["ch0", *output_channels], ["small_out"], axis=1, name="assemble_9x9"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task288_roof_rays",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task288_roof_rays",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task288_roof_rays")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task288.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task288 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task288.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task288.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
