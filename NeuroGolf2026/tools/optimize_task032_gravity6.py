#!/usr/bin/env python3
"""Build a compact clean ONNX model for NeuroGolf task032.

Task032 applies column-wise gravity to non-background cells on a square grid of
size 4, 5, or 6.  The ONNX graph works on the static 6x6 crop, infers the valid
height from the one-hot background channel, and enumerates source/target row
placements with count-below tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
SMALL = 6


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axis_height", [2]),
        const_i64("axis_width", [3]),
        const_i64("axis_height_width", [2, 3]),
        const_f32("zero_scalar", 0.0),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 24, 24]),
        *slice_4d("small", [0, 0, 0, 0], [1, 10, SMALL, SMALL]),
        *slice_4d("small_colors", [0, 1, 0, 0], [1, 10, SMALL, SMALL]),
    ]
    for i in range(SMALL):
        inits.extend(slice_4d(f"row{i}", [0, 0, i, 0], [1, 9, i + 1, SMALL]))
        inits.extend(slice_4d(f"below{i}", [0, 0, i, 0], [1, 1, SMALL, SMALL]))
    for t in range(SMALL):
        inits.append(const_f32(f"target{t}", float(t)))
        inits.append(const_i64(f"pad_row{t}", [0, 0, t, 0, 0, 0, SMALL - 1 - t, 0]))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "small", "small"),
        make_slice("input", "colors", "small_colors"),
        helper.make_node("ReduceSum", ["small", "axis_channel"], ["valid"], keepdims=1, name="valid_mask"),
        helper.make_node("ReduceMax", ["valid"], ["valid_rows"], axes=[3], keepdims=1, name="valid_rows"),
        helper.make_node("ReduceSum", ["valid_rows", "axis_height_width"], ["height"], keepdims=1, name="grid_height"),
        helper.make_node("ReduceSum", ["colors", "axis_channel"], ["occupied"], keepdims=1, name="occupied"),
    ]

    placed_terms: list[str] = []
    for i in range(SMALL):
        nodes.append(make_slice("colors", f"row{i}", f"row{i}"))
        nodes.append(make_slice("occupied", f"below{i}", f"below{i}"))
        nodes.append(helper.make_node("ReduceSum", [f"below{i}", "axis_height"], [f"count_below{i}"], keepdims=1, name=f"count_below{i}"))
        for t in range(SMALL):
            nodes.extend(
                [
                    helper.make_node("Sub", ["height", f"target{t}"], [f"needed_i{i}_t{t}"], name=f"needed_i{i}_t{t}"),
                    helper.make_node("Equal", [f"count_below{i}", f"needed_i{i}_t{t}"], [f"cond_i{i}_t{t}"], name=f"cond_i{i}_t{t}"),
                    helper.make_node("Cast", [f"cond_i{i}_t{t}"], [f"cond_f_i{i}_t{t}"], to=TensorProto.FLOAT, name=f"cond_cast_i{i}_t{t}"),
                    helper.make_node("Mul", [f"row{i}", f"cond_f_i{i}_t{t}"], [f"piece_i{i}_t{t}"], name=f"piece_i{i}_t{t}"),
                    helper.make_node("Pad", [f"piece_i{i}_t{t}", f"pad_row{t}", "zero_scalar"], [f"placed_i{i}_t{t}"], name=f"placed_i{i}_t{t}"),
                ]
            )
            placed_terms.append(f"placed_i{i}_t{t}")

    nodes.extend(
        [
            helper.make_node("Sum", placed_terms, ["out_colors_raw"], name="sum_placed_colors"),
            helper.make_node("Greater", ["out_colors_raw", "zero_scalar"], ["out_colors_bool"], name="clip_colors_bool"),
            helper.make_node("Cast", ["out_colors_bool"], ["out_colors"], to=TensorProto.FLOAT, name="clip_colors"),
            helper.make_node("ReduceSum", ["out_colors", "axis_channel"], ["out_occupied"], keepdims=1, name="out_occupied"),
            helper.make_node("Sub", ["valid", "out_occupied"], ["background"], name="background_channel"),
            helper.make_node("Concat", ["background", "out_colors"], ["out6"], axis=1, name="assemble_out6"),
            helper.make_node("Pad", ["out6", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task032_gravity6",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task032_gravity6",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task032_gravity6_v103"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task032.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task032 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task032.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task032.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task032 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
