#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task013.

Two colored edge seeds define alternating row or column stripes.  The input
background channel carries the true grid extent, so generated stripes are
masked back to the original variable-size grid inside the 30x30 contract.
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
GAPS = range(2, 7)


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_init(prefix: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{prefix}_starts", starts), const_i64(f"{prefix}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def add_col_shift_inits(inits: list[onnx.TensorProto], prefix: str, offset: int, channels: int) -> None:
    if offset >= 0:
        starts = [0, 0, 0, 0]
        ends = [1, channels, 1, 30 - offset]
        pads = [0, 0, 0, offset, 0, 0, 0, 0]
    else:
        k = -offset
        starts = [0, 0, 0, k]
        ends = [1, channels, 1, 30]
        pads = [0, 0, 0, 0, 0, 0, 0, k]
    inits.extend(slice_init(prefix, starts, ends))
    inits.append(const_i64(f"{prefix}_pad", pads))


def add_row_shift_inits(inits: list[onnx.TensorProto], prefix: str, offset: int, channels: int) -> None:
    if offset >= 0:
        starts = [0, 0, 0, 0]
        ends = [1, channels, 30 - offset, 1]
        pads = [0, 0, offset, 0, 0, 0, 0, 0]
    else:
        k = -offset
        starts = [0, 0, k, 0]
        ends = [1, channels, 30, 1]
        pads = [0, 0, 0, 0, 0, 0, k, 0]
    inits.extend(slice_init(prefix, starts, ends))
    inits.append(const_i64(f"{prefix}_pad", pads))


def shift_nodes(src: str, dst: str, prefix: str) -> list[onnx.NodeProto]:
    return [
        make_slice(src, f"{dst}_part", prefix),
        helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
    ]


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("bg_starts", [0, 0, 0, 0]),
        const_i64("bg_ends", [1, 1, 30, 30]),
        const_i64("color_starts", [0, 1, 0, 0]),
        const_i64("color_ends", [1, 10, 30, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axis_h", [2]),
        const_i64("axis_w", [3]),
        const_i64("axes_pair", [1, 2, 3]),
        const_i64("axes_hw", [2, 3]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones", np.ones((1, 1, 30, 30), dtype=np.float32)),
    ]

    # Column/row shifts used for pair detection and repeat-by-2d expansion.
    for d in GAPS:
        add_col_shift_inits(inits, f"col_pair_sample_{d}", -d, 1)
        add_row_shift_inits(inits, f"row_pair_sample_{d}", -d, 1)
        for step in range(0, 30, 2 * d):
            add_col_shift_inits(inits, f"col_repeat_{d}_{step}", step, 9)
            add_row_shift_inits(inits, f"row_repeat_{d}_{step}", step, 9)

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "bg_starts", "bg_ends", "axes4", "steps4"], ["background_in"], name="slice_background"),
        helper.make_node("Slice", ["input", "color_starts", "color_ends", "axes4", "steps4"], ["colors_in"], name="drop_background"),
        helper.make_node("Greater", ["colors_in", "half"], ["seed_bool"], name="seed_bool"),
        helper.make_node("Cast", ["seed_bool"], ["seeds"], to=TensorProto.FLOAT, name="seed_cast"),
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["inside_count"], keepdims=1, name="inside_count"),
        helper.make_node("Greater", ["inside_count", "half"], ["inside_bool"], name="inside_bool"),
        helper.make_node("Cast", ["inside_bool"], ["inside"], to=TensorProto.FLOAT, name="inside_cast"),
        helper.make_node("ReduceSum", ["inside", "axis_w"], ["row_inside_count"], keepdims=1, name="row_inside_count"),
        helper.make_node("Greater", ["row_inside_count", "half"], ["row_inside_bool"], name="row_inside_bool"),
        helper.make_node("Cast", ["row_inside_bool"], ["row_inside"], to=TensorProto.FLOAT, name="row_inside_cast"),
        helper.make_node("ReduceSum", ["inside", "axis_h"], ["col_inside_count"], keepdims=1, name="col_inside_count"),
        helper.make_node("Greater", ["col_inside_count", "half"], ["col_inside_bool"], name="col_inside_bool"),
        helper.make_node("Cast", ["col_inside_bool"], ["col_inside"], to=TensorProto.FLOAT, name="col_inside_cast"),
        helper.make_node("ReduceSum", ["row_inside", "axes_hw"], ["height_count"], keepdims=1, name="height_count"),
        helper.make_node("ReduceSum", ["col_inside", "axes_hw"], ["width_count"], keepdims=1, name="width_count"),
        helper.make_node("Less", ["height_count", "width_count"], ["col_orient_bool"], name="col_orient_bool"),
        helper.make_node("Greater", ["height_count", "width_count"], ["row_orient_bool"], name="row_orient_bool"),
        helper.make_node("Cast", ["col_orient_bool"], ["col_orient"], to=TensorProto.FLOAT, name="col_orient_cast"),
        helper.make_node("Cast", ["row_orient_bool"], ["row_orient"], to=TensorProto.FLOAT, name="row_orient_cast"),
        helper.make_node("ReduceSum", ["seeds", "axis_h"], ["seed_cols"], keepdims=1, name="seed_cols"),
        helper.make_node("ReduceSum", ["seed_cols", "axis_channel"], ["seed_cols_any"], keepdims=1, name="seed_cols_any"),
        helper.make_node("Greater", ["seed_cols_any", "half"], ["seed_cols_any_bool"], name="seed_cols_any_bool"),
        helper.make_node("Cast", ["seed_cols_any_bool"], ["seed_cols_any_f"], to=TensorProto.FLOAT, name="seed_cols_any_cast"),
        helper.make_node("ReduceSum", ["seeds", "axis_w"], ["seed_rows"], keepdims=1, name="seed_rows"),
        helper.make_node("ReduceSum", ["seed_rows", "axis_channel"], ["seed_rows_any"], keepdims=1, name="seed_rows_any"),
        helper.make_node("Greater", ["seed_rows_any", "half"], ["seed_rows_any_bool"], name="seed_rows_any_bool"),
        helper.make_node("Cast", ["seed_rows_any_bool"], ["seed_rows_any_f"], to=TensorProto.FLOAT, name="seed_rows_any_cast"),
    ]

    col_branches: list[str] = []
    row_branches: list[str] = []
    for d in GAPS:
        col_sample = f"col_pair_sample_{d}_out"
        nodes.extend(shift_nodes("seed_cols_any_f", col_sample, f"col_pair_sample_{d}"))
        nodes.append(helper.make_node("Mul", ["seed_cols_any_f", col_sample], [f"col_pair_hits_{d}"], name=f"col_pair_hits_{d}"))
        nodes.append(helper.make_node("ReduceSum", [f"col_pair_hits_{d}", "axes_pair"], [f"col_pair_count_{d}"], keepdims=1, name=f"col_pair_count_{d}"))
        nodes.append(helper.make_node("Greater", [f"col_pair_count_{d}", "half"], [f"col_pair_bool_{d}"], name=f"col_pair_bool_{d}"))
        nodes.append(helper.make_node("Cast", [f"col_pair_bool_{d}"], [f"col_gate_{d}"], to=TensorProto.FLOAT, name=f"col_gate_{d}"))

        repeats: list[str] = []
        for step in range(0, 30, 2 * d):
            shifted = f"col_repeat_{d}_{step}_out"
            nodes.extend(shift_nodes("seed_cols", shifted, f"col_repeat_{d}_{step}"))
            repeats.append(shifted)
        nodes.append(helper.make_node("Sum", repeats, [f"col_repeat_sum_{d}"], name=f"col_repeat_sum_{d}"))
        nodes.append(helper.make_node("Mul", [f"col_repeat_sum_{d}", f"col_gate_{d}"], [f"col_branch_{d}"], name=f"col_branch_{d}"))
        col_branches.append(f"col_branch_{d}")

        row_sample = f"row_pair_sample_{d}_out"
        nodes.extend(shift_nodes("seed_rows_any_f", row_sample, f"row_pair_sample_{d}"))
        nodes.append(helper.make_node("Mul", ["seed_rows_any_f", row_sample], [f"row_pair_hits_{d}"], name=f"row_pair_hits_{d}"))
        nodes.append(helper.make_node("ReduceSum", [f"row_pair_hits_{d}", "axes_pair"], [f"row_pair_count_{d}"], keepdims=1, name=f"row_pair_count_{d}"))
        nodes.append(helper.make_node("Greater", [f"row_pair_count_{d}", "half"], [f"row_pair_bool_{d}"], name=f"row_pair_bool_{d}"))
        nodes.append(helper.make_node("Cast", [f"row_pair_bool_{d}"], [f"row_gate_{d}"], to=TensorProto.FLOAT, name=f"row_gate_{d}"))

        repeats = []
        for step in range(0, 30, 2 * d):
            shifted = f"row_repeat_{d}_{step}_out"
            nodes.extend(shift_nodes("seed_rows", shifted, f"row_repeat_{d}_{step}"))
            repeats.append(shifted)
        nodes.append(helper.make_node("Sum", repeats, [f"row_repeat_sum_{d}"], name=f"row_repeat_sum_{d}"))
        nodes.append(helper.make_node("Mul", [f"row_repeat_sum_{d}", f"row_gate_{d}"], [f"row_branch_{d}"], name=f"row_branch_{d}"))
        row_branches.append(f"row_branch_{d}")

    nodes.extend(
        [
            helper.make_node("Sum", col_branches, ["col_pattern_1x30"], name="col_pattern_1x30"),
            helper.make_node("Mul", ["col_pattern_1x30", "col_orient"], ["col_pattern_oriented"], name="select_col_orientation"),
            helper.make_node("Mul", ["col_pattern_oriented", "row_inside"], ["col_pattern"], name="broadcast_col_pattern"),
            helper.make_node("Sum", row_branches, ["row_pattern_30x1"], name="row_pattern_30x1"),
            helper.make_node("Mul", ["row_pattern_30x1", "row_orient"], ["row_pattern_oriented"], name="select_row_orientation"),
            helper.make_node("Mul", ["row_pattern_oriented", "col_inside"], ["row_pattern"], name="broadcast_row_pattern"),
            helper.make_node("Sum", ["col_pattern", "row_pattern"], ["raw_colors"], name="sum_patterns"),
            helper.make_node("Mul", ["raw_colors", "inside"], ["raw_colors_inside"], name="mask_to_original_grid"),
            helper.make_node("Greater", ["raw_colors_inside", "half"], ["colors_bool"], name="colors_bool"),
            helper.make_node("Cast", ["colors_bool"], ["colors"], to=TensorProto.FLOAT, name="colors_cast"),
            helper.make_node("ReduceSum", ["colors", "axis_channel"], ["occupied_count"], keepdims=1, name="occupied_count"),
            helper.make_node("Greater", ["occupied_count", "half"], ["occupied_bool"], name="occupied_bool"),
            helper.make_node("Cast", ["occupied_bool"], ["occupied"], to=TensorProto.FLOAT, name="occupied_cast"),
            helper.make_node("Sub", ["ones", "occupied"], ["empty_mask"], name="empty_mask"),
            helper.make_node("Mul", ["background_in", "empty_mask"], ["ch0"], name="background_channel"),
            helper.make_node("Concat", ["ch0", "colors"], ["output"], axis=1, name="assemble_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task013_edge_stripes",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task013_edge_stripes",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task013_edge_stripes_v96"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task013.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task013 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task013.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task013.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task013 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
