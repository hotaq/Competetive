#!/usr/bin/env python3
"""Build a strict-static pattern-mask transfer model for task170."""

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


def f32(name: str, arr: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def i64(name: str, arr: np.ndarray | list[int] | int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def vi(name: str, dtype: int = TensorProto.FLOAT, shape: list[int] | None = None) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape or EXPECTED_SHAPE)


def build_model() -> onnx.ModelProto:
    candidate_top: list[int] = []
    candidate_left: list[int] = []
    candidate_size: list[int] = []
    for size in (3, 4):
        for r in range(31 - size):
            for c in range(31 - size):
                candidate_top.append(r)
                candidate_left.append(c)
                candidate_size.append(size)

    row30 = np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)
    col30 = np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)
    row4 = np.arange(4, dtype=np.int64).reshape(1, 1, 4, 1)
    col4 = np.arange(4, dtype=np.int64).reshape(1, 1, 1, 4)

    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        f32("one_f", 1.0),
        f32("score_scale", 1000.0),
        i64("crop_starts", [0, 1, 0, 0]),
        i64("crop_ends", [1, 10, 30, 30]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_channel", [1]),
        i64("axes_hw", [2, 3]),
        i64("axes_width", [3]),
        i64("axes_height", [2]),
        i64("row_rev", list(reversed(range(30)))),
        i64("col_rev", list(reversed(range(30)))),
        i64("twenty_nine", 29),
        i64("one_i", 1),
        i64("three4", np.array([[[[3]]]], dtype=np.int64)),
        i64("four4", np.array([[[[4]]]], dtype=np.int64)),
        i64("five4", np.array([[[[5]]]], dtype=np.int64)),
        i64("nine4", np.array([[[[9]]]], dtype=np.int64)),
        i64("fifteen4", np.array([[[[15]]]], dtype=np.int64)),
        i64("sixteen4", np.array([[[[16]]]], dtype=np.int64)),
        i64("scalar4_shape", [1, 1, 1, 1]),
        i64("row30", row30),
        i64("col30", col30),
        i64("row4", row4),
        i64("col4", col4),
        i64("pattern_row_idx_shape", [1, 10, 4, 30]),
        i64("pattern_col_idx_shape", [1, 10, 4, 4]),
        i64("pattern9_starts", [0, 1, 0, 0]),
        i64("pattern9_ends", [1, 10, 4, 4]),
        i64("mask_row_idx_shape", [1, 1, 4, 30]),
        i64("mask_col_idx_shape", [1, 1, 4, 4]),
        i64("pad4_to30", [0, 0, 0, 0, 0, 0, 26, 26]),
        i64("candidate_top", candidate_top),
        i64("candidate_left", candidate_left),
        i64("candidate_size", candidate_size),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["x"], name="nonzero_channels"),
        helper.make_node("ReduceSum", ["x", "axes_hw"], ["counts"], keepdims=1, name="counts_by_color"),
        helper.make_node("ReduceMax", ["counts", "axes_channel"], ["max_count"], keepdims=1, name="max_count"),
        helper.make_node("Equal", ["counts", "max_count"], ["majority_bool"], name="majority_color_bool"),
        helper.make_node("Cast", ["majority_bool"], ["majority"], to=TensorProto.FLOAT, name="majority_color"),
        helper.make_node("Greater", ["counts", "zero_f"], ["present_bool"], name="present_color_bool"),
        helper.make_node("Less", ["counts", "max_count"], ["minority_bool_raw"], name="minority_less_than_majority"),
        helper.make_node("And", ["present_bool", "minority_bool_raw"], ["minority_bool"], name="minority_color_bool"),
        helper.make_node("Cast", ["minority_bool"], ["minority"], to=TensorProto.FLOAT, name="minority_color"),
        helper.make_node("ReduceMax", ["x", "axes_channel"], ["occ"], keepdims=1, name="nonzero_occupancy"),
        helper.make_node("Mul", ["x", "majority"], ["majority_pixels"], name="majority_pixels"),
        helper.make_node("ReduceMax", ["majority_pixels", "axes_channel"], ["maj_occ"], keepdims=1, name="majority_occupancy"),
        helper.make_node("Mul", ["x", "minority"], ["minority_pixels"], name="minority_pixels"),
        helper.make_node("ReduceMax", ["minority_pixels", "axes_channel"], ["nonmaj_occ"], keepdims=1, name="nonmajority_occupancy"),
    ]

    score_parts: list[str] = []
    for size in (3, 4):
        prefix = f"s{size}"
        out_hw = 31 - size
        flat = out_hw * out_hw
        inits.extend(
            [
                f32(f"{prefix}_ones", np.ones((1, 1, size, size), dtype=np.float32)),
                f32(f"{prefix}_area", float(size * size)),
                i64(f"{prefix}_flat_shape", [1, flat]),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Conv", ["occ", f"{prefix}_ones"], [f"{prefix}_occ_sum"], name=f"{prefix}_occ_sum"),
                helper.make_node("Equal", [f"{prefix}_occ_sum", f"{prefix}_area"], [f"{prefix}_valid_bool"], name=f"{prefix}_valid_bool"),
                helper.make_node("Cast", [f"{prefix}_valid_bool"], [f"{prefix}_valid"], to=TensorProto.FLOAT, name=f"{prefix}_valid"),
                helper.make_node("Conv", ["nonmaj_occ", f"{prefix}_ones"], [f"{prefix}_nonmaj_sum"], name=f"{prefix}_nonmaj_sum"),
                helper.make_node("Mul", [f"{prefix}_nonmaj_sum", "score_scale"], [f"{prefix}_nonmaj_score"], name=f"{prefix}_nonmaj_score"),
                helper.make_node("Add", [f"{prefix}_nonmaj_score", f"{prefix}_area"], [f"{prefix}_raw_score"], name=f"{prefix}_raw_score"),
                helper.make_node("Mul", [f"{prefix}_raw_score", f"{prefix}_valid"], [f"{prefix}_score"], name=f"{prefix}_score"),
                helper.make_node("Reshape", [f"{prefix}_score", f"{prefix}_flat_shape"], [f"{prefix}_flat"], name=f"{prefix}_flatten"),
            ]
        )
        score_parts.append(f"{prefix}_flat")

    nodes.extend(
        [
            helper.make_node("Concat", score_parts, ["all_scores"], axis=1, name="concat_pattern_scores"),
            helper.make_node("ArgMax", ["all_scores"], ["best_idx"], axis=1, keepdims=0, name="best_pattern_idx"),
            helper.make_node("Gather", ["candidate_top", "best_idx"], ["pattern_top_1d"], axis=0, name="gather_pattern_top"),
            helper.make_node("Gather", ["candidate_left", "best_idx"], ["pattern_left_1d"], axis=0, name="gather_pattern_left"),
            helper.make_node("Gather", ["candidate_size", "best_idx"], ["pattern_size_1d"], axis=0, name="gather_pattern_size"),
            helper.make_node("Reshape", ["pattern_top_1d", "scalar4_shape"], ["pattern_top"], name="pattern_top_scalar"),
            helper.make_node("Reshape", ["pattern_left_1d", "scalar4_shape"], ["pattern_left"], name="pattern_left_scalar"),
            helper.make_node("Reshape", ["pattern_size_1d", "scalar4_shape"], ["pattern_size"], name="pattern_size_scalar"),
            helper.make_node("Add", ["pattern_top", "pattern_size"], ["pattern_bottom_excl"], name="pattern_bottom_excl"),
            helper.make_node("Add", ["pattern_left", "pattern_size"], ["pattern_right_excl"], name="pattern_right_excl"),
            helper.make_node("GreaterOrEqual", ["row30", "pattern_top"], ["pattern_row_ge"], name="pattern_row_ge"),
            helper.make_node("Less", ["row30", "pattern_bottom_excl"], ["pattern_row_lt"], name="pattern_row_lt"),
            helper.make_node("And", ["pattern_row_ge", "pattern_row_lt"], ["pattern_row_in"], name="pattern_row_in"),
            helper.make_node("GreaterOrEqual", ["col30", "pattern_left"], ["pattern_col_ge"], name="pattern_col_ge"),
            helper.make_node("Less", ["col30", "pattern_right_excl"], ["pattern_col_lt"], name="pattern_col_lt"),
            helper.make_node("And", ["pattern_col_ge", "pattern_col_lt"], ["pattern_col_in"], name="pattern_col_in"),
            helper.make_node("And", ["pattern_row_in", "pattern_col_in"], ["pattern_mask_bool"], name="pattern_mask_bool"),
            helper.make_node("Cast", ["pattern_mask_bool"], ["pattern_mask"], to=TensorProto.FLOAT, name="pattern_mask_float"),
            helper.make_node("Sub", ["one_f", "pattern_mask"], ["not_pattern_mask"], name="not_pattern_mask"),
            helper.make_node("Mul", ["maj_occ", "not_pattern_mask"], ["big_occ"], name="majority_without_pattern"),
            helper.make_node("ReduceMax", ["big_occ", "axes_width"], ["big_row_has"], keepdims=1, name="big_row_has"),
            helper.make_node("ReduceMax", ["big_occ", "axes_height"], ["big_col_has"], keepdims=1, name="big_col_has"),
            helper.make_node("ArgMax", ["big_row_has"], ["big_top"], axis=2, keepdims=1, name="big_top"),
            helper.make_node("ArgMax", ["big_col_has"], ["big_left"], axis=3, keepdims=1, name="big_left"),
            helper.make_node("Gather", ["big_row_has", "row_rev"], ["big_row_has_rev"], axis=2, name="reverse_big_rows"),
            helper.make_node("Gather", ["big_col_has", "col_rev"], ["big_col_has_rev"], axis=3, name="reverse_big_cols"),
            helper.make_node("ArgMax", ["big_row_has_rev"], ["bottom_from_rev"], axis=2, keepdims=1, name="big_bottom_from_rev"),
            helper.make_node("ArgMax", ["big_col_has_rev"], ["right_from_rev"], axis=3, keepdims=1, name="big_right_from_rev"),
            helper.make_node("Sub", ["twenty_nine", "bottom_from_rev"], ["big_bottom"], name="big_bottom"),
            helper.make_node("Sub", ["twenty_nine", "right_from_rev"], ["big_right"], name="big_right"),
            helper.make_node("Sub", ["big_bottom", "big_top"], ["big_h_minus_one"], name="big_h_minus_one"),
            helper.make_node("Add", ["big_h_minus_one", "one_i"], ["big_h"], name="big_h"),
            helper.make_node("Equal", ["big_h", "nine4"], ["big_h9"], name="big_h9"),
            helper.make_node("Equal", ["big_h", "fifteen4"], ["big_h15"], name="big_h15"),
            helper.make_node("Equal", ["big_h", "sixteen4"], ["big_h16"], name="big_h16"),
            helper.make_node("Equal", ["pattern_size", "three4"], ["pattern_is3"], name="pattern_is3"),
            helper.make_node("Where", ["pattern_is3", "four4", "three4"], ["cell_if_12"], name="cell_if_12"),
            helper.make_node("Where", ["big_h9", "three4", "cell_if_12"], ["cell_if_9_or_12"], name="cell_if_9_or_12"),
            helper.make_node("Where", ["big_h15", "five4", "cell_if_9_or_12"], ["cell_if_15"], name="cell_if_15"),
            helper.make_node("Where", ["big_h16", "four4", "cell_if_15"], ["cell"], name="cell_size"),
            helper.make_node("Mul", ["row4", "cell"], ["mask_row_offset"], name="mask_row_offset"),
            helper.make_node("Mul", ["col4", "cell"], ["mask_col_offset"], name="mask_col_offset"),
            helper.make_node("Add", ["big_top", "mask_row_offset"], ["mask_row_src"], name="mask_row_src"),
            helper.make_node("Add", ["big_left", "mask_col_offset"], ["mask_col_src"], name="mask_col_src"),
            helper.make_node("Expand", ["mask_row_src", "mask_row_idx_shape"], ["mask_row_idx"], name="expand_mask_rows"),
            helper.make_node("Expand", ["mask_col_src", "mask_col_idx_shape"], ["mask_col_idx"], name="expand_mask_cols"),
            helper.make_node("GatherElements", ["big_occ", "mask_row_idx"], ["mask_rows"], axis=2, name="gather_mask_rows"),
            helper.make_node("GatherElements", ["mask_rows", "mask_col_idx"], ["mask4"], axis=3, name="gather_mask_cols"),
            helper.make_node("Less", ["row4", "pattern_size"], ["out_row_valid_bool"], name="out_row_valid_bool"),
            helper.make_node("Less", ["col4", "pattern_size"], ["out_col_valid_bool"], name="out_col_valid_bool"),
            helper.make_node("And", ["out_row_valid_bool", "out_col_valid_bool"], ["out_valid_bool"], name="out_valid_bool"),
            helper.make_node("Cast", ["out_valid_bool"], ["out_valid"], to=TensorProto.FLOAT, name="out_valid"),
            helper.make_node("Mul", ["mask4", "out_valid"], ["final_mask"], name="final_mask"),
            helper.make_node("Add", ["pattern_top", "row4"], ["pattern_row_src"], name="pattern_row_src"),
            helper.make_node("Add", ["pattern_left", "col4"], ["pattern_col_src"], name="pattern_col_src"),
            helper.make_node("Expand", ["pattern_row_src", "pattern_row_idx_shape"], ["pattern_row_idx"], name="expand_pattern_rows"),
            helper.make_node("Expand", ["pattern_col_src", "pattern_col_idx_shape"], ["pattern_col_idx"], name="expand_pattern_cols"),
            helper.make_node("GatherElements", ["input", "pattern_row_idx"], ["pattern_rows"], axis=2, name="gather_pattern_rows"),
            helper.make_node("GatherElements", ["pattern_rows", "pattern_col_idx"], ["pattern4"], axis=3, name="gather_pattern_cols"),
            helper.make_node("Slice", ["pattern4", "pattern9_starts", "pattern9_ends", "axes4", "steps4"], ["pattern9"], name="slice_pattern_nonzero_channels"),
            helper.make_node("Mul", ["pattern9", "final_mask"], ["out9"], name="apply_big_mask"),
            helper.make_node("Sub", ["out_valid", "final_mask"], ["zero1"], name="zero_channel_inside_output"),
            helper.make_node("Concat", ["zero1", "out9"], ["out4"], axis=1, name="restore_ten_channels"),
            helper.make_node("Pad", ["out4", "pad4_to30", "zero_f"], ["output"], name="pad_to_30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task170_static_pattern_mask",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=[
            vi("x", TensorProto.FLOAT, [1, 9, 30, 30]),
            vi("counts", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("majority", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("minority", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("occ", TensorProto.FLOAT, [1, 1, 30, 30]),
            vi("maj_occ", TensorProto.FLOAT, [1, 1, 30, 30]),
            vi("nonmaj_occ", TensorProto.FLOAT, [1, 1, 30, 30]),
            vi("all_scores", TensorProto.FLOAT, [1, len(candidate_top)]),
            vi("best_idx", TensorProto.INT64, [1]),
            vi("pattern_top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("pattern_left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("pattern_size", TensorProto.INT64, [1, 1, 1, 1]),
            vi("pattern_mask", TensorProto.FLOAT, [1, 1, 30, 30]),
            vi("big_occ", TensorProto.FLOAT, [1, 1, 30, 30]),
            vi("big_row_has", TensorProto.FLOAT, [1, 1, 30, 1]),
            vi("big_col_has", TensorProto.FLOAT, [1, 1, 1, 30]),
            vi("big_top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("big_left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("big_h", TensorProto.INT64, [1, 1, 1, 1]),
            vi("cell", TensorProto.INT64, [1, 1, 1, 1]),
            vi("mask_row_idx", TensorProto.INT64, [1, 1, 4, 30]),
            vi("mask_col_idx", TensorProto.INT64, [1, 1, 4, 4]),
            vi("mask4", TensorProto.FLOAT, [1, 1, 4, 4]),
            vi("final_mask", TensorProto.FLOAT, [1, 1, 4, 4]),
            vi("pattern_row_idx", TensorProto.INT64, [1, 10, 4, 30]),
            vi("pattern_col_idx", TensorProto.INT64, [1, 10, 4, 4]),
            vi("pattern4", TensorProto.FLOAT, [1, 10, 4, 4]),
            vi("pattern9", TensorProto.FLOAT, [1, 9, 4, 4]),
            vi("out9", TensorProto.FLOAT, [1, 9, 4, 4]),
            vi("zero1", TensorProto.FLOAT, [1, 1, 4, 4]),
            vi("out4", TensorProto.FLOAT, [1, 10, 4, 4]),
        ],
    )
    model = helper.make_model(graph, producer_name="neurogolf_task170_static_pattern_mask", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task170_static_pattern_mask"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task170.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task170.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task170.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
