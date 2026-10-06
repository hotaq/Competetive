#!/usr/bin/env python3
"""Build a strict-static rectangle-bank model for task396."""

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


def border_kernel(h: int, w: int) -> np.ndarray:
    k = np.zeros((1, 1, h, w), dtype=np.float32)
    k[:, :, 0, :] = 1.0
    k[:, :, h - 1, :] = 1.0
    k[:, :, :, 0] = 1.0
    k[:, :, :, w - 1] = 1.0
    return k


def build_model() -> onnx.ModelProto:
    row_base = np.arange(8, dtype=np.int64).reshape(1, 1, 8, 1)
    col_base = np.arange(8, dtype=np.int64).reshape(1, 1, 1, 8)

    candidate_top: list[int] = []
    candidate_left: list[int] = []
    candidate_h: list[int] = []
    candidate_w: list[int] = []
    for h in range(3, 9):
        for w in range(3, 9):
            for r in range(19 - h):
                for c in range(19 - w):
                    candidate_top.append(r)
                    candidate_left.append(c)
                    candidate_h.append(h)
                    candidate_w.append(w)

    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        f32("score_scale", 1000.0),
        i64("crop_starts", [0, 1, 0, 0]),
        i64("crop_ends", [1, 10, 18, 18]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_channel", [1]),
        i64("axes_hw", [2, 3]),
        i64("zero_i", 0),
        i64("one_i", 1),
        i64("seventeen_i", 17),
        i64("row_base", row_base),
        i64("col_base", col_base),
        i64("row_idx_shape", [1, 1, 8, 18]),
        i64("col_idx_shape", [1, 1, 8, 8]),
        i64("scalar4_shape", [1, 1, 1, 1]),
        i64("pad8_to30", [0, 0, 0, 0, 0, 0, 22, 22]),
        i64("candidate_top", candidate_top),
        i64("candidate_left", candidate_left),
        i64("candidate_h", candidate_h),
        i64("candidate_w", candidate_w),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["x"], name="crop_nonzero_channels_18"),
        helper.make_node("ReduceSum", ["x", "axes_hw"], ["counts"], keepdims=1, name="counts_by_color"),
        helper.make_node("ReduceMax", ["counts", "axes_channel"], ["max_count"], keepdims=1, name="max_count"),
        helper.make_node("Greater", ["counts", "zero_f"], ["present_bool"], name="present"),
        helper.make_node("Less", ["counts", "max_count"], ["minority_bool_raw"], name="count_less_than_max"),
        helper.make_node("And", ["present_bool", "minority_bool_raw"], ["minority_bool"], name="minority_color_bool"),
        helper.make_node("Cast", ["minority_bool"], ["minority"], to=TensorProto.FLOAT, name="minority_color"),
        helper.make_node("Equal", ["counts", "max_count"], ["majority_bool"], name="majority_color_bool"),
        helper.make_node("Cast", ["majority_bool"], ["majority"], to=TensorProto.FLOAT, name="majority_color"),
        helper.make_node("Mul", ["x", "majority"], ["majority_pixels"], name="majority_pixels"),
        helper.make_node("ReduceMax", ["majority_pixels", "axes_channel"], ["maj_occ"], keepdims=1, name="majority_occupancy"),
        helper.make_node("Mul", ["x", "minority"], ["minority_pixels"], name="minority_pixels"),
        helper.make_node("ReduceMax", ["minority_pixels", "axes_channel"], ["min_occ"], keepdims=1, name="minority_occupancy"),
        helper.make_node("ReduceMax", ["x", "axes_channel"], ["all_occ"], keepdims=1, name="all_nonzero_occupancy"),
    ]

    score_parts: list[str] = []
    for h in range(3, 9):
        for w in range(3, 9):
            prefix = f"r{h}x{w}"
            out_h = 19 - h
            out_w = 19 - w
            flat = out_h * out_w
            border = 2 * h + 2 * w - 4
            inits.extend(
                [
                    f32(f"{prefix}_border_kernel", border_kernel(h, w)),
                    f32(f"{prefix}_ones_kernel", np.ones((1, 1, h, w), dtype=np.float32)),
                    f32(f"{prefix}_border_count", float(border)),
                    f32(f"{prefix}_area", float(h * w)),
                    i64(f"{prefix}_flat_shape", [1, flat]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Conv", ["maj_occ", f"{prefix}_border_kernel"], [f"{prefix}_border_sum"], name=f"{prefix}_border_sum"),
                    helper.make_node("Equal", [f"{prefix}_border_sum", f"{prefix}_border_count"], [f"{prefix}_valid_bool"], name=f"{prefix}_valid_bool"),
                    helper.make_node("Cast", [f"{prefix}_valid_bool"], [f"{prefix}_valid"], to=TensorProto.FLOAT, name=f"{prefix}_valid"),
                    helper.make_node("Conv", ["min_occ", f"{prefix}_ones_kernel"], [f"{prefix}_minority_count"], name=f"{prefix}_minority_count"),
                    helper.make_node("Mul", [f"{prefix}_minority_count", "score_scale"], [f"{prefix}_minority_score"], name=f"{prefix}_minority_score"),
                    helper.make_node("Add", [f"{prefix}_minority_score", f"{prefix}_area"], [f"{prefix}_raw_score"], name=f"{prefix}_raw_score"),
                    helper.make_node("Mul", [f"{prefix}_raw_score", f"{prefix}_valid"], [f"{prefix}_score"], name=f"{prefix}_score"),
                    helper.make_node("Reshape", [f"{prefix}_score", f"{prefix}_flat_shape"], [f"{prefix}_flat"], name=f"{prefix}_flatten"),
                ]
            )
            score_parts.append(f"{prefix}_flat")

    nodes.extend(
        [
            helper.make_node("Concat", score_parts, ["all_scores"], axis=1, name="concat_scores"),
            helper.make_node("ArgMax", ["all_scores"], ["best_idx"], axis=1, keepdims=0, name="best_candidate"),
            helper.make_node("Gather", ["candidate_top", "best_idx"], ["top_1d"], axis=0, name="gather_top"),
            helper.make_node("Gather", ["candidate_left", "best_idx"], ["left_1d"], axis=0, name="gather_left"),
            helper.make_node("Gather", ["candidate_h", "best_idx"], ["height_1d"], axis=0, name="gather_height"),
            helper.make_node("Gather", ["candidate_w", "best_idx"], ["width_1d"], axis=0, name="gather_width"),
            helper.make_node("Reshape", ["top_1d", "scalar4_shape"], ["top"], name="top_scalar"),
            helper.make_node("Reshape", ["left_1d", "scalar4_shape"], ["left"], name="left_scalar"),
            helper.make_node("Reshape", ["height_1d", "scalar4_shape"], ["height"], name="height_scalar"),
            helper.make_node("Reshape", ["width_1d", "scalar4_shape"], ["width"], name="width_scalar"),
            helper.make_node("Add", ["top", "row_base"], ["row_src_raw"], name="row_src_raw"),
            helper.make_node("Add", ["left", "col_base"], ["col_src_raw"], name="col_src_raw"),
            helper.make_node("Less", ["row_base", "height"], ["row_valid_bool"], name="row_valid_bool"),
            helper.make_node("Less", ["col_base", "width"], ["col_valid_bool"], name="col_valid_bool"),
            helper.make_node("Cast", ["row_valid_bool"], ["row_valid"], to=TensorProto.FLOAT, name="row_valid"),
            helper.make_node("Cast", ["col_valid_bool"], ["col_valid"], to=TensorProto.FLOAT, name="col_valid"),
            helper.make_node("Min", ["row_src_raw", "seventeen_i"], ["row_src_clip"], name="row_clip"),
            helper.make_node("Min", ["col_src_raw", "seventeen_i"], ["col_src_clip"], name="col_clip"),
            helper.make_node("Expand", ["row_src_clip", "row_idx_shape"], ["row_idx"], name="expand_row_idx"),
            helper.make_node("Expand", ["col_src_clip", "col_idx_shape"], ["col_idx"], name="expand_col_idx"),
            helper.make_node("GatherElements", ["all_occ", "row_idx"], ["row_gathered"], axis=2, name="gather_rows"),
            helper.make_node("GatherElements", ["row_gathered", "col_idx"], ["occ8_raw"], axis=3, name="gather_cols"),
            helper.make_node("Mul", ["row_valid", "col_valid"], ["valid"], name="valid_mask"),
            helper.make_node("Mul", ["occ8_raw", "valid"], ["occ8"], name="mask_to_bbox"),
            helper.make_node("Mul", ["occ8", "minority"], ["out9"], name="paint_minority_color"),
            helper.make_node("Sub", ["valid", "occ8"], ["zero1"], name="zero_color_inside_bbox"),
            helper.make_node("Concat", ["zero1", "out9"], ["out8"], axis=1, name="restore_channels"),
            helper.make_node("Pad", ["out8", "pad8_to30", "zero_f"], ["output"], name="pad_to_30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task396_static_rect_bank",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=[
            vi("x", TensorProto.FLOAT, [1, 9, 18, 18]),
            vi("counts", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("minority", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("majority", TensorProto.FLOAT, [1, 9, 1, 1]),
            vi("maj_occ", TensorProto.FLOAT, [1, 1, 18, 18]),
            vi("min_occ", TensorProto.FLOAT, [1, 1, 18, 18]),
            vi("all_occ", TensorProto.FLOAT, [1, 1, 18, 18]),
            vi("all_scores", TensorProto.FLOAT, [1, len(candidate_top)]),
            vi("best_idx", TensorProto.INT64, [1]),
            vi("top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("height", TensorProto.INT64, [1, 1, 1, 1]),
            vi("width", TensorProto.INT64, [1, 1, 1, 1]),
            vi("row_idx", TensorProto.INT64, [1, 1, 8, 18]),
            vi("col_idx", TensorProto.INT64, [1, 1, 8, 8]),
            vi("row_gathered", TensorProto.FLOAT, [1, 1, 8, 18]),
            vi("occ8", TensorProto.FLOAT, [1, 1, 8, 8]),
            vi("out9", TensorProto.FLOAT, [1, 9, 8, 8]),
            vi("out8", TensorProto.FLOAT, [1, 10, 8, 8]),
        ],
    )
    model = helper.make_model(graph, producer_name="neurogolf_task396_static_rect_bank", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task396_static_rect_bank"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task396.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task396.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task396.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
