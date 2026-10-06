#!/usr/bin/env python3
"""Build a strict-static nonzero-bbox crop + horizontal flip model for task177."""

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
    row_base = np.arange(8, dtype=np.int64).reshape(1, 1, 8, 1)
    col_base = np.arange(8, dtype=np.int64).reshape(1, 1, 1, 8)
    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        i64("crop_starts", [0, 1, 0, 0]),
        i64("crop_ends", [1, 10, 20, 20]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_channel", [1]),
        i64("axes_width", [3]),
        i64("axes_height", [2]),
        i64("first_channel_starts", [0, 0, 0, 0]),
        i64("first_channel_ends", [1, 1, 8, 8]),
        i64("row_base", row_base),
        i64("col_base", col_base),
        i64("col_reverse", list(reversed(range(20)))),
        i64("zero_i", 0),
        i64("one_i", 1),
        i64("nineteen", 19),
        i64("twenty", 20),
        i64("row_idx_shape", [1, 9, 8, 20]),
        i64("col_idx_shape", [1, 9, 8, 8]),
        i64("pad_spatial", [0, 0, 0, 0, 0, 0, 22, 22]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["x"], name="crop_nonzero_channels_20"),
        helper.make_node("ReduceMax", ["x", "axes_channel"], ["occ"], keepdims=1, name="nonzero_occupancy"),
        helper.make_node("ReduceMax", ["occ", "axes_width"], ["row_has"], keepdims=1, name="row_has"),
        helper.make_node("ReduceMax", ["occ", "axes_height"], ["col_has"], keepdims=1, name="col_has"),
        helper.make_node("ArgMax", ["row_has"], ["top"], axis=2, keepdims=1, name="top_row"),
        helper.make_node("ArgMax", ["col_has"], ["left"], axis=3, keepdims=1, name="left_col"),
        helper.make_node("Gather", ["col_has", "col_reverse"], ["col_has_rev"], axis=3, name="reverse_col_has"),
        helper.make_node("ArgMax", ["col_has_rev"], ["right_from_rev"], axis=3, keepdims=1, name="right_from_reversed"),
        helper.make_node("Sub", ["nineteen", "right_from_rev"], ["right"], name="right_col"),
        helper.make_node("Sub", ["right", "left"], ["width_minus_one"], name="width_minus_one"),
        helper.make_node("Add", ["width_minus_one", "one_i"], ["width"], name="bbox_width"),
        helper.make_node("Add", ["row_base", "top"], ["row_src_raw"], name="row_src_raw"),
        helper.make_node("Sub", ["right", "col_base"], ["col_src_raw"], name="flipped_col_src_raw"),
        helper.make_node("Less", ["row_src_raw", "twenty"], ["row_valid_bool"], name="row_valid_bool"),
        helper.make_node("Less", ["col_base", "width"], ["col_valid_bool"], name="col_valid_bool"),
        helper.make_node("Cast", ["row_valid_bool"], ["row_valid"], to=TensorProto.FLOAT, name="row_valid_float"),
        helper.make_node("Cast", ["col_valid_bool"], ["col_valid"], to=TensorProto.FLOAT, name="col_valid_float"),
        helper.make_node("Min", ["row_src_raw", "nineteen"], ["row_src_clip"], name="row_src_clip"),
        helper.make_node("Max", ["col_src_raw", "zero_i"], ["col_src_low_clip"], name="col_src_low_clip"),
        helper.make_node("Min", ["col_src_low_clip", "nineteen"], ["col_src_clip"], name="col_src_clip"),
        helper.make_node("Expand", ["row_src_clip", "row_idx_shape"], ["row_idx"], name="expand_row_idx"),
        helper.make_node("Expand", ["col_src_clip", "col_idx_shape"], ["col_idx"], name="expand_col_idx"),
        helper.make_node("GatherElements", ["x", "row_idx"], ["row_gathered"], axis=2, name="gather_rows"),
        helper.make_node("GatherElements", ["row_gathered", "col_idx"], ["cropped9_raw"], axis=3, name="gather_cols"),
        helper.make_node("Mul", ["row_valid", "col_valid"], ["valid_mask"], name="valid_mask"),
        helper.make_node("Mul", ["cropped9_raw", "valid_mask"], ["cropped9"], name="mask_invalid_edges"),
        helper.make_node(
            "Slice",
            ["cropped9", "first_channel_starts", "first_channel_ends", "axes4", "steps4"],
            ["first_channel"],
            name="slice_first_channel",
        ),
        helper.make_node("Mul", ["first_channel", "zero_f"], ["zero1"], name="zero_background_channel"),
        helper.make_node(
            "Concat",
            ["zero1", "cropped9"],
            ["out20"],
            axis=1,
            name="restore_ten_channels",
        ),
        helper.make_node("Pad", ["out20", "pad_spatial", "zero_f"], ["output"], name="pad_to_30"),
    ]

    graph = helper.make_graph(
        nodes,
        "task177_static_gather_crop",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=[
            vi("x", TensorProto.FLOAT, [1, 9, 20, 20]),
            vi("occ", TensorProto.FLOAT, [1, 1, 20, 20]),
            vi("row_has", TensorProto.FLOAT, [1, 1, 20, 1]),
            vi("col_has", TensorProto.FLOAT, [1, 1, 1, 20]),
            vi("top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("right", TensorProto.INT64, [1, 1, 1, 1]),
            vi("width", TensorProto.INT64, [1, 1, 1, 1]),
            vi("row_src_raw", TensorProto.INT64, [1, 1, 8, 1]),
            vi("col_src_raw", TensorProto.INT64, [1, 1, 1, 8]),
            vi("row_idx", TensorProto.INT64, [1, 9, 8, 20]),
            vi("col_idx", TensorProto.INT64, [1, 9, 8, 8]),
            vi("row_gathered", TensorProto.FLOAT, [1, 9, 8, 20]),
            vi("cropped9", TensorProto.FLOAT, [1, 9, 8, 8]),
            vi("out20", TensorProto.FLOAT, [1, 10, 8, 8]),
        ],
    )
    model = helper.make_model(graph, producer_name="neurogolf_task177_static_gather_crop", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task177_static_gather_crop"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task177.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task177.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task177.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
