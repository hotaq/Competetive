#!/usr/bin/env python3
"""Build a strict-static missing 5x5 patch recovery model for task351."""

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
    rev5 = np.arange(4, -1, -1, dtype=np.int64)
    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        f32("block_weight", np.ones((1, 1, 5, 5), dtype=np.float32)),
        f32("twentyfive_f", 25.0),
        i64("crop16_starts", [0, 0, 0, 0]),
        i64("crop16_ends", [1, 10, 16, 16]),
        i64("ch3_starts", [0, 3, 0, 0]),
        i64("ch3_ends", [1, 4, 16, 16]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_width", [3]),
        i64("axes_height", [2]),
        i64("eleven_i", 11),
        i64("rev_row_base", rev5.reshape(1, 1, 5, 1)),
        i64("rev_col_base", rev5.reshape(1, 1, 1, 5)),
        i64("row_idx_shape", [1, 10, 5, 16]),
        i64("col_idx_shape", [1, 10, 5, 5]),
        i64("pad5_to30", [0, 0, 0, 0, 0, 0, 25, 25]),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop16_starts", "crop16_ends", "axes4", "steps4"], ["x16"], name="crop_16"),
        helper.make_node("Slice", ["input", "ch3_starts", "ch3_ends", "axes4", "steps4"], ["ch3"], name="slice_ch3"),
        helper.make_node("Conv", ["ch3", "block_weight"], ["block_sum"], name="score_5x5_block"),
        helper.make_node("Equal", ["block_sum", "twentyfive_f"], ["block_bool"], name="block_is_solid3"),
        helper.make_node("Cast", ["block_bool"], ["block"], to=TensorProto.FLOAT, name="block_float"),
        helper.make_node("ReduceMax", ["block", "axes_width"], ["row_has"], keepdims=1, name="row_has_block"),
        helper.make_node("ReduceMax", ["block", "axes_height"], ["col_has"], keepdims=1, name="col_has_block"),
        helper.make_node("ArgMax", ["row_has"], ["block_top"], axis=2, keepdims=1, name="block_top"),
        helper.make_node("ArgMax", ["col_has"], ["block_left"], axis=3, keepdims=1, name="block_left"),
        helper.make_node("Sub", ["eleven_i", "block_top"], ["src_top"], name="opposite_top"),
        helper.make_node("Sub", ["eleven_i", "block_left"], ["src_left"], name="opposite_left"),
        helper.make_node("Add", ["src_top", "rev_row_base"], ["row_src"], name="row_src_rot180"),
        helper.make_node("Add", ["src_left", "rev_col_base"], ["col_src"], name="col_src_rot180"),
        helper.make_node("Expand", ["row_src", "row_idx_shape"], ["row_idx"], name="expand_row_idx"),
        helper.make_node("Expand", ["col_src", "col_idx_shape"], ["col_idx"], name="expand_col_idx"),
        helper.make_node("GatherElements", ["x16", "row_idx"], ["row_gathered"], axis=2, name="gather_rows"),
        helper.make_node("GatherElements", ["row_gathered", "col_idx"], ["out5"], axis=3, name="gather_cols"),
        helper.make_node("Pad", ["out5", "pad5_to30", "zero_f"], ["output"], name="pad_to_30"),
    ]
    graph = helper.make_graph(
        nodes,
        "task351_static_missing_patch",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=[
            vi("x16", TensorProto.FLOAT, [1, 10, 16, 16]),
            vi("ch3", TensorProto.FLOAT, [1, 1, 16, 16]),
            vi("block_sum", TensorProto.FLOAT, [1, 1, 12, 12]),
            vi("block", TensorProto.FLOAT, [1, 1, 12, 12]),
            vi("row_has", TensorProto.FLOAT, [1, 1, 12, 1]),
            vi("col_has", TensorProto.FLOAT, [1, 1, 1, 12]),
            vi("block_top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("block_left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("src_top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("src_left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("row_idx", TensorProto.INT64, [1, 10, 5, 16]),
            vi("col_idx", TensorProto.INT64, [1, 10, 5, 5]),
            vi("row_gathered", TensorProto.FLOAT, [1, 10, 5, 16]),
            vi("out5", TensorProto.FLOAT, [1, 10, 5, 5]),
        ],
    )
    model = helper.make_model(graph, producer_name="neurogolf_task351_static_missing_patch", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task351_static_missing_patch"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task351.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task351.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task351.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
