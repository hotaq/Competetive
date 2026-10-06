#!/usr/bin/env python3
"""Build a compact clean ONNX model for NeuroGolf task004.

Rule from the public discussion: for each non-background color, detect the
single bottom-right L-corner, keep that corner's row and column fixed, and
shift every other colored cell one column to the right inside the original
canvas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


CHANNELS = 10
NONZERO_CHANNELS = 9
HEIGHT = 30
WIDTH = 30
CROP = 16
EXPECTED_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]


@dataclass(frozen=True)
class Row:
    task: str
    right: int
    wrong: int
    examples: int
    macs: int | str
    memory: int | str
    params: int | str
    cost: int | str
    points: float | str
    file_bytes: int
    reason: str


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    # Conv output at (r, c) sums input[r, c] + input[r, c-1] + input[r-1, c].
    corner_weight = np.zeros((NONZERO_CHANNELS, 1, 2, 2), dtype=np.float32)
    corner_weight[:, 0, 0, 1] = 1.0
    corner_weight[:, 0, 1, 0] = 1.0
    corner_weight[:, 0, 1, 1] = 1.0
    # Depthwise 1x2 convolution implements a right shift with left padding.
    shift_weight = np.zeros((NONZERO_CHANNELS, 1, 1, 2), dtype=np.float32)
    shift_weight[:, 0, 0, 0] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("nz_starts", [0, 1, 0, 0]),
        const_i64("nz_ends", [1, CHANNELS, CROP, CROP]),
        const_i64("bg_starts", [0, 0, 0, 0]),
        const_i64("bg_ends", [1, 1, CROP, CROP]),
        const_i64("out_pads", [0, 0, 0, 0, 0, 0, HEIGHT - CROP, WIDTH - CROP]),
        const_f32("corner_weight", corner_weight),
        const_f32("shift_weight", shift_weight),
        const_f32("zero", 0.0),
        const_f32("two_half", 2.5),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Slice",
            ["input", "nz_starts", "nz_ends", "axes4", "steps4"],
            ["x"],
            name="crop_nonzero_channels",
        ),
        helper.make_node(
            "Slice",
            ["input", "bg_starts", "bg_ends", "axes4", "steps4"],
            ["input_background"],
            name="crop_background_channel",
        ),
        helper.make_node("ReduceMax", ["x"], ["input_nonzero_any"], axes=[1], keepdims=1, name="input_nonzero_occupancy"),
        helper.make_node("Add", ["input_background", "input_nonzero_any"], ["valid"], name="valid_inside_canvas"),
        helper.make_node(
            "Conv",
            ["x", "corner_weight"],
            ["corner_score"],
            name="detect_bottom_right_l_corners",
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0],
            group=NONZERO_CHANNELS,
        ),
        helper.make_node("ReduceMax", ["corner_score"], ["row_score"], axes=[3], keepdims=1, name="corner_rows"),
        helper.make_node("ReduceMax", ["corner_score"], ["col_score"], axes=[2], keepdims=1, name="corner_cols"),
        helper.make_node("Greater", ["row_score", "two_half"], ["row_has"], name="row_has_corner"),
        helper.make_node("Greater", ["col_score", "two_half"], ["col_has"], name="col_has_corner"),
        helper.make_node("Or", ["row_has", "col_has"], ["keep_region"], name="keep_row_or_col"),
        helper.make_node("Where", ["keep_region", "x", "zero"], ["keep"], name="keep_fixed_row_col"),
        helper.make_node(
            "Where",
            ["keep_region", "zero", "x"],
            ["move"],
            name="remove_kept_cells",
        ),
        helper.make_node(
            "Conv",
            ["move", "shift_weight"],
            ["shifted"],
            name="shift_move_right",
            kernel_shape=[1, 2],
            pads=[0, 1, 0, 0],
            group=NONZERO_CHANNELS,
        ),
        helper.make_node("Mul", ["shifted", "valid"], ["shifted_valid"], name="drop_shift_outside_canvas"),
        helper.make_node("Add", ["keep", "shifted_valid"], ["nonzero_out"], name="merge_kept_and_shifted"),
        helper.make_node("ReduceMax", ["nonzero_out"], ["any_nonzero"], axes=[1], keepdims=1, name="nonzero_occupancy"),
        helper.make_node("Sub", ["valid", "any_nonzero"], ["background"], name="background_inside_canvas"),
        helper.make_node("Concat", ["background", "nonzero_out"], ["out16"], axis=1, name="assemble_16x16"),
        helper.make_node("Pad", ["out16", "out_pads", "zero"], ["output"], name="pad_back_to_30"),
    ]

    graph = helper.make_graph(
        nodes,
        "task004_corner_shift",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task004_corner_shift",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-zip",
        type=Path,
        default=Path("generated/metric_v2_v84_plus_latin_formula_v85/submission.zip"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task004_corner_shift_v86b"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "task004.onnx"
    zip_path = args.out_dir / "submission.zip"
    report_path = args.out_dir / "report.csv"

    model = make_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task004.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    if wrong:
        row = Row("task004", right, wrong, len(examples), "", "", "", "", "", len(raw), reason)
    else:
        try:
            macs, memory, params, cost = score_network(model_path)
            row = Row("task004", right, wrong, len(examples), macs, memory, params, cost, task_points(cost), len(raw), "ok")
            print(f"task004: ok cost={cost} points={task_points(cost):.6f} bytes={len(raw)}")
        except Exception as exc:  # noqa: BLE001
            row = Row("task004", right, wrong, len(examples), "", "", "", "", "", len(raw), f"profile_error:{exc}")

    with zipfile.ZipFile(args.base_zip) as base_zf:
        members = {Path(name).name: base_zf.read(name) for name in base_zf.namelist()}
    if row.reason == "ok":
        members["task004.onnx"] = raw
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, member_raw in sorted(members.items()):
            zf.writestr(name, member_raw)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)

    print(f"verify: right={right} wrong={wrong} examples={len(examples)} reason={row.reason}")
    print(f"model: {model_path}")
    print(f"zip: {zip_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
