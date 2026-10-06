#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task024."""

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


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes_rows", [1, 3]),
        const_i64("axes_cols", [1, 2]),
        const_i64("axis_h", [2]),
        const_i64("axis_w", [3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("half", 0.5),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["active_rows"], axes=[1, 3], keepdims=1, name="active_rows"),
        helper.make_node("ReduceMax", ["input"], ["active_cols"], axes=[1, 2], keepdims=1, name="active_cols"),
    ]

    for color in (0, 1, 2, 3):
        inits.extend(
            [
                const_i64(f"ch{color}_starts", [0, color, 0, 0]),
                const_i64(f"ch{color}_ends", [1, color + 1, 30, 30]),
            ]
        )
        nodes.append(
            helper.make_node(
                "Slice",
                ["input", f"ch{color}_starts", f"ch{color}_ends", "axes4", "steps4"],
                [f"ch{color}"],
                name=f"slice_ch{color}",
            )
        )

    nodes.extend(
        [
            helper.make_node("ReduceMax", ["ch1"], ["row_seed1"], axes=[3], keepdims=1, name="row_seed1"),
            helper.make_node("Mul", ["row_seed1", "active_cols"], ["line1"], name="draw_color1_rows"),
            helper.make_node("ReduceMax", ["ch3"], ["row_seed3"], axes=[3], keepdims=1, name="row_seed3"),
            helper.make_node("Mul", ["row_seed3", "active_cols"], ["line3"], name="draw_color3_rows"),
            helper.make_node("ReduceMax", ["ch2"], ["col_seed2"], axes=[2], keepdims=1, name="col_seed2"),
            helper.make_node("Mul", ["active_rows", "col_seed2"], ["line2_raw"], name="draw_color2_cols_raw"),
            helper.make_node("Sum", ["line1", "line3"], ["row_line_sum"], name="sum_row_lines"),
            helper.make_node("Greater", ["row_line_sum", "half"], ["row_line_bool"], name="row_line_any_bool"),
            helper.make_node("Cast", ["row_line_bool"], ["row_line_any"], to=TensorProto.FLOAT, name="row_line_any_float"),
            helper.make_node("Sub", ["one", "row_line_any"], ["keep_col2"], name="row_priority_mask"),
            helper.make_node("Mul", ["line2_raw", "keep_col2"], ["line2"], name="draw_color2_cols"),
            helper.make_node("Sum", ["line1", "line2", "line3"], ["line_sum"], name="sum_lines"),
            helper.make_node("Greater", ["line_sum", "half"], ["line_any_bool"], name="line_any_bool"),
            helper.make_node("Cast", ["line_any_bool"], ["line_any"], to=TensorProto.FLOAT, name="line_any_float"),
            helper.make_node("Sub", ["one", "line_any"], ["keep_bg"], name="keep_background_mask"),
            helper.make_node("Mul", ["ch0", "keep_bg"], ["out_ch0"], name="new_background"),
            helper.make_node("Sub", ["ch0", "ch0"], ["zero_ch"], name="zero_channel"),
            helper.make_node(
                "Concat",
                ["out_ch0", "line1", "line2", "line3", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "zero_ch"],
                ["output"],
                axis=1,
                name="assemble_output",
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task024_row_col_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task024_row_col_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task024_row_col_lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task024.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task024 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task024.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task024.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
