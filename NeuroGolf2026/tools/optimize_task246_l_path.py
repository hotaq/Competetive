#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task246."""

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


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_i64_scalar(name: str, value: int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def ch_slice(color: int) -> list[onnx.TensorProto]:
    return [
        const_i64(f"ch{color}_starts", [0, color, 0, 0]),
        const_i64(f"ch{color}_ends", [1, color + 1, 30, 30]),
    ]


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("tile_rows", [1, 1, 30, 1]),
        const_i64("tile_cols", [1, 1, 1, 30]),
        const_f32("half", 0.5),
        *ch_slice(0),
        *ch_slice(2),
        *ch_slice(3),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["ch0_in"], name="slice_ch0"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_ch2"),
        helper.make_node("Slice", ["input", "ch3_starts", "ch3_ends", "axes4", "steps4"], ["ch3"], name="slice_ch3"),
        helper.make_node("ReduceMax", ["ch2"], ["row2"], axes=[3], keepdims=1, name="row2"),
        helper.make_node("ReduceMax", ["ch2"], ["col2"], axes=[2], keepdims=1, name="col2"),
        helper.make_node("ReduceMax", ["ch3"], ["row3"], axes=[3], keepdims=1, name="row3"),
        helper.make_node("ReduceMax", ["ch3"], ["col3"], axes=[2], keepdims=1, name="col3"),
        helper.make_node("Tile", ["row2", "tile_cols"], ["row2_full"], name="row2_full"),
        helper.make_node("Tile", ["col3", "tile_rows"], ["col3_full"], name="col3_full"),
        helper.make_node("CumSum", ["col2", "axis_w"], ["cum_col2"], name="cum_col2"),
        helper.make_node("CumSum", ["col3", "axis_w"], ["cum_col3"], name="cum_col3"),
        helper.make_node("Sub", ["cum_col2", "cum_col3"], ["col_diff"], name="col_diff"),
        helper.make_node("Abs", ["col_diff"], ["col_xor"], name="col_xor"),
        helper.make_node("Add", ["col_xor", "col3"], ["col_add_corner"], name="col_add_corner"),
        helper.make_node("Sub", ["col_add_corner", "col2"], ["col_between_raw"], name="col_between_raw"),
        helper.make_node("Greater", ["col_between_raw", "half"], ["col_between_bool"], name="col_between_bool"),
        helper.make_node("Cast", ["col_between_bool"], ["col_between"], to=TensorProto.FLOAT, name="col_between"),
        helper.make_node("Tile", ["col_between", "tile_rows"], ["col_between_full"], name="col_between_full"),
        helper.make_node("Mul", ["row2_full", "col_between_full"], ["h_path"], name="h_path"),
        helper.make_node("CumSum", ["row2", "axis_h"], ["cum_row2"], name="cum_row2"),
        helper.make_node("CumSum", ["row3", "axis_h"], ["cum_row3"], name="cum_row3"),
        helper.make_node("Sub", ["cum_row2", "cum_row3"], ["row_diff"], name="row_diff"),
        helper.make_node("Abs", ["row_diff"], ["row_xor"], name="row_xor"),
        helper.make_node("Add", ["row_xor", "row2"], ["row_add_corner"], name="row_add_corner"),
        helper.make_node("Sub", ["row_add_corner", "row3"], ["row_between_raw"], name="row_between_raw"),
        helper.make_node("Greater", ["row_between_raw", "half"], ["row_between_bool"], name="row_between_bool"),
        helper.make_node("Cast", ["row_between_bool"], ["row_between"], to=TensorProto.FLOAT, name="row_between"),
        helper.make_node("Tile", ["row_between", "tile_cols"], ["row_between_full"], name="row_between_full"),
        helper.make_node("Mul", ["col3_full", "row_between_full"], ["v_path"], name="v_path"),
        helper.make_node("Add", ["h_path", "v_path"], ["path_raw"], name="path_raw"),
        helper.make_node("Greater", ["path_raw", "half"], ["path_bool"], name="path_bool"),
        helper.make_node("Cast", ["path_bool"], ["ch8"], to=TensorProto.FLOAT, name="ch8"),
        helper.make_node("Sub", ["ch0_in", "ch8"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ch0_in", "ch0_in"], ["zero"], name="zero"),
        helper.make_node("Concat", ["ch0", "zero", "ch2", "ch3", "zero", "zero", "zero", "zero", "ch8", "zero"], ["output"], axis=1, name="assemble"),
    ]
    graph = helper.make_graph(
        nodes,
        "task246_l_path",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task246_l_path",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task246_l_path")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task246.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task246 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task246.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task246.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
