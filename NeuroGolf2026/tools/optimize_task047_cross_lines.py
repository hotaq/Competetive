#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task047."""

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


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch7_starts", [0, 7, 0, 0]),
        const_i64("ch7_ends", [1, 8, 9, 9]),
        const_i64("ch8_starts", [0, 8, 0, 0]),
        const_i64("ch8_ends", [1, 9, 9, 9]),
        const_i64("tile_row", [1, 1, 1, 9]),
        const_i64("tile_col", [1, 1, 9, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 21, 21]),
        const_f32("ones9", np.ones((1, 1, 9, 9), dtype=np.float32)),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "ch7_starts", "ch7_ends", "axes4", "steps4"], ["seed7"], name="slice_seed7"),
        helper.make_node("Slice", ["input", "ch8_starts", "ch8_ends", "axes4", "steps4"], ["seed8"], name="slice_seed8"),
        helper.make_node("ReduceMax", ["seed7"], ["row7"], axes=[3], keepdims=1, name="row7_any"),
        helper.make_node("ReduceMax", ["seed7"], ["col7"], axes=[2], keepdims=1, name="col7_any"),
        helper.make_node("ReduceMax", ["seed8"], ["row8"], axes=[3], keepdims=1, name="row8_any"),
        helper.make_node("ReduceMax", ["seed8"], ["col8"], axes=[2], keepdims=1, name="col8_any"),
        helper.make_node("Tile", ["row7", "tile_row"], ["row7_full"], name="tile_row7"),
        helper.make_node("Tile", ["col7", "tile_col"], ["col7_full"], name="tile_col7"),
        helper.make_node("Tile", ["row8", "tile_row"], ["row8_full"], name="tile_row8"),
        helper.make_node("Tile", ["col8", "tile_col"], ["col8_full"], name="tile_col8"),
        helper.make_node("Add", ["row7_full", "col7_full"], ["line7_raw"], name="line7_raw"),
        helper.make_node("Add", ["row8_full", "col8_full"], ["line8_raw"], name="line8_raw"),
        helper.make_node("Greater", ["line7_raw", "half"], ["line7_bool"], name="line7_bool"),
        helper.make_node("Greater", ["line8_raw", "half"], ["line8_bool"], name="line8_bool"),
        helper.make_node("Cast", ["line7_bool"], ["line7"], to=TensorProto.FLOAT, name="line7"),
        helper.make_node("Cast", ["line8_bool"], ["line8"], to=TensorProto.FLOAT, name="line8"),
        helper.make_node("Mul", ["line7", "line8"], ["ch2"], name="intersections"),
        helper.make_node("Sub", ["ones9", "ch2"], ["not_ch2"], name="not_intersection"),
        helper.make_node("Mul", ["line7", "not_ch2"], ["ch7"], name="out7"),
        helper.make_node("Mul", ["line8", "not_ch2"], ["ch8"], name="out8"),
        helper.make_node("Add", ["line7", "line8"], ["union_plus_inter"], name="union_sum"),
        helper.make_node("Sub", ["union_plus_inter", "ch2"], ["union"], name="union"),
        helper.make_node("Sub", ["ones9", "union"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ones9", "ones9"], ["zero9"], name="zero9"),
    ]
    channel_outputs = ["ch0", "zero9", "ch2", "zero9", "zero9", "zero9", "zero9", "ch7", "ch8", "zero9"]
    nodes.extend(
        [
            helper.make_node("Concat", channel_outputs, ["small_out"], axis=1, name="assemble_9x9"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task047_cross_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task047_cross_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task047_cross_lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task047.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task047 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task047.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task047.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
