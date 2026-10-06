#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task249."""

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
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_5x10_to_30", [0, 0, 0, 0, 0, 0, 25, 20]),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["active_rows"], axes=[1, 3], keepdims=1, name="active_rows"),
        helper.make_node("ReduceMax", ["input"], ["active_cols"], axes=[1, 2], keepdims=1, name="active_cols"),
        helper.make_node("ReduceSum", ["active_rows", "axes_all"], ["row_count"], keepdims=1, name="row_count"),
        helper.make_node("ReduceSum", ["active_cols", "axes_all"], ["col_count"], keepdims=1, name="col_count"),
    ]

    branches: list[str] = []
    for h in (3, 4, 5):
        for w in (3, 4, 5):
            prefix = f"b{h}x{w}"
            inits.extend(
                [
                    const_f32(f"{prefix}_h", np.ones((1, 1, 1, 1), dtype=np.float32) * h),
                    const_f32(f"{prefix}_w", np.ones((1, 1, 1, 1), dtype=np.float32) * w),
                    const_i64(f"{prefix}_starts", [0, 0, 0, 0]),
                    const_i64(f"{prefix}_ends", [1, 10, h, w]),
                    const_i64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, 5 - h, 10 - (2 * w)]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Equal", ["row_count", f"{prefix}_h"], [f"{prefix}_h_ok"], name=f"{prefix}_h_ok"),
                    helper.make_node("Equal", ["col_count", f"{prefix}_w"], [f"{prefix}_w_ok"], name=f"{prefix}_w_ok"),
                    helper.make_node("And", [f"{prefix}_h_ok", f"{prefix}_w_ok"], [f"{prefix}_size_ok_bool"], name=f"{prefix}_size_ok_bool"),
                    helper.make_node("Cast", [f"{prefix}_size_ok_bool"], [f"{prefix}_size_ok"], to=TensorProto.FLOAT, name=f"{prefix}_size_ok"),
                    helper.make_node("Slice", ["input", f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [f"{prefix}_crop"], name=f"{prefix}_crop"),
                    helper.make_node("Concat", [f"{prefix}_crop", f"{prefix}_crop"], [f"{prefix}_repeat"], axis=3, name=f"{prefix}_repeat_twice"),
                    helper.make_node("Pad", [f"{prefix}_repeat", f"{prefix}_pad", "zero"], [f"{prefix}_padded"], name=f"{prefix}_pad_5x10"),
                    helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_size_ok"], [f"{prefix}_selected"], name=f"{prefix}_select"),
                ]
            )
            branches.append(f"{prefix}_selected")

    nodes.extend(
        [
            helper.make_node("Sum", branches, ["small_out"], name="sum_size_branches"),
            helper.make_node("Pad", ["small_out", "pad_5x10_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task249_repeat_width",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task249_repeat_width",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task249_repeat_width")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task249.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task249 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task249.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task249.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
