#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task282."""

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
    weight = np.zeros((2, 1, 3, 3), dtype=np.float32)
    weight[0, 0, 0, 1] = 1.0
    weight[0, 0, 1, 0] = 1.0
    weight[0, 0, 1, 2] = 1.0
    weight[0, 0, 2, 1] = 1.0
    weight[1, 0, 0, 0] = 1.0
    weight[1, 0, 0, 2] = 1.0
    weight[1, 0, 2, 0] = 1.0
    weight[1, 0, 2, 2] = 1.0
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("seed_starts", [0, 5, 0, 0]),
        const_i64("seed_ends", [1, 6, 9, 9]),
        const_i64("ch1_starts", [0, 0, 0, 0]),
        const_i64("ch1_ends", [1, 1, 9, 9]),
        const_i64("ch5_starts", [0, 1, 0, 0]),
        const_i64("ch5_ends", [1, 2, 9, 9]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 21, 21]),
        const_f32("weight", weight),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones9", np.ones((1, 1, 9, 9), dtype=np.float32)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "seed_starts", "seed_ends", "axes4", "steps4"], ["seed5"], name="slice_seed5"),
        helper.make_node("Conv", ["seed5", "weight"], ["paint_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="paint_stencil"),
        helper.make_node("Slice", ["paint_raw", "ch1_starts", "ch1_ends", "axes4", "steps4"], ["ch1_raw"], name="slice_ch1"),
        helper.make_node("Slice", ["paint_raw", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5_raw"], name="slice_ch5"),
        helper.make_node("Greater", ["ch1_raw", "half"], ["ch1_bool"], name="ch1_bool"),
        helper.make_node("Greater", ["ch5_raw", "half"], ["ch5_bool"], name="ch5_bool"),
        helper.make_node("Cast", ["ch1_bool"], ["ch1"], to=TensorProto.FLOAT, name="ch1"),
        helper.make_node("Cast", ["ch5_bool"], ["ch5"], to=TensorProto.FLOAT, name="ch5"),
        helper.make_node("Add", ["ch1", "ch5"], ["union_raw"], name="union_raw"),
        helper.make_node("Greater", ["union_raw", "half"], ["union_bool"], name="union_bool"),
        helper.make_node("Cast", ["union_bool"], ["union"], to=TensorProto.FLOAT, name="union"),
        helper.make_node("Sub", ["ones9", "union"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ones9", "ones9"], ["zero9"], name="zero9"),
        helper.make_node("Concat", ["ch0", "ch1", "zero9", "zero9", "zero9", "ch5", "zero9", "zero9", "zero9", "zero9"], ["small_out"], axis=1, name="assemble_9x9"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
    ]
    graph = helper.make_graph(
        nodes,
        "task282_hollow_stencil",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task282_hollow_stencil",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task282_hollow_stencil")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task282.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task282 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task282.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task282.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
