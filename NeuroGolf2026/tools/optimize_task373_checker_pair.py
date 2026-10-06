#!/usr/bin/env python3
"""Build a compact clean ONNX model for NeuroGolf task373."""

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


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    top_mask = np.zeros((1, 1, 2, 6), dtype=np.float32)
    bottom_mask = np.zeros((1, 1, 2, 6), dtype=np.float32)
    for c in range(6):
        top_mask[0, 0, 0, c] = 1.0 if c % 2 == 0 else 0.0
        top_mask[0, 0, 1, c] = 1.0 if c % 2 == 1 else 0.0
        bottom_mask[0, 0, :, c] = 1.0 - top_mask[0, 0, :, c]

    inits = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("top_starts", [0, 0, 0, 0]),
        const_i64("top_ends", [1, 10, 1, 1]),
        const_i64("bottom_starts", [0, 0, 1, 0]),
        const_i64("bottom_ends", [1, 10, 2, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 28, 24]),
        const_f32("zero_scalar", 0.0),
        const_f32("top_mask", top_mask),
        const_f32("bottom_mask", bottom_mask),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "top_starts", "top_ends", "axes4", "steps4"], ["top_color"], name="top_color"),
        helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends", "axes4", "steps4"], ["bottom_color"], name="bottom_color"),
        helper.make_node("Mul", ["top_color", "top_mask"], ["top_part"], name="top_part"),
        helper.make_node("Mul", ["bottom_color", "bottom_mask"], ["bottom_part"], name="bottom_part"),
        helper.make_node("Add", ["top_part", "bottom_part"], ["small_out"], name="small_out"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
    ]
    graph = helper.make_graph(nodes, "task373_checker_pair", [tensor_info("input")], [tensor_info("output")], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task373_checker_pair",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task373_checker_pair")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task373.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task373 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    (out_dir / "task373.onnx").write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task373.onnx", raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
