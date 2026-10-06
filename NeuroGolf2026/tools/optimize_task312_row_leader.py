#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task312."""

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
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 12, 12]),
        const_i64("lead_starts", [0, 0, 0, 0]),
        const_i64("lead_ends", [1, 10, 12, 1]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 12, 12]),
        const_i64("lead_nonzero_starts", [0, 1, 0, 0]),
        const_i64("lead_nonzero_ends", [1, 10, 12, 1]),
        const_i64("axis_channel", [1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 18, 18]),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_12x12"),
        helper.make_node("Slice", ["small", "lead_starts", "lead_ends", "axes4", "steps4"], ["lead_color"], name="slice_row_leader"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["is5"], name="slice_color5"),
        helper.make_node("Slice", ["lead_color", "lead_nonzero_starts", "lead_nonzero_ends", "axes4", "steps4"], ["lead_nonzero_channels"], name="drop_leader_background"),
        helper.make_node("ReduceSum", ["lead_nonzero_channels", "axis_channel"], ["lead_nonzero"], keepdims=1, name="leader_is_nonzero"),
        helper.make_node("Mul", ["is5", "lead_nonzero"], ["replace_mask"], name="replace_only_nonzero_leaders"),
        helper.make_node("Sub", ["one", "replace_mask"], ["keep_mask"], name="keep_non_replaced"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_original_cells"),
        helper.make_node("Mul", ["lead_color", "replace_mask"], ["paint"], name="paint_leader_color"),
        helper.make_node("Add", ["kept", "paint"], ["small_out"], name="apply_row_leader"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]

    graph = helper.make_graph(
        nodes,
        "task312_row_leader",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task312_row_leader",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task312_row_leader")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task312.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task312 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task312.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task312.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
