#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task352."""

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
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 30, 30]),
        const_i64("seed2_starts", [0, 2, 0, 0]),
        const_i64("seed2_ends", [1, 3, 30, 30]),
        const_f32("dilate_weight", weight),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Slice", ["input", "seed2_starts", "seed2_ends", "axes4", "steps4"], ["seed2"], name="slice_seed2"),
        helper.make_node("Conv", ["seed2", "dilate_weight"], ["near_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="dilate_seed2"),
        helper.make_node("Greater", ["near_raw", "half"], ["near_bool"], name="near_bool"),
        helper.make_node("Cast", ["near_bool"], ["near"], to=TensorProto.FLOAT, name="near_float"),
        helper.make_node("Mul", ["near", "bg"], ["paint1"], name="paint_background_near_nonzero"),
        helper.make_node("Sub", ["one", "paint1"], ["keep_bg"], name="keep_background_mask"),
        helper.make_node("Mul", ["bg", "keep_bg"], ["out_ch0"], name="background_after_paint"),
    ]

    channel_outputs: list[str] = []
    for color in range(10):
        if color == 0:
            channel_outputs.append("out_ch0")
            continue
        starts = f"ch{color}_starts"
        ends = f"ch{color}_ends"
        inits.extend(
            [
                const_i64(starts, [0, color, 0, 0]),
                const_i64(ends, [1, color + 1, 30, 30]),
            ]
        )
        base = f"base_ch{color}"
        nodes.append(helper.make_node("Slice", ["input", starts, ends, "axes4", "steps4"], [base], name=f"slice_ch{color}"))
        if color == 1:
            nodes.append(helper.make_node("Add", [base, "paint1"], ["out_ch1"], name="add_paint1"))
            channel_outputs.append("out_ch1")
        else:
            channel_outputs.append(base)

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_channels"))

    graph = helper.make_graph(
        nodes,
        "task352_dilate_nonzero",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task352_dilate_nonzero",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task352_dilate_nonzero")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task352.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task352 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task352.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task352.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
