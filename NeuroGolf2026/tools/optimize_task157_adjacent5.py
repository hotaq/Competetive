#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task157."""

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
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 15]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 10, 15]),
        const_i64("seed5_starts", [0, 5, 0, 0]),
        const_i64("seed5_ends", [1, 6, 10, 15]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 15]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_f32("adjacent5_weight", weight),
        const_f32("zero", 0.0),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x15"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Slice", ["input", "seed5_starts", "seed5_ends", "axes4", "steps4"], ["seed5"], name="slice_color5"),
        helper.make_node("Conv", ["seed5", "adjacent5_weight"], ["near5_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="detect_near_color5"),
        helper.make_node("Greater", ["near5_raw", "half"], ["near5_bool"], name="near5_bool"),
        helper.make_node("Cast", ["near5_bool"], ["near5"], to=TensorProto.FLOAT, name="near5_float"),
        helper.make_node("Mul", ["near5", "bg"], ["paint1"], name="paint_only_background"),
        helper.make_node("Sub", ["one", "paint1"], ["keep_bg"], name="keep_background"),
        helper.make_node("Mul", ["bg", "keep_bg"], ["bg_after_paint"], name="background_after_paint"),
        helper.make_node("Add", ["bg_after_paint", "seed5"], ["out_ch0"], name="erase_color5_to_background"),
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
                const_i64(ends, [1, color + 1, 10, 15]),
            ]
        )
        base = f"base_ch{color}"
        nodes.append(helper.make_node("Slice", ["small", starts, ends, "axes4", "steps4"], [base], name=f"slice_ch{color}"))
        if color == 1:
            nodes.append(helper.make_node("Add", [base, "paint1"], ["out_ch1"], name="add_painted_color1"))
            channel_outputs.append("out_ch1")
        elif color == 5:
            nodes.append(helper.make_node("Mul", [base, "zero"], ["out_ch5"], name="erase_color5"))
            channel_outputs.append("out_ch5")
        else:
            channel_outputs.append(base)

    nodes.extend(
        [
            helper.make_node("Concat", channel_outputs, ["small_out"], axis=1, name="assemble_channels"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task157_adjacent5",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task157_adjacent5",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task157_adjacent5")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task157.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task157 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task157.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task157.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
