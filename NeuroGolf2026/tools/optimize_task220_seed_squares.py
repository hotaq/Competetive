#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task220."""

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
    square = np.ones((3, 1, 3, 3), dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("seed_starts", [0, 2, 0, 0]),
        const_i64("seed_ends", [1, 9, 30, 30]),
        const_i64("bg_starts", [0, 0, 0, 0]),
        const_i64("bg_ends", [1, 1, 30, 30]),
        const_i64("paint1_starts", [0, 0, 0, 0]),
        const_i64("paint1_ends", [1, 1, 30, 30]),
        const_i64("paint6_starts", [0, 1, 0, 0]),
        const_i64("paint6_ends", [1, 2, 30, 30]),
        const_i64("paint4_starts", [0, 2, 0, 0]),
        const_i64("paint4_ends", [1, 3, 30, 30]),
        const_i64("axis_channel", [1]),
        const_f32("square_weight", square),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "seed_starts", "seed_ends", "axes4", "steps4"], ["seed_2_to_8"], name="slice_seed_range"),
        helper.make_node("Slice", ["input", "bg_starts", "bg_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Slice", ["seed_2_to_8", "paint1_starts", "paint1_ends", "axes4", "steps4"], ["seed2"], name="slice_seed2"),
        helper.make_node("Slice", ["seed_2_to_8", "paint6_starts", "paint6_ends", "axes4", "steps4"], ["seed3"], name="slice_seed3"),
    ]
    # The seed range is channels 2..8, so channel 8 sits at local index 6.
    inits.extend(
        [
            const_i64("seed8_starts", [0, 6, 0, 0]),
            const_i64("seed8_ends", [1, 7, 30, 30]),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Slice", ["seed_2_to_8", "seed8_starts", "seed8_ends", "axes4", "steps4"], ["seed8"], name="slice_seed8"),
            helper.make_node("Concat", ["seed2", "seed3", "seed8"], ["seeds238"], axis=1, name="assemble_seed_channels"),
            helper.make_node("Conv", ["seeds238", "square_weight"], ["paint_raw"], group=3, kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="draw_square_masks"),
            helper.make_node("Greater", ["paint_raw", "half"], ["paint_bool"], name="paint_bool"),
            helper.make_node("Cast", ["paint_bool"], ["paint_all"], to=TensorProto.FLOAT, name="paint_float"),
            helper.make_node("Mul", ["paint_all", "bg"], ["paint_bg"], name="paint_only_background"),
            helper.make_node("ReduceSum", ["paint_bg", "axis_channel"], ["paint_any"], keepdims=1, name="paint_any_channel"),
            helper.make_node("Sub", ["one", "paint_any"], ["keep_bg"], name="keep_background_mask"),
            helper.make_node("Mul", ["bg", "keep_bg"], ["out_ch0"], name="background_after_paint"),
            helper.make_node("Slice", ["paint_bg", "paint1_starts", "paint1_ends", "axes4", "steps4"], ["paint1"], name="slice_paint1"),
            helper.make_node("Slice", ["paint_bg", "paint6_starts", "paint6_ends", "axes4", "steps4"], ["paint6"], name="slice_paint6"),
            helper.make_node("Slice", ["paint_bg", "paint4_starts", "paint4_ends", "axes4", "steps4"], ["paint4"], name="slice_paint4"),
        ]
    )

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
            nodes.append(helper.make_node("Add", [base, "paint1"], ["out_ch1"], name="add_color1_square"))
            channel_outputs.append("out_ch1")
        elif color == 4:
            nodes.append(helper.make_node("Add", [base, "paint4"], ["out_ch4"], name="add_color4_square"))
            channel_outputs.append("out_ch4")
        elif color == 6:
            nodes.append(helper.make_node("Add", [base, "paint6"], ["out_ch6"], name="add_color6_square"))
            channel_outputs.append("out_ch6")
        else:
            channel_outputs.append(base)

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_channels"))

    graph = helper.make_graph(
        nodes,
        "task220_seed_squares",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task220_seed_squares",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task220_seed_squares")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task220.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task220 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task220.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task220.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
