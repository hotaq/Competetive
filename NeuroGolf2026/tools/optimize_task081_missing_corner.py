#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task081."""

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
    weight = np.zeros((4, 1, 2, 2), dtype=np.float32)
    # Windows with one missing corner, ordered TL/TR/BL/BR missing.
    weight[0, 0, 0, 1] = weight[0, 0, 1, 0] = weight[0, 0, 1, 1] = 1.0
    weight[1, 0, 0, 0] = weight[1, 0, 1, 0] = weight[1, 0, 1, 1] = 1.0
    weight[2, 0, 0, 0] = weight[2, 0, 0, 1] = weight[2, 0, 1, 1] = 1.0
    weight[3, 0, 0, 0] = weight[3, 0, 0, 1] = weight[3, 0, 1, 0] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 7, 7]),
        const_i64("seed8_starts", [0, 8, 0, 0]),
        const_i64("seed8_ends", [1, 9, 7, 7]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 7, 7]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_tl", [0, 0, 0, 0, 0, 0, 1, 1]),
        const_i64("pad_tr", [0, 0, 0, 1, 0, 0, 1, 0]),
        const_i64("pad_bl", [0, 0, 1, 0, 0, 0, 0, 1]),
        const_i64("pad_br", [0, 0, 1, 1, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 23, 23]),
        const_f32("corner_weight", weight),
        const_f32("two_half", 2.5),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_7x7"),
        helper.make_node("Slice", ["input", "seed8_starts", "seed8_ends", "axes4", "steps4"], ["ch8"], name="slice_color8"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Conv", ["ch8", "corner_weight"], ["corner_counts"], kernel_shape=[2, 2], name="count_three_corners"),
        helper.make_node("Greater", ["corner_counts", "two_half"], ["corner_bool"], name="is_three_corner"),
        helper.make_node("Cast", ["corner_bool"], ["corner_hits"], to=TensorProto.FLOAT, name="corner_hits_float"),
    ]

    padded_masks: list[str] = []
    for idx, pad in enumerate(("pad_tl", "pad_tr", "pad_bl", "pad_br")):
        starts = f"mask{idx}_starts"
        ends = f"mask{idx}_ends"
        inits.extend([const_i64(starts, [0, idx, 0, 0]), const_i64(ends, [1, idx + 1, 6, 6])])
        nodes.extend(
            [
                helper.make_node("Slice", ["corner_hits", starts, ends, "axes4", "steps4"], [f"mask{idx}_6x6"], name=f"slice_mask_{idx}"),
                helper.make_node("Pad", [f"mask{idx}_6x6", pad, "zero"], [f"mask{idx}_7x7"], name=f"place_mask_{idx}"),
            ]
        )
        padded_masks.append(f"mask{idx}_7x7")

    nodes.extend(
        [
            helper.make_node("Sum", padded_masks, ["paint_sum"], name="sum_missing_corner_masks"),
            helper.make_node("Greater", ["paint_sum", "half"], ["paint_bool"], name="paint_any_bool"),
            helper.make_node("Cast", ["paint_bool"], ["paint_any_raw"], to=TensorProto.FLOAT, name="paint_any_float"),
            helper.make_node("Mul", ["paint_any_raw", "bg"], ["paint"], name="paint_only_background"),
            helper.make_node("Sub", ["one", "paint"], ["keep_bg"], name="keep_background_mask"),
            helper.make_node("Mul", ["bg", "keep_bg"], ["out_ch0"], name="new_background"),
        ]
    )

    channel_outputs: list[str] = []
    for color in range(10):
        if color == 0:
            channel_outputs.append("out_ch0")
            continue
        starts = f"ch{color}_starts"
        ends = f"ch{color}_ends"
        inits.extend([const_i64(starts, [0, color, 0, 0]), const_i64(ends, [1, color + 1, 7, 7])])
        base = f"base_ch{color}"
        nodes.append(helper.make_node("Slice", ["small", starts, ends, "axes4", "steps4"], [base], name=f"slice_ch{color}"))
        if color == 1:
            nodes.append(helper.make_node("Add", [base, "paint"], ["out_ch1"], name="add_missing_corner_color"))
            channel_outputs.append("out_ch1")
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
        "task081_missing_corner",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task081_missing_corner",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task081_missing_corner")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task081.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task081 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task081.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task081.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
