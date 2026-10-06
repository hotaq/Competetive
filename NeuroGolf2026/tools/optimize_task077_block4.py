#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task077."""

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
    bottom_right = np.zeros((1, 1, 3, 3), dtype=np.float32)
    bottom_right[0, 0, 2, 2] = 1.0
    block_cells = np.zeros((1, 1, 3, 3), dtype=np.float32)
    block_cells[0, 0, 1, 1] = 1.0
    block_cells[0, 0, 1, 0] = 1.0
    block_cells[0, 0, 0, 1] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 30, 30]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 30, 30]),
        const_f32("bottom_right_weight", bottom_right),
        const_f32("block_cells_weight", block_cells),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["occupied"], keepdims=1, name="occupied_any_color"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["ch0"], name="slice_ch0"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_ch2"),
        helper.make_node("Sub", ["occupied", "ch0"], ["non_bg"], name="non_background"),
        helper.make_node("Sub", ["non_bg", "ch2"], ["fg"], name="foreground_not_color2"),
        helper.make_node("Conv", ["ch2", "bottom_right_weight"], ["bottom_right_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="detect_2_as_bottom_right"),
        helper.make_node("Greater", ["bottom_right_raw", "half"], ["bottom_right_bool"], name="bottom_right_bool"),
        helper.make_node("Cast", ["bottom_right_bool"], ["bottom_right_mask"], to=TensorProto.FLOAT, name="bottom_right_float"),
        helper.make_node("Mul", ["fg", "bottom_right_mask"], ["top_left_mask"], name="valid_top_left_cells"),
        helper.make_node("Conv", ["top_left_mask", "block_cells_weight"], ["change_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="expand_to_three_cells"),
        helper.make_node("Greater", ["change_raw", "half"], ["change_bool"], name="change_bool"),
        helper.make_node("Cast", ["change_bool"], ["change_candidates"], to=TensorProto.FLOAT, name="change_float"),
        helper.make_node("Mul", ["fg", "change_candidates"], ["change"], name="change_only_foreground"),
        helper.make_node("Sub", ["one", "change"], ["keep"], name="keep_original_foreground"),
    ]

    channel_outputs: list[str] = []
    for color in range(10):
        starts = f"ch{color}_starts"
        ends = f"ch{color}_ends"
        if color not in (0, 2):
            inits.extend(
                [
                    const_i64(starts, [0, color, 0, 0]),
                    const_i64(ends, [1, color + 1, 30, 30]),
                ]
            )
            base = f"base_ch{color}"
            nodes.append(helper.make_node("Slice", ["input", starts, ends, "axes4", "steps4"], [base], name=f"slice_ch{color}"))
        elif color == 0:
            base = "ch0"
        else:
            base = "ch2"

        if color == 4:
            nodes.append(helper.make_node("Add", [base, "change"], ["out_ch4"], name="paint_color4"))
            channel_outputs.append("out_ch4")
        elif color in (0, 2):
            channel_outputs.append(base)
        else:
            nodes.append(helper.make_node("Mul", [base, "keep"], [f"out_ch{color}"], name=f"clear_changed_ch{color}"))
            channel_outputs.append(f"out_ch{color}")

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_channels"))

    graph = helper.make_graph(
        nodes,
        "task077_block4",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task077_block4",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task077_block4")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task077.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task077 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task077.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task077.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
