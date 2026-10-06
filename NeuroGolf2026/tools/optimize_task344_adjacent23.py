#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task344."""

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
    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0
    cross[0, 0, 2, 1] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 30, 30]),
        const_i64("ch3_starts", [0, 3, 0, 0]),
        const_i64("ch3_ends", [1, 4, 30, 30]),
        const_f32("cross_weight", cross),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_color2"),
        helper.make_node("Slice", ["input", "ch3_starts", "ch3_ends", "axes4", "steps4"], ["ch3"], name="slice_color3"),
        helper.make_node("Conv", ["ch2", "cross_weight"], ["near2_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="near_color2"),
        helper.make_node("Conv", ["ch3", "cross_weight"], ["near3_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="near_color3"),
        helper.make_node("Greater", ["near2_raw", "half"], ["near2_bool"], name="near2_bool"),
        helper.make_node("Greater", ["near3_raw", "half"], ["near3_bool"], name="near3_bool"),
        helper.make_node("Cast", ["near2_bool"], ["near2"], to=TensorProto.FLOAT, name="near2_float"),
        helper.make_node("Cast", ["near3_bool"], ["near3"], to=TensorProto.FLOAT, name="near3_float"),
        helper.make_node("Mul", ["ch2", "near3"], ["remove2"], name="remove_2_next_to_3"),
        helper.make_node("Mul", ["ch3", "near2"], ["change3"], name="change_3_next_to_2"),
        helper.make_node("Sub", ["one", "near3"], ["keep2_mask"], name="keep2_mask"),
        helper.make_node("Sub", ["one", "near2"], ["keep3_mask"], name="keep3_mask"),
        helper.make_node("Mul", ["ch2", "keep2_mask"], ["out_ch2"], name="keep_color2"),
        helper.make_node("Mul", ["ch3", "keep3_mask"], ["out_ch3"], name="keep_color3"),
    ]

    channel_outputs: list[str] = []
    for color in range(10):
        if color == 2:
            channel_outputs.append("out_ch2")
            continue
        if color == 3:
            channel_outputs.append("out_ch3")
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
        if color == 0:
            nodes.append(helper.make_node("Add", [base, "remove2"], ["out_ch0"], name="removed_2_to_background"))
            channel_outputs.append("out_ch0")
        elif color == 8:
            nodes.append(helper.make_node("Add", [base, "change3"], ["out_ch8"], name="changed_3_to_color8"))
            channel_outputs.append("out_ch8")
        else:
            channel_outputs.append(base)

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_channels"))

    graph = helper.make_graph(
        nodes,
        "task344_adjacent23",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task344_adjacent23",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task344_adjacent23")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task344.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task344 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task344.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task344.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
