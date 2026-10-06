#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task230."""

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
    block = np.ones((1, 1, 2, 2), dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 30, 30]),
        const_i64("seed5_starts", [0, 5, 0, 0]),
        const_i64("seed5_ends", [1, 6, 30, 30]),
        const_i64("m1_starts", [0, 0, 1, 1]),
        const_i64("m1_ends", [1, 1, 30, 30]),
        const_i64("m2_starts", [0, 0, 1, 0]),
        const_i64("m2_ends", [1, 1, 30, 28]),
        const_i64("m3_starts", [0, 0, 0, 1]),
        const_i64("m3_ends", [1, 1, 28, 30]),
        const_i64("m4_starts", [0, 0, 0, 0]),
        const_i64("m4_ends", [1, 1, 28, 28]),
        const_i64("m1_pad", [0, 0, 0, 0, 0, 0, 1, 1]),
        const_i64("m2_pad", [0, 0, 0, 2, 0, 0, 1, 0]),
        const_i64("m3_pad", [0, 0, 2, 0, 0, 0, 0, 1]),
        const_i64("m4_pad", [0, 0, 2, 2, 0, 0, 0, 0]),
        const_f32("block_weight", block),
        const_f32("half3", 3.5),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Slice", ["input", "seed5_starts", "seed5_ends", "axes4", "steps4"], ["ch5"], name="slice_color5"),
        helper.make_node("Conv", ["ch5", "block_weight"], ["block_raw"], kernel_shape=[2, 2], pads=[0, 0, 1, 1], name="detect_2x2_blocks"),
        helper.make_node("Greater", ["block_raw", "half3"], ["block_bool"], name="block_bool"),
        helper.make_node("Cast", ["block_bool"], ["top_left"], to=TensorProto.FLOAT, name="top_left_float"),
    ]

    for marker in (1, 2, 3, 4):
        nodes.extend(
            [
                helper.make_node("Slice", ["top_left", f"m{marker}_starts", f"m{marker}_ends", "axes4", "steps4"], [f"m{marker}_crop"], name=f"marker{marker}_crop"),
                helper.make_node("Pad", [f"m{marker}_crop", f"m{marker}_pad", "zero"], [f"m{marker}_raw"], name=f"marker{marker}_shift"),
                helper.make_node("Mul", [f"m{marker}_raw", "bg"], [f"marker{marker}"], name=f"marker{marker}_only_background"),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Sum", ["marker1", "marker2", "marker3", "marker4"], ["marker_any"], name="sum_markers"),
            helper.make_node("Sub", ["one", "marker_any"], ["keep_bg"], name="keep_background_mask"),
            helper.make_node("Mul", ["bg", "keep_bg"], ["out_ch0"], name="background_after_markers"),
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
        if color in (1, 2, 3, 4):
            nodes.append(helper.make_node("Add", [base, f"marker{color}"], [f"out_ch{color}"], name=f"add_marker{color}"))
            channel_outputs.append(f"out_ch{color}")
        else:
            channel_outputs.append(base)

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_channels"))

    graph = helper.make_graph(
        nodes,
        "task230_block_markers",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task230_block_markers",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task230_block_markers")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task230.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task230 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task230.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task230.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
