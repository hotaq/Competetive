#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task342.

The task has a 10x10 grid with a 2x2 color-8 block and four singleton
colored markers, one in each quadrant around the block. The output erases the
input and places the quadrant marker colors into the matching cells of the
2x2 block.
"""

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


def const_i64_scalar(name: str, value: int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def shift_nodes(src: str, dst: str, direction: str) -> list[onnx.NodeProto]:
    return [
        make_slice(src, f"{dst}_part", f"shift_{direction}"),
        helper.make_node("Pad", [f"{dst}_part", f"pad_{direction}", "zero_scalar"], [dst], name=f"pad_{dst}"),
    ]


def mul_chain(nodes: list[onnx.NodeProto], names: list[str], out: str, prefix: str) -> None:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        nxt = out if idx == len(names) - 1 else f"{prefix}_mul{idx}"
        nodes.append(helper.make_node("Mul", [cur, name], [nxt], name=f"{prefix}_mul_{idx}"))
        cur = nxt


def exclusive_quadrant_nodes(src: str, prefix: str, reverse_h: bool, reverse_w: bool) -> list[onnx.NodeProto]:
    nodes: list[onnx.NodeProto] = []
    cur = src
    if reverse_h:
        nodes.append(helper.make_node("Gather", [cur, "rev10"], [f"{prefix}_rev_h"], axis=2, name=f"{prefix}_rev_h"))
        cur = f"{prefix}_rev_h"
    if reverse_w:
        nodes.append(helper.make_node("Gather", [cur, "rev10"], [f"{prefix}_rev_w"], axis=3, name=f"{prefix}_rev_w"))
        cur = f"{prefix}_rev_w"
    nodes.extend(
        [
            helper.make_node("CumSum", [cur, "axis_h"], [f"{prefix}_cum_h"], name=f"{prefix}_cum_h"),
            helper.make_node("CumSum", [f"{prefix}_cum_h", "axis_w"], [f"{prefix}_cum_hw"], name=f"{prefix}_cum_hw"),
            make_slice(f"{prefix}_cum_hw", f"{prefix}_excl_part", "prefix_excl"),
            helper.make_node("Pad", [f"{prefix}_excl_part", "pad_top_left", "zero_scalar"], [f"{prefix}_excl"], name=f"{prefix}_excl"),
        ]
    )
    cur = f"{prefix}_excl"
    if reverse_w:
        nodes.append(helper.make_node("Gather", [cur, "rev10"], [f"{prefix}_unrev_w"], axis=3, name=f"{prefix}_unrev_w"))
        cur = f"{prefix}_unrev_w"
    if reverse_h:
        nodes.append(helper.make_node("Gather", [cur, "rev10"], [f"{prefix}_unrev_h"], axis=2, name=f"{prefix}_unrev_h"))
        cur = f"{prefix}_unrev_h"
    nodes.extend(
        [
            helper.make_node("Greater", [cur, "half"], [f"{prefix}_bool"], name=f"{prefix}_bool"),
            helper.make_node("Cast", [f"{prefix}_bool"], [prefix], to=TensorProto.FLOAT, name=f"{prefix}_cast"),
        ]
    )
    return nodes


def make_model() -> onnx.ModelProto:
    channel_mask = np.ones((1, 10, 1, 1), dtype=np.float32)
    channel_mask[:, 0, :, :] = 0.0
    channel_mask[:, 8, :, :] = 0.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("axis_channel", [1]),
        const_i64("rev10", list(reversed(range(10)))),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones10", np.ones((1, 1, 10, 10), dtype=np.float32)),
        const_f32("marker_channel_mask", channel_mask),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_i64("pad_top_left", [0, 0, 1, 1, 0, 0, 0, 0]),
        *slice_4d("small", [0, 0, 0, 0], [1, 10, 10, 10]),
        *slice_4d("ch8", [0, 8, 0, 0], [1, 9, 10, 10]),
        *slice_4d("prefix_excl", [0, 0, 0, 0], [1, 10, 9, 9]),
        *slice_4d("shift_e", [0, 0, 0, 1], [1, 1, 10, 10]),
        *slice_4d("shift_w", [0, 0, 0, 0], [1, 1, 10, 9]),
        *slice_4d("shift_s", [0, 0, 1, 0], [1, 1, 10, 10]),
        *slice_4d("shift_n", [0, 0, 0, 0], [1, 1, 9, 10]),
        *slice_4d("shift_se", [0, 0, 1, 1], [1, 1, 10, 10]),
        *slice_4d("shift_sw", [0, 0, 1, 0], [1, 1, 10, 9]),
        *slice_4d("shift_ne", [0, 0, 0, 1], [1, 1, 9, 10]),
        *slice_4d("shift_nw", [0, 0, 0, 0], [1, 1, 9, 9]),
        const_i64("pad_e", [0, 0, 0, 0, 0, 0, 0, 1]),
        const_i64("pad_w", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("pad_s", [0, 0, 0, 0, 0, 0, 1, 0]),
        const_i64("pad_n", [0, 0, 1, 0, 0, 0, 0, 0]),
        const_i64("pad_se", [0, 0, 0, 0, 0, 0, 1, 1]),
        const_i64("pad_sw", [0, 0, 0, 1, 0, 0, 1, 0]),
        const_i64("pad_ne", [0, 0, 1, 0, 0, 0, 0, 1]),
        const_i64("pad_nw", [0, 0, 1, 1, 0, 0, 0, 0]),
    ]
    for color in range(1, 10):
        inits.extend(slice_4d(f"out_ch{color}", [0, color, 0, 0], [1, color + 1, 10, 10]))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "small", "small"),
        helper.make_node("Mul", ["small", "marker_channel_mask"], ["markers"], name="drop_background_and_block"),
        make_slice("small", "block8", "ch8"),
        *shift_nodes("block8", "east", "e"),
        *shift_nodes("block8", "west", "w"),
        *shift_nodes("block8", "south", "s"),
        *shift_nodes("block8", "north", "n"),
        *shift_nodes("block8", "se", "se"),
        *shift_nodes("block8", "sw", "sw"),
        *shift_nodes("block8", "ne", "ne"),
        *shift_nodes("block8", "nw", "nw"),
    ]
    mul_chain(nodes, ["block8", "east", "south", "se"], "tl_cell", "tl")
    mul_chain(nodes, ["block8", "west", "south", "sw"], "tr_cell", "tr")
    mul_chain(nodes, ["block8", "east", "north", "ne"], "bl_cell", "bl")
    mul_chain(nodes, ["block8", "west", "north", "nw"], "br_cell", "br")
    nodes.extend(exclusive_quadrant_nodes("markers", "q_nw", reverse_h=False, reverse_w=False))
    nodes.extend(exclusive_quadrant_nodes("markers", "q_ne", reverse_h=False, reverse_w=True))
    nodes.extend(exclusive_quadrant_nodes("markers", "q_sw", reverse_h=True, reverse_w=False))
    nodes.extend(exclusive_quadrant_nodes("markers", "q_se", reverse_h=True, reverse_w=True))
    nodes.extend(
        [
            helper.make_node("Mul", ["q_nw", "tl_cell"], ["paint_tl"], name="paint_tl"),
            helper.make_node("Mul", ["q_ne", "tr_cell"], ["paint_tr"], name="paint_tr"),
            helper.make_node("Mul", ["q_sw", "bl_cell"], ["paint_bl"], name="paint_bl"),
            helper.make_node("Mul", ["q_se", "br_cell"], ["paint_br"], name="paint_br"),
            helper.make_node("Sum", ["paint_tl", "paint_tr", "paint_bl", "paint_br"], ["painted"], name="painted_colors"),
            helper.make_node("ReduceSum", ["painted", "axis_channel"], ["occupied"], keepdims=1, name="occupied"),
            helper.make_node("Sub", ["ones10", "occupied"], ["ch0"], name="background"),
        ]
    )
    channels = ["ch0"]
    for color in range(1, 10):
        nodes.append(make_slice("painted", f"ch{color}", f"out_ch{color}"))
        channels.append(f"ch{color}")
    nodes.extend(
        [
            helper.make_node("Concat", channels, ["small_out"], axis=1, name="assemble_10x10"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task342_quadrant_block",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task342_quadrant_block",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task342_quadrant_block")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task342.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task342 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task342.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task342.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
