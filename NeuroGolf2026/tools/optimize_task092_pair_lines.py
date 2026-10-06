#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task092.

Each nonzero color appears twice. If both markers share a row, draw the
horizontal span between them. If they share a column, draw the vertical span
between them. Vertical spans win at intersections.
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
H = 30
W = 30


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


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("axes_h", [2]),
        const_i64("axes_w", [3]),
        const_i64("axis_channel", [1]),
        const_i64("rev_h", list(reversed(range(H)))),
        const_i64("rev_w", list(reversed(range(W)))),
        const_i64("tile_h", [1, 1, H, 1]),
        const_i64("tile_w", [1, 1, 1, W]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 0, 0]),
        const_f32("half", 0.5),
        const_f32("one_scalar", 1.0),
        const_f32("one_hw", np.ones((1, 1, H, W), dtype=np.float32)),
        const_f32("zero_scalar", 0.0),
        *slice_4d("small", [0, 0, 0, 0], [1, 10, H, W]),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, H, W]),
    ]
    for color in range(1, 10):
        inits.extend(slice_4d(f"ch{color}", [0, color, 0, 0], [1, color + 1, H, W]))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "small", "small"),
        make_slice("small", "ch0_in", "ch0"),
        helper.make_node("ReduceSum", ["small", "axis_channel"], ["inside"], keepdims=1, name="inside_shape"),
    ]
    hlines: list[str] = []
    vlines: list[str] = []
    for color in range(1, 10):
        ch = f"ch{color}"
        nodes.extend(
            [
                make_slice("small", ch, f"ch{color}"),
                helper.make_node("ReduceSum", [ch, "axes_w"], [f"{ch}_row_count"], keepdims=1, name=f"{ch}_row_count"),
                helper.make_node("Greater", [f"{ch}_row_count", "half"], [f"{ch}_row_nonzero_bool"], name=f"{ch}_row_nonzero_bool"),
                helper.make_node("Greater", [f"{ch}_row_count", "one_scalar"], [f"{ch}_row_pair_bool"], name=f"{ch}_row_pair_bool"),
                helper.make_node("Cast", [f"{ch}_row_pair_bool"], [f"{ch}_row_pair"], to=TensorProto.FLOAT, name=f"{ch}_row_pair"),
                helper.make_node("Tile", [f"{ch}_row_pair", "tile_w"], [f"{ch}_row_pair_full"], name=f"{ch}_row_pair_full"),
                helper.make_node("CumSum", [ch, "axis_w"], [f"{ch}_left_count"], name=f"{ch}_left_count"),
                helper.make_node("Greater", [f"{ch}_left_count", "half"], [f"{ch}_left_bool"], name=f"{ch}_left_bool"),
                helper.make_node("Cast", [f"{ch}_left_bool"], [f"{ch}_left"], to=TensorProto.FLOAT, name=f"{ch}_left"),
                helper.make_node("Gather", [ch, "rev_w"], [f"{ch}_rev_w"], axis=3, name=f"{ch}_rev_w"),
                helper.make_node("CumSum", [f"{ch}_rev_w", "axis_w"], [f"{ch}_right_count_rev"], name=f"{ch}_right_count_rev"),
                helper.make_node("Gather", [f"{ch}_right_count_rev", "rev_w"], [f"{ch}_right_count"], axis=3, name=f"{ch}_right_unrev"),
                helper.make_node("Greater", [f"{ch}_right_count", "half"], [f"{ch}_right_bool"], name=f"{ch}_right_bool"),
                helper.make_node("Cast", [f"{ch}_right_bool"], [f"{ch}_right"], to=TensorProto.FLOAT, name=f"{ch}_right"),
                helper.make_node("Mul", [f"{ch}_left", f"{ch}_right"], [f"{ch}_hspan"], name=f"{ch}_hspan"),
                helper.make_node("Mul", [f"{ch}_hspan", f"{ch}_row_pair_full"], [f"hline{color}"], name=f"hline{color}"),
                helper.make_node("ReduceSum", [ch, "axes_h"], [f"{ch}_col_count"], keepdims=1, name=f"{ch}_col_count"),
                helper.make_node("Greater", [f"{ch}_col_count", "one_scalar"], [f"{ch}_col_pair_bool"], name=f"{ch}_col_pair_bool"),
                helper.make_node("Cast", [f"{ch}_col_pair_bool"], [f"{ch}_col_pair"], to=TensorProto.FLOAT, name=f"{ch}_col_pair"),
                helper.make_node("Tile", [f"{ch}_col_pair", "tile_h"], [f"{ch}_col_pair_full"], name=f"{ch}_col_pair_full"),
                helper.make_node("CumSum", [ch, "axis_h"], [f"{ch}_up_count"], name=f"{ch}_up_count"),
                helper.make_node("Greater", [f"{ch}_up_count", "half"], [f"{ch}_up_bool"], name=f"{ch}_up_bool"),
                helper.make_node("Cast", [f"{ch}_up_bool"], [f"{ch}_up"], to=TensorProto.FLOAT, name=f"{ch}_up"),
                helper.make_node("Gather", [ch, "rev_h"], [f"{ch}_rev_h"], axis=2, name=f"{ch}_rev_h"),
                helper.make_node("CumSum", [f"{ch}_rev_h", "axis_h"], [f"{ch}_down_count_rev"], name=f"{ch}_down_count_rev"),
                helper.make_node("Gather", [f"{ch}_down_count_rev", "rev_h"], [f"{ch}_down_count"], axis=2, name=f"{ch}_down_unrev"),
                helper.make_node("Greater", [f"{ch}_down_count", "half"], [f"{ch}_down_bool"], name=f"{ch}_down_bool"),
                helper.make_node("Cast", [f"{ch}_down_bool"], [f"{ch}_down"], to=TensorProto.FLOAT, name=f"{ch}_down"),
                helper.make_node("Mul", [f"{ch}_up", f"{ch}_down"], [f"{ch}_vspan"], name=f"{ch}_vspan"),
                helper.make_node("Mul", [f"{ch}_vspan", f"{ch}_col_pair_full"], [f"vline{color}"], name=f"vline{color}"),
            ]
        )
        hlines.append(f"hline{color}")
        vlines.append(f"vline{color}")

    nodes.extend(
        [
            helper.make_node("Sum", vlines, ["v_union_raw"], name="v_union_raw"),
            helper.make_node("Greater", ["v_union_raw", "half"], ["v_union_bool"], name="v_union_bool"),
            helper.make_node("Cast", ["v_union_bool"], ["v_union"], to=TensorProto.FLOAT, name="v_union"),
            helper.make_node("Sub", ["one_hw", "v_union"], ["not_v_union"], name="not_v_union"),
        ]
    )
    final_channels: list[str] = []
    for color in range(1, 10):
        nodes.extend(
            [
                helper.make_node("Mul", [f"hline{color}", "not_v_union"], [f"hline{color}_masked"], name=f"hline{color}_masked"),
                helper.make_node("Add", [f"vline{color}", f"hline{color}_masked"], [f"out{color}"], name=f"out{color}"),
            ]
        )
        final_channels.append(f"out{color}")

    nodes.extend(
        [
            helper.make_node("Sum", final_channels, ["occupied_raw"], name="occupied_raw"),
            helper.make_node("Greater", ["occupied_raw", "half"], ["occupied_bool"], name="occupied_bool"),
            helper.make_node("Cast", ["occupied_bool"], ["occupied"], to=TensorProto.FLOAT, name="occupied"),
            helper.make_node("Sub", ["inside", "occupied"], ["ch0"], name="background"),
            helper.make_node("Concat", ["ch0", *final_channels], ["small_out"], axis=1, name="assemble_30x30"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task092_pair_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task092_pair_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task092_pair_lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task092.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task092 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task092.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task092.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
