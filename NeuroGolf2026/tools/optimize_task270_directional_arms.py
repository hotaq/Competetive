#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task270."""

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


def slice_channel(prefix: str, color: int) -> list[onnx.TensorProto]:
    return [
        const_i64(f"{prefix}_starts", [0, color, 0, 0]),
        const_i64(f"{prefix}_ends", [1, color + 1, 15, 15]),
    ]


def direction_nodes(prefix: str, center: str, marker: str, out_color: str) -> list[onnx.NodeProto]:
    return [
        helper.make_node("CumSum", [marker, "axis_h"], [f"{prefix}_cum_h"], name=f"{prefix}_cum_h"),
        helper.make_node("ReduceSum", [marker, "axes_h_reduce"], [f"{prefix}_total_col"], keepdims=1, name=f"{prefix}_total_col"),
        helper.make_node("Tile", [f"{prefix}_total_col", "tile_col_total"], [f"{prefix}_total_col_full"], name=f"{prefix}_total_col_full"),
        helper.make_node("Sub", [f"{prefix}_total_col_full", f"{prefix}_cum_h"], [f"{prefix}_below_count"], name=f"{prefix}_below_count"),
        helper.make_node("Greater", [f"{prefix}_cum_h", "half"], [f"{prefix}_above_bool"], name=f"{prefix}_above_bool"),
        helper.make_node("Greater", [f"{prefix}_below_count", "half"], [f"{prefix}_below_bool"], name=f"{prefix}_below_bool"),
        helper.make_node("Cast", [f"{prefix}_above_bool"], [f"{prefix}_above"], to=TensorProto.FLOAT, name=f"{prefix}_above"),
        helper.make_node("Cast", [f"{prefix}_below_bool"], [f"{prefix}_below"], to=TensorProto.FLOAT, name=f"{prefix}_below"),
        helper.make_node("Mul", [center, f"{prefix}_above"], [f"{prefix}_center_above"], name=f"{prefix}_center_above"),
        helper.make_node("Mul", [center, f"{prefix}_below"], [f"{prefix}_center_below"], name=f"{prefix}_center_below"),
        helper.make_node("Slice", [f"{prefix}_center_above", "shift_n_starts", "shift_n_ends", "axes4", "steps4"], [f"{prefix}_north_part"], name=f"{prefix}_north_part"),
        helper.make_node("Slice", [f"{prefix}_center_below", "shift_s_starts", "shift_s_ends", "axes4", "steps4"], [f"{prefix}_south_part"], name=f"{prefix}_south_part"),
        helper.make_node("Pad", [f"{prefix}_north_part", "pad_north", "zero_scalar"], [f"{prefix}_north"], name=f"{prefix}_north"),
        helper.make_node("Pad", [f"{prefix}_south_part", "pad_south", "zero_scalar"], [f"{prefix}_south"], name=f"{prefix}_south"),
        helper.make_node("CumSum", [marker, "axis_w"], [f"{prefix}_cum_w"], name=f"{prefix}_cum_w"),
        helper.make_node("ReduceSum", [marker, "axes_w_reduce"], [f"{prefix}_total_row"], keepdims=1, name=f"{prefix}_total_row"),
        helper.make_node("Tile", [f"{prefix}_total_row", "tile_row_total"], [f"{prefix}_total_row_full"], name=f"{prefix}_total_row_full"),
        helper.make_node("Sub", [f"{prefix}_total_row_full", f"{prefix}_cum_w"], [f"{prefix}_right_count"], name=f"{prefix}_right_count"),
        helper.make_node("Greater", [f"{prefix}_cum_w", "half"], [f"{prefix}_left_bool"], name=f"{prefix}_left_bool"),
        helper.make_node("Greater", [f"{prefix}_right_count", "half"], [f"{prefix}_right_bool"], name=f"{prefix}_right_bool"),
        helper.make_node("Cast", [f"{prefix}_left_bool"], [f"{prefix}_left"], to=TensorProto.FLOAT, name=f"{prefix}_left"),
        helper.make_node("Cast", [f"{prefix}_right_bool"], [f"{prefix}_right"], to=TensorProto.FLOAT, name=f"{prefix}_right"),
        helper.make_node("Mul", [center, f"{prefix}_left"], [f"{prefix}_center_left"], name=f"{prefix}_center_left"),
        helper.make_node("Mul", [center, f"{prefix}_right"], [f"{prefix}_center_right"], name=f"{prefix}_center_right"),
        helper.make_node("Slice", [f"{prefix}_center_left", "shift_w_starts", "shift_w_ends", "axes4", "steps4"], [f"{prefix}_west_part"], name=f"{prefix}_west_part"),
        helper.make_node("Slice", [f"{prefix}_center_right", "shift_e_starts", "shift_e_ends", "axes4", "steps4"], [f"{prefix}_east_part"], name=f"{prefix}_east_part"),
        helper.make_node("Pad", [f"{prefix}_west_part", "pad_west", "zero_scalar"], [f"{prefix}_west"], name=f"{prefix}_west"),
        helper.make_node("Pad", [f"{prefix}_east_part", "pad_east", "zero_scalar"], [f"{prefix}_east"], name=f"{prefix}_east"),
        helper.make_node("Sum", [f"{prefix}_north", f"{prefix}_south", f"{prefix}_west", f"{prefix}_east"], [out_color], name=f"{prefix}_arms"),
    ]


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("axes_h_reduce", [2]),
        const_i64("axes_w_reduce", [3]),
        const_i64("tile_col_total", [1, 1, 15, 1]),
        const_i64("tile_row_total", [1, 1, 1, 15]),
        const_i64("shift_n_starts", [0, 0, 1, 0]),
        const_i64("shift_n_ends", [1, 1, 15, 15]),
        const_i64("shift_s_starts", [0, 0, 0, 0]),
        const_i64("shift_s_ends", [1, 1, 14, 15]),
        const_i64("shift_w_starts", [0, 0, 0, 1]),
        const_i64("shift_w_ends", [1, 1, 15, 15]),
        const_i64("shift_e_starts", [0, 0, 0, 0]),
        const_i64("shift_e_ends", [1, 1, 15, 14]),
        const_i64("pad_north", [0, 0, 0, 0, 0, 0, 1, 0]),
        const_i64("pad_south", [0, 0, 1, 0, 0, 0, 0, 0]),
        const_i64("pad_west", [0, 0, 0, 0, 0, 0, 0, 1]),
        const_i64("pad_east", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 15, 15]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones15", np.ones((1, 1, 15, 15), dtype=np.float32)),
        *slice_channel("center1", 1),
        *slice_channel("center2", 2),
        *slice_channel("marker3", 3),
        *slice_channel("marker7", 7),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "center1_starts", "center1_ends", "axes4", "steps4"], ["ch1"], name="slice_center1"),
        helper.make_node("Slice", ["input", "center2_starts", "center2_ends", "axes4", "steps4"], ["ch2"], name="slice_center2"),
        helper.make_node("Slice", ["input", "marker3_starts", "marker3_ends", "axes4", "steps4"], ["marker3"], name="slice_marker3"),
        helper.make_node("Slice", ["input", "marker7_starts", "marker7_ends", "axes4", "steps4"], ["marker7"], name="slice_marker7"),
        *direction_nodes("c1", "ch1", "marker7", "ch7"),
        *direction_nodes("c2", "ch2", "marker3", "ch3"),
        helper.make_node("Sum", ["ch1", "ch2", "ch3", "ch7"], ["occupied"], name="occupied"),
        helper.make_node("Sub", ["ones15", "occupied"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ones15", "ones15"], ["zero15"], name="zero15"),
    ]
    channels = ["ch0", "ch1", "ch2", "ch3", "zero15", "zero15", "zero15", "ch7", "zero15", "zero15"]
    nodes.extend(
        [
            helper.make_node("Concat", channels, ["small_out"], axis=1, name="assemble_15x15"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task270_directional_arms",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task270_directional_arms",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task270_directional_arms")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task270.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task270 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task270.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task270.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
