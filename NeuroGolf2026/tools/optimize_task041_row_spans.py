#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task041."""

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


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_w", 3),
        const_i64("axes_w_reduce", [3]),
        const_i64("tile_cols", [1, 1, 1, 10]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("half", 0.5),
        const_f32("two", 2.0),
        const_f32("one_point_five", 1.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones10", np.ones((1, 1, 10, 10), dtype=np.float32)),
    ]
    nodes: list[onnx.NodeProto] = []
    fills: list[str] = []
    for color in range(1, 10):
        inits.extend(
            [
                const_i64(f"ch{color}_starts", [0, color, 0, 0]),
                const_i64(f"ch{color}_ends", [1, color + 1, 10, 10]),
            ]
        )
        ch = f"ch{color}"
        nodes.extend(
            [
                helper.make_node("Slice", ["input", f"ch{color}_starts", f"ch{color}_ends", "axes4", "steps4"], [ch], name=f"slice_ch{color}"),
                helper.make_node("CumSum", [ch, "axis_w"], [f"{ch}_cum"], name=f"{ch}_cum"),
                helper.make_node("Greater", [f"{ch}_cum", "half"], [f"{ch}_after_first_bool"], name=f"{ch}_after_first_bool"),
                helper.make_node("Less", [f"{ch}_cum", "two"], [f"{ch}_before_second_bool"], name=f"{ch}_before_second_bool"),
                helper.make_node("Cast", [f"{ch}_after_first_bool"], [f"{ch}_after_first"], to=TensorProto.FLOAT, name=f"{ch}_after_first"),
                helper.make_node("Cast", [f"{ch}_before_second_bool"], [f"{ch}_before_second"], to=TensorProto.FLOAT, name=f"{ch}_before_second"),
                helper.make_node("Mul", [f"{ch}_after_first", f"{ch}_before_second"], [f"{ch}_between_raw"], name=f"{ch}_between_raw"),
                helper.make_node("ReduceSum", [ch, "axes_w_reduce"], [f"{ch}_row_count"], keepdims=1, name=f"{ch}_row_count"),
                helper.make_node("Greater", [f"{ch}_row_count", "one_point_five"], [f"{ch}_row_pair_bool"], name=f"{ch}_row_pair_bool"),
                helper.make_node("Cast", [f"{ch}_row_pair_bool"], [f"{ch}_row_pair"], to=TensorProto.FLOAT, name=f"{ch}_row_pair"),
                helper.make_node("Tile", [f"{ch}_row_pair", "tile_cols"], [f"{ch}_row_pair_full"], name=f"{ch}_row_pair_full"),
                helper.make_node("Mul", [f"{ch}_between_raw", f"{ch}_row_pair_full"], [f"{ch}_between"], name=f"{ch}_between"),
                helper.make_node("Add", [f"{ch}_between", ch], [f"{ch}_fill_raw"], name=f"{ch}_fill_raw"),
                helper.make_node("Greater", [f"{ch}_fill_raw", "half"], [f"{ch}_fill_bool"], name=f"{ch}_fill_bool"),
                helper.make_node("Cast", [f"{ch}_fill_bool"], [f"{ch}_fill"], to=TensorProto.FLOAT, name=f"{ch}_fill"),
            ]
        )
        fills.append(f"{ch}_fill")
    nodes.extend(
        [
            helper.make_node("Sum", fills, ["union_raw"], name="union_raw"),
            helper.make_node("Greater", ["union_raw", "half"], ["union_bool"], name="union_bool"),
            helper.make_node("Cast", ["union_bool"], ["union"], to=TensorProto.FLOAT, name="union"),
            helper.make_node("Sub", ["ones10", "union"], ["ch0"], name="background"),
            helper.make_node("Sub", ["ones10", "ones10"], ["zero10"], name="zero10"),
            helper.make_node(
                "Concat",
                ["ch0"] + fills,
                ["small_out"],
                axis=1,
                name="assemble_10x10",
            ),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task041_row_spans",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task041_row_spans",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task041_row_spans")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task041.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task041 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task041.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task041.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
