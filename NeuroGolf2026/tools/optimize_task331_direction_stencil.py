#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task331."""

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
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("seed_starts", [0, 1, 0, 0]),
        const_i64("seed_ends", [1, 2, 10, 10]),
        const_i64("north_starts", [0, 0, 1, 0]),
        const_i64("north_ends", [1, 1, 10, 10]),
        const_i64("south_starts", [0, 0, 0, 0]),
        const_i64("south_ends", [1, 1, 9, 10]),
        const_i64("west_starts", [0, 0, 0, 1]),
        const_i64("west_ends", [1, 1, 10, 10]),
        const_i64("east_starts", [0, 0, 0, 0]),
        const_i64("east_ends", [1, 1, 10, 9]),
        const_i64("pad_north", [0, 0, 0, 0, 0, 0, 1, 0]),
        const_i64("pad_south", [0, 0, 1, 0, 0, 0, 0, 0]),
        const_i64("pad_west", [0, 0, 0, 0, 0, 0, 0, 1]),
        const_i64("pad_east", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("zero_scalar", 0.0),
        const_f32("ones10", np.ones((1, 1, 10, 10), dtype=np.float32)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "seed_starts", "seed_ends", "axes4", "steps4"], ["ch1"], name="slice_seed1"),
        helper.make_node("Slice", ["ch1", "north_starts", "north_ends", "axes4", "steps4"], ["north_part"], name="north_part"),
        helper.make_node("Slice", ["ch1", "south_starts", "south_ends", "axes4", "steps4"], ["south_part"], name="south_part"),
        helper.make_node("Slice", ["ch1", "west_starts", "west_ends", "axes4", "steps4"], ["west_part"], name="west_part"),
        helper.make_node("Slice", ["ch1", "east_starts", "east_ends", "axes4", "steps4"], ["east_part"], name="east_part"),
        helper.make_node("Pad", ["north_part", "pad_north", "zero_scalar"], ["ch2"], name="paint_north_2"),
        helper.make_node("Pad", ["south_part", "pad_south", "zero_scalar"], ["ch8"], name="paint_south_8"),
        helper.make_node("Pad", ["west_part", "pad_west", "zero_scalar"], ["ch7"], name="paint_west_7"),
        helper.make_node("Pad", ["east_part", "pad_east", "zero_scalar"], ["ch6"], name="paint_east_6"),
        helper.make_node("Sum", ["ch1", "ch2", "ch6", "ch7", "ch8"], ["occupied"], name="occupied"),
        helper.make_node("Sub", ["ones10", "occupied"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ones10", "ones10"], ["zero10"], name="zero10"),
    ]
    channels = ["ch0", "ch1", "ch2", "zero10", "zero10", "zero10", "ch6", "ch7", "ch8", "zero10"]
    nodes.extend(
        [
            helper.make_node("Concat", channels, ["small_out"], axis=1, name="assemble_10x10"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task331_direction_stencil",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task331_direction_stencil",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task331_direction_stencil")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task331.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task331 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task331.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task331.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
