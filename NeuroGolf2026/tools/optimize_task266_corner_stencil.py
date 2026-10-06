#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task266."""

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
        const_i64("seed_starts", [0, 2, 0, 0]),
        const_i64("seed_ends", [1, 3, 3, 5]),
        const_i64("nw_starts", [0, 0, 1, 1]),
        const_i64("nw_ends", [1, 1, 3, 5]),
        const_i64("ne_starts", [0, 0, 1, 0]),
        const_i64("ne_ends", [1, 1, 3, 4]),
        const_i64("sw_starts", [0, 0, 0, 1]),
        const_i64("sw_ends", [1, 1, 2, 5]),
        const_i64("se_starts", [0, 0, 0, 0]),
        const_i64("se_ends", [1, 1, 2, 4]),
        const_i64("pad_nw", [0, 0, 0, 0, 0, 0, 1, 1]),
        const_i64("pad_ne", [0, 0, 0, 1, 0, 0, 1, 0]),
        const_i64("pad_sw", [0, 0, 1, 0, 0, 0, 0, 1]),
        const_i64("pad_se", [0, 0, 1, 1, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 25]),
        const_f32("zero_scalar", 0.0),
        const_f32("ones", np.ones((1, 1, 3, 5), dtype=np.float32)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "seed_starts", "seed_ends", "axes4", "steps4"], ["seed2"], name="slice_seed2"),
        helper.make_node("Slice", ["seed2", "nw_starts", "nw_ends", "axes4", "steps4"], ["nw_part"], name="nw_part"),
        helper.make_node("Slice", ["seed2", "ne_starts", "ne_ends", "axes4", "steps4"], ["ne_part"], name="ne_part"),
        helper.make_node("Slice", ["seed2", "sw_starts", "sw_ends", "axes4", "steps4"], ["sw_part"], name="sw_part"),
        helper.make_node("Slice", ["seed2", "se_starts", "se_ends", "axes4", "steps4"], ["se_part"], name="se_part"),
        helper.make_node("Pad", ["nw_part", "pad_nw", "zero_scalar"], ["ch3"], name="paint_nw_3"),
        helper.make_node("Pad", ["ne_part", "pad_ne", "zero_scalar"], ["ch6"], name="paint_ne_6"),
        helper.make_node("Pad", ["sw_part", "pad_sw", "zero_scalar"], ["ch8"], name="paint_sw_8"),
        helper.make_node("Pad", ["se_part", "pad_se", "zero_scalar"], ["ch7"], name="paint_se_7"),
        helper.make_node("Sum", ["ch3", "ch6", "ch7", "ch8"], ["occupied"], name="occupied"),
        helper.make_node("Sub", ["ones", "occupied"], ["ch0"], name="background"),
        helper.make_node("Sub", ["ones", "ones"], ["zero"], name="zero"),
        helper.make_node("Concat", ["ch0", "zero", "zero", "ch3", "zero", "zero", "ch6", "ch7", "ch8", "zero"], ["small_out"], axis=1, name="assemble_3x5"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
    ]
    graph = helper.make_graph(
        nodes,
        "task266_corner_stencil",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task266_corner_stencil",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task266_corner_stencil")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task266.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task266 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task266.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task266.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
