#!/usr/bin/env python3
"""Build clean static ONNX models for NeuroGolf tasks 095 and 317."""

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


def make_model(task: str) -> onnx.ModelProto:
    include_center_as_five = task == "task095"
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    if include_center_as_five:
        weight[0, 0, 1, 1] = 0.0
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("seed_starts", [0, 5, 0, 0]),
        const_i64("seed_ends", [1, 6, 9, 9]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 21, 21]),
        const_f32("weight", weight),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones9", np.ones((1, 1, 9, 9), dtype=np.float32)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "seed_starts", "seed_ends", "axes4", "steps4"], ["seed5"], name="slice_seed5"),
        helper.make_node("Conv", ["seed5", "weight"], ["paint_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="paint_stencil"),
        helper.make_node("Greater", ["paint_raw", "half"], ["paint_bool"], name="paint_bool"),
        helper.make_node("Cast", ["paint_bool"], ["paint"], to=TensorProto.FLOAT, name="paint"),
    ]
    if include_center_as_five:
        nodes.extend(
            [
                helper.make_node("Sub", ["ones9", "seed5"], ["not_seed"], name="not_seed"),
                helper.make_node("Mul", ["paint", "not_seed"], ["ch1"], name="ch1_neighbors"),
                helper.make_node("Identity", ["seed5"], ["ch5"], name="ch5_center"),
                helper.make_node("Add", ["ch1", "ch5"], ["union"], name="union"),
                helper.make_node("Sub", ["ones9", "union"], ["ch0"], name="background"),
                helper.make_node("Sub", ["ones9", "ones9"], ["zero9"], name="zero9"),
                helper.make_node("Concat", ["ch0", "ch1", "zero9", "zero9", "zero9", "ch5", "zero9", "zero9", "zero9", "zero9"], ["small_out"], axis=1, name="assemble_095"),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Sub", ["ones9", "paint"], ["ch0"], name="background"),
                helper.make_node("Sub", ["ones9", "ones9"], ["zero9"], name="zero9"),
                helper.make_node("Concat", ["ch0", "paint", "zero9", "zero9", "zero9", "zero9", "zero9", "zero9", "zero9", "zero9"], ["small_out"], axis=1, name="assemble_317"),
            ]
        )
    nodes.append(helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"))
    graph = helper.make_graph(
        nodes,
        f"{task}_stencil",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name=f"neurogolf_{task}_stencil",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task095_317_stencils")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}
    for task in ("task095", "task317"):
        model = make_model(task)
        raw = model.SerializeToString()
        examples = all_examples(json.loads(Path(f"data/latest/extracted/{task}.json").read_text()))
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(f"{task} verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
        if wrong:
            raise SystemExit(1)
        (out_dir / f"{task}.onnx").write_bytes(raw)
        raws[task] = raw
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task, raw in raws.items():
            zf.writestr(f"{task}.onnx", raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
