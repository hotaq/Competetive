#!/usr/bin/env python3
"""Build a clean static ONNX model for task231 horizontal pattern repeat.

The task repeats each row's short horizontal period until the output is twice
as wide as the input.  Inputs are 5 rows high, widths are 6..10, and the public
examples use periods 1, 2, or 3.
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


def vi(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_all", [0, 1, 2, 3]),
        f32("zero", 0.0),
        f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["active_cols"], axes=[1, 2], keepdims=1, name="active_cols"),
        helper.make_node("ReduceSum", ["active_cols", "axes_all"], ["width"], keepdims=1, name="width_count"),
    ]

    branches: list[str] = []
    for width in range(6, 11):
        prior_period_ok: list[str] = []
        for period in (1, 2, 3):
            prefix = f"w{width}_p{period}"
            out_width = 2 * width
            repeat_count = (out_width + period - 1) // period
            repeated_inputs = [f"{prefix}_period"] * repeat_count
            inits.extend(
                [
                    f32(f"{prefix}_target", float(width)),
                    i64(f"{prefix}_starts", [0, 0, 0, 0]),
                    i64(f"{prefix}_period_ends", [1, 10, 5, period]),
                    i64(f"{prefix}_input_ends", [1, 10, 5, width]),
                    i64(f"{prefix}_out_ends", [1, 10, 5, out_width]),
                    i64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, 25, 30 - out_width]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Equal", ["width", f"{prefix}_target"], [f"{prefix}_width_ok"], name=f"{prefix}_width_ok"),
                    helper.make_node("Slice", ["input", f"{prefix}_starts", f"{prefix}_period_ends", "axes4", "steps4"], [f"{prefix}_period"], name=f"{prefix}_period"),
                    helper.make_node("Slice", ["input", f"{prefix}_starts", f"{prefix}_input_ends", "axes4", "steps4"], [f"{prefix}_input"], name=f"{prefix}_input"),
                    helper.make_node("Concat", repeated_inputs, [f"{prefix}_repeat_long"], axis=3, name=f"{prefix}_repeat_long"),
                    helper.make_node("Slice", [f"{prefix}_repeat_long", f"{prefix}_starts", f"{prefix}_input_ends", "axes4", "steps4"], [f"{prefix}_repeat_input"], name=f"{prefix}_repeat_input"),
                    helper.make_node("Sub", [f"{prefix}_input", f"{prefix}_repeat_input"], [f"{prefix}_diff"], name=f"{prefix}_diff"),
                    helper.make_node("Abs", [f"{prefix}_diff"], [f"{prefix}_abs"], name=f"{prefix}_abs"),
                    helper.make_node("ReduceSum", [f"{prefix}_abs", "axes_all"], [f"{prefix}_err"], keepdims=1, name=f"{prefix}_err"),
                    helper.make_node("Equal", [f"{prefix}_err", "zero"], [f"{prefix}_period_ok"], name=f"{prefix}_period_ok"),
                ]
            )
            period_gate = f"{prefix}_period_ok"
            for earlier in prior_period_ok:
                not_earlier = f"{prefix}_not_{earlier}"
                nodes.extend(
                    [
                        helper.make_node("Not", [earlier], [not_earlier], name=not_earlier),
                        helper.make_node("And", [period_gate, not_earlier], [f"{prefix}_period_after_{earlier}"], name=f"{prefix}_period_after_{earlier}"),
                    ]
                )
                period_gate = f"{prefix}_period_after_{earlier}"
            prior_period_ok.append(f"{prefix}_period_ok")

            nodes.extend(
                [
                    helper.make_node("And", [f"{prefix}_width_ok", period_gate], [f"{prefix}_select_bool"], name=f"{prefix}_select_bool"),
                    helper.make_node("Cast", [f"{prefix}_select_bool"], [f"{prefix}_gate"], to=TensorProto.FLOAT, name=f"{prefix}_gate"),
                    helper.make_node("Slice", [f"{prefix}_repeat_long", f"{prefix}_starts", f"{prefix}_out_ends", "axes4", "steps4"], [f"{prefix}_repeat"], name=f"{prefix}_repeat_out"),
                    helper.make_node("Pad", [f"{prefix}_repeat", f"{prefix}_pad", "zero"], [f"{prefix}_padded"], name=f"{prefix}_pad"),
                    helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_gate"], [f"{prefix}_selected"], name=f"{prefix}_select"),
                ]
            )
            branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_width_branches"))

    graph = helper.make_graph(
        nodes,
        "task231_repeat_width",
        [vi("input")],
        [vi("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task231_repeat_width",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task231_repeat_width")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = make_model().SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task231.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task231 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    (out_dir / "task231.onnx").write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task231.onnx", raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
