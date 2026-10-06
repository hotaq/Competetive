#!/usr/bin/env python3
"""Build compact Latin-pattern fill models for task305 and task061."""

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
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
TASKS = {
    "task305": ("add", 16, 16),
    "task061": ("mul", 18, 18),
}


def input_vi() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)


def output_vi() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("output", TensorProto.BOOL, EXPECTED_SHAPE)


def i64(name: str, value: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def bconst(name: str, value: bool) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.bool_), name=name)


def formula_grid(kind: str, h: int, w: int) -> np.ndarray:
    rows = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    cols = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
    if kind == "add":
        return rows + cols
    if kind == "mul":
        return rows * cols
    raise ValueError(kind)


def make_model(task: str, kind: str, h: int, w: int) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        i64("starts", [0, 0, 0, 0]),
        i64("ends", [1, 10, h, w]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - h, 30 - w]),
        f32("channel_ids", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        f32("formula", formula_grid(kind, h, w)),
        f32("one", 1.0),
        bconst("false", False),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes4", "steps4"], ["small"], name="slice_work_area"),
        helper.make_node("Mul", ["small", "channel_ids"], ["weighted"], name="weight_channels"),
        helper.make_node("ReduceMax", ["weighted"], ["n"], axes=[0, 1, 2, 3], keepdims=1, name="max_color_n"),
        helper.make_node("Div", ["formula", "n"], ["quotient"], name="formula_div_n"),
        helper.make_node("Floor", ["quotient"], ["quotient_floor"], name="quotient_floor"),
        helper.make_node("Mul", ["quotient_floor", "n"], ["multiple"], name="multiple_n"),
        helper.make_node("Sub", ["formula", "multiple"], ["remainder"], name="remainder"),
        helper.make_node("Add", ["remainder", "one"], ["color_index"], name="color_index"),
        helper.make_node("Equal", ["color_index", "channel_ids"], ["small_out"], name="one_hot_bool"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "false"], ["output"], name="pad_to_output"),
    ]
    graph = helper.make_graph(nodes, f"{task}_latin_formula", [input_vi()], [output_vi()], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_latin_formula_tasks",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_latin_formula_tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}
    for task, (kind, h, w) in TASKS.items():
        model_path = out_dir / f"{task}.onnx"
        raw = make_model(task, kind, h, w).SerializeToString()
        model_path.write_bytes(raw)
        examples = all_examples(json.loads(Path(f"data/latest/extracted/{task}.json").read_text()))
        right, wrong, reason = verify_model_bytes(raw, examples)
        macs, memory, params, cost = score_network(model_path)
        print(
            f"{task} verify: right={right} wrong={wrong} examples={len(examples)} "
            f"reason={reason} cost={cost} points={task_points(cost):.6f}"
        )
        if wrong:
            raise SystemExit(1)
        raws[f"{task}.onnx"] = raw
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(raws.items()):
            zf.writestr(name, raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
