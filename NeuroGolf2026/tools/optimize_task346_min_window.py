#!/usr/bin/env python3
"""Build a compact clean ONNX model for NeuroGolf task346.

The task has two non-background colors.  The answer is the color whose largest
3x3 same-color window count is smaller; the distractor color forms a dense
object while the answer color is scattered.
"""

from __future__ import annotations

import argparse
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


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | list[float] | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("crop_starts", [0, 1, 0, 0]),
        const_i64("crop_ends", [1, 10, 12, 12]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_f32("window_weight", np.ones((9, 1, 3, 3), dtype=np.float32)),
        const_f32("one", 1.0),
        const_f32("absent_penalty", 100.0),
        const_f32("zero_scalar", 0.0),
        const_f32("zero_cell", np.zeros((1, 1, 1, 1), dtype=np.float32)),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 29, 29]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["colors12"], name="slice_colors12"),
        helper.make_node("Conv", ["colors12", "window_weight"], ["window_counts"], group=9, name="same_color_3x3_counts"),
        helper.make_node("ReduceMax", ["window_counts"], ["max_window"], axes=[2, 3], keepdims=1, name="max_window_by_color"),
        helper.make_node("ReduceMax", ["colors12"], ["present"], axes=[2, 3], keepdims=1, name="present_by_color"),
        helper.make_node("Sub", ["one", "present"], ["absent"], name="absent_mask"),
        helper.make_node("Mul", ["absent", "absent_penalty"], ["penalty"], name="absent_penalty"),
        helper.make_node("Add", ["max_window", "penalty"], ["score"], name="penalized_score"),
        helper.make_node("ReduceMin", ["score"], ["best_score"], axes=[1], keepdims=1, name="best_score"),
        helper.make_node("Equal", ["score", "best_score"], ["winner_bool"], name="winner_equal_best"),
        helper.make_node("Cast", ["winner_bool"], ["winner_colors"], to=TensorProto.FLOAT, name="winner_cast"),
        helper.make_node("Concat", ["zero_cell", "winner_colors"], ["answer_cell"], axis=1, name="prepend_background_channel"),
        helper.make_node("Pad", ["answer_cell", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
    ]

    graph = helper.make_graph(
        nodes,
        "task346_min_window",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task346_min_window",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task346_min_window_v101"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task346.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task346 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task346.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task346.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task346 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
