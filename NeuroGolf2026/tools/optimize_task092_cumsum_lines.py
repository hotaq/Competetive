#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task092.

Same-color endpoint pairs define horizontal or vertical line segments.  This
uses cumulative sums over all non-background color channels to fill between the
first and last endpoint in each row/column.
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


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("bg_starts", [0, 0, 0, 0]),
        const_i64("bg_ends", [1, 1, 30, 30]),
        const_i64("slice_starts", [0, 1, 0, 0]),
        const_i64("slice_ends", [1, 10, 30, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_h", 2),
        const_i64("axis_w", 3),
        const_i64("axis_channel", [1]),
        const_i64("reduce_h", [2]),
        const_i64("reduce_w", [3]),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
        const_f32("ones", np.ones((1, 1, 30, 30), dtype=np.float32)),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "bg_starts", "bg_ends", "axes4", "steps4"], ["background_in"], name="slice_background"),
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "axes4", "steps4"], ["colors_in"], name="drop_background"),
        helper.make_node("Greater", ["colors_in", "half"], ["seed_bool"], name="seed_bool"),
        helper.make_node("Cast", ["seed_bool"], ["seeds"], to=TensorProto.FLOAT, name="seed_cast"),
        helper.make_node("ReduceSum", ["seeds", "axis_channel"], ["seed_count"], keepdims=1, name="seed_count"),
        helper.make_node("Less", ["seed_count", "half"], ["no_seed"], name="no_seed"),
        helper.make_node("CumSum", ["seeds", "axis_w"], ["cum_w"], name="cum_w"),
        helper.make_node("ReduceSum", ["seeds", "reduce_w"], ["row_total"], keepdims=1, name="row_total"),
        helper.make_node("Greater", ["row_total", "one"], ["row_pair"], name="row_pair"),
        helper.make_node("Greater", ["cum_w", "half"], ["left_seen"], name="left_seen"),
        helper.make_node("Sub", ["row_total", "cum_w"], ["right_after"], name="right_after"),
        helper.make_node("Add", ["right_after", "seeds"], ["right_inclusive"], name="right_inclusive"),
        helper.make_node("Greater", ["right_inclusive", "half"], ["right_seen"], name="right_seen"),
        helper.make_node("And", ["left_seen", "right_seen"], ["h_span_a"], name="h_span_a"),
        helper.make_node("And", ["h_span_a", "row_pair"], ["h_span"], name="h_span"),
        helper.make_node("CumSum", ["seeds", "axis_h"], ["cum_h"], name="cum_h"),
        helper.make_node("ReduceSum", ["seeds", "reduce_h"], ["col_total"], keepdims=1, name="col_total"),
        helper.make_node("Greater", ["col_total", "one"], ["col_pair"], name="col_pair"),
        helper.make_node("Greater", ["cum_h", "half"], ["top_seen"], name="top_seen"),
        helper.make_node("Sub", ["col_total", "cum_h"], ["below_after"], name="below_after"),
        helper.make_node("Add", ["below_after", "seeds"], ["below_inclusive"], name="below_inclusive"),
        helper.make_node("Greater", ["below_inclusive", "half"], ["below_seen"], name="below_seen"),
        helper.make_node("And", ["top_seen", "below_seen"], ["v_span_a"], name="v_span_a"),
        helper.make_node("And", ["v_span_a", "col_pair"], ["v_span"], name="v_span"),
        helper.make_node("Cast", ["v_span"], ["v_span_f"], to=TensorProto.FLOAT, name="v_span_cast"),
        helper.make_node("ReduceSum", ["v_span_f", "axis_channel"], ["v_span_count"], keepdims=1, name="v_span_count"),
        helper.make_node("Less", ["v_span_count", "half"], ["no_vertical_span"], name="no_vertical_span"),
        helper.make_node("And", ["h_span", "no_vertical_span"], ["h_span_masked"], name="h_span_masked"),
        helper.make_node("Or", ["h_span_masked", "v_span"], ["span_any"], name="span_any"),
        helper.make_node("And", ["span_any", "no_seed"], ["paint_span"], name="paint_span"),
        helper.make_node("Or", ["seed_bool", "paint_span"], ["colors_bool"], name="colors_bool"),
        helper.make_node("Cast", ["colors_bool"], ["colors"], to=TensorProto.FLOAT, name="colors_cast"),
        helper.make_node("ReduceSum", ["colors", "axis_channel"], ["occupied_count"], keepdims=1, name="occupied_count"),
        helper.make_node("Greater", ["occupied_count", "half"], ["occupied_bool"], name="occupied_bool"),
        helper.make_node("Cast", ["occupied_bool"], ["occupied"], to=TensorProto.FLOAT, name="occupied_cast"),
        helper.make_node("Sub", ["ones", "occupied"], ["empty_mask"], name="empty_mask"),
        helper.make_node("Mul", ["background_in", "empty_mask"], ["ch0"], name="background_channel"),
        helper.make_node("Concat", ["ch0", "colors"], ["output"], axis=1, name="assemble_output"),
    ]

    graph = helper.make_graph(
        nodes,
        "task092_cumsum_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task092_cumsum_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task092_cumsum_lines_v95"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task092.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task092 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task092.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task092.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task092 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
