#!/usr/bin/env python3
"""Build a clean static pair-line model for NeuroGolf task161."""

from __future__ import annotations

import argparse
import csv
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
H = 30
W = 30


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def or_many(inputs: list[str], output: str, nodes: list[onnx.NodeProto]) -> None:
    cur = inputs[0]
    for i, name in enumerate(inputs[1:], start=1):
        dst = output if i == len(inputs) - 1 else f"{output}_{i}"
        nodes.append(helper.make_node("Or", [cur, name], [dst], name=f"or_{dst}"))
        cur = dst


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axes_hw", [2, 3]),
        const_i64("axes_w", [3]),
        const_i64("axes_h", [2]),
        const_i64("axis_h_scalar", 2),
        const_i64("axis_w_scalar", 3),
        const_i64("tile_w", [1, 1, 1, W]),
        const_i64("tile_h", [1, 1, H, 1]),
        const_i64("rev_h", list(reversed(range(H)))),
        const_i64("rev_w", list(reversed(range(W)))),
        const_f32("half", 0.5),
        const_f32("one_half", 1.5),
        const_f32("three_half", 3.5),
        const_f32("four_half", 4.5),
        const_f32("zero_scalar", 0.0),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, H, W]),
    ]
    for color in range(1, 10):
        inits.extend(slice_4d(f"ch{color}", [0, color, 0, 0], [1, color + 1, H, W]))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["inside"], keepdims=1, name="inside_mask"),
        helper.make_node("ReduceMax", ["inside"], ["row_inside"], axes=[3], keepdims=1, name="row_inside"),
        helper.make_node("ReduceMax", ["inside"], ["col_inside"], axes=[2], keepdims=1, name="col_inside"),
        helper.make_node("ReduceSum", ["row_inside", "axes_h"], ["input_h"], keepdims=1, name="input_h"),
        helper.make_node("ReduceSum", ["col_inside", "axes_w"], ["input_w"], keepdims=1, name="input_w"),
        make_slice("input", "ch0_in", "ch0"),
    ]
    channel_lines: list[str] = []
    for color in range(1, 10):
        p = f"c{color}"
        nodes.extend(
            [
                make_slice("input", p, f"ch{color}"),
                helper.make_node("ReduceSum", [p, "axes_hw"], [f"{p}_count"], keepdims=1, name=f"{p}_count"),
                helper.make_node("Greater", [f"{p}_count", "three_half"], [f"{p}_count_ge4"], name=f"{p}_count_ge4"),
                helper.make_node("Less", [f"{p}_count", "four_half"], [f"{p}_count_lt5"], name=f"{p}_count_lt5"),
                helper.make_node("And", [f"{p}_count_ge4", f"{p}_count_lt5"], [f"{p}_exact4"], name=f"{p}_exact4"),
                helper.make_node("ReduceSum", [p, "axes_w"], [f"{p}_row_count"], keepdims=1, name=f"{p}_row_count"),
                helper.make_node("Greater", [f"{p}_row_count", "one_half"], [f"{p}_row_pair"], name=f"{p}_row_pair"),
                helper.make_node("Cast", [f"{p}_row_pair"], [f"{p}_row_pair_f"], to=TensorProto.FLOAT, name=f"{p}_row_pair_f"),
                helper.make_node("ReduceSum", [f"{p}_row_pair_f", "axes_h"], [f"{p}_row_pair_n"], keepdims=1, name=f"{p}_row_pair_n"),
                helper.make_node("Greater", [f"{p}_row_pair_n", "half"], [f"{p}_has_row_pair"], name=f"{p}_has_row_pair"),
                helper.make_node("Greater", [f"{p}_row_pair_n", "one_half"], [f"{p}_two_row_pairs"], name=f"{p}_two_row_pairs"),
                helper.make_node("ReduceSum", [p, "axes_h"], [f"{p}_col_count"], keepdims=1, name=f"{p}_col_count"),
                helper.make_node("Greater", [f"{p}_col_count", "one_half"], [f"{p}_col_pair"], name=f"{p}_col_pair"),
                helper.make_node("Cast", [f"{p}_col_pair"], [f"{p}_col_pair_f"], to=TensorProto.FLOAT, name=f"{p}_col_pair_f"),
                helper.make_node("ReduceSum", [f"{p}_col_pair_f", "axes_w"], [f"{p}_col_pair_n"], keepdims=1, name=f"{p}_col_pair_n"),
                helper.make_node("Greater", [f"{p}_col_pair_n", "half"], [f"{p}_has_col_pair"], name=f"{p}_has_col_pair"),
                helper.make_node("Greater", [f"{p}_col_pair_n", "one_half"], [f"{p}_two_col_pairs"], name=f"{p}_two_col_pairs"),
                helper.make_node("And", [f"{p}_exact4", f"{p}_has_row_pair"], [f"{p}_exact4_has_row"], name=f"{p}_exact4_has_row"),
                helper.make_node("And", [f"{p}_exact4_has_row", f"{p}_has_col_pair"], [f"{p}_target"], name=f"{p}_target"),
                helper.make_node("And", [f"{p}_two_row_pairs", f"{p}_two_col_pairs"], [f"{p}_rect"], name=f"{p}_rect"),
                helper.make_node("Not", [f"{p}_rect"], [f"{p}_not_rect"], name=f"{p}_not_rect"),
                helper.make_node("CumSum", [f"{p}_row_pair_f", "axis_h_scalar"], [f"{p}_row_left"], name=f"{p}_row_left"),
                helper.make_node("Gather", [f"{p}_row_pair_f", "rev_h"], [f"{p}_row_rev"], axis=2, name=f"{p}_row_rev"),
                helper.make_node("CumSum", [f"{p}_row_rev", "axis_h_scalar"], [f"{p}_row_right_rev"], name=f"{p}_row_right_rev"),
                helper.make_node("Gather", [f"{p}_row_right_rev", "rev_h"], [f"{p}_row_right"], axis=2, name=f"{p}_row_right"),
                helper.make_node("Greater", [f"{p}_row_left", "half"], [f"{p}_row_left_b"], name=f"{p}_row_left_b"),
                helper.make_node("Greater", [f"{p}_row_right", "half"], [f"{p}_row_right_b"], name=f"{p}_row_right_b"),
                helper.make_node("And", [f"{p}_row_left_b", f"{p}_row_right_b"], [f"{p}_row_span_b"], name=f"{p}_row_span_b"),
                helper.make_node("Cast", [f"{p}_row_span_b"], [f"{p}_row_span_f"], to=TensorProto.FLOAT, name=f"{p}_row_span_f"),
                helper.make_node("ReduceSum", [f"{p}_row_span_f", "axes_h"], [f"{p}_row_span"], keepdims=1, name=f"{p}_row_span"),
                helper.make_node("CumSum", [f"{p}_col_pair_f", "axis_w_scalar"], [f"{p}_col_left"], name=f"{p}_col_left"),
                helper.make_node("Gather", [f"{p}_col_pair_f", "rev_w"], [f"{p}_col_rev"], axis=3, name=f"{p}_col_rev"),
                helper.make_node("CumSum", [f"{p}_col_rev", "axis_w_scalar"], [f"{p}_col_right_rev"], name=f"{p}_col_right_rev"),
                helper.make_node("Gather", [f"{p}_col_right_rev", "rev_w"], [f"{p}_col_right"], axis=3, name=f"{p}_col_right"),
                helper.make_node("Greater", [f"{p}_col_left", "half"], [f"{p}_col_left_b"], name=f"{p}_col_left_b"),
                helper.make_node("Greater", [f"{p}_col_right", "half"], [f"{p}_col_right_b"], name=f"{p}_col_right_b"),
                helper.make_node("And", [f"{p}_col_left_b", f"{p}_col_right_b"], [f"{p}_col_span_b"], name=f"{p}_col_span_b"),
                helper.make_node("Cast", [f"{p}_col_span_b"], [f"{p}_col_span_f"], to=TensorProto.FLOAT, name=f"{p}_col_span_f"),
                helper.make_node("ReduceSum", [f"{p}_col_span_f", "axes_w"], [f"{p}_col_span"], keepdims=1, name=f"{p}_col_span"),
                helper.make_node("Equal", [f"{p}_col_span", "input_w"], [f"{p}_cols_span_full"], name=f"{p}_cols_span_full"),
                helper.make_node("Equal", [f"{p}_row_span", "input_h"], [f"{p}_rows_span_full"], name=f"{p}_rows_span_full"),
                helper.make_node("Or", [f"{p}_not_rect", f"{p}_cols_span_full"], [f"{p}_draw_rows"], name=f"{p}_draw_rows"),
                helper.make_node("Or", [f"{p}_not_rect", f"{p}_rows_span_full"], [f"{p}_draw_cols"], name=f"{p}_draw_cols"),
                helper.make_node("And", [f"{p}_row_pair", f"{p}_draw_rows"], [f"{p}_row_draw"], name=f"{p}_row_draw"),
                helper.make_node("And", [f"{p}_col_pair", f"{p}_draw_cols"], [f"{p}_col_draw"], name=f"{p}_col_draw"),
                helper.make_node("Tile", [f"{p}_row_draw", "tile_w"], [f"{p}_row_full_b"], name=f"{p}_row_full_b"),
                helper.make_node("Tile", [f"{p}_col_draw", "tile_h"], [f"{p}_col_full_b"], name=f"{p}_col_full_b"),
                helper.make_node("Or", [f"{p}_row_full_b", f"{p}_col_full_b"], [f"{p}_line_raw_b"], name=f"{p}_line_raw_b"),
                helper.make_node("And", [f"{p}_line_raw_b", f"{p}_target"], [f"{p}_line_target_b"], name=f"{p}_line_target_b"),
                helper.make_node("Cast", [f"{p}_line_target_b"], [f"{p}_line_target"], to=TensorProto.FLOAT, name=f"{p}_line_target"),
                helper.make_node("Mul", [f"{p}_line_target", "inside"], [f"out{color}"], name=f"out{color}"),
            ]
        )
        channel_lines.append(f"out{color}")

    nodes.extend(
        [
            helper.make_node("Sum", channel_lines, ["nonzero_raw"], name="nonzero_raw"),
            helper.make_node("Greater", ["nonzero_raw", "half"], ["nonzero_b"], name="nonzero_b"),
            helper.make_node("Cast", ["nonzero_b"], ["nonzero"], to=TensorProto.FLOAT, name="nonzero"),
            helper.make_node("Sub", ["inside", "nonzero"], ["out0"], name="out0"),
            helper.make_node("Concat", ["out0", *channel_lines], ["output"], axis=1, name="assemble_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task161_pair_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task161_pair_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def write_zip_with_task(base_zip: Path, out_zip: Path, task_name: str, raw: bytes) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for name in sorted(names):
            zout.writestr(name, raw if name == task_name else zin.read(name))
        if task_name not in names:
            zout.writestr(task_name, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize task161 with a clean target-color pair-line graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v89_plus_task273_v90/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task161_pair_lines_v91"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task161.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task161 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task161.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task161 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task161.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task161.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task161",
                "right": right,
                "wrong": wrong,
                "macs": macs,
                "memory": memory,
                "params": params,
                "cost": cost,
                "points": f"{points:.6f}",
                "onnx_bytes": len(raw),
            }
        )
    print(f"wrote {onnx_path}")
    print(f"one-task zip: {one_task_zip}")
    print(f"full zip: {full_zip}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
