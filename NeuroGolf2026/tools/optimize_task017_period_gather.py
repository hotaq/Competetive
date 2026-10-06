#!/usr/bin/env python3
"""Build a clean static period-completion model for NeuroGolf task017."""

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
H = 21
W = 21
PERIODS = [2, 4, 5, 6, 7, 8, 9]


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


def max_chain(inputs: list[str], output: str, nodes: list[onnx.NodeProto]) -> None:
    cur = inputs[0]
    for i, name in enumerate(inputs[1:], start=1):
        dst = output if i == len(inputs) - 1 else f"{output}_{i}"
        nodes.append(helper.make_node("Max", [cur, name], [dst], name=f"max_{dst}"))
        cur = dst


def period_candidate_nodes(
    period: int,
    prior_ok: str | None,
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
) -> tuple[str, str]:
    prefix = f"p{period}"
    class_values: list[str] = []
    for rr in range(period):
        for cc in range(period):
            flat_indices = [r * W + c for r in range(rr, H, period) for c in range(cc, W, period)]
            name = f"{prefix}_{rr}_{cc}"
            inits.append(const_i64(f"{name}_idx", flat_indices))
            nodes.extend(
                [
                    helper.make_node("Gather", ["xflat", f"{name}_idx"], [f"{name}_pixels"], axis=2, name=f"{name}_gather"),
                    helper.make_node("ReduceMax", [f"{name}_pixels"], [f"{name}_color"], axes=[2], keepdims=1, name=f"{name}_color"),
                ]
            )
            class_values.append(f"{name}_color")

    out_map = [(r % period) * period + (c % period) for r in range(H) for c in range(W)]
    inits.extend(
        [
            const_i64(f"{prefix}_out_idx", out_map),
            const_f32(f"{prefix}_class_count", float(period * period)),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Concat", class_values, [f"{prefix}_template"], axis=2, name=f"{prefix}_template"),
            helper.make_node("ReduceSum", [f"{prefix}_template", "axes_template"], [f"{prefix}_sum"], keepdims=0, name=f"{prefix}_sum"),
            helper.make_node("Equal", [f"{prefix}_sum", f"{prefix}_class_count"], [f"{prefix}_ok_raw"], name=f"{prefix}_ok_raw"),
        ]
    )
    if prior_ok is None:
        nodes.append(helper.make_node("Identity", [f"{prefix}_ok_raw"], [f"{prefix}_selected_b"], name=f"{prefix}_selected_b"))
        next_prior = f"{prefix}_ok_raw"
    else:
        nodes.extend(
            [
                helper.make_node("Not", [prior_ok], [f"{prefix}_prior_not"], name=f"{prefix}_prior_not"),
                helper.make_node("And", [f"{prefix}_ok_raw", f"{prefix}_prior_not"], [f"{prefix}_selected_b"], name=f"{prefix}_selected_b"),
                helper.make_node("Or", [prior_ok, f"{prefix}_ok_raw"], [f"{prefix}_prior_next"], name=f"{prefix}_prior_next"),
            ]
        )
        next_prior = f"{prefix}_prior_next"

    nodes.extend(
        [
            helper.make_node("Gather", [f"{prefix}_template", f"{prefix}_out_idx"], [f"{prefix}_out_flat"], axis=2, name=f"{prefix}_out_flat"),
            helper.make_node("Reshape", [f"{prefix}_out_flat", "shape_1_9_21_21"], [f"{prefix}_out"], name=f"{prefix}_out"),
            helper.make_node("Cast", [f"{prefix}_selected_b"], [f"{prefix}_selected"], to=TensorProto.FLOAT, name=f"{prefix}_selected"),
            helper.make_node("Mul", [f"{prefix}_out", f"{prefix}_selected"], [f"{prefix}_weighted"], name=f"{prefix}_weighted"),
        ]
    )
    return f"{prefix}_weighted", next_prior


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axes_template", [1, 2]),
        const_i64("axis_channel", [1]),
        const_i64("shape_1_9_441", [1, 9, H * W]),
        const_i64("shape_1_9_21_21", [1, 9, H, W]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - H, 30 - W]),
        const_f32("zero_scalar", 0.0),
        *slice_4d("x19", [0, 1, 0, 0], [1, 10, H, W]),
    ]
    nodes: list[onnx.NodeProto] = [
        make_slice("input", "x19", "x19"),
        helper.make_node("Reshape", ["x19", "shape_1_9_441"], ["xflat"], name="xflat"),
    ]

    candidates: list[str] = []
    prior_ok: str | None = None
    for period in PERIODS:
        candidate, prior_ok = period_candidate_nodes(period, prior_ok, nodes, inits)
        candidates.append(candidate)

    max_chain(candidates, "out19", nodes)
    nodes.extend(
        [
            helper.make_node("ReduceSum", ["out19", "axis_channel"], ["nonzero"], keepdims=1, name="nonzero"),
            helper.make_node("Sub", ["nonzero", "nonzero"], ["out0"], name="out0"),
            helper.make_node("Concat", ["out0", "out19"], ["out10"], axis=1, name="out10"),
            helper.make_node("Pad", ["out10", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task017_period_gather",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task017_period_gather",
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
    parser = argparse.ArgumentParser(description="Optimize task017 with a clean period-template gather graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v90_plus_task020_161_v91/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task017_period_gather_v92"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task017.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task017 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task017.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task017 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task017.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task017.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task017",
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
