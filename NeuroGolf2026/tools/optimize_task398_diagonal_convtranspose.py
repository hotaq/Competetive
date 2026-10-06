#!/usr/bin/env python3
"""Build a clean static diagonal-expansion model for NeuroGolf task398."""

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


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def diagonal_kernel(n: int) -> np.ndarray:
    kernel = np.zeros((9, 1, n, n), dtype=np.float32)
    for ch in range(9):
        for r in range(n):
            kernel[ch, 0, r, n - 1 - r] = 1.0
    return kernel


def make_model() -> onnx.ModelProto:
    row_grid = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1).repeat(W, axis=3)
    col_grid = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W).repeat(H, axis=2)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axes_count", [1, 2, 3]),
        const_i64("axis_channel", [1]),
        const_f32("zero_scalar", 0.0),
        const_f32("five", 5.0),
        const_f32("row_grid", row_grid),
        const_f32("col_grid", col_grid),
        *slice_4d("row_nonzero", [0, 1, 0, 0], [1, 10, 1, 5]),
    ]

    for k in range(1, 6):
        n = 5 * k
        inits.extend(slice_4d(f"crop_{n}", [0, 0, 0, 0], [1, 9, n, n]))
        inits.append(const_i64(f"pad_{n}", [0, 0, 0, 0, 0, 0, H - n, W - n]))
        inits.append(const_f32(f"count_{k}", float(k)))
        inits.append(const_f32(f"w_{n}", diagonal_kernel(n)))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "row_nonzero", "row_nonzero"),
        helper.make_node("ReduceSum", ["row_nonzero", "axes_count"], ["nonzero_count"], keepdims=1, name="nonzero_count"),
        helper.make_node("Mul", ["nonzero_count", "five"], ["canvas_size"], name="canvas_size"),
        helper.make_node("Less", ["row_grid", "canvas_size"], ["row_inside"], name="row_inside"),
        helper.make_node("Less", ["col_grid", "canvas_size"], ["col_inside"], name="col_inside"),
        helper.make_node("And", ["row_inside", "col_inside"], ["canvas_bool"], name="canvas_bool"),
        helper.make_node("Cast", ["canvas_bool"], ["canvas"], to=TensorProto.FLOAT, name="canvas"),
    ]

    branch_terms: list[str] = []
    for k in range(1, 6):
        n = 5 * k
        nodes.extend(
            [
                helper.make_node(
                    "ConvTranspose",
                    ["row_nonzero", f"w_{n}"],
                    [f"diag_raw_{n}"],
                    group=9,
                    name=f"diag_raw_{n}",
                ),
                make_slice(f"diag_raw_{n}", f"diag_crop_{n}", f"crop_{n}"),
                helper.make_node("Pad", [f"diag_crop_{n}", f"pad_{n}", "zero_scalar"], [f"diag_30_{n}"], name=f"diag_30_{n}"),
                helper.make_node("Equal", ["nonzero_count", f"count_{k}"], [f"gate_{k}_bool"], name=f"gate_{k}_bool"),
                helper.make_node("Cast", [f"gate_{k}_bool"], [f"gate_{k}"], to=TensorProto.FLOAT, name=f"gate_{k}"),
                helper.make_node("Mul", [f"diag_30_{n}", f"gate_{k}"], [f"diag_term_{n}"], name=f"diag_term_{n}"),
            ]
        )
        branch_terms.append(f"diag_term_{n}")

    nodes.extend(
        [
            helper.make_node("Sum", branch_terms, ["nonzero_out"], name="nonzero_out"),
            helper.make_node("ReduceSum", ["nonzero_out", "axis_channel"], ["occupied"], keepdims=1, name="occupied"),
            helper.make_node("Sub", ["canvas", "occupied"], ["out0"], name="out0"),
            helper.make_node("Concat", ["out0", "nonzero_out"], ["output"], axis=1, name="assemble_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task398_diagonal_convtranspose",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task398_diagonal_convtranspose",
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
    parser = argparse.ArgumentParser(description="Optimize task398 with a clean diagonal ConvTranspose graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v88_plus_task325_v89/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task398_diagonal_convtranspose_v90"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task398.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task398 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task398.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task398 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task398.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task398.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task398",
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
