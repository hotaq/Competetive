#!/usr/bin/env python3
"""Build a clean static 4-neighbor hole-fill model for NeuroGolf task002."""

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
ITERS = 20


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


def make_model() -> onnx.ModelProto:
    cross4 = np.asarray([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32)
    cross5 = np.asarray([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_f32("cross4", cross4),
        const_f32("cross5", cross5),
        const_f32("half", 0.5),
        const_f32("three_half", 3.5),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, H, W]),
        *slice_4d("ch3", [0, 3, 0, 0], [1, 4, H, W]),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["inside"], keepdims=1, name="inside"),
        make_slice("input", "bg", "ch0"),
        make_slice("input", "ch3", "ch3"),
        helper.make_node("Conv", ["inside", "cross4"], ["inside_n4"], pads=[1, 1, 1, 1], name="inside_neighbor4_count"),
        helper.make_node("Less", ["inside_n4", "three_half"], ["boundary_b"], name="grid_boundary_bool"),
        helper.make_node("Cast", ["boundary_b"], ["boundary"], to=TensorProto.FLOAT, name="grid_boundary"),
        helper.make_node("Mul", ["bg", "boundary"], ["ext0"], name="exterior_seed"),
    ]

    cur = "ext0"
    for i in range(ITERS):
        nodes.extend(
            [
                helper.make_node("Conv", [cur, "cross5"], [f"ext_count_{i}"], pads=[1, 1, 1, 1], name=f"ext_count_{i}"),
                helper.make_node("Greater", [f"ext_count_{i}", "half"], [f"ext_b_{i}"], name=f"ext_b_{i}"),
                helper.make_node("Cast", [f"ext_b_{i}"], [f"ext_f_{i}"], to=TensorProto.FLOAT, name=f"ext_f_{i}"),
                helper.make_node("Mul", [f"ext_f_{i}", "bg"], [f"ext{i + 1}"], name=f"ext{i + 1}"),
            ]
        )
        cur = f"ext{i + 1}"

    nodes.extend(
        [
            helper.make_node("Sub", ["bg", cur], ["hole"], name="hole_interior"),
            helper.make_node("Sub", ["bg", "hole"], ["out0"], name="background_out"),
            helper.make_node("Sub", ["hole", "hole"], ["zero"], name="zero"),
        ]
    )
    zero_channels = [f"z{i}" for i in range(7)]
    for name in zero_channels:
        nodes.append(helper.make_node("Identity", ["zero"], [name], name=name))
    nodes.append(
        helper.make_node(
            "Concat",
            ["out0", "z0", "z1", "ch3", "hole", "z2", "z3", "z4", "z5", "z6"],
            ["output"],
            axis=1,
            name="assemble_output",
        )
    )

    graph = helper.make_graph(
        nodes,
        "task002_hole_fill",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task002_hole_fill",
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
    parser = argparse.ArgumentParser(description="Optimize task002 with clean 4-neighbor hole fill.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v90_plus_task020_161_v91/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task002_hole_fill_v92"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task002.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task002 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task002.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task002 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task002.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task002.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task002",
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
