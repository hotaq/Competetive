#!/usr/bin/env python3
"""Build a clean static CumSum fill model for NeuroGolf task367."""

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


def const_i64_scalar(name: str, value: int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("rev_h", list(reversed(range(H)))),
        const_i64("rev_w", list(reversed(range(W)))),
        const_f32("half", 0.5),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, H, W]),
        *slice_4d("ch5", [0, 5, 0, 0], [1, 6, H, W]),
    ]

    zero_channels = [f"z{i}" for i in range(8)]
    nodes: list[onnx.NodeProto] = [
        make_slice("input", "x0", "ch0"),
        make_slice("input", "x5", "ch5"),
        helper.make_node("Sub", ["x0", "x0"], ["zero"], name="zero"),
        helper.make_node("Greater", ["x0", "half"], ["is_background"], name="is_background"),
        helper.make_node("CumSum", ["x5", "axis_w"], ["left_count"], name="left_count"),
        helper.make_node("Greater", ["left_count", "half"], ["left_has"], name="left_has"),
        helper.make_node("Gather", ["x5", "rev_w"], ["x5_rev_w"], axis=3, name="x5_rev_w"),
        helper.make_node("CumSum", ["x5_rev_w", "axis_w"], ["right_count_rev"], name="right_count_rev"),
        helper.make_node("Gather", ["right_count_rev", "rev_w"], ["right_count"], axis=3, name="right_unrev"),
        helper.make_node("Greater", ["right_count", "half"], ["right_has"], name="right_has"),
        helper.make_node("CumSum", ["x5", "axis_h"], ["up_count"], name="up_count"),
        helper.make_node("Greater", ["up_count", "half"], ["up_has"], name="up_has"),
        helper.make_node("Gather", ["x5", "rev_h"], ["x5_rev_h"], axis=2, name="x5_rev_h"),
        helper.make_node("CumSum", ["x5_rev_h", "axis_h"], ["down_count_rev"], name="down_count_rev"),
        helper.make_node("Gather", ["down_count_rev", "rev_h"], ["down_count"], axis=2, name="down_unrev"),
        helper.make_node("Greater", ["down_count", "half"], ["down_has"], name="down_has"),
        helper.make_node("And", ["left_has", "right_has"], ["hspan"], name="hspan"),
        helper.make_node("And", ["up_has", "down_has"], ["vspan"], name="vspan"),
        helper.make_node("And", ["hspan", "vspan"], ["inside_frame"], name="inside_frame"),
        helper.make_node("And", ["inside_frame", "is_background"], ["paint_mask"], name="paint_mask"),
        helper.make_node("Where", ["paint_mask", "zero", "x0"], ["out0"], name="out0_background"),
        helper.make_node("Where", ["paint_mask", "x0", "zero"], ["out4"], name="out4_paint"),
    ]
    for name in zero_channels:
        nodes.append(helper.make_node("Identity", ["zero"], [name], name=name))
    nodes.append(
        helper.make_node(
            "Concat",
            ["out0", "z0", "z1", "z2", "out4", "x5", "z3", "z4", "z5", "z6"],
            ["output"],
            axis=1,
            name="assemble_output",
        )
    )

    graph = helper.make_graph(
        nodes,
        "task367_cumsum_fill",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task367_cumsum_fill",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def write_zip_with_task(base_zip: Path, out_zip: Path, task_name: str, raw: bytes) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            if name == task_name:
                zout.writestr(name, raw)
            else:
                zout.writestr(name, zin.read(name))
        if task_name not in set(zin.namelist()):
            zout.writestr(task_name, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize task367 with a clean CumSum span-fill graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v87_all_generated_guard_sweep_v88/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task367_cumsum_fill_v89"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task367.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task367 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task367.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task367 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task367.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task367.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task367",
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
