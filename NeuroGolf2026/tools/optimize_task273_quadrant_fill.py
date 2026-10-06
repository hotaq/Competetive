#!/usr/bin/env python3
"""Build a clean static quadrant-fill model for NeuroGolf task273."""

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


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def quadrant_nodes(src: str, prefix: str, *, rev_h: bool, rev_w: bool) -> list[onnx.NodeProto]:
    nodes: list[onnx.NodeProto] = []
    cur = src
    if rev_h:
        nodes.append(helper.make_node("Gather", [cur, "rev30"], [f"{prefix}_rev_h"], axis=2, name=f"{prefix}_rev_h"))
        cur = f"{prefix}_rev_h"
    if rev_w:
        nodes.append(helper.make_node("Gather", [cur, "rev30"], [f"{prefix}_rev_w"], axis=3, name=f"{prefix}_rev_w"))
        cur = f"{prefix}_rev_w"
    nodes.extend(
        [
            helper.make_node("CumSum", [cur, "axis_h"], [f"{prefix}_cum_h"], name=f"{prefix}_cum_h"),
            helper.make_node("CumSum", [f"{prefix}_cum_h", "axis_w"], [f"{prefix}_cum_hw"], name=f"{prefix}_cum_hw"),
            make_slice(f"{prefix}_cum_hw", f"{prefix}_body", "minus_last"),
            helper.make_node("Pad", [f"{prefix}_body", "pad_top_left", "zero_scalar"], [f"{prefix}_shift"], name=f"{prefix}_shift"),
        ]
    )
    cur = f"{prefix}_shift"
    if rev_w:
        nodes.append(helper.make_node("Gather", [cur, "rev30"], [f"{prefix}_unrev_w"], axis=3, name=f"{prefix}_unrev_w"))
        cur = f"{prefix}_unrev_w"
    if rev_h:
        nodes.append(helper.make_node("Gather", [cur, "rev30"], [f"{prefix}_unrev_h"], axis=2, name=f"{prefix}_unrev_h"))
        cur = f"{prefix}_unrev_h"
    nodes.append(helper.make_node("Greater", [cur, "half"], [f"{prefix}_has"], name=f"{prefix}_has"))
    return nodes


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("rev30", list(reversed(range(30)))),
        const_i64("pad_top_left", [0, 0, 1, 1, 0, 0, 0, 0]),
        const_f32("zero_scalar", 0.0),
        const_f32("half", 0.5),
        *slice_4d("ch0", [0, 0, 0, 0], [1, 1, H, W]),
        *slice_4d("ch4", [0, 4, 0, 0], [1, 5, H, W]),
        *slice_4d("minus_last", [0, 0, 0, 0], [1, 1, H - 1, W - 1]),
    ]

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "x0", "ch0"),
        make_slice("input", "x4", "ch4"),
        helper.make_node("Sub", ["x0", "x0"], ["zero"], name="zero"),
        helper.make_node("Greater", ["x0", "half"], ["is_background"], name="is_background"),
    ]
    nodes.extend(quadrant_nodes("x4", "nw", rev_h=False, rev_w=False))
    nodes.extend(quadrant_nodes("x4", "ne", rev_h=False, rev_w=True))
    nodes.extend(quadrant_nodes("x4", "sw", rev_h=True, rev_w=False))
    nodes.extend(quadrant_nodes("x4", "se", rev_h=True, rev_w=True))
    nodes.extend(
        [
            helper.make_node("And", ["nw_has", "ne_has"], ["top_corners"], name="top_corners"),
            helper.make_node("And", ["sw_has", "se_has"], ["bottom_corners"], name="bottom_corners"),
            helper.make_node("And", ["top_corners", "bottom_corners"], ["inside_rect"], name="inside_rect"),
            helper.make_node("And", ["inside_rect", "is_background"], ["paint_mask"], name="paint_mask"),
            helper.make_node("Where", ["paint_mask", "zero", "x0"], ["out0"], name="out0"),
            helper.make_node("Where", ["paint_mask", "x0", "zero"], ["out2"], name="out2"),
        ]
    )
    zero_channels = [f"z{i}" for i in range(7)]
    for name in zero_channels:
        nodes.append(helper.make_node("Identity", ["zero"], [name], name=name))
    nodes.append(
        helper.make_node(
            "Concat",
            ["out0", "z0", "out2", "z1", "x4", "z2", "z3", "z4", "z5", "z6"],
            ["output"],
            axis=1,
            name="assemble_output",
        )
    )

    graph = helper.make_graph(
        nodes,
        "task273_quadrant_fill",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task273_quadrant_fill",
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
    parser = argparse.ArgumentParser(description="Optimize task273 with a clean quadrant-fill graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v88_plus_task325_v89/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task273_quadrant_fill_v90"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task273.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task273 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task273.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task273 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task273.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task273.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task273",
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
