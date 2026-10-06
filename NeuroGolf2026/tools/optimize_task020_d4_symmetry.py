#!/usr/bin/env python3
"""Build a clean static D4-symmetry completion model for NeuroGolf task020."""

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
H = 10
W = 10
B = 5


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


def selected_candidate_nodes(top: int, left: int, nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto]) -> str:
    prefix = f"c{top}_{left}"
    inits.extend(
        [
            *slice_4d(prefix, [0, 0, top, left], [1, 9, top + B, left + B]),
            const_i64(f"{prefix}_pads", [0, 0, top, left, 0, 0, H - top - B, W - left - B]),
            const_i64(f"{prefix}_top", top),
            const_i64(f"{prefix}_left", left),
        ]
    )
    nodes.extend(
        [
            make_slice("x19", f"{prefix}_box", prefix),
            helper.make_node("Gather", [f"{prefix}_box", "rev5"], [f"{prefix}_fh"], axis=2, name=f"{prefix}_fh"),
            helper.make_node("Gather", [f"{prefix}_box", "rev5"], [f"{prefix}_fw"], axis=3, name=f"{prefix}_fw"),
            helper.make_node("Gather", [f"{prefix}_fh", "rev5"], [f"{prefix}_fb"], axis=3, name=f"{prefix}_fb"),
            helper.make_node("Transpose", [f"{prefix}_box"], [f"{prefix}_tr"], perm=[0, 1, 3, 2], name=f"{prefix}_tr"),
            helper.make_node("Gather", [f"{prefix}_tr", "rev5"], [f"{prefix}_tr_fh"], axis=2, name=f"{prefix}_tr_fh"),
            helper.make_node("Gather", [f"{prefix}_tr", "rev5"], [f"{prefix}_tr_fw"], axis=3, name=f"{prefix}_tr_fw"),
            helper.make_node("Gather", [f"{prefix}_tr_fh", "rev5"], [f"{prefix}_tr_fb"], axis=3, name=f"{prefix}_tr_fb"),
        ]
    )
    max_chain(
        [
            f"{prefix}_box",
            f"{prefix}_fh",
            f"{prefix}_fw",
            f"{prefix}_fb",
            f"{prefix}_tr",
            f"{prefix}_tr_fh",
            f"{prefix}_tr_fw",
            f"{prefix}_tr_fb",
        ],
        f"{prefix}_box_d4",
        nodes,
    )
    nodes.extend(
        [
            helper.make_node("Pad", [f"{prefix}_box_d4", f"{prefix}_pads", "zero_scalar"], [f"{prefix}_d4"], name=f"{prefix}_pad10"),
            helper.make_node("Equal", ["top", f"{prefix}_top"], [f"{prefix}_top_ok"], name=f"{prefix}_top_ok"),
            helper.make_node("Equal", ["left", f"{prefix}_left"], [f"{prefix}_left_ok"], name=f"{prefix}_left_ok"),
            helper.make_node("And", [f"{prefix}_top_ok", f"{prefix}_left_ok"], [f"{prefix}_ok"], name=f"{prefix}_ok"),
            helper.make_node("Where", [f"{prefix}_ok", f"{prefix}_d4", "zero19"], [f"{prefix}_selected"], name=f"{prefix}_select"),
        ]
    )
    return f"{prefix}_selected"


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("rev5", [4, 3, 2, 1, 0]),
        const_i64("pad10_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("zero_scalar", 0.0),
        const_f32("one10", np.ones((1, 1, H, W), dtype=np.float32)),
        *slice_4d("crop10", [0, 0, 0, 0], [1, 10, H, W]),
        *slice_4d("nonzero_ch", [0, 1, 0, 0], [1, 10, H, W]),
    ]

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "x10", "crop10"),
        make_slice("x10", "x19", "nonzero_ch"),
        helper.make_node("ReduceMax", ["x19"], ["occ"], axes=[1], keepdims=1, name="occupancy"),
        helper.make_node("ReduceMax", ["occ"], ["row_has"], axes=[3], keepdims=1, name="row_has"),
        helper.make_node("ReduceMax", ["occ"], ["col_has"], axes=[2], keepdims=1, name="col_has"),
        helper.make_node("ArgMax", ["row_has"], ["top"], axis=2, keepdims=1, name="top_row"),
        helper.make_node("ArgMax", ["col_has"], ["left"], axis=3, keepdims=1, name="left_col"),
        helper.make_node("Sub", ["x19", "x19"], ["zero19"], name="zero19"),
    ]

    selected: list[str] = []
    # Across all task020 examples the active 5x5 bbox starts in rows/cols 1..4.
    for top in range(1, 5):
        for left in range(1, 5):
            selected.append(selected_candidate_nodes(top, left, nodes, inits))
    max_chain(selected, "out19", nodes)
    nodes.extend(
        [
            helper.make_node("ReduceMax", ["out19"], ["nonzero_out"], axes=[1], keepdims=1, name="nonzero_out"),
            helper.make_node("Sub", ["one10", "nonzero_out"], ["out0"], name="background0"),
            helper.make_node("Concat", ["out0", "out19"], ["out10"], axis=1, name="assemble10"),
            helper.make_node("Pad", ["out10", "pad10_to_30", "zero_scalar"], ["output"], name="pad_output30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task020_d4_symmetry",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task020_d4_symmetry",
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
    parser = argparse.ArgumentParser(description="Optimize task020 with a clean D4-symmetry completion graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v89_plus_task273_v90/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task020_d4_symmetry_v91"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task020.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task020 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task020.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task020 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task020.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task020.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task020",
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
