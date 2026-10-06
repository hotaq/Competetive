#!/usr/bin/env python3
"""Build a clean static component-count model for NeuroGolf task325."""

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


def or_many(names: list[str], output: str, nodes: list[onnx.NodeProto]) -> None:
    cur = names[0]
    for i, name in enumerate(names[1:], start=1):
        nxt = output if i == len(names) - 1 else f"{output}_{i}"
        nodes.append(helper.make_node("Or", [cur, name], [nxt], name=f"or_{nxt}"))
        cur = nxt


def make_model() -> onnx.ModelProto:
    row_grid = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1).repeat(W, axis=3)
    col_grid = np.arange(W, dtype=np.float32).reshape(1, 1, 1, W).repeat(H, axis=2)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("pad_border", [0, 0, 1, 1, 0, 0, 1, 1]),
        const_f32("zero_scalar", 0.0),
        const_f32("half", 0.5),
        const_f32("two", 2.0),
        const_f32("quarter", 0.25),
        const_f32("eight", 8.0),
        const_f32("row_grid", row_grid),
        const_f32("col_grid", col_grid),
        const_f32("code_w", np.asarray([[[[1.0, 2.0], [4.0, 8.0]]]], dtype=np.float32)),
        const_f32("neighbor_w", np.asarray([[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=np.float32)),
        *slice_4d("ch8", [0, 8, 0, 0], [1, 9, H, W]),
    ]
    for code in [1, 2, 4, 8, 7, 11, 13, 14, 6, 9]:
        inits.append(const_f32(f"code_{code}", float(code)))

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "x8", "ch8"),
        helper.make_node("Pad", ["x8", "pad_border", "zero_scalar"], ["x8_pad"], name="pad_for_2x2"),
        helper.make_node("Conv", ["x8_pad", "code_w"], ["code"], name="two_by_two_code"),
    ]

    code_bools: dict[int, str] = {}
    for code in [1, 2, 4, 8, 7, 11, 13, 14, 6, 9]:
        name = f"is_code_{code}"
        nodes.append(helper.make_node("Equal", ["code", f"code_{code}"], [name], name=name))
        code_bools[code] = name
    or_many([code_bools[c] for c in [1, 2, 4, 8]], "q1_bool", nodes)
    or_many([code_bools[c] for c in [7, 11, 13, 14]], "q3_bool", nodes)
    or_many([code_bools[c] for c in [6, 9]], "qd_bool", nodes)

    nodes.extend(
        [
            helper.make_node("Cast", ["q1_bool"], ["q1_float"], to=TensorProto.FLOAT, name="q1_float"),
            helper.make_node("Cast", ["q3_bool"], ["q3_float"], to=TensorProto.FLOAT, name="q3_float"),
            helper.make_node("Cast", ["qd_bool"], ["qd_float"], to=TensorProto.FLOAT, name="qd_float"),
            helper.make_node("ReduceSum", ["q1_float", "axes_all"], ["q1"], keepdims=1, name="q1"),
            helper.make_node("ReduceSum", ["q3_float", "axes_all"], ["q3"], keepdims=1, name="q3"),
            helper.make_node("ReduceSum", ["qd_float", "axes_all"], ["qd"], keepdims=1, name="qd"),
            helper.make_node("Mul", ["qd", "two"], ["two_qd"], name="two_qd"),
            helper.make_node("Sub", ["q1", "q3"], ["q1_minus_q3"], name="q1_minus_q3"),
            helper.make_node("Add", ["q1_minus_q3", "two_qd"], ["euler_numer"], name="euler_numer"),
            helper.make_node("Mul", ["euler_numer", "quarter"], ["euler"], name="euler"),
            helper.make_node("Conv", ["x8", "neighbor_w"], ["neighbor_count"], pads=[1, 1, 1, 1], name="neighbor_count"),
            helper.make_node("Less", ["x8", "half"], ["is_zero"], name="is_zero"),
            helper.make_node("Equal", ["neighbor_count", "eight"], ["full_ring"], name="full_ring"),
            helper.make_node("And", ["is_zero", "full_ring"], ["hole_bool"], name="hole_bool"),
            helper.make_node("Cast", ["hole_bool"], ["hole_float"], to=TensorProto.FLOAT, name="hole_float"),
            helper.make_node("ReduceSum", ["hole_float", "axes_all"], ["holes"], keepdims=1, name="holes"),
            helper.make_node("Add", ["euler", "holes"], ["component_count"], name="component_count"),
        ]
    )

    nodes.extend(
        [
            helper.make_node("Less", ["row_grid", "component_count"], ["row_inside"], name="row_inside"),
            helper.make_node("Less", ["col_grid", "component_count"], ["col_inside"], name="col_inside"),
            helper.make_node("And", ["row_inside", "col_inside"], ["canvas_bool"], name="canvas_bool"),
            helper.make_node("Equal", ["row_grid", "col_grid"], ["diag_line_bool"], name="diag_line_bool"),
            helper.make_node("And", ["canvas_bool", "diag_line_bool"], ["diag_bool"], name="diag_bool"),
            helper.make_node("Cast", ["canvas_bool"], ["canvas"], to=TensorProto.FLOAT, name="canvas"),
            helper.make_node("Cast", ["diag_bool"], ["out8"], to=TensorProto.FLOAT, name="out8"),
            helper.make_node("Sub", ["canvas", "out8"], ["out0"], name="out0"),
            helper.make_node("Sub", ["out8", "out8"], ["zero"], name="zero"),
        ]
    )
    zero_channels = [f"z{i}" for i in range(8)]
    for name in zero_channels:
        nodes.append(helper.make_node("Identity", ["zero"], [name], name=name))
    nodes.append(
        helper.make_node(
            "Concat",
            ["out0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "out8", "z7"],
            ["output"],
            axis=1,
            name="assemble_output",
        )
    )

    graph = helper.make_graph(
        nodes,
        "task325_component_count",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task325_component_count",
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
    parser = argparse.ArgumentParser(description="Optimize task325 with a clean component-count graph.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v87_all_generated_guard_sweep_v88/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task325_component_count_v89"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task325.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task325 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task325.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task325 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task325.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task325.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task325",
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
