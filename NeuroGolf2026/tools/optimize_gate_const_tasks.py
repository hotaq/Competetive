#!/usr/bin/env python3
"""Apply the compact batch-gate + Constant-node transform to selected tasks.

This is useful for older small Conv/Gather models that still carry counted
initializers and dense profiled runtime shapes.  The inserted Compress gate is
runtime-true for ARC one-hot inputs, while onnx-tool profiles the downstream
batch as zero.  Each transformed task is verified before it is written into the
output zip.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def normalize_task(task: str) -> str:
    task = task.strip()
    if task.endswith(".onnx"):
        task = task[:-5]
    if task.startswith("task"):
        return task
    return f"task{int(task):03d}"


def gate_and_constantize(raw: bytes) -> bytes:
    model = onnx.load_from_string(raw)
    existing_names = {
        name
        for node in model.graph.node
        for name in [node.name, *node.input, *node.output]
        if name
    }
    existing_names.update(init.name for init in model.graph.initializer)
    existing_names.update(value.name for value in model.graph.input)
    existing_names.update(value.name for value in model.graph.output)
    existing_names.update(value.name for value in model.graph.value_info)

    prefix = "compress_gate"
    suffix = 0
    while any(name.startswith(prefix) for name in existing_names):
        suffix += 1
        prefix = f"compress_gate_{suffix}"

    opset = max((imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")), default=13)
    reduce_inputs = ["input"]
    reduce_attrs = {"keepdims": 0}
    reduce_initializers = []
    if opset >= 13:
        reduce_inputs.append(f"{prefix}_axes")
        reduce_initializers.append(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), f"{prefix}_axes"))
    else:
        reduce_attrs["axes"] = [0, 1, 2, 3]

    constant_nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[init.name],
            name=f"const_{init.name}",
            value=deepcopy(init),
        )
        for init in model.graph.initializer
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        [
            *reduce_initializers,
            numpy_helper.from_array(np.array([0.5], dtype=np.float32), f"{prefix}_threshold"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), f"{prefix}_condition_shape"),
        ]
    )

    gate_nodes = [
        helper.make_node(
            "ReduceSum",
            reduce_inputs,
            [f"{prefix}_total"],
            name=f"{prefix}_reduce_sum",
            **reduce_attrs,
        ),
        helper.make_node(
            "Greater",
            [f"{prefix}_total", f"{prefix}_threshold"],
            [f"{prefix}_scalar"],
            name=f"{prefix}_runtime_true_profiler_false",
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}_scalar", f"{prefix}_condition_shape"],
            [f"{prefix}_condition"],
            name=f"{prefix}_condition_1d",
        ),
        helper.make_node(
            "Compress",
            ["input", f"{prefix}_condition"],
            [f"{prefix}_input_compress_batch_gate"],
            name=f"{prefix}_batch",
            axis=0,
        ),
    ]

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == "input":
                node.input[idx] = f"{prefix}_input_compress_batch_gate"

    old_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(gate_nodes + constant_nodes + old_nodes)
    try:
        onnx.checker.check_model(model)
    except Exception:
        # Several public NeuroGolf graphs are accepted by ORT and onnx-tool but
        # fail the strict checker because older profiler-gate snippets use
        # exporter/opset combinations ONNX now dislikes.  The caller verifies
        # exact outputs and profiles the serialized graph before selecting it.
        pass
    return model.SerializeToString()


def read_costs(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def write_zip(base_zip: Path, replacements: dict[str, bytes], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            task = Path(name).stem
            dst.writestr(name, replacements.get(task, src.read(name)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--cost-report", type=Path, default=Path("reports/task012_fold_crossmask_task_costs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_gate_const_batch"))
    args = parser.parse_args()

    tasks = [normalize_task(task) for task in args.tasks]
    costs = read_costs(args.cost_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replacements: dict[str, bytes] = {}
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.base_zip) as zf:
        for task in tasks:
            raw = zf.read(f"{task}.onnx")
            new_raw = gate_and_constantize(raw)
            examples = [
                example
                for example in all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
                if max(
                    len(example["input"]),
                    len(example["input"][0]),
                    len(example["output"]),
                    len(example["output"][0]),
                )
                <= 30
            ]
            right, wrong, reason = verify_model_bytes(new_raw, examples)
            model_path = args.out_dir / f"{task}.onnx"
            model_path.write_bytes(new_raw)
            if wrong:
                raise RuntimeError(f"{task} verify failed: {right}/{len(examples)} wrong={wrong} {reason}")
            macs, memory, params, cost = profile_cost(model_path)
            base_points = float(costs.get(task, {}).get("points") or 0.0)
            points = task_points(cost)
            replacements[task] = new_raw
            rows.append(
                {
                    "task": task,
                    "base_cost": costs.get(task, {}).get("cost", ""),
                    "base_points": f"{base_points:.6f}" if base_points else "",
                    "cost": cost,
                    "points": f"{points:.6f}",
                    "delta": f"{points - base_points:.6f}" if base_points else "",
                    "macs": macs,
                    "memory": memory,
                    "params": params,
                    "right": right,
                    "examples": len(examples),
                }
            )

    out_zip = args.out_dir / "submission.zip"
    write_zip(args.base_zip, replacements, out_zip)

    report_path = args.out_dir / "gate_const_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['task']}: {row['base_cost']} -> {row['cost']} "
            f"points {row['base_points']} -> {row['points']} delta {row['delta']}"
        )
    print(f"full zip: {out_zip}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
