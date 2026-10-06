#!/usr/bin/env python3
"""Apply an exact Reshape->Cast(bool) to Cast(bool)->Reshape rewrite."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import audit_candidate, profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def bool_reshape_pass(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    out_to_node = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    rewrites: dict[str, tuple[onnx.NodeProto, onnx.NodeProto]] = {}
    skip_names: set[str] = set()
    for cast in model.graph.node:
        if cast.op_type != "Cast" or not cast.input:
            continue
        to_bool = any(attr.name == "to" and attr.i == onnx.TensorProto.BOOL for attr in cast.attribute)
        if not to_bool:
            continue
        reshape = out_to_node.get(cast.input[0])
        if reshape is None or reshape.op_type != "Reshape" or len(reshape.input) < 2:
            continue
        if len(consumers.get(reshape.output[0], [])) != 1:
            continue
        bool_mid = f"{cast.output[0]}__pre_reshape_bool"
        new_cast = helper.make_node(
            "Cast",
            [reshape.input[0]],
            [bool_mid],
            name=f"{cast.name or cast.output[0]}_before_reshape",
            to=onnx.TensorProto.BOOL,
        )
        new_reshape = helper.make_node(
            "Reshape",
            [bool_mid, reshape.input[1]],
            [cast.output[0]],
            name=f"{reshape.name or reshape.output[0]}_after_bool",
        )
        rewrites[cast.name or cast.output[0]] = (new_cast, new_reshape)
        skip_names.add(reshape.name or reshape.output[0])

    if not rewrites:
        return model, 0

    new_model = copy.deepcopy(model)
    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        node_key = node.name or node.output[0]
        if node_key in skip_names:
            continue
        if node_key in rewrites:
            new_nodes.extend(rewrites[node_key])
            continue
        new_nodes.append(copy.deepcopy(node))

    del new_model.graph.node[:]
    new_model.graph.node.extend(new_nodes)
    del new_model.graph.value_info[:]
    new_model.producer_name = f"{model.producer_name}_bool_reshape"
    try:
        onnx.checker.check_model(new_model)
    except Exception:
        # Some public models are accepted by runtime but not strict checker
        # because their exporters reused names. Verification below is decisive.
        pass
    return new_model, len(rewrites)


def read_tasks(report: Path, max_cost: int, min_points: float) -> list[str]:
    tasks: list[str] = []
    with report.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("reason") != "ok" or not row.get("cost"):
                continue
            cost = int(row["cost"])
            points = task_points(cost)
            if points >= min_points and cost <= max_cost:
                tasks.append(row["task"])
    return sorted(set(tasks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch6/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--report", type=Path, default=Path("reports/clean_batch6_task_costs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/bool_reshape_pass"))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-cost", type=int, default=2500)
    parser.add_argument("--min-points", type=float, default=17.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks if args.tasks else read_tasks(args.report, args.max_cost, args.min_points)
    rows: list[dict[str, object]] = []

    with zipfile.ZipFile(args.base_zip) as zf:
        for task in tasks:
            member = f"{task}.onnx"
            try:
                base_raw = zf.read(member)
            except KeyError:
                continue
            model, rewrites = bool_reshape_pass(onnx.load_from_string(base_raw))
            if rewrites == 0:
                continue
            raw = model.SerializeToString()
            task_dir = args.out_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)
            model_path = task_dir / member
            model_path.write_bytes(raw)
            examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
            right, wrong, reason = verify_model_bytes(raw, examples)
            macs, memory, params, cost = profile_cost(model_path)
            base_audit = audit_candidate(task=task, raw=base_raw, work_path=args.out_dir / "base" / member, data_root=args.data_root)
            base_cost = int(base_audit["cost"])
            rows.append(
                {
                    "task": task,
                    "rewrites": rewrites,
                    "right": right,
                    "wrong": wrong,
                    "reason": reason,
                    "base_cost": base_cost,
                    "cost": cost,
                    "delta": task_points(cost) - task_points(base_cost),
                    "points": task_points(cost),
                    "model_path": str(model_path),
                }
            )

    rows.sort(key=lambda row: (-float(row["delta"]), row["task"]))
    report_path = args.out_dir / "report.csv"
    if rows:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"tasks scanned: {len(tasks)}")
    print(f"candidates: {len(rows)}")
    print(f"report: {report_path}")
    for row in rows[:30]:
        print(
            f"{row['task']} rewrites={row['rewrites']} {row['base_cost']}->{row['cost']} "
            f"delta={float(row['delta']):.6f} points={float(row['points']):.6f} "
            f"wrong={row['wrong']}"
        )


if __name__ == "__main__":
    main()
