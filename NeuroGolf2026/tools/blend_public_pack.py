#!/usr/bin/env python3
"""Blend verified public-pack ONNX candidates into a local NeuroGolf zip.

The script is intentionally conservative:
- only considers tasks in a selected triage bucket,
- requires static 1x10x30x30 input/output shapes,
- rejects banned ONNX ops,
- requires ONNX Runtime exact match on train + test + arc-gen,
- selects the public candidate only when its profiled score improves the base.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import onnx
import onnx_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
MAX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
ALLOWED_DOMAINS = {"", "ai.onnx"}


def task_points(cost: int | None) -> float:
    if cost is None or cost <= 0:
        return 0.0
    return max(1.0, 25.0 - math.log(cost))


def scorer_loophole_risk(model: onnx.ModelProto) -> str | None:
    if len(model.functions):
        return "model_functions_disallowed"

    custom_opsets = sorted({opset.domain for opset in model.opset_import if opset.domain not in ALLOWED_DOMAINS})
    if custom_opsets:
        return "custom_opset_domain:" + ",".join(custom_opsets)[:160]

    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ALLOWED_DOMAINS})
    if custom_domains:
        return "custom_domain_node:" + ",".join(custom_domains)[:160]

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return "graph_attribute_disallowed"

    io_names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    collisions = sorted(io_names & initializer_names)
    if collisions:
        return "io_initializer_collision:" + ",".join(collisions)[:160]

    return None


def model_shapes_and_ops(model: onnx.ModelProto) -> tuple[list[int], list[int], set[str]]:
    in_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    out_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    ops = {node.op_type for node in model.graph.node}
    return in_shape, out_shape, ops


def staticize_graph_io(raw: bytes) -> bytes:
    model = onnx.load_from_string(raw)
    for value_info in list(model.graph.input) + list(model.graph.output):
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) != len(EXPECTED_SHAPE):
            continue
        for dim, value in zip(dims, EXPECTED_SHAPE):
            dim.dim_param = ""
            dim.dim_value = value
    onnx.checker.check_model(model)
    return model.SerializeToString()


def profile_cost(path: Path) -> tuple[int, int, int, int]:
    model = onnx_tool.loadmodel(str(path), {"verbose": False})
    graph = model.graph
    graph.graph_reorder_nodes()
    graph.shape_infer(None)
    graph.profile()
    if not graph.valid_profile:
        raise ValueError("invalid onnx_tool profile")
    for key in graph.nodemap.keys():
        node = graph.nodemap[key]
        if node.op_type in BANNED_OPS:
            raise ValueError(f"banned op {node.op_type}")
        if node.memory < 0:
            raise ValueError("negative memory in onnx_tool profile")
    macs = int(sum(graph.macs))
    memory = int(graph.memory)
    params = int(graph.params)
    return macs, memory, params, macs + memory + params


def load_bucket_tasks(triage_csv: Path, bucket: str) -> list[str]:
    tasks: list[str] = []
    with triage_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if bucket == "ALL" or row["bucket"] == bucket:
                tasks.append(row["task"])
    return sorted(tasks)


def load_public_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task_id"]: row for row in csv.DictReader(f)}


def examples_fit_runtime_shape(examples: list[dict[str, list[list[int]]]]) -> bool:
    for example in examples:
        for key in ("input", "output"):
            grid = example[key]
            if not grid or not grid[0]:
                return False
            if len(grid) > 30 or len(grid[0]) > 30:
                return False
    return True


def audit_candidate(
    *,
    task: str,
    raw: bytes,
    work_path: Path,
    data_root: Path,
) -> dict[str, str | int | float]:
    work_path.parent.mkdir(parents=True, exist_ok=True)
    work_path.write_bytes(raw)
    model = onnx.load_from_string(raw)
    in_shape, out_shape, ops = model_shapes_and_ops(model)
    result: dict[str, str | int | float] = {
        "task": task,
        "file_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "input_shape": "x".join(map(str, in_shape)),
        "output_shape": "x".join(map(str, out_shape)),
        "ops": ";".join(sorted(ops)),
        "banned_ops": ";".join(sorted(ops & BANNED_OPS)),
        "right": 0,
        "wrong": 0,
        "examples": 0,
        "macs": "",
        "memory": "",
        "params": "",
        "cost": "",
        "points": "",
        "staticized_graph_io": "",
        "reason": "",
    }
    if len(raw) > MAX_BYTES:
        result["reason"] = "oversized"
        return result
    risk = scorer_loophole_risk(model)
    if risk:
        result["reason"] = risk
        return result
    if in_shape != EXPECTED_SHAPE or out_shape != EXPECTED_SHAPE:
        result["reason"] = "non_static_expected_shape"
        return result
    if ops & BANNED_OPS:
        result["reason"] = "banned_ops"
        return result

    examples = all_examples(json.loads((data_root / f"{task}.json").read_text()))
    if not examples_fit_runtime_shape(examples):
        result["examples"] = len(examples)
        result["reason"] = "example_grid_exceeds_30"
        return result
    right, wrong, reason = verify_model_bytes(raw, examples)
    result.update({"right": right, "wrong": wrong, "examples": len(examples)})
    if wrong:
        result["reason"] = reason or "local_verify_failed"
        return result

    try:
        macs, memory, params, cost = profile_cost(work_path)
    except Exception as exc:
        result["reason"] = f"profile_error:{exc}"
        return result
    result.update(
        {
            "macs": macs,
            "memory": memory,
            "params": params,
            "cost": cost,
            "points": f"{task_points(cost):.6f}",
            "reason": "ok",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend public NeuroGolf candidates into a local submission.")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--triage-csv", type=Path, default=Path("reports/task_triage.csv"))
    parser.add_argument("--bucket", default="A_exact")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/a_exact_submission.zip"))
    parser.add_argument(
        "--public-zip",
        type=Path,
        default=Path("data/public/thisray-neurogolf-4743-93-submission-task-table/submission_zip_exact_bytes.bin"),
    )
    parser.add_argument(
        "--public-table",
        type=Path,
        default=Path("data/public/thisray-neurogolf-4743-93-submission-task-table/task_by_task_release_table.csv"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("generated/public_blend_a_exact"))
    parser.add_argument("--zip-out", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/public_blend_a_exact_report.csv"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_bucket_tasks(args.triage_csv, args.bucket)
    public_table = load_public_table(args.public_table)
    rows: list[dict[str, str | int | float]] = []
    selected: dict[str, bytes] = {}

    with zipfile.ZipFile(args.base_zip) as base_zf, zipfile.ZipFile(args.public_zip) as public_zf:
        base_members = {name: base_zf.read(name) for name in base_zf.namelist()}
        selected.update(base_members)

        for task in tasks:
            member = f"{task}.onnx"
            base_raw = base_members.get(member)
            base_audit: dict[str, str | int | float] = {}
            if base_raw is not None:
                base_audit = audit_candidate(
                    task=task,
                    raw=base_raw,
                    work_path=args.work_dir / "base" / member,
                    data_root=args.data_root,
                )
            public_row = public_table.get(task, {})
            public_member = public_row.get("zip_member") or member
            if public_member not in public_zf.namelist():
                row = {
                    "task": task,
                    "action": "keep_base",
                    "reason": "missing public member",
                    "base_cost": base_audit.get("cost", ""),
                    "base_points": base_audit.get("points", ""),
                }
                rows.append(row)
                continue

            public_raw = public_zf.read(public_member)
            cand_audit = audit_candidate(
                task=task,
                raw=public_raw,
                work_path=args.work_dir / "candidate" / member,
                data_root=args.data_root,
            )
            selected_public_raw = public_raw
            if cand_audit["reason"] == "non_static_expected_shape":
                try:
                    staticized_raw = staticize_graph_io(public_raw)
                    staticized_audit = audit_candidate(
                        task=task,
                        raw=staticized_raw,
                        work_path=args.work_dir / "staticized_candidate" / member,
                        data_root=args.data_root,
                    )
                    if staticized_audit["reason"] == "ok":
                        staticized_audit["staticized_graph_io"] = "yes"
                        cand_audit = staticized_audit
                        selected_public_raw = staticized_raw
                except Exception as exc:
                    cand_audit["reason"] = f"staticize_error:{exc}"

            base_cost = int(base_audit["cost"]) if str(base_audit.get("cost", "")).isdigit() else None
            cand_cost = int(cand_audit["cost"]) if str(cand_audit.get("cost", "")).isdigit() else None
            base_pts = task_points(base_cost)
            cand_pts = task_points(cand_cost)
            delta = cand_pts - base_pts
            if base_raw is None and cand_audit["reason"] == "ok":
                action = "add_public"
            elif cand_audit["reason"] == "ok" and delta > 1e-9:
                action = "replace_public"
            else:
                action = "keep_base"
            if action in {"add_public", "replace_public"}:
                selected[member] = selected_public_raw

            rows.append(
                {
                    "task": task,
                    "action": action,
                    "base_cost": base_cost or "",
                    "base_points": f"{base_pts:.6f}" if base_cost else "",
                    "candidate_cost": cand_cost or "",
                    "candidate_points": f"{cand_pts:.6f}" if cand_cost else "",
                    "delta_points": f"{delta:.6f}" if cand_cost and base_cost else "",
                    "candidate_reason": cand_audit["reason"],
                    "candidate_ops": cand_audit["ops"],
                    "candidate_right": cand_audit["right"],
                    "candidate_wrong": cand_audit["wrong"],
                    "candidate_examples": cand_audit["examples"],
                    "candidate_caveat": ";".join(
                        caveat
                        for caveat in [
                            "profile_below_9000" if cand_cost is not None and cand_cost < 9000 else "",
                            "staticized_graph_io" if cand_audit.get("staticized_graph_io") else "",
                        ]
                        if caveat
                    ),
                    "source_name": public_row.get("source_name", ""),
                    "source_layer": public_row.get("source_layer", ""),
                    "zip_member": public_member,
                    "sha256": cand_audit["sha256"],
                }
            )

    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member in sorted(selected):
            zf.writestr(member, selected[member])

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    replaced = [row for row in rows if row.get("action") == "replace_public"]
    added = [row for row in rows if row.get("action") == "add_public"]
    total_base = sum(float(row.get("base_points") or 0) for row in rows)
    total_selected = sum(
        float(row.get("candidate_points") or 0)
        if row.get("action") in {"add_public", "replace_public"}
        else float(row.get("base_points") or 0)
        for row in rows
    )
    selected_rows = replaced + added
    caveats = [f"{row['task']}:{row.get('candidate_caveat')}" for row in selected_rows if row.get("candidate_caveat")]

    print(f"bucket: {args.bucket}")
    print(f"tasks: {len(tasks)}")
    print(f"added: {len(added)}")
    print(f"replaced: {len(replaced)}")
    print(f"zip entries: {len(selected)}")
    print(f"base points: {total_base:.6f}")
    print(f"selected points: {total_selected:.6f}")
    print(f"delta: {total_selected - total_base:.6f}")
    print(f"candidate caveats: {', '.join(caveats) if caveats else 'none'}")
    print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
