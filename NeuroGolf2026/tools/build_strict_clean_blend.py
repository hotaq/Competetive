#!/usr/bin/env python3
"""Build a future-proof NeuroGolf blend using the discussion-694772 guard.

This intentionally optimizes for clean validator behavior, not current public LB:
- base members are kept only when they pass strict static/runtime checks,
- public candidates are considered only if a prior compare report marked them
  locally correct and they also pass the strict guard,
- selection is by official-style local cost among clean candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from tools.audit_metric_change_risk import audit_member, is_submission_task
from tools.blend_public_pack import task_points
from tools.compare_candidate_zips import read_zip_members


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def as_int(value: object) -> int | None:
    text = str(value)
    return int(text) if text.isdigit() else None


def load_costs(path: Path) -> dict[str, int]:
    costs: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task = row.get("task", "")
            cost = as_int(row.get("cost", ""))
            if task and cost:
                costs[task] = cost
    return costs


def load_base_risk(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def read_compare_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def clean_risk(row: dict[str, object]) -> bool:
    return str(row.get("clean_static_runtime", "")) == "1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict clean blend from audited candidates.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--base-cost-csv", type=Path, default=Path("reports/clean_best6270_94_task_costs.csv"))
    parser.add_argument("--base-risk-csv", type=Path, default=Path("reports/metric_change_risk_clean6270_94.csv"))
    parser.add_argument(
        "--compare-csv",
        action="append",
        type=Path,
        default=[Path("reports/pulled_public_compare_after6270_94.csv")],
    )
    parser.add_argument("--runtime-timeout", type=float, default=5.0)
    parser.add_argument("--zip-out", type=Path, default=Path("generated/strict_clean_blend_694772.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/strict_clean_blend_694772.csv"))
    args = parser.parse_args()

    base_members = read_zip_members(args.base_zip)
    base_costs = load_costs(args.base_cost_csv)
    base_risk = load_base_risk(args.base_risk_csv)

    selected: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    risk_cache: dict[str, dict[str, object]] = {}

    for member, raw in sorted(base_members.items()):
        if not is_submission_task(member):
            continue
        task = Path(member).stem
        cost = base_costs.get(task)
        risk = base_risk.get(task, {})
        if cost is None or not clean_risk(risk):
            continue
        selected[task] = {
            "task": task,
            "member": member,
            "source": "base",
            "candidate_zip": str(args.base_zip),
            "cost": cost,
            "points": task_points(cost),
            "file_bytes": len(raw),
            "sha256": sha(raw),
            "raw": raw,
            "constant_elements": risk.get("constant_elements", ""),
            "dynamic_value_info_count": risk.get("dynamic_value_info_count", ""),
        }

    zip_cache: dict[Path, dict[str, bytes]] = {}
    for row in read_compare_rows(args.compare_csv):
        if row.get("candidate_reason") != "ok":
            continue
        task = row.get("task", "")
        member = f"{task}.onnx"
        if not is_submission_task(member):
            continue
        cost = as_int(row.get("candidate_cost", ""))
        if cost is None:
            continue
        zip_path = Path(row.get("candidate_zip", ""))
        if not zip_path.exists():
            rows.append(
                {
                    "task": task,
                    "source": row.get("source", ""),
                    "candidate_zip": str(zip_path),
                    "decision": "skip_missing_zip",
                }
            )
            continue
        if zip_path not in zip_cache:
            zip_cache[zip_path] = read_zip_members(zip_path)
        raw = zip_cache[zip_path].get(member)
        if raw is None:
            rows.append(
                {
                    "task": task,
                    "source": row.get("source", ""),
                    "candidate_zip": str(zip_path),
                    "decision": "skip_missing_member",
                }
            )
            continue

        digest = sha(raw)
        if digest not in risk_cache:
            risk_cache[digest] = audit_member(member, raw, args.runtime_timeout)
        risk = risk_cache[digest]
        if not clean_risk(risk):
            rows.append(
                {
                    "task": task,
                    "source": row.get("source", ""),
                    "candidate_zip": str(zip_path),
                    "candidate_cost": cost,
                    "decision": "skip_not_strict_clean",
                    "dynamic_value_info_count": risk.get("dynamic_value_info_count", ""),
                    "runtime_matches_declared": risk.get("runtime_matches_declared", ""),
                    "runtime_error": risk.get("runtime_error", ""),
                    "constant_elements": risk.get("constant_elements", ""),
                    "sha256": digest,
                }
            )
            continue

        old = selected.get(task)
        decision = "skip_not_better"
        if old is None or (cost, len(raw), row.get("source", "")) < (
            int(old["cost"]),
            int(old["file_bytes"]),
            str(old["source"]),
        ):
            selected[task] = {
                "task": task,
                "member": member,
                "source": row.get("source", ""),
                "candidate_zip": str(zip_path),
                "cost": cost,
                "points": task_points(cost),
                "file_bytes": len(raw),
                "sha256": digest,
                "raw": raw,
                "constant_elements": risk.get("constant_elements", ""),
                "dynamic_value_info_count": risk.get("dynamic_value_info_count", ""),
            }
            decision = "select"

        rows.append(
            {
                "task": task,
                "source": row.get("source", ""),
                "candidate_zip": str(zip_path),
                "candidate_cost": cost,
                "decision": decision,
                "dynamic_value_info_count": risk.get("dynamic_value_info_count", ""),
                "runtime_matches_declared": risk.get("runtime_matches_declared", ""),
                "runtime_error": risk.get("runtime_error", ""),
                "constant_elements": risk.get("constant_elements", ""),
                "sha256": digest,
            }
        )

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task, item in sorted(selected.items()):
            zf.writestr(str(item["member"]), item["raw"])

    selected_rows = [
        {key: value for key, value in item.items() if key != "raw"}
        for item in sorted(selected.values(), key=lambda item: str(item["task"]))
    ]
    report_rows = selected_rows + rows
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    if report_rows:
        fieldnames = sorted({key for row in report_rows for key in row.keys()})
        with args.report_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
    else:
        args.report_out.write_text("", encoding="utf-8")

    selected_public = [row for row in selected_rows if row["source"] != "base"]
    print(f"base_strict_tasks: {sum(1 for row in selected_rows if row['source'] == 'base')}")
    print(f"public_strict_replacements_or_adds: {len(selected_public)}")
    print(f"selected_tasks: {len(selected_rows)}")
    print(f"selected_points_estimate: {sum(float(row['points']) for row in selected_rows):.6f}")
    print(f"audited_unique_candidate_payloads: {len(risk_cache)}")
    print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
