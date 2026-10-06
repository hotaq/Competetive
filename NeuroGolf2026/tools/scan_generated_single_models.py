#!/usr/bin/env python3
"""Audit generated task*.onnx files and compare them with a current report."""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import audit_candidate, task_points  # noqa: E402

TASK_RE = re.compile(r"task(\d{3})\.onnx$")


def as_int(value: str | None) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def as_float(value: str | None) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return 0.0


def read_current(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument(
        "--exclude-substr",
        action="append",
        default=[
            "/audit_",
            "/compare_",
            "/scan_",
            "/ready_",
            "_cost_audit/",
            "_compare_work/",
            "/tmp_",
        ],
        help="Substring to skip in candidate paths. May be repeated.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--work-dir", type=Path, default=Path("generated/scan_generated_single_models_work"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/generated_single_model_compare.csv"))
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    current = read_current(args.current_report)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    patterns = args.glob or ["generated/**/task*.onnx"]
    paths = sorted({Path(match) for pattern in patterns for match in glob.glob(pattern, recursive=True)})

    rows: list[dict[str, object]] = []
    for path in paths:
        match = TASK_RE.match(path.name)
        if not match:
            continue
        task = f"task{match.group(1)}"
        if task not in current:
            continue
        path_text = "/" + str(path)
        if any(skip in path_text for skip in args.exclude_substr):
            continue
        base_cost = as_int(current[task].get("cost"))
        if not base_cost:
            continue
        try:
            audit = audit_candidate(task=task, raw=path.read_bytes(), work_path=args.work_dir / task / path.name, data_root=args.data_root)
        except Exception as exc:  # noqa: BLE001 - scanner should continue.
            rows.append(
                {
                    "task": task,
                    "source_path": str(path),
                    "base_cost": base_cost,
                    "base_points": f"{task_points(base_cost):.6f}",
                    "candidate_cost": "",
                    "candidate_points": "",
                    "delta_points": "",
                    "candidate_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        cand_cost = as_int(str(audit.get("cost", "")))
        cand_points = task_points(cand_cost) if cand_cost else 0.0
        base_points = task_points(base_cost)
        delta = cand_points - base_points if cand_cost else 0.0
        if delta < args.min_delta:
            continue
        rows.append(
            {
                "task": task,
                "source_path": str(path),
                "base_cost": base_cost,
                "base_points": f"{base_points:.6f}",
                "candidate_cost": cand_cost or "",
                "candidate_points": f"{cand_points:.6f}" if cand_cost else "",
                "delta_points": f"{delta:.6f}" if cand_cost else "",
                "candidate_reason": audit.get("reason", ""),
                "candidate_ops": audit.get("ops", ""),
                "candidate_right": audit.get("right", ""),
                "candidate_wrong": audit.get("wrong", ""),
                "candidate_examples": audit.get("examples", ""),
                "candidate_file_bytes": len(path.read_bytes()),
                "candidate_sha256": audit.get("sha256", ""),
            }
        )

    rows.sort(key=lambda row: (-as_float(str(row.get("delta_points", ""))), row["task"], row["source_path"]))
    if rows:
        with args.report_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        args.report_out.write_text("", encoding="utf-8")

    print(f"paths scanned: {len(paths)}")
    print(f"rows kept: {len(rows)}")
    print(f"report: {args.report_out}")
    for row in rows[:30]:
        print(
            f"{row['task']}: {row['base_cost']} -> {row['candidate_cost']} "
            f"delta={row['delta_points']} path={row['source_path']} "
            f"reason={row['candidate_reason']}"
        )


if __name__ == "__main__":
    main()
