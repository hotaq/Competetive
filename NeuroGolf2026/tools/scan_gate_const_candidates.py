#!/usr/bin/env python3
"""Probe the batch-gate + Constant-node transform across many tasks.

Unlike optimize_gate_const_tasks.py, this script keeps scanning when a task
fails verification or profiling, so it can be used as a wide candidate finder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.optimize_gate_const_tasks import gate_and_constantize, normalize_task  # noqa: E402


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


def read_report(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_tasks(rows: list[dict[str, str]], max_cost: int, min_points: float, include_existing_compress: bool) -> list[str]:
    tasks: list[str] = []
    for row in rows:
        cost = as_int(row.get("cost"))
        points = as_float(row.get("points"))
        if not cost or cost > max_cost or points < min_points or points >= 20:
            continue
        if not include_existing_compress and "Compress" in row.get("ops", ""):
            continue
        tasks.append(row["task"])
    return sorted(set(tasks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/probe_gate_const_candidates"))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-cost", type=int, default=5000)
    parser.add_argument("--min-points", type=float, default=15.0)
    parser.add_argument("--include-existing-compress", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = read_report(args.report)
    by_task = {row["task"]: row for row in report_rows}
    tasks = [normalize_task(task) for task in args.tasks] if args.tasks else select_tasks(
        report_rows,
        args.max_cost,
        args.min_points,
        args.include_existing_compress,
    )

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.base_zip) as zf:
        for task in tasks:
            base = by_task.get(task, {})
            out_row: dict[str, object] = {
                "task": task,
                "base_cost": base.get("cost", ""),
                "base_points": base.get("points", ""),
                "cost": "",
                "points": "",
                "delta": "",
                "macs": "",
                "memory": "",
                "params": "",
                "right": "",
                "wrong": "",
                "examples": "",
                "status": "error",
                "reason": "",
            }
            try:
                raw = zf.read(f"{task}.onnx")
                new_raw = gate_and_constantize(raw)
                task_json = json.loads((args.data_root / f"{task}.json").read_text())
                examples = [
                    example
                    for example in all_examples(task_json)
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
                macs, memory, params, cost = profile_cost(model_path)
                base_points = as_float(base.get("points"))
                points = task_points(cost)
                out_row.update(
                    {
                        "cost": cost,
                        "points": f"{points:.6f}",
                        "delta": f"{points - base_points:.6f}",
                        "macs": macs,
                        "memory": memory,
                        "params": params,
                        "right": right,
                        "wrong": wrong,
                        "examples": len(examples),
                        "status": "ok" if wrong == 0 else "wrong",
                        "reason": reason,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - scanner should continue.
                out_row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(out_row)

    rows.sort(
        key=lambda row: (
            row["status"] != "ok",
            -as_float(str(row.get("delta", ""))),
            as_int(str(row.get("cost", ""))) or 10**12,
            str(row["task"]),
        )
    )
    report_path = args.out_dir / "scan_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"tasks scanned: {len(tasks)}")
    print(f"ok: {sum(row['status'] == 'ok' for row in rows)}")
    print(f"report: {report_path}")
    for row in rows[:30]:
        print(
            f"{row['task']}: {row['base_cost']} -> {row['cost']} "
            f"points {row['base_points']} -> {row['points']} "
            f"delta {row['delta']} status={row['status']} reason={row['reason']}"
        )


if __name__ == "__main__":
    main()
