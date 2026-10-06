#!/usr/bin/env python3
"""Summarize the current NeuroGolf staged batch by 20+ task coverage."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


def task_points(cost: int | None) -> float:
    if cost is None or cost <= 0:
        return 0.0
    return max(1.0, 25.0 - math.log(cost))


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-report", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("generated/ready_20plus_many"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/current_20plus_many.md"))
    parser.add_argument("--csv-out", type=Path, default=Path("reports/current_20plus_many.csv"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.cost_report.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cost = parse_int(row.get("cost", ""))
            if cost is None:
                continue
            row["cost_int"] = str(cost)
            row["points_calc"] = f"{task_points(cost):.6f}"
            rows.append(row)

    rows.sort(key=lambda r: (int(r["cost_int"]), r["task"]))
    plus20 = [r for r in rows if int(r["cost_int"]) <= 148]
    near = [r for r in rows if 148 < int(r["cost_int"]) <= 500]
    total_points = sum(task_points(int(r["cost_int"])) for r in rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    staged_zip = args.out_dir / "submission.zip"
    shutil.copyfile(args.source_zip, staged_zip)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "cost_int", "points_calc", "bucket", "ops", "right", "wrong", "examples", "reason"]
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in plus20:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# Current 20+ Staged Batch",
        "",
        f"- Source zip: `{args.source_zip}`",
        f"- Staged zip: `{staged_zip}`",
        f"- Cost report: `{args.cost_report}`",
        f"- Local estimated points from scored rows: `{total_points:.6f}`",
        f"- Tasks at 20+ (`cost <= 148`): `{len(plus20)}`",
        f"- Near-20 tasks (`149 <= cost <= 500`): `{len(near)}`",
        "",
        "## 20+ Tasks",
        "",
        "| task | cost | points | bucket |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in plus20:
        lines.append(
            f"| {row['task']} | {row['cost_int']} | {row['points_calc']} | {row.get('bucket', '')} |"
        )
    lines.extend(
        [
            "",
            "## Near-20 Queue",
            "",
            "| task | cost | points | bucket | ops |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in near:
        ops = row.get("ops", "")
        if len(ops) > 80:
            ops = ops[:77] + "..."
        lines.append(
            f"| {row['task']} | {row['cost_int']} | {row['points_calc']} | {row.get('bucket', '')} | {ops} |"
        )
    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"20+ tasks: {len(plus20)}")
    print(f"near-20 tasks: {len(near)}")
    print(f"local estimated points: {total_points:.6f}")
    print(f"staged zip: {staged_zip}")
    print(f"report: {args.report_out}")
    print(f"csv: {args.csv_out}")


if __name__ == "__main__":
    main()
