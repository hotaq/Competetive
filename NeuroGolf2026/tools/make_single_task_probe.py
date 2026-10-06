#!/usr/bin/env python3
"""Create NeuroGolf single-task probe submissions from a blended zip."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
import zipfile
from pathlib import Path


def task_member(task: str) -> str:
    task = task.strip()
    if task.endswith(".onnx"):
        return Path(task).name
    if task.startswith("task"):
        return f"{task}.onnx"
    return f"task{int(task):03d}.onnx"


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def build_probe(base_zip: Path, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src:
        raw = src.read(member)
    (out_dir / member).write_bytes(raw)
    out_zip = out_dir / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, raw)
    return out_zip


def write_queue(report_csv: Path, queue_out: Path, *, min_points: float, static_only: bool) -> list[str]:
    rows = load_rows(report_csv)
    queue: list[dict[str, str]] = []
    for task, row in rows.items():
        if row.get("reason") != "ok" or not row.get("points"):
            continue
        if float(row["points"]) < min_points:
            continue
        if static_only and row.get("static_expected_shape") != "1":
            continue
        queue.append(row)
    queue.sort(key=lambda row: (-float(row["points"]), int(row["cost"]), row["task"]))

    fields = [
        "task",
        "cost",
        "points",
        "static_expected_shape",
        "input_shape",
        "output_shape",
        "families",
        "notes",
        "sha256",
    ]
    queue_out.parent.mkdir(parents=True, exist_ok=True)
    with queue_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in queue:
            writer.writerow({field: row.get(field, "") for field in fields})
    return [row["task"] for row in queue]


def submit(zip_path: Path, message: str, kaggle_python: str) -> None:
    cmd = [
        kaggle_python,
        "-m",
        "kaggle.cli",
        "competitions",
        "submit",
        "-c",
        "neurogolf-2026",
        "-f",
        str(zip_path),
        "-m",
        message,
    ]
    subprocess.run(cmd, check=True)


def poll(limit: int, delay: int, kaggle_python: str) -> None:
    for i in range(limit):
        result = subprocess.run(
            [kaggle_python, "-m", "kaggle.cli", "competitions", "submissions", "-c", "neurogolf-2026"],
            check=True,
            text=True,
            capture_output=True,
        )
        lines = result.stdout.splitlines()[:8]
        print("\n".join(lines))
        if any("SubmissionStatus.COMPLETE" in line for line in lines[3:4]):
            return
        if i + 1 < limit:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--report-csv", type=Path, default=Path("reports/task399_parity_count_task_costs.csv"))
    parser.add_argument("--queue-out", type=Path, default=Path("reports/single_file_probe_queue.csv"))
    parser.add_argument("--min-points", type=float, default=20.0)
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument("--task", action="append", help="task number/name/member; may be repeated")
    parser.add_argument("--out-root", type=Path, default=Path("generated/probes"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--kaggle-python", default="/usr/bin/python3")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-limit", type=int, default=20)
    parser.add_argument("--poll-delay", type=int, default=20)
    args = parser.parse_args()

    queued = write_queue(
        args.report_csv,
        args.queue_out,
        min_points=args.min_points,
        static_only=not args.include_dynamic,
    )
    print(f"queue: {args.queue_out} ({len(queued)} tasks)")

    if not args.task:
        print("top queued tasks:", ", ".join(queued[:20]))
        return

    for task in args.task:
        member = task_member(task)
        task_name = member[:-5]
        out_dir = args.out_root / task_name
        out_zip = build_probe(args.base_zip, member, out_dir)
        print(f"probe: {out_zip} ({out_zip.stat().st_size} bytes)")
        if args.submit:
            message = f"Single file {task_name} 20pt probe"
            submit(out_zip, message, args.kaggle_python)
            if args.poll:
                poll(args.poll_limit, args.poll_delay, args.kaggle_python)


if __name__ == "__main__":
    main()
