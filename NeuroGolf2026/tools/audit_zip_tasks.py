#!/usr/bin/env python3
"""Audit and optionally extract selected task members from a submission zip."""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import audit_candidate, task_points


def read_member(zf: zipfile.ZipFile, task: str) -> bytes:
    wanted = f"{task}.onnx"
    for name in zf.namelist():
        if Path(name).name.lower() == wanted:
            return zf.read(name)
    raise FileNotFoundError(wanted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--work-dir", type=Path, default=Path("generated/audit_zip_tasks_work"))
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as zf:
        for raw_task in args.task:
            task = Path(raw_task).stem.lower()
            raw = read_member(zf, task)
            if args.extract_dir:
                (args.extract_dir / f"{task}.onnx").write_bytes(raw)
            audit = audit_candidate(
                task=task,
                raw=raw,
                work_path=args.work_dir / args.zip.parent.name / f"{task}.onnx",
                data_root=args.data_root,
            )
            cost_text = str(audit.get("cost", ""))
            cost = int(cost_text) if cost_text.isdigit() else 0
            rows.append(
                {
                    "zip": str(args.zip),
                    "task": task,
                    "reason": audit.get("reason", ""),
                    "cost": cost or "",
                    "points": f"{task_points(cost):.6f}" if cost else "",
                    "ops": audit.get("ops", ""),
                    "right": audit.get("right", ""),
                    "wrong": audit.get("wrong", ""),
                    "examples": audit.get("examples", ""),
                    "file_bytes": len(raw),
                    "sha256": audit.get("sha256", ""),
                    "extracted": str(args.extract_dir / f"{task}.onnx") if args.extract_dir else "",
                }
            )

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with args.report_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(
            f"{row['task']}: reason={row['reason']} cost={row['cost']} "
            f"points={row['points']} right={row['right']} wrong={row['wrong']} "
            f"ops={row['ops']} extracted={row['extracted']}"
        )


if __name__ == "__main__":
    main()
