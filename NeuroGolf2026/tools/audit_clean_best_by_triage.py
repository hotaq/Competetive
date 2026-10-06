#!/usr/bin/env python3
"""Audit the current clean submission against triage metadata.

This is a targeting tool, not a model generator.  It profiles every ONNX file in
the chosen base zip, joins the result with task triage, and emits a ranked view
of cheap-looking tasks where a custom clean model may still be worth building.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import (
    BANNED_OPS,
    EXPECTED_SHAPE,
    MAX_BYTES,
    model_shapes_and_ops,
    profile_cost,
    task_points,
)
from tools.generate_a_exact_submission import all_examples, verify_model_bytes


def read_triage(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def read_zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {
            Path(name).name[:-5]: zf.read(name)
            for name in zf.namelist()
            if Path(name).name.startswith("task") and Path(name).name.endswith(".onnx")
        }


def as_int(value: str) -> int | None:
    return int(value) if str(value).isdigit() else None


def shape_text(shape: list[int]) -> str:
    return "x".join(str(dim or "?") for dim in shape)


def audit_member(task: str, raw: bytes, work_path: Path, data_root: Path) -> dict[str, object]:
    work_path.parent.mkdir(parents=True, exist_ok=True)
    work_path.write_bytes(raw)
    in_shape, out_shape, ops = model_shapes_and_ops(raw)
    result: dict[str, object] = {
        "task": task,
        "file_bytes": len(raw),
        "input_shape": shape_text(in_shape),
        "output_shape": shape_text(out_shape),
        "static_expected_shape": int(in_shape == EXPECTED_SHAPE and out_shape == EXPECTED_SHAPE),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ops": ";".join(sorted(ops)),
        "banned_ops": ";".join(sorted(ops & BANNED_OPS)),
        "right": "",
        "wrong": "",
        "examples": "",
        "macs": "",
        "memory": "",
        "params": "",
        "cost": "",
        "points": "",
        "reason": "",
    }
    if len(raw) > MAX_BYTES:
        result["reason"] = "oversized"
        return result
    if ops & BANNED_OPS:
        result["reason"] = "banned_ops"
        return result

    examples = [
        example
        for example in all_examples(json.loads((data_root / f"{task}.json").read_text()))
        if max(len(example["input"]), len(example["input"][0]), len(example["output"]), len(example["output"][0])) <= 30
    ]
    if not examples:
        result["reason"] = "no_examples_fit_runtime_shape"
        return result
    try:
        right, wrong, reason = verify_model_bytes(raw, examples)
    except Exception as exc:
        result["reason"] = f"verify_error:{exc}"
        return result
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


def score_priority(row: dict[str, object]) -> tuple[float, int, int]:
    bucket_rank = {"A_exact": 4, "B_template": 3, "C_medium": 2, "D_hard": 1}.get(str(row["bucket"]), 0)
    cost = int(row["cost"] or 0)
    difficulty = int(row["difficulty"] or 999)
    missing_bonus = 1_000_000 if row["reason"] == "missing_from_zip" else 0
    return (bucket_rank * 1_000_000 + missing_bonus + max(cost, 0) - difficulty * 1000, -difficulty, cost)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clean best NeuroGolf zip by triage bucket.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--triage-csv", type=Path, default=Path("reports/task_triage.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("generated/clean_best_cost_audit"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/clean_best_task_costs.csv"))
    parser.add_argument("--targets-out", type=Path, default=Path("reports/clean_best_target_rank.md"))
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.targets_out.parent.mkdir(parents=True, exist_ok=True)

    triage = read_triage(args.triage_csv)
    members = read_zip_members(args.base_zip)
    rows: list[dict[str, object]] = []

    for task in sorted(triage):
        meta = triage[task]
        raw = members.get(task)
        if raw is None:
            audit: dict[str, object] = {
                "task": task,
                "file_bytes": "",
                "input_shape": "",
                "output_shape": "",
                "static_expected_shape": "",
                "sha256": "",
                "ops": "",
                "banned_ops": "",
                "right": "",
                "wrong": "",
                "examples": meta["example_n"],
                "macs": "",
                "memory": "",
                "params": "",
                "cost": "",
                "points": "",
                "reason": "missing_from_zip",
            }
        else:
            audit = audit_member(
                task=task,
                raw=raw,
                work_path=args.work_dir / f"{task}.onnx",
                data_root=args.data_root,
            )

        cost = as_int(str(audit.get("cost", "")))
        points = task_points(cost) if cost else 0.0
        rows.append(
            {
                "task": task,
                "task_num": meta["task_num"],
                "bucket": meta["bucket"],
                "difficulty": meta["difficulty"],
                "families": meta["families"],
                "notes": meta["notes"],
                "input_shapes": meta["input_shapes"],
                "output_shapes": meta["output_shapes"],
                "same_shape_all": meta["same_shape_all"],
                "input_colors": meta["input_colors"],
                "output_colors": meta["output_colors"],
                "new_output_colors": meta["new_output_colors"],
                "cost": cost or "",
                "points": f"{points:.6f}" if cost else "",
                "macs": audit.get("macs", ""),
                "memory": audit.get("memory", ""),
                "params": audit.get("params", ""),
                "file_bytes": audit.get("file_bytes", ""),
                "input_shape": audit.get("input_shape", ""),
                "output_shape": audit.get("output_shape", ""),
                "static_expected_shape": audit.get("static_expected_shape", ""),
                "ops": audit.get("ops", ""),
                "banned_ops": audit.get("banned_ops", ""),
                "right": audit.get("right", ""),
                "wrong": audit.get("wrong", ""),
                "examples": audit.get("examples", ""),
                "reason": audit.get("reason", ""),
                "sha256": audit.get("sha256", ""),
            }
        )

    fieldnames = list(rows[0].keys())
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["reason"] == "ok"]
    static_ok_rows = [row for row in ok_rows if row["static_expected_shape"] == 1]
    total_points = sum(float(row["points"] or 0) for row in rows)
    bucket_counts: dict[str, dict[str, object]] = {}
    for row in rows:
        bucket = str(row["bucket"])
        item = bucket_counts.setdefault(bucket, {"tasks": 0, "ok": 0, "points": 0.0})
        item["tasks"] = int(item["tasks"]) + 1
        item["ok"] = int(item["ok"]) + (1 if row["reason"] == "ok" else 0)
        item["points"] = float(item["points"]) + float(row["points"] or 0)

    ranked = sorted(rows, key=score_priority, reverse=True)[: args.top_n]
    easy_ranked = [
        row
        for row in ranked
        if row["bucket"] in {"A_exact", "B_template"} or int(row["difficulty"] or 999) <= 16
    ][:40]

    lines = [
        "# Clean Best Target Rank",
        "",
        f"Base zip: `{args.base_zip}`",
        f"Audited tasks: {len(rows)}",
        f"Locally valid tasks: {len(ok_rows)}",
        f"Static 1x10x30x30 valid tasks: {len(static_ok_rows)}",
        f"Local estimated points: {total_points:.6f}",
        "",
        "## Bucket Totals",
        "",
        "| Bucket | Tasks | OK | Points |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in ["A_exact", "B_template", "C_medium", "D_hard"]:
        item = bucket_counts.get(bucket, {"tasks": 0, "ok": 0, "points": 0.0})
        lines.append(f"| {bucket} | {item['tasks']} | {item['ok']} | {float(item['points']):.6f} |")

    lines.extend(
        [
            "",
            "## Highest Priority Clean Targets",
            "",
            "| Task | Bucket | Difficulty | Cost | Points | Static | IO Shape | Shapes | Families | Notes | Reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in easy_ranked:
        notes = str(row["notes"]).replace("|", "/")
        families = str(row["families"]).replace("|", "/")
        shapes = f"{row['input_shapes']} -> {row['output_shapes']}".replace("|", "/")
        io_shape = f"{row['input_shape']} -> {row['output_shape']}"
        lines.append(
            f"| {row['task']} | {row['bucket']} | {row['difficulty']} | {row['cost']} | "
            f"{row['points']} | {row['static_expected_shape']} | {io_shape} | {shapes} | "
            f"{families} | {notes} | {row['reason']} |"
        )

    args.targets_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"tasks": len(rows), "ok": len(ok_rows), "points": total_points}, indent=2))
    print(f"report: {args.report_out}")
    print(f"targets: {args.targets_out}")


if __name__ == "__main__":
    main()
