#!/usr/bin/env python3
"""Compare public NeuroGolf submission zips against a current base zip."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

from tools.blend_public_pack import audit_candidate, task_points


def read_zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        members: dict[str, bytes] = {}
        for name in zf.namelist():
            member = Path(name).name.lower()
            if member.startswith("task") and member.endswith(".onnx"):
                members[member] = zf.read(name)
        return members


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_name(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def as_int(value: object) -> int | None:
    text = str(value)
    return int(text) if text.isdigit() else None


def safe_audit_candidate(**kwargs: object) -> dict[str, object]:
    """Audit a candidate without allowing one malformed ONNX to abort a scan."""

    try:
        return audit_candidate(**kwargs)
    except Exception as exc:  # noqa: BLE001 - report untrusted public ONNX failures as data
        return {
            "reason": "audit_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "ops": "",
            "right": "",
            "wrong": "",
            "examples": "",
            "sha256": sha(kwargs["raw"]),  # type: ignore[index]
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate submission zips against a base.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--candidate-glob", action="append", default=["data/public/**/submission.zip"])
    parser.add_argument("--work-dir", type=Path, default=Path("generated/candidate_compare_work"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/candidate_compare.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("reports/candidate_compare_summary.csv"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    base_members = read_zip_members(args.base_zip)
    base_sha = {member: sha(raw) for member, raw in base_members.items()}
    base_audit_cache: dict[str, dict[str, object]] = {}
    candidate_paths = sorted(
        {
            Path(match)
            for pattern in args.candidate_glob
            for match in glob.glob(pattern, recursive=True)
        }
    )

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "source": "",
            "candidate_zip": "",
            "diff": 0,
            "candidate_ok": 0,
            "better": 0,
            "worse": 0,
            "same_local": 0,
            "base_missing": 0,
            "missing_task_json": 0,
            "local_delta_sum": 0.0,
            "cost1": 0,
        }
    )

    for candidate_zip in candidate_paths:
        if candidate_zip.resolve() == args.base_zip.resolve():
            continue
        name = source_name(candidate_zip)
        candidate_members = read_zip_members(candidate_zip)
        item = summary[name]
        item["source"] = name
        item["candidate_zip"] = str(candidate_zip)
        for member, raw in sorted(candidate_members.items()):
            task = member[:-5]
            if base_sha.get(member) == sha(raw):
                continue

            item["diff"] = int(item["diff"]) + 1
            task_json = args.data_root / f"{task}.json"
            if not task_json.exists():
                item["missing_task_json"] = int(item["missing_task_json"]) + 1
                rows.append(
                    {
                        "source": name,
                        "candidate_zip": str(candidate_zip),
                        "task": task,
                        "base_cost": "",
                        "base_points": "",
                        "candidate_cost": "",
                        "candidate_points": "",
                        "delta_points": "",
                        "candidate_reason": "missing_task_json",
                        "candidate_ops": "",
                        "candidate_right": "",
                        "candidate_wrong": "",
                        "candidate_examples": "",
                        "candidate_error": "",
                        "candidate_file_bytes": len(raw),
                        "candidate_sha256": sha(raw),
                    }
                )
                continue

            cand_audit = safe_audit_candidate(
                task=task,
                raw=raw,
                work_path=args.work_dir / name / "candidate" / member,
                data_root=args.data_root,
            )
            base_raw = base_members.get(member)
            base_audit: dict[str, object] = {}
            if base_raw is None:
                item["base_missing"] = int(item["base_missing"]) + 1
            else:
                if member not in base_audit_cache:
                    base_audit_cache[member] = safe_audit_candidate(
                        task=task,
                        raw=base_raw,
                        work_path=args.work_dir / "base" / member,
                        data_root=args.data_root,
                    )
                base_audit = base_audit_cache[member]

            base_cost = as_int(base_audit.get("cost", ""))
            cand_cost = as_int(cand_audit.get("cost", ""))
            base_points = task_points(base_cost)
            cand_points = task_points(cand_cost)
            delta = cand_points - base_points if base_cost and cand_cost else ""
            if cand_audit.get("reason") == "ok":
                item["candidate_ok"] = int(item["candidate_ok"]) + 1
            if cand_cost == 1:
                item["cost1"] = int(item["cost1"]) + 1
            if isinstance(delta, float):
                item["local_delta_sum"] = float(item["local_delta_sum"]) + delta
                if delta > 1e-9:
                    item["better"] = int(item["better"]) + 1
                elif delta < -1e-9:
                    item["worse"] = int(item["worse"]) + 1
                else:
                    item["same_local"] = int(item["same_local"]) + 1

            rows.append(
                {
                    "source": name,
                    "candidate_zip": str(candidate_zip),
                    "task": task,
                    "base_cost": base_cost or "",
                    "base_points": f"{base_points:.6f}" if base_cost else "",
                    "candidate_cost": cand_cost or "",
                    "candidate_points": f"{cand_points:.6f}" if cand_cost else "",
                    "delta_points": f"{delta:.6f}" if isinstance(delta, float) else "",
                    "candidate_reason": cand_audit.get("reason", ""),
                    "candidate_ops": cand_audit.get("ops", ""),
                    "candidate_right": cand_audit.get("right", ""),
                    "candidate_wrong": cand_audit.get("wrong", ""),
                    "candidate_examples": cand_audit.get("examples", ""),
                    "candidate_error": cand_audit.get("error", ""),
                    "candidate_file_bytes": len(raw),
                    "candidate_sha256": cand_audit.get("sha256", sha(raw)),
                }
            )

    if rows:
        fieldnames = list(rows[0].keys())
        with args.report_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        args.report_out.write_text("", encoding="utf-8")

    summary_rows = sorted(summary.values(), key=lambda row: str(row["source"]))
    for row in summary_rows:
        row["local_delta_sum"] = f"{float(row['local_delta_sum']):.6f}"
    if summary_rows:
        with args.summary_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        args.summary_out.write_text("", encoding="utf-8")

    print(f"candidate_zips: {len(candidate_paths)}")
    print(f"diff_rows: {len(rows)}")
    print(f"report: {args.report_out}")
    print(f"summary: {args.summary_out}")


if __name__ == "__main__":
    main()
