#!/usr/bin/env python3
"""Build a focused NeuroGolf work queue from audited task costs.

The queue follows the current competition strategy:
- keep true 20+ candidates separate,
- treat crop/pad reductions as 15-17 point lane work,
- ignore six-figure-cost tasks unless a replacement clears a meaningful gain.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


HIGH_POTENTIAL_FAMILIES = (
    "color_remap",
    "crop_exact_same_offset",
    "geometric_exact",
    "split_binary_relation",
)

CROP_LANE_FAMILIES = (
    "crop_exact_variable_offset",
    "crop_exact_same_offset",
    "reduce_or_crop",
    "geometric_exact",
    "split_binary_relation",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value: str) -> int:
    return int(float(value)) if value else 0


def as_float(value: str) -> float:
    return float(value) if value else 0.0


def has_family(row: dict[str, str], families: Iterable[str]) -> bool:
    text = row.get("families", "")
    return any(family in text for family in families)


def task_sort(row: dict[str, str]) -> tuple[int, str]:
    return as_int(row.get("cost", "")), row.get("task", "")


def fmt_task(row: dict[str, str]) -> str:
    notes = row.get("notes", "")
    suffix = f" | {notes}" if notes else ""
    return (
        f"| {row.get('task', '')} | {row.get('bucket', '')} | "
        f"{as_int(row.get('cost', ''))} | {as_float(row.get('points', '')):.3f} | "
        f"{row.get('static_expected_shape', '')} | {row.get('families', '')}{suffix} |"
    )


def candidate_sort(row: dict[str, str]) -> tuple[float, str]:
    return -as_float(row.get("delta_points", "")), row.get("task", "")


def is_clean_candidate(row: dict[str, str]) -> bool:
    source = row.get("source", "").lower()
    task = row.get("task", "")
    cost = as_float(row.get("candidate_cost", ""))
    if task == "task001" and cost <= 1:
        return False
    blocked_terms = ("hacked", "exploit", "zero-mac", "zero_mac")
    return not any(term in source for term in blocked_terms)


def best_candidates_by_task(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        task = row.get("task", "")
        if not task:
            continue
        old = best.get(task)
        if old is None:
            best[task] = row
            continue
        new_key = (
            -as_float(row.get("delta_points", "")),
            as_float(row.get("candidate_cost", "")),
            row.get("source", ""),
        )
        old_key = (
            -as_float(old.get("delta_points", "")),
            as_float(old.get("candidate_cost", "")),
            old.get("source", ""),
        )
        if new_key < old_key:
            best[task] = row
    return sorted(best.values(), key=candidate_sort)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", type=Path, default=Path("reports/clean_batch8_task_costs.csv"))
    parser.add_argument(
        "--candidate-compare",
        type=Path,
        default=Path("reports/candidate_compare_after_batch8.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("reports/sharp_work_queue.md"))
    args = parser.parse_args()

    rows = read_csv(args.costs)
    for row in rows:
        row["cost_i"] = str(as_int(row.get("cost", "")))
        row["points_f"] = f"{as_float(row.get('points', '')):.9f}"

    current_20_clean = sorted(
        [
            row
            for row in rows
            if row.get("static_expected_shape") == "1"
            and as_int(row.get("cost", "")) <= 148
            and as_float(row.get("points", "")) >= 20
        ],
        key=task_sort,
    )
    current_20_nonstatic = sorted(
        [
            row
            for row in rows
            if row.get("static_expected_shape") != "1"
            and as_int(row.get("cost", "")) <= 148
            and as_float(row.get("points", "")) >= 20
        ],
        key=task_sort,
    )
    near_20 = sorted(
        [
            row
            for row in rows
            if row.get("static_expected_shape") == "1"
            and 148 < as_int(row.get("cost", "")) <= 800
            and (
                has_family(row, HIGH_POTENTIAL_FAMILIES)
                or "same_shape_rule" in row.get("families", "")
                or "small_grid" in row.get("families", "")
                or "expand_or_tile" in row.get("families", "")
            )
        ],
        key=task_sort,
    )
    crop_lane = sorted(
        [
            row
            for row in rows
            if 15 <= as_float(row.get("points", "")) < 17
            and as_int(row.get("cost", "")) <= 20000
            and has_family(row, CROP_LANE_FAMILIES)
        ],
        key=lambda row: (-as_float(row.get("points", "")), as_int(row.get("cost", "")), row.get("task", "")),
    )
    drop_lane = sorted(
        [
            row
            for row in rows
            if as_int(row.get("cost", "")) >= 100000
        ],
        key=lambda row: (-as_int(row.get("cost", "")), row.get("task", "")),
    )

    candidates = read_csv(args.candidate_compare)
    positive_candidates = best_candidates_by_task(
        [
            row
            for row in candidates
            if row.get("candidate_reason") == "ok"
            and as_float(row.get("delta_points", "")) > 0.5
            and row.get("candidate_cost")
            and is_clean_candidate(row)
        ],
    )

    lines: list[str] = []
    lines.append("# Sharp NeuroGolf Work Queue")
    lines.append("")
    lines.append(f"Cost file: `{args.costs}`")
    lines.append(f"Candidate compare: `{args.candidate_compare}`")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append("- `20+` needs `cost <= exp(5) ~= 148.4`.")
    lines.append("- `+0.5` score needs at least a `39.3%` cost cut because `new_cost <= old_cost * exp(-0.5)`.")
    lines.append("- Six-figure tasks stay out of the queue unless a verified candidate clears `+0.5` local points.")
    lines.append("")

    lines.append(f"## Current Clean Static 20+ Tasks ({len(current_20_clean)})")
    lines.append("")
    lines.append("| Task | Bucket | Cost | Points | Static | Families |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    lines.extend(fmt_task(row) for row in current_20_clean)
    lines.append("")

    lines.append(f"## Public Low-Cost But Non-Static 20+ Tasks ({len(current_20_nonstatic)})")
    lines.append("")
    lines.append("Useful as score evidence, but not part of the strict clean lane unless rebuilt with static IO.")
    lines.append("")
    lines.append("| Task | Bucket | Cost | Points | Static | Families |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    lines.extend(fmt_task(row) for row in current_20_nonstatic)
    lines.append("")

    lines.append(f"## 20+ Watchlist ({len(near_20)})")
    lines.append("")
    lines.append("These are close enough that a tiny lookup/remap/gather rewrite can matter.")
    lines.append("")
    lines.append("| Task | Bucket | Cost | Points | Static | Families |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    lines.extend(fmt_task(row) for row in near_20)
    lines.append("")

    lines.append(f"## Crop 15-17 Lane ({len(crop_lane)})")
    lines.append("")
    lines.append("Use these for bounded crop/pad cleanup only; do not chase 20+ unless the graph becomes tiny.")
    lines.append("")
    lines.append("| Task | Bucket | Cost | Points | Static | Families |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    lines.extend(fmt_task(row) for row in crop_lane[:80])
    lines.append("")

    lines.append(f"## Candidate Replacements Over +0.5 ({len(positive_candidates)})")
    lines.append("")
    if positive_candidates:
        lines.append("| Task | Delta | Base Cost | Candidate Cost | Source | Zip |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for row in positive_candidates[:80]:
            lines.append(
                f"| {row.get('task', '')} | {as_float(row.get('delta_points', '')):.3f} | "
                f"{row.get('base_cost', '')} | {row.get('candidate_cost', '')} | "
                f"{row.get('source', '')} | `{row.get('candidate_zip', '')}` |"
            )
    else:
        lines.append("No verified `>0.5` candidate replacements found yet.")
    lines.append("")

    lines.append(f"## Drop Unless Big-Cut ({len(drop_lane)})")
    lines.append("")
    lines.append("| Task | Cost | Points | Needed Cost For +0.5 | Families |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in drop_lane[:80]:
        cost = as_int(row.get("cost", ""))
        needed = math.floor(cost * math.exp(-0.5))
        lines.append(
            f"| {row.get('task', '')} | {cost} | {as_float(row.get('points', '')):.3f} | "
            f"{needed} | {row.get('families', '')} |"
        )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        " ".join(
            [
                f"current_20_clean={len(current_20_clean)}",
                f"current_20_nonstatic={len(current_20_nonstatic)}",
                f"near_20={len(near_20)}",
                f"crop_lane={len(crop_lane)}",
                f"positives={len(positive_candidates)}",
                f"drop={len(drop_lane)}",
            ]
        )
    )


if __name__ == "__main__":
    main()
