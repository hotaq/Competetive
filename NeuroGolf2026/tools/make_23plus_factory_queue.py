#!/usr/bin/env python3
"""Build a factory-level queue for pushing NeuroGolf tasks toward 23+.

This report is intentionally clean-lane only: it highlights where a reusable
generator can improve many tasks at once, and calls out where 23+ would require
sub-8 official cost rather than a normal per-task tweak.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


POINT_THRESHOLDS = (23, 22, 21, 20, 19, 18, 17, 16, 15)
FACTORY_FAMILIES = (
    "same_shape_rule",
    "small_grid",
    "low_color",
    "new_output_colors",
    "expand_or_tile",
)
STRUCTURAL_FAMILIES = (
    "color_remap",
    "crop_exact_same_offset",
    "geometric_exact",
    "split_binary_relation",
)
CROP_FAMILIES = (
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


def as_int(value: str | None) -> int:
    if not value:
        return 0
    return int(float(value))


def as_float(value: str | None) -> float:
    if not value:
        return 0.0
    return float(value)


def score_from_cost(cost: float) -> float:
    if cost <= 0:
        return 25.0
    return max(1.0, 25.0 - math.log(cost))


def cost_for_score(score: float) -> float:
    return math.exp(25.0 - score)


def has_family(row: dict[str, str], families: Iterable[str]) -> bool:
    text = row.get("families", "")
    return any(family in text for family in families)


def split_families(text: str) -> list[str]:
    return [part for part in text.split(";") if part]


def is_static(row: dict[str, str]) -> bool:
    return row.get("static_expected_shape") == "1"


def is_scored(row: dict[str, str]) -> bool:
    return row.get("reason") == "ok" and as_int(row.get("cost")) > 0 and as_float(row.get("points")) > 0


def by_cost(row: dict[str, str]) -> tuple[int, str]:
    return as_int(row.get("cost")), row.get("task", "")


def by_points_then_cost(row: dict[str, str]) -> tuple[float, int, str]:
    return (-as_float(row.get("points")), as_int(row.get("cost")), row.get("task", ""))


def md_task_table(rows: list[dict[str, str]], limit: int = 40) -> list[str]:
    lines = [
        "| Task | Bucket | Cost | Points | Static | Families | Ops |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        ops = row.get("ops", "")
        if len(ops) > 70:
            ops = ops[:67] + "..."
        lines.append(
            f"| {row.get('task', '')} | {row.get('bucket', '')} | "
            f"{as_int(row.get('cost'))} | {as_float(row.get('points')):.3f} | "
            f"{row.get('static_expected_shape', '')} | {row.get('families', '')} | {ops} |"
        )
    if len(rows) > limit:
        lines.append(f"| ... | ... | {len(rows) - limit} more | ... | ... | ... | ... |")
    return lines


def read_best_public_candidates(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    good: list[dict[str, str]] = []
    for row in rows:
        if row.get("candidate_reason") != "ok":
            continue
        source = row.get("source", "").lower()
        if any(term in source for term in ("exploit", "zero-mac", "zero_mac", "hacked")):
            continue
        delta = as_float(row.get("delta_points"))
        if delta <= 0:
            continue
        good.append(row)
    good.sort(key=lambda row: (-as_float(row.get("delta_points")), row.get("task", "")))
    return good


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--costs",
        type=Path,
        default=Path("reports/task292_two_stage_compact_task_costs.csv"),
    )
    parser.add_argument(
        "--candidate-compare",
        type=Path,
        default=Path("reports/candidate_compare_after_batch8.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/23plus_factory_queue.md"),
    )
    args = parser.parse_args()

    rows = [row for row in read_csv(args.costs) if row.get("task", "").startswith("task")]
    if not rows:
        raise SystemExit(f"no task rows found in {args.costs}")

    scored_rows = [row for row in rows if is_scored(row)]
    static_rows = [row for row in scored_rows if is_static(row)]
    nonstatic_rows = [row for row in scored_rows if not is_static(row)]
    valid_rows = [row for row in rows if row.get("reason") == "ok"]
    total_points = sum(as_float(row.get("points")) for row in scored_rows)

    threshold_counts_all = {
        threshold: sum(as_float(row.get("points")) >= threshold for row in scored_rows)
        for threshold in POINT_THRESHOLDS
    }
    threshold_counts_static = {
        threshold: sum(is_static(row) and as_float(row.get("points")) >= threshold for row in scored_rows)
        for threshold in POINT_THRESHOLDS
    }

    top_static = sorted(static_rows, key=by_cost)
    top_all = sorted(scored_rows, key=by_cost)
    best_static_cost = as_int(top_static[0].get("cost")) if top_static else 0
    best_all_cost = as_int(top_all[0].get("cost")) if top_all else 0

    near_23_structural = sorted(
        [
            row
            for row in scored_rows
            if as_float(row.get("points")) >= 20
            and has_family(row, STRUCTURAL_FAMILIES + FACTORY_FAMILIES)
        ],
        key=by_points_then_cost,
    )
    factorable_bank = sorted(
        [
            row
            for row in static_rows
            if 148 < as_int(row.get("cost")) <= 3000
            and ("Compress" in row.get("ops", "") or "Where" in row.get("ops", ""))
            and has_family(row, FACTORY_FAMILIES + STRUCTURAL_FAMILIES)
        ],
        key=by_cost,
    )
    near_20_factory = sorted(
        [
            row
            for row in static_rows
            if 148 < as_int(row.get("cost")) <= 900
            and has_family(row, FACTORY_FAMILIES + STRUCTURAL_FAMILIES)
        ],
        key=by_cost,
    )
    dynamic_staticization = sorted(
        [
            row
            for row in nonstatic_rows
            if as_float(row.get("points")) >= 20
            or as_int(row.get("cost")) <= 200
        ],
        key=by_cost,
    )
    crop_lane = sorted(
        [
            row
            for row in static_rows
            if 15 <= as_float(row.get("points")) < 17
            and as_int(row.get("cost")) <= 20000
            and has_family(row, CROP_FAMILIES)
        ],
        key=by_points_then_cost,
    )
    drop_unless_factory = sorted(
        [row for row in scored_rows if as_int(row.get("cost")) >= 100000],
        key=lambda row: (-as_int(row.get("cost")), row.get("task", "")),
    )

    family_counter = Counter()
    for row in scored_rows:
        for family in split_families(row.get("families", "")):
            family_counter[family] += 1

    under_20_family_counter = Counter()
    for row in scored_rows:
        if as_float(row.get("points")) >= 20:
            continue
        for family in split_families(row.get("families", "")):
            under_20_family_counter[family] += 1

    public_candidates = read_best_public_candidates(args.candidate_compare)
    public_big_gains = [
        row
        for row in public_candidates
        if as_float(row.get("delta_points")) >= 0.5
        and as_float(row.get("candidate_cost")) <= as_float(row.get("base_cost")) * math.exp(-0.5)
    ]

    target_23_cost = cost_for_score(23)
    target_20_cost = cost_for_score(20)
    current_gap_to_23_each = 400 * 23 - total_points
    current_gap_to_20_each = 400 * 20 - total_points

    lines: list[str] = []
    lines.append("# 23+ Factory Queue")
    lines.append("")
    lines.append(f"Cost file: `{args.costs}`")
    lines.append(f"Candidate compare: `{args.candidate_compare}`")
    lines.append("")
    lines.append("## Score Ceiling Math")
    lines.append("")
    lines.append(f"- `23+` requires `cost <= exp(2) = {target_23_cost:.6f}`, so integer cost must be `<= 7`.")
    lines.append(f"- `20+` requires `cost <= exp(5) = {target_20_cost:.6f}`, so integer cost must be `<= 148`.")
    lines.append(f"- Current local estimated total: `{total_points:.6f}`.")
    lines.append(f"- Gap to `20 * 400 = 8000`: `{current_gap_to_20_each:.3f}` points.")
    lines.append(f"- Gap to `23 * 400 = 9200`: `{current_gap_to_23_each:.3f}` points.")
    lines.append(f"- Best current static cost: `{best_static_cost}` -> `{score_from_cost(best_static_cost):.3f}` points.")
    lines.append(f"- Best current any-shape cost: `{best_all_cost}` -> `{score_from_cost(best_all_cost):.3f}` points.")
    if best_static_cost > target_23_cost:
        lines.append(
            f"- Clean `23+` is below the current static floor by `x{best_static_cost / target_23_cost:.1f}` cost on the best task."
        )
    lines.append("")
    lines.append("Interpretation: a clean all-task `23+` push is not a per-file optimization problem. It would require a new sub-8-cost representation for whole families. Anything that reports hundreds of `25.00` tasks via zero-volume tensors or uncounted constants belongs to metric-bug territory, not this clean factory lane.")
    lines.append("")

    lines.append("## Current Coverage")
    lines.append("")
    lines.append(f"- Total tasks: `{len(rows)}`")
    lines.append(f"- Locally valid rows: `{len(valid_rows)}`")
    lines.append(f"- Scored rows: `{len(scored_rows)}`")
    lines.append(f"- Static `1x10x30x30` scored rows: `{len(static_rows)}`")
    lines.append(f"- Non-static scored rows: `{len(nonstatic_rows)}`")
    lines.append("")
    lines.append("| Threshold | All Tasks | Static Tasks |")
    lines.append("| ---: | ---: | ---: |")
    for threshold in POINT_THRESHOLDS:
        lines.append(
            f"| >= {threshold} | {threshold_counts_all[threshold]} | {threshold_counts_static[threshold]} |"
        )
    lines.append("")

    lines.append("## Best Current Costs")
    lines.append("")
    lines.extend(md_task_table(top_all[:20], limit=20))
    lines.append("")

    lines.append("## Family Counts Still Under 20")
    lines.append("")
    lines.append("| Family | Under-20 Tasks | All Tasks |")
    lines.append("| --- | ---: | ---: |")
    for family, count in under_20_family_counter.most_common():
        lines.append(f"| {family} | {count} | {family_counter[family]} |")
    lines.append("")

    lines.append(f"## Lane A: Factorable Bank / Predicate Sharing ({len(factorable_bank)})")
    lines.append("")
    lines.append("Use the task292 trick as a generator: split one giant exact-output predicate bank into shared axes such as size, phase, color, orientation, and local pattern. This is the sharpest clean batch lane because it can cut many constant/Compress/Where graphs without changing semantics.")
    lines.append("")
    lines.extend(md_task_table(factorable_bank, limit=60))
    lines.append("")

    lines.append(f"## Lane B: Near-20 Factory Targets ({len(near_20_factory)})")
    lines.append("")
    lines.append("These are close enough that a reusable generator should beat manual one-offs. Minimum useful move is `+0.5`, so the new cost must be at most about `60.65%` of current cost.")
    lines.append("")
    lines.append("| Task | Cost | Points | Needed Cost For +0.5 | Families |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in near_20_factory[:80]:
        cost = as_int(row.get("cost"))
        needed = math.floor(cost * math.exp(-0.5))
        lines.append(
            f"| {row.get('task', '')} | {cost} | {as_float(row.get('points')):.3f} | {needed} | {row.get('families', '')} |"
        )
    lines.append("")

    lines.append(f"## Lane C: Dynamic Low-Cost Staticization ({len(dynamic_staticization)})")
    lines.append("")
    lines.append("These are already low-cost public-style models, but not strict static IO. Rebuilding them cleanly may recover 20-22 points, though not 23+ unless the cost drops below 8.")
    lines.append("")
    lines.extend(md_task_table(dynamic_staticization, limit=60))
    lines.append("")

    lines.append(f"## Lane D: Structural 20+ Study Set ({len(near_23_structural)})")
    lines.append("")
    lines.append("Study these to identify reusable primitives. They are not necessarily next submissions; they are examples of what a compact clean representation currently looks like.")
    lines.append("")
    lines.extend(md_task_table(near_23_structural, limit=60))
    lines.append("")

    lines.append(f"## Lane E: Crop / Geometry 15-17 Cleanup ({len(crop_lane)})")
    lines.append("")
    lines.append("Only worth batch work if one generator applies across many tasks. Do not hand-chase unless expected gain is at least `+0.5`.")
    lines.append("")
    lines.extend(md_task_table(crop_lane, limit=80))
    lines.append("")

    lines.append(f"## Public Candidate Big-Cuts Still Worth Checking ({len(public_big_gains)})")
    lines.append("")
    if public_big_gains:
        lines.append("| Task | Delta | Base Cost | Candidate Cost | Source | Zip |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for row in public_big_gains[:60]:
            lines.append(
                f"| {row.get('task', '')} | {as_float(row.get('delta_points')):.3f} | "
                f"{row.get('base_cost', '')} | {row.get('candidate_cost', '')} | "
                f"{row.get('source', '')} | `{row.get('candidate_zip', '')}` |"
            )
    else:
        lines.append("No clean public candidate currently clears the `+0.5` big-cut rule.")
    lines.append("")

    lines.append(f"## Drop Unless Generator-Level Breakthrough ({len(drop_unless_factory)})")
    lines.append("")
    lines.append("These are too expensive for manual scoring gains. Touch them only when a family generator changes the cost class.")
    lines.append("")
    lines.append("| Task | Cost | Points | Needed Cost For +0.5 | Families |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in drop_unless_factory[:80]:
        cost = as_int(row.get("cost"))
        needed = math.floor(cost * math.exp(-0.5))
        lines.append(
            f"| {row.get('task', '')} | {cost} | {as_float(row.get('points')):.3f} | {needed} | {row.get('families', '')} |"
        )
    lines.append("")

    lines.append("## Next Generator Spec")
    lines.append("")
    lines.append("1. Build one analyzer that reads a task JSON and infers shared predicate axes from all train/test/arc-gen examples: dimensions, color set, row/column counts, local pattern signatures, and phase/orientation.")
    lines.append("2. Build one `bank_factorizer` that replaces one predicate per output with a product of shared axes, then emits the smallest exact ONNX graph it can verify locally.")
    lines.append("3. Run it across Lane A and Lane B first; submit only batches with verified local gain `> +0.5` and no metric-risk flags.")
    lines.append("4. Treat `23+` as a research target for new primitives, not the immediate submission gate. The immediate gate is raising many tasks to `20-22` without exploit behavior.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"total_points={total_points:.6f}")
    print(
        " ".join(
            [
                f"ge23={threshold_counts_all[23]}",
                f"ge20={threshold_counts_all[20]}",
                f"static_ge20={threshold_counts_static[20]}",
                f"factorable={len(factorable_bank)}",
                f"near20={len(near_20_factory)}",
                f"dynamic={len(dynamic_staticization)}",
                f"crop={len(crop_lane)}",
                f"drop={len(drop_unless_factory)}",
            ]
        )
    )


if __name__ == "__main__":
    main()
