#!/usr/bin/env python3
"""Build a current metric-v2 feasibility report for 20+ NeuroGolf tasks."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POINT_THRESHOLDS = (23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11)
NEAR_COST_LIMIT = 5000
DATA_ROOT = Path("data/latest/extracted")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clean: list[dict[str, str]] = []
    for row in rows:
        if row.get("reason") != "ok" or not row.get("cost"):
            continue
        row["cost_int"] = str(int(float(row["cost"])))
        row["points_calc"] = f"{score_from_cost(int(row['cost_int'])):.6f}"
        clean.append(row)
    return clean


def score_from_cost(cost: int) -> float:
    if cost <= 0:
        return 25.0
    return max(1.0, 25.0 - math.log(cost))


def cost_for_score(score: float) -> int:
    return math.floor(math.exp(25.0 - score))


def as_int(row: dict[str, str], key: str = "cost_int") -> int:
    return int(float(row.get(key) or 0))


def as_float(row: dict[str, str], key: str = "points") -> float:
    return float(row.get(key) or 0.0)


def lane_for(row: dict[str, str]) -> str:
    cost = as_int(row)
    ops = row.get("ops", "")
    if cost <= cost_for_score(20):
        if ops == "Gather":
            return "current_20plus_channel_gather"
        if ops == "Constant;Transpose":
            return "current_20plus_direct_transpose"
        return "current_20plus_other"
    if cost <= 1500 and "Pad" in ops and "Slice" in ops:
        return "near20_tiny_crop_pad"
    if cost <= NEAR_COST_LIMIT and ("Pad" in ops or "Slice" in ops):
        return "17to19_crop_or_small_static"
    if "Conv" in ops or "Reduce" in ops or "MatMul" in ops:
        return "not_20plus_logic_heavy"
    return "not_20plus_other"


def md_table(rows: list[dict[str, str]], *, limit: int = 40) -> list[str]:
    lines = [
        "| task | cost | points | lane | ops |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        ops = row.get("ops", "")
        if len(ops) > 76:
            ops = ops[:73] + "..."
        lines.append(
            f"| {row.get('task', '')} | {as_int(row)} | {as_float(row):.6f} | "
            f"{row.get('factory_lane', '')} | `{ops}` |"
        )
    if len(rows) > limit:
        lines.append(f"| ... | ... | ... | {len(rows) - limit} more | ... |")
    return lines


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "source",
        "cost_int",
        "points",
        "factory_lane",
        "ops",
        "macs",
        "memory",
        "params",
        "file_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def current_tasks(rows: list[dict[str, str]], *, lane_prefix: str) -> set[str]:
    return {row["task"] for row in rows if row.get("factory_lane", "").startswith(lane_prefix)}


def scan_single_axis_gather() -> list[tuple[str, int, int, int]]:
    from tools.generate_a_exact_submission import all_examples
    from tools.metric_v2_select import examples_fit_30
    from tools.optimize_single_axis_gather_autoscan import infer_single_axis_gather

    import json

    hits: list[tuple[str, int, int, int]] = []
    for task_path in sorted(DATA_ROOT.glob("task*.json")):
        task = json.loads(task_path.read_text())
        examples = all_examples(task)
        if not examples_fit_30(examples):
            continue
        inferred = infer_single_axis_gather(examples)
        if inferred is None:
            continue
        axis, indices = inferred
        hits.append((task_path.stem, axis, len(indices), len(examples)))
    return hits


def scan_geometric_exact() -> list[tuple[str, str, int]]:
    from tools.metric_v2_select import examples_fit_30
    from tools.triage_tasks import all_examples, exact_transform_tags

    import json

    hits: list[tuple[str, str, int]] = []
    for task_path in sorted(DATA_ROOT.glob("task*.json")):
        task = json.loads(task_path.read_text())
        examples = all_examples(task)
        if not examples_fit_30(examples):
            continue
        tags = [tag for tag in exact_transform_tags(examples) if tag != "identity"]
        for tag in tags:
            hits.append((task_path.stem, tag, len(examples)))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("generated/task004_corner_shift_v86c/metric_v2_selected.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/metric_v2_20plus_feasibility.md"),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("reports/metric_v2_20plus_feasibility.csv"),
    )
    args = parser.parse_args()

    rows = read_rows(args.selected)
    for row in rows:
        row["factory_lane"] = lane_for(row)

    rows_by_cost = sorted(rows, key=lambda r: (as_int(r), r["task"]))
    plus20 = [r for r in rows_by_cost if as_int(r) <= cost_for_score(20)]
    near20 = [r for r in rows_by_cost if cost_for_score(20) < as_int(r) <= NEAR_COST_LIMIT]
    numeric_near20 = [r for r in rows_by_cost if cost_for_score(20) < as_int(r) <= 1500]
    direct_20 = [r for r in plus20 if r["factory_lane"].startswith("current_20plus")]

    total_points = sum(as_float(r) for r in rows)
    threshold_counts = {
        threshold: sum(as_float(row) >= threshold for row in rows)
        for threshold in POINT_THRESHOLDS
    }
    lane_counts = Counter(row["factory_lane"] for row in rows)
    plus20_ops = Counter(row.get("ops", "") for row in plus20)
    gather_hits = scan_single_axis_gather()
    geometric_hits = scan_geometric_exact()
    selected_gathers = current_tasks(rows, lane_prefix="current_20plus_channel_gather")
    missed_gathers = [hit for hit in gather_hits if hit[0] not in selected_gathers]

    write_csv(args.csv_out, rows_by_cost)

    lines: list[str] = []
    lines.append("# Metric-v2 20+ Feasibility")
    lines.append("")
    lines.append(f"- Source selected report: `{args.selected}`")
    lines.append(f"- Scored tasks in current local best: `{len(rows)}`")
    lines.append(f"- Current local metric-v2 score: `{total_points:.6f}`")
    lines.append(f"- Average over 400 tasks: `{total_points / 400.0:.6f}`")
    lines.append(f"- `20+` hard cost threshold: `cost <= {cost_for_score(20)}`")
    lines.append(f"- `19+` hard cost threshold: `cost <= {cost_for_score(19)}`")
    lines.append(f"- `18+` hard cost threshold: `cost <= {cost_for_score(18)}`")
    lines.append("")
    lines.append("## Current Counts")
    lines.append("")
    for threshold in POINT_THRESHOLDS:
        lines.append(f"- Tasks at `{threshold}+`: `{threshold_counts[threshold]}`")
    lines.append("")
    lines.append("## Practical Conclusion")
    lines.append("")
    lines.append(
        "Clean `20+` requires total `params + static-memory + MACs <= 148`. "
        "Under the April 28 metric update, one ordinary 30x30 float activation is already far above this. "
        "So the only repeatable clean 20+ family is a graph that returns the output almost directly: "
        "channel `Gather`, direct geometric `Transpose`, or an equally tiny no-intermediate transform."
    )
    lines.append("")
    lines.append(
        "For tasks that need object detection, masks, padding, reductions, or convolution, the clean target should be "
        "`13-18` first, then `18-19` for tiny crop/pad cases. Trying to force every task to 20+ cleanly is not a "
        "good work queue; it implies a new scoring loophole or a radically different representation."
    )
    lines.append("")
    lines.append("## 20+ Factory Lanes")
    lines.append("")
    lines.append("1. `channel_gather`: same-shape color remap or channel permutation. Current cost floor: `10-30`, score `21.6-22.7`.")
    lines.append("2. `direct_transpose`: exact full-grid transpose/rotation-like tasks with no pad/slice body. Current cost: `5`, score `23.39` locally.")
    lines.append("3. `tiny_crop_pad`: near miss only. It lands around `18-19.6` because `Pad/Slice` static tensors still count.")
    lines.append("4. `logic_heavy`: Conv/Reduce/Where/MatMul tasks. These are not 20+ candidates under clean metric-v2.")
    lines.append("")
    lines.append("## Fresh Direct-Family Rescan")
    lines.append("")
    lines.append(
        f"- Single-axis `Gather` scan found `{len(gather_hits)}` tasks; missed by current selected 20+ set: `{len(missed_gathers)}`."
    )
    if gather_hits:
        lines.append(
            "- Gather hits: "
            + ", ".join(f"`{task}:axis{axis}/n{length}`" for task, axis, length, _ in gather_hits)
        )
    if missed_gathers:
        lines.append(
            "- Missed gather hits: "
            + ", ".join(f"`{task}:axis{axis}/n{length}`" for task, axis, length, _ in missed_gathers)
        )
    lines.append(f"- Exact geometric scan found `{len(geometric_hits)}` non-identity tasks.")
    if geometric_hits:
        lines.append(
            "- Geometric hits: "
            + ", ".join(f"`{task}:{tag}`" for task, tag, _ in geometric_hits)
        )
    lines.append(
        "- Only direct `transpose` naturally stays output-direct on the padded tensor. "
        "Flips and rotations need crop/pad or a small intermediate to keep the valid grid at the top-left, "
        "which already pushes cost above the `20+` threshold."
    )
    lines.append("")
    lines.append("## Lane Counts")
    lines.append("")
    for lane, count in sorted(lane_counts.items()):
        lines.append(f"- `{lane}`: `{count}`")
    lines.append("")
    lines.append("## Current 20+ Tasks")
    lines.append("")
    lines.extend(md_table(direct_20, limit=80))
    lines.append("")
    lines.append("20+ op patterns:")
    for ops, count in plus20_ops.most_common():
        lines.append(f"- `{ops}`: `{count}`")
    lines.append("")
    lines.append("## Closest Numeric Near Misses")
    lines.append("")
    lines.extend(md_table(numeric_near20, limit=40))
    lines.append("")
    lines.append("## Wider 17-19 Static Queue")
    lines.append("")
    lines.extend(md_table(near20, limit=60))
    lines.append("")
    lines.append("## Next Work Queue")
    lines.append("")
    lines.append("1. Re-scan all tasks for exact `Gather` channel remaps and replace any missed same-shape color-map tasks.")
    lines.append("2. Re-scan geometric tasks for direct output `Transpose`/`Gather` models that avoid `Pad` entirely.")
    lines.append("3. Inspect near misses `task326`, `task135`, `task087`, `task140`, and `task380` for removable constants or output-direct rewrites.")
    lines.append("4. Keep complex tasks on the clean 13-18 optimization lane; task004-style object logic will not become 20+ without a new primitive.")
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rows: {len(rows)}")
    print(f"score: {total_points:.6f}")
    print(f"20+ tasks: {len(plus20)}")
    print(f"near <=1500: {len(numeric_near20)}")
    print(f"report: {args.out}")
    print(f"csv: {args.csv_out}")


if __name__ == "__main__":
    main()
