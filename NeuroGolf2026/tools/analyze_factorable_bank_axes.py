#!/usr/bin/env python3
"""Find shared predicate axes for compact NeuroGolf bank models.

The goal is to stop hand-picking one task at a time.  For every lane-A/B task,
this script searches for a small set of reusable predicates that separates the
observed output bank: size, color counts, and a few input cell probes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable


FACTORY_FAMILIES = (
    "same_shape_rule",
    "small_grid",
    "low_color",
    "new_output_colors",
    "expand_or_tile",
    "reduce_or_crop",
)
STRUCTURAL_FAMILIES = (
    "color_remap",
    "crop_exact_same_offset",
    "geometric_exact",
    "split_binary_relation",
)


Grid = list[list[int]]
Example = dict[str, Grid]
FeatureValue = int | tuple[int, ...]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: str | None) -> int:
    if not value:
        return 0
    return int(float(value))


def as_float(value: str | None) -> float:
    if not value:
        return 0.0
    return float(value)


def has_family(row: dict[str, str], families: tuple[str, ...]) -> bool:
    text = row.get("families", "")
    return any(family in text for family in families)


def load_examples(path: Path) -> list[Example]:
    task = json.loads(path.read_text())
    return task["train"] + task["test"] + task["arc-gen"]


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def grid_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def color_counts(grid: Grid) -> tuple[int, ...]:
    counts = [0] * 10
    for row in grid:
        for value in row:
            counts[value] += 1
    return tuple(counts)


def cell_value(grid: Grid, r: int, c: int) -> int:
    h, w = shape(grid)
    if r >= h or c >= w:
        return -1
    return grid[r][c]


def row_counts(grid: Grid, r: int, color: int) -> int:
    h, _ = shape(grid)
    if r >= h:
        return -1
    return sum(1 for value in grid[r] if value == color)


def col_counts(grid: Grid, c: int, color: int) -> int:
    h, w = shape(grid)
    if c >= w:
        return -1
    return sum(1 for r in range(h) if grid[r][c] == color)


def build_feature_map(examples: list[Example]) -> dict[str, list[FeatureValue]]:
    inputs = [example["input"] for example in examples]
    max_h = max(shape(grid)[0] for grid in inputs)
    max_w = max(shape(grid)[1] for grid in inputs)
    counts = [color_counts(grid) for grid in inputs]

    features: dict[str, list[FeatureValue]] = {
        "height": [shape(grid)[0] for grid in inputs],
        "width": [shape(grid)[1] for grid in inputs],
        "area": [shape(grid)[0] * shape(grid)[1] for grid in inputs],
    }
    for color in range(10):
        values = [count[color] for count in counts]
        if len(set(values)) > 1:
            features[f"count_c{color}"] = values

    # Cell probes are the most common factorization axis for compact bank models.
    for r in range(max_h):
        for c in range(max_w):
            values = [cell_value(grid, r, c) for grid in inputs]
            if len(set(values)) > 1:
                features[f"cell_{r}_{c}"] = values

    # Row/column color counts catch stripe, separator, and phase rules without
    # needing one predicate for every individual cell.
    max_rows_for_counts = min(max_h, 12)
    max_cols_for_counts = min(max_w, 12)
    active_colors = sorted({value for grid in inputs for row in grid for value in row})
    for color in active_colors:
        for r in range(max_rows_for_counts):
            values = [row_counts(grid, r, color) for grid in inputs]
            if len(set(values)) > 1:
                features[f"row{r}_count_c{color}"] = values
        for c in range(max_cols_for_counts):
            values = [col_counts(grid, c, color) for grid in inputs]
            if len(set(values)) > 1:
                features[f"col{c}_count_c{color}"] = values

    return features


def conflict_stats(keys: list[Hashable], outputs: list[Hashable]) -> tuple[int, int, int]:
    buckets: dict[Hashable, set[Hashable]] = defaultdict(set)
    for key, output in zip(keys, outputs):
        buckets[key].add(output)
    conflicting_buckets = sum(len(values) > 1 for values in buckets.values())
    extra_outputs = sum(max(0, len(values) - 1) for values in buckets.values())
    return conflicting_buckets, extra_outputs, len(buckets)


def selected_keys(selected: list[str], features: dict[str, list[FeatureValue]], n: int) -> list[Hashable]:
    if not selected:
        return [()] * n
    return [tuple(features[name][idx] for name in selected) for idx in range(n)]


def greedy_select_features(
    features: dict[str, list[FeatureValue]],
    outputs: list[Hashable],
    max_features: int,
) -> tuple[list[str], int, int, int]:
    n = len(outputs)
    selected: list[str] = []
    remaining = set(features)

    keys = selected_keys(selected, features, n)
    _, extra, bucket_count = conflict_stats(keys, outputs)

    while extra > 0 and remaining and len(selected) < max_features:
        best_name = ""
        best_tuple: tuple[int, int, int, str] | None = None
        for name in remaining:
            test_selected = selected + [name]
            test_keys = selected_keys(test_selected, features, n)
            conflicting, test_extra, test_bucket_count = conflict_stats(test_keys, outputs)
            unique_values = len(set(features[name]))
            candidate = (test_extra, conflicting, unique_values, name)
            if best_tuple is None or candidate < best_tuple:
                best_tuple = candidate
                best_name = name
        if best_tuple is None or best_tuple[0] >= extra:
            break
        selected.append(best_name)
        remaining.remove(best_name)
        _, extra, bucket_count = conflict_stats(selected_keys(selected, features, n), outputs)

    conflicting, extra, bucket_count = conflict_stats(selected_keys(selected, features, n), outputs)
    return selected, conflicting, extra, bucket_count


def lane_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("reason") != "ok" or row.get("static_expected_shape") != "1":
            continue
        cost = as_int(row.get("cost"))
        if cost <= 148 or cost > 3000:
            continue
        if "Compress" not in row.get("ops", "") and "Where" not in row.get("ops", ""):
            continue
        if not has_family(row, FACTORY_FAMILIES + STRUCTURAL_FAMILIES):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: (as_int(row.get("cost")), row.get("task", "")))


def classify(row: dict[str, object]) -> str:
    if row["extra_conflicts"] == 0 and row["selected_feature_count"] <= 4 and row["bank_groups"] <= 80:
        return "build_first"
    if row["extra_conflicts"] == 0 and row["selected_feature_count"] <= 8:
        return "build_second"
    if row["extra_conflicts"] == 0:
        return "many_features"
    return "needs_richer_features"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", type=Path, default=Path("reports/task292_two_stage_compact_task_costs.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-csv", type=Path, default=Path("reports/factorable_bank_axes.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/factorable_bank_axes.md"))
    parser.add_argument("--max-features", type=int, default=8)
    args = parser.parse_args()

    cost_rows = read_csv(args.costs)
    tasks = lane_rows(cost_rows)

    report_rows: list[dict[str, object]] = []
    for row in tasks:
        task = row["task"]
        examples = load_examples(args.data_root / f"{task}.json")
        outputs = [grid_key(example["output"]) for example in examples]
        features = build_feature_map(examples)
        selected, conflicting, extra, bank_groups = greedy_select_features(features, outputs, args.max_features)
        output_groups = len(set(outputs))
        feature_types = Counter(name.split("_", 1)[0] for name in selected)
        result: dict[str, object] = {
            "task": task,
            "bucket": row.get("bucket", ""),
            "cost": as_int(row.get("cost")),
            "points": f"{as_float(row.get('points')):.6f}",
            "examples": len(examples),
            "unique_outputs": output_groups,
            "candidate_features": len(features),
            "selected_feature_count": len(selected),
            "selected_features": ";".join(selected),
            "feature_types": ";".join(f"{key}:{value}" for key, value in sorted(feature_types.items())),
            "conflicting_buckets": conflicting,
            "extra_conflicts": extra,
            "bank_groups": bank_groups,
            "families": row.get("families", ""),
        }
        result["lane"] = classify(result)
        report_rows.append(result)

    report_rows.sort(
        key=lambda row: (
            {"build_first": 0, "build_second": 1, "many_features": 2, "needs_richer_features": 3}[str(row["lane"])],
            int(row["selected_feature_count"]),
            int(row["cost"]),
            str(row["task"]),
        )
    )
    write_csv(args.out_csv, report_rows)

    lines: list[str] = []
    lines.append("# Factorable Bank Axes")
    lines.append("")
    lines.append(f"Cost file: `{args.costs}`")
    lines.append(f"Data root: `{args.data_root}`")
    lines.append("")
    lanes = Counter(str(row["lane"]) for row in report_rows)
    lines.append("## Summary")
    lines.append("")
    for lane, count in lanes.most_common():
        lines.append(f"- `{lane}`: `{count}`")
    lines.append("")
    lines.append("| Task | Lane | Cost | Points | Unique Outputs | Features | Bank Groups | Selected Features |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in report_rows:
        selected = str(row["selected_features"])
        if len(selected) > 90:
            selected = selected[:87] + "..."
        lines.append(
            f"| {row['task']} | {row['lane']} | {row['cost']} | {row['points']} | "
            f"{row['unique_outputs']} | {row['selected_feature_count']} | {row['bank_groups']} | {selected} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `build_first`: the observed output bank is separated by <=4 cheap axes; this is the best target for automatic ONNX bank factorization.")
    lines.append("- `build_second`: separated by <=8 axes; still promising, but the emitted graph may need grouped predicates or a small decision tree.")
    lines.append("- `many_features`: exact separation exists but likely overfits unless the features map to an obvious ARC rule.")
    lines.append("- `needs_richer_features`: add rule-specific axes such as connected components, bounding boxes, symmetry, or periodicity.")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"tasks={len(tasks)}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    print(" ".join(f"{lane}={count}" for lane, count in sorted(lanes.items())))


if __name__ == "__main__":
    main()
