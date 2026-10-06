#!/usr/bin/env python3
"""
Triage all NeuroGolf tasks and rank easy static-ONNX targets.

The output is intentionally heuristic. It does not solve tasks; it finds
families that are likely cheap to encode as static ONNX graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


Grid = list[list[int]]
Example = dict[str, Grid]

STARTER_TASKS = {6, 95, 127, 261, 331}


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def area(grid: Grid) -> int:
    h, w = shape(grid)
    return h * w


def colors(grid: Grid) -> set[int]:
    return {cell for row in grid for cell in row}


def color_counts(grid: Grid) -> Counter[int]:
    return Counter(cell for row in grid for cell in row)


def flatten(grid: Grid) -> list[int]:
    return [cell for row in grid for cell in row]


def grid_to_tuple(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def flip_v(grid: Grid) -> Grid:
    return list(reversed([row[:] for row in grid]))


def rot90(grid: Grid) -> Grid:
    return flip_h(transpose(grid))


def rot180(grid: Grid) -> Grid:
    return flip_v(flip_h(grid))


def rot270(grid: Grid) -> Grid:
    return flip_v(transpose(grid))


def anti_transpose(grid: Grid) -> Grid:
    return rot180(transpose(grid))


TRANSFORMS = {
    "identity": lambda g: [row[:] for row in g],
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}


def all_examples(task: dict[str, list[Example]]) -> list[Example]:
    return task["train"] + task["test"] + task["arc-gen"]


def format_shapes(shapes: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"{h}x{w}" for h, w in sorted(set(shapes)))


def format_colors(values: Iterable[int]) -> str:
    return "".join(str(v) for v in sorted(set(values)))


def exact_transform_tags(examples: list[Example]) -> list[str]:
    possible: set[str] | None = None
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        matching = {name for name, fn in TRANSFORMS.items() if fn(inp) == out}
        possible = matching if possible is None else possible & matching
        if not possible:
            return []
    return sorted(possible or [])


def color_remap(examples: list[Example]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        if shape(inp) != shape(out):
            return None
        for in_color, out_color in zip(flatten(inp), flatten(out)):
            if in_color in mapping and mapping[in_color] != out_color:
                return None
            mapping[in_color] = out_color
    return mapping


def find_subgrid_positions(inp: Grid, out: Grid) -> set[tuple[int, int]]:
    in_h, in_w = shape(inp)
    out_h, out_w = shape(out)
    if out_h > in_h or out_w > in_w:
        return set()
    target = grid_to_tuple(out)
    positions: set[tuple[int, int]] = set()
    for r in range(in_h - out_h + 1):
        for c in range(in_w - out_w + 1):
            sub = tuple(tuple(inp[r + rr][c : c + out_w]) for rr in range(out_h))
            if sub == target:
                positions.add((r, c))
    return positions


def exact_crop_tags(examples: list[Example]) -> list[str]:
    all_positions: list[set[tuple[int, int]]] = []
    for ex in examples:
        positions = find_subgrid_positions(ex["input"], ex["output"])
        if not positions:
            return []
        all_positions.append(positions)
    common = set.intersection(*all_positions) if all_positions else set()
    return ["crop_exact_same_offset"] if common else ["crop_exact_variable_offset"]


def full_separator_lines(grid: Grid) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape(grid)
    col_lines: list[tuple[int, int]] = []
    row_lines: list[tuple[int, int]] = []
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1:
            col_lines.append((c, col[0]))
    for r in range(h):
        row = grid[r]
        if len(set(row)) == 1:
            row_lines.append((r, row[0]))
    return col_lines, row_lines


def split_candidates(ex: Example) -> list[tuple[str, int, Grid, Grid]]:
    inp = ex["input"]
    out_h, out_w = shape(ex["output"])
    in_h, in_w = shape(inp)
    candidates: list[tuple[str, int, Grid, Grid]] = []
    col_lines, row_lines = full_separator_lines(inp)
    for c, sep_color in col_lines:
        left_w = c
        right_w = in_w - c - 1
        if left_w <= 0 or left_w != right_w:
            continue
        if (in_h, left_w) != (out_h, out_w):
            continue
        left = [row[:c] for row in inp]
        right = [row[c + 1 :] for row in inp]
        candidates.append(("vertical", sep_color, left, right))
    for r, sep_color in row_lines:
        top_h = r
        bottom_h = in_h - r - 1
        if top_h <= 0 or top_h != bottom_h:
            continue
        if (top_h, in_w) != (out_h, out_w):
            continue
        top = inp[:r]
        bottom = inp[r + 1 :]
        candidates.append(("horizontal", sep_color, top, bottom))
    return candidates


def infer_binary_split_relation(examples: list[Example]) -> str | None:
    per_example = [split_candidates(ex) for ex in examples]
    if not per_example or any(not cands for cands in per_example):
        return None

    keys = sorted(set.intersection(*(set((ori, color) for ori, color, _, _ in cands) for cands in per_example)))
    for orientation, sep_color in keys:
        triples: set[tuple[int, int, int]] = set()
        for ex, cands in zip(examples, per_example):
            match = next((cand for cand in cands if cand[0] == orientation and cand[1] == sep_color), None)
            if match is None:
                break
            _, _, left, right = match
            out = ex["output"]
            for r in range(shape(out)[0]):
                for c in range(shape(out)[1]):
                    triples.add((left[r][c], right[r][c], out[r][c]))
        else:
            colors_seen = set()
            for lval, rval, oval in triples:
                colors_seen.add(lval)
                colors_seen.add(rval)
                colors_seen.add(oval)
            relation_modes = [
                ("and", lambda lval, rval, a, b: lval == a and rval == b),
                ("or", lambda lval, rval, a, b: lval == a or rval == b),
                ("xor", lambda lval, rval, a, b: (lval == a) ^ (rval == b)),
            ]
            for mode, predicate in relation_modes:
                for a in range(10):
                    for b in range(10):
                        hit = {oval for lval, rval, oval in triples if predicate(lval, rval, a, b)}
                        miss = {oval for lval, rval, oval in triples if not predicate(lval, rval, a, b)}
                        if len(hit) == 1 and len(miss) == 1 and hit != miss:
                            out_color = next(iter(hit))
                            bg_color = next(iter(miss))
                            return f"split_{orientation}_sep{sep_color}_{mode}_{a}_{b}_to_{out_color}_else_{bg_color}"
    return None


def component_count(grid: Grid) -> int:
    h, w = shape(grid)
    if h == 0 or w == 0:
        return 0
    counts = color_counts(grid)
    background = counts.most_common(1)[0][0]
    seen: set[tuple[int, int]] = set()
    total = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == background or (r, c) in seen:
                continue
            total += 1
            color = grid[r][c]
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return total


def estimate_difficulty(row: dict[str, Any], families: list[str]) -> tuple[int, str]:
    if "identity" in families:
        return 1, "A_exact"
    if "color_remap" in families:
        return 5, "A_exact"
    if "geometric_exact" in families:
        return 8, "A_exact"
    if "split_binary_relation" in families:
        return 10, "A_exact"
    if "crop_exact_same_offset" in families:
        return 12, "A_exact"
    if "crop_exact_variable_offset" in families:
        return 18, "B_template"

    score = 30
    if row["same_shape_all"] == "1":
        score -= 6
    if row["fixed_output_shape"] == "1":
        score -= 3
    if row["max_input_area"] <= 100:
        score -= 4
    elif row["max_input_area"] >= 600:
        score += 8
    if row["input_color_count"] <= 3:
        score -= 3
    elif row["input_color_count"] >= 7:
        score += 4
    if row["output_color_count"] <= 3:
        score -= 2
    if row["has_separator"] == "1":
        score -= 2
    if row["new_output_colors"]:
        score += 2
    if row["shape_variety"] > 2:
        score += 5
    if row["avg_input_components"] >= 8:
        score += 4

    if score <= 20:
        bucket = "B_template"
    elif score <= 32:
        bucket = "C_medium"
    else:
        bucket = "D_hard"
    return max(score, 2), bucket


def analyze_task(path: Path) -> dict[str, Any]:
    task_num = int(path.stem.replace("task", ""))
    task = json.loads(path.read_text())
    examples = all_examples(task)

    input_shapes = [shape(ex["input"]) for ex in examples]
    output_shapes = [shape(ex["output"]) for ex in examples]
    input_colors = set().union(*(colors(ex["input"]) for ex in examples))
    output_colors = set().union(*(colors(ex["output"]) for ex in examples))
    new_output_colors = output_colors - input_colors
    same_shape_all = all(inp == out for inp, out in zip(input_shapes, output_shapes))
    fixed_input_shape = len(set(input_shapes)) == 1
    fixed_output_shape = len(set(output_shapes)) == 1
    output_smaller_all = all(area(ex["output"]) < area(ex["input"]) for ex in examples)
    output_larger_all = all(area(ex["output"]) > area(ex["input"]) for ex in examples)
    max_input_area = max(h * w for h, w in input_shapes)
    max_output_area = max(h * w for h, w in output_shapes)

    col_sep_counter: Counter[int] = Counter()
    row_sep_counter: Counter[int] = Counter()
    for ex in examples:
        col_lines, row_lines = full_separator_lines(ex["input"])
        col_sep_counter.update(color for _, color in col_lines if color != 0)
        row_sep_counter.update(color for _, color in row_lines if color != 0)
    has_separator = bool(col_sep_counter or row_sep_counter)

    families: list[str] = []
    notes: list[str] = []

    transforms = exact_transform_tags(examples)
    if transforms:
        if transforms == ["identity"]:
            families.append("identity")
        else:
            families.append("geometric_exact")
        notes.append("exact_transform=" + "|".join(transforms))

    mapping = color_remap(examples)
    if mapping is not None:
        if any(k != v for k, v in mapping.items()):
            families.append("color_remap")
            notes.append("color_map=" + ",".join(f"{k}->{v}" for k, v in sorted(mapping.items())))
        elif "identity" not in families:
            families.append("identity")

    crop_tags = exact_crop_tags(examples)
    families.extend(tag for tag in crop_tags if tag not in families)

    split_relation = infer_binary_split_relation(examples)
    if split_relation:
        families.append("split_binary_relation")
        notes.append(split_relation)

    if same_shape_all and "color_remap" not in families and "geometric_exact" not in families:
        families.append("same_shape_rule")
    if output_smaller_all and not crop_tags:
        families.append("reduce_or_crop")
    if output_larger_all:
        families.append("expand_or_tile")
    if has_separator and "split_binary_relation" not in families:
        families.append("separator_structure")
    if max_input_area <= 100 and max_output_area <= 100:
        families.append("small_grid")
    if len(input_colors) <= 3:
        families.append("low_color")
    if new_output_colors:
        families.append("new_output_colors")

    avg_components = sum(component_count(ex["input"]) for ex in examples) / len(examples)
    max_components = max(component_count(ex["input"]) for ex in examples)

    row: dict[str, Any] = {
        "task": f"task{task_num:03d}",
        "task_num": task_num,
        "starter_suggested": "1" if task_num in STARTER_TASKS else "0",
        "train_n": len(task["train"]),
        "test_n": len(task["test"]),
        "arc_gen_n": len(task["arc-gen"]),
        "example_n": len(examples),
        "input_shapes": format_shapes(input_shapes),
        "output_shapes": format_shapes(output_shapes),
        "same_shape_all": "1" if same_shape_all else "0",
        "fixed_input_shape": "1" if fixed_input_shape else "0",
        "fixed_output_shape": "1" if fixed_output_shape else "0",
        "output_smaller_all": "1" if output_smaller_all else "0",
        "output_larger_all": "1" if output_larger_all else "0",
        "max_input_area": max_input_area,
        "max_output_area": max_output_area,
        "input_colors": format_colors(input_colors),
        "output_colors": format_colors(output_colors),
        "new_output_colors": format_colors(new_output_colors),
        "input_color_count": len(input_colors),
        "output_color_count": len(output_colors),
        "has_separator": "1" if has_separator else "0",
        "separator_col_colors": ",".join(f"{k}:{v}" for k, v in sorted(col_sep_counter.items())),
        "separator_row_colors": ",".join(f"{k}:{v}" for k, v in sorted(row_sep_counter.items())),
        "shape_variety": len(set(input_shapes + output_shapes)),
        "avg_input_components": round(avg_components, 2),
        "max_input_components": max_components,
        "families": ";".join(dict.fromkeys(families)),
        "notes": " | ".join(notes),
    }
    difficulty, bucket = estimate_difficulty(row, families)
    row["difficulty"] = difficulty
    row["bucket"] = bucket
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bucket_counts = Counter(row["bucket"] for row in rows)
    family_counts: Counter[str] = Counter()
    for row in rows:
        for family in row["families"].split(";"):
            if family:
                family_counts[family] += 1

    easiest = sorted(rows, key=lambda r: (r["difficulty"], r["task_num"]))[:60]
    starter = [row for row in rows if row["starter_suggested"] == "1"]

    lines = [
        "# NeuroGolf Task Triage Summary",
        "",
        f"Total tasks: {len(rows)}",
        "",
        "## Buckets",
        "",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- {bucket}: {count}")

    lines.extend(["", "## Family Counts", ""])
    for family, count in family_counts.most_common():
        lines.append(f"- {family}: {count}")

    def add_table(title: str, table_rows: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Task | Difficulty | Bucket | Shapes | Colors | Families | Notes |",
                "| --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for row in table_rows:
            shapes = f"{row['input_shapes']} -> {row['output_shapes']}"
            colors_text = f"{row['input_colors']} -> {row['output_colors']}"
            lines.append(
                f"| {row['task']} | {row['difficulty']} | {row['bucket']} | "
                f"{shapes} | {colors_text} | {row['families']} | {row['notes']} |"
            )

    add_table("Top 60 Easiest Heuristic Targets", easiest)
    add_table("Official Starter Tasks", sorted(starter, key=lambda r: r["task_num"]))

    lines.extend(
        [
            "",
            "## Recommended First Lane",
            "",
            "1. Keep task006 as the first self-built static ONNX proof.",
            "2. Implement exact families first: color_remap, geometric_exact, split_binary_relation, crop_exact_same_offset.",
            "3. Use B_template tasks next only after a reusable ONNX template exists.",
            "4. Treat D_hard tasks as manual rescue backlog.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage NeuroGolf task JSON files.")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    task_files = sorted(args.data_root.glob("task*.json"))
    if len(task_files) != 400:
        raise SystemExit(f"Expected 400 task JSON files under {args.data_root}, found {len(task_files)}")

    rows = [analyze_task(path) for path in task_files]
    rows = sorted(rows, key=lambda r: (r["difficulty"], r["task_num"]))

    csv_path = args.out_dir / "task_triage.csv"
    json_path = args.out_dir / "task_triage.json"
    summary_path = args.out_dir / "task_triage_summary.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary(rows, summary_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print("top 20:")
    for row in rows[:20]:
        print(f"{row['task']} difficulty={row['difficulty']} bucket={row['bucket']} families={row['families']}")


if __name__ == "__main__":
    main()
