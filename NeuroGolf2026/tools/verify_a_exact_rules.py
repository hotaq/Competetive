#!/usr/bin/env python3
"""
Pure-Python verifier for A_exact task rules before ONNX generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from triage_tasks import (  # noqa: E402
    TRANSFORMS,
    all_examples,
    grid_to_tuple,
    infer_binary_split_relation,
    shape,
)


Grid = list[list[int]]


def parse_color_map(note: str) -> dict[int, int]:
    match = re.search(r"color_map=([^|]+)", note)
    if not match:
        raise ValueError(f"missing color_map note: {note}")
    mapping: dict[int, int] = {}
    for item in match.group(1).strip().split(","):
        src, dst = item.split("->")
        mapping[int(src)] = int(dst)
    return mapping


def parse_exact_transform(note: str) -> str:
    match = re.search(r"exact_transform=([^| ]+)", note)
    if not match:
        raise ValueError(f"missing exact_transform note: {note}")
    return match.group(1)


def parse_split_relation(note: str) -> tuple[str, int, str, int, int, int, int]:
    pattern = r"split_(vertical|horizontal)_sep(\d+)_(and|or|xor)_(\d+)_(\d+)_to_(\d+)_else_(\d+)"
    match = re.search(pattern, note)
    if not match:
        raise ValueError(f"missing split relation note: {note}")
    orientation, sep, mode, a, b, out_color, else_color = match.groups()
    return orientation, int(sep), mode, int(a), int(b), int(out_color), int(else_color)


def infer_common_crop_offset(examples: list[dict[str, Grid]]) -> tuple[int, int]:
    common: set[tuple[int, int]] | None = None
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        in_h, in_w = shape(inp)
        out_h, out_w = shape(out)
        target = grid_to_tuple(out)
        positions: set[tuple[int, int]] = set()
        for r in range(in_h - out_h + 1):
            for c in range(in_w - out_w + 1):
                candidate = tuple(tuple(inp[r + rr][c : c + out_w]) for rr in range(out_h))
                if candidate == target:
                    positions.add((r, c))
        common = positions if common is None else common & positions
    if not common:
        raise ValueError("no common crop offset")
    return sorted(common)[0]


def color_remap(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping[cell] for cell in row] for row in grid]


def split_relation(grid: Grid, note: str) -> Grid:
    orientation, sep_color, mode, a, b, out_color, else_color = parse_split_relation(note)
    h, w = shape(grid)
    if orientation == "vertical":
        sep = (w - 1) // 2
        assert all(grid[r][sep] == sep_color for r in range(h))
        left = [row[:sep] for row in grid]
        right = [row[sep + 1 :] for row in grid]
    else:
        sep = (h - 1) // 2
        assert all(cell == sep_color for cell in grid[sep])
        left = grid[:sep]
        right = grid[sep + 1 :]

    out: Grid = []
    for r in range(len(left)):
        row: list[int] = []
        for c in range(len(left[0])):
            lval = left[r][c]
            rval = right[r][c]
            if mode == "and":
                hit = lval == a and rval == b
            elif mode == "or":
                hit = lval == a or rval == b
            elif mode == "xor":
                hit = (lval == a) ^ (rval == b)
            else:
                raise ValueError(mode)
            row.append(out_color if hit else else_color)
        out.append(row)
    return out


def crop_same_offset(grid: Grid, offset: tuple[int, int], out_shape: tuple[int, int]) -> Grid:
    r0, c0 = offset
    h, w = out_shape
    return [row[c0 : c0 + w] for row in grid[r0 : r0 + h]]


def predict(row: dict[str, str], examples: list[dict[str, Grid]], grid: Grid, expected_shape: tuple[int, int]) -> Grid:
    families = row["families"].split(";")
    note = row["notes"]
    if "color_remap" in families:
        return color_remap(grid, parse_color_map(note))
    if "geometric_exact" in families:
        return TRANSFORMS[parse_exact_transform(note)](grid)
    if "split_binary_relation" in families:
        return split_relation(grid, note)
    if "crop_exact_same_offset" in families:
        return crop_same_offset(grid, infer_common_crop_offset(examples), expected_shape)
    raise ValueError(f"unsupported A_exact row: {row['task']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--triage-csv", type=Path, default=Path("reports/task_triage.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/a_exact_rule_report.csv"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.triage_csv.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["bucket"] == "A_exact"]

    report: list[dict[str, str | int]] = []
    for row in rows:
        task = json.loads((args.data_root / f"{row['task']}.json").read_text())
        examples = all_examples(task)
        right = 0
        wrong = 0
        first_error = ""
        try:
            # Re-infer split relation once so stale notes are caught.
            if "split_binary_relation" in row["families"].split(";"):
                inferred = infer_binary_split_relation(examples)
                if inferred != row["notes"]:
                    raise ValueError(f"split note stale: {row['notes']} != {inferred}")
            for idx, ex in enumerate(examples):
                pred = predict(row, examples, ex["input"], shape(ex["output"]))
                if pred == ex["output"]:
                    right += 1
                else:
                    wrong += 1
                    if not first_error:
                        first_error = f"mismatch example {idx}"
            status = "ok" if wrong == 0 else "failed"
        except Exception as exc:
            status = "error"
            first_error = str(exc)
        report.append(
            {
                "task": row["task"],
                "families": row["families"],
                "status": status,
                "right": right,
                "wrong": wrong,
                "examples": len(examples),
                "reason": first_error,
                "notes": row["notes"],
            }
        )
        print(f"{row['task']}: {status} {right}/{len(examples)} {first_error}")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
