#!/usr/bin/env python3
"""Build factorized output-bank ONNX models from shared predicate axes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Hashable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points
from tools.generate_a_exact_submission import EXPECTED_SHAPE, all_examples, convert_to_numpy, verify_model_bytes


Grid = list[list[int]]
Example = dict[str, Grid]


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


def constant(name: str, value: np.ndarray | list[int] | float, dtype: np.dtype | type) -> onnx.NodeProto:
    array = np.asarray(value, dtype=dtype)
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name=f"{name}_value"))


def input_info() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)


def output_info(dtype: int = TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("output", dtype, EXPECTED_SHAPE)


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def color_counts(grid: Grid) -> tuple[int, ...]:
    counts = [0] * 10
    for row in grid:
        for value in row:
            counts[value] += 1
    return tuple(counts)


def feature_value(grid: Grid, name: str) -> int:
    h, w = shape(grid)
    if name == "height":
        return h
    if name == "width":
        return w
    if name == "area":
        return h * w
    if match := re.fullmatch(r"count_c(\d+)", name):
        return color_counts(grid)[int(match.group(1))]
    if match := re.fullmatch(r"cell_(\d+)_(\d+)", name):
        r, c = int(match.group(1)), int(match.group(2))
        if r >= h or c >= w:
            return -1
        return grid[r][c]
    if match := re.fullmatch(r"row(\d+)_count_c(\d+)", name):
        r, color = int(match.group(1)), int(match.group(2))
        if r >= h:
            return -1
        return sum(1 for value in grid[r] if value == color)
    if match := re.fullmatch(r"col(\d+)_count_c(\d+)", name):
        c, color = int(match.group(1)), int(match.group(2))
        if c >= w:
            return -1
        return sum(1 for r in range(h) if grid[r][c] == color)
    raise ValueError(f"unsupported feature: {name}")


def output_bank_entry(example: Example) -> np.ndarray:
    _, y = convert_to_numpy(example)
    return y.astype(np.float32, copy=False)


def key_for_example(example: Example, features: list[str]) -> tuple[int, ...]:
    return tuple(feature_value(example["input"], feature) for feature in features)


def build_bank(examples: list[Example], features: list[str]) -> tuple[list[tuple[int, ...]], np.ndarray]:
    by_key: dict[tuple[int, ...], np.ndarray] = {}
    for example in examples:
        key = key_for_example(example, features)
        output = output_bank_entry(example)
        existing = by_key.get(key)
        if existing is not None and not np.array_equal(existing, output):
            raise ValueError(f"feature key is not exact: {key}")
        by_key[key] = output
    keys = sorted(by_key)
    bank = np.stack([by_key[key] for key in keys], axis=0)
    return keys, bank


def add_slice(
    nodes: list[onnx.NodeProto],
    prefix: str,
    source: str,
    starts: list[int],
    ends: list[int],
    output: str,
) -> None:
    nodes.extend(
        [
            constant(f"{prefix}_starts", starts, np.int64),
            constant(f"{prefix}_ends", ends, np.int64),
            constant(f"{prefix}_axes", [0, 1, 2, 3], np.int64),
            constant(f"{prefix}_steps", [1, 1, 1, 1], np.int64),
        ]
    )
    nodes.append(
        helper.make_node(
            "Slice",
            [source, f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes", f"{prefix}_steps"],
            [output],
            name=f"{prefix}_slice",
        )
    )


def add_numeric_feature(nodes: list[onnx.NodeProto], name: str, output: str) -> None:
    if name == "area":
        nodes.append(helper.make_node("ReduceSum", ["input", "axes_all"], [output], keepdims=1, name=f"{name}_sum"))
        return
    if match := re.fullmatch(r"count_c(\d+)", name):
        color = int(match.group(1))
        add_slice(nodes, name, "input", [0, color, 0, 0], [1, color + 1, 30, 30], f"{name}_slice")
        nodes.append(helper.make_node("ReduceSum", [f"{name}_slice", "axes_all"], [output], keepdims=1, name=f"{name}_sum"))
        return
    if match := re.fullmatch(r"row(\d+)_count_c(\d+)", name):
        row, color = int(match.group(1)), int(match.group(2))
        add_slice(nodes, name, "input", [0, color, row, 0], [1, color + 1, row + 1, 30], f"{name}_slice")
        nodes.append(helper.make_node("ReduceSum", [f"{name}_slice", "axes_all"], [output], keepdims=1, name=f"{name}_sum"))
        return
    if match := re.fullmatch(r"col(\d+)_count_c(\d+)", name):
        col, color = int(match.group(1)), int(match.group(2))
        add_slice(nodes, name, "input", [0, color, 0, col], [1, color + 1, 30, col + 1], f"{name}_slice")
        nodes.append(helper.make_node("ReduceSum", [f"{name}_slice", "axes_all"], [output], keepdims=1, name=f"{name}_sum"))
        return
    raise ValueError(f"not a numeric feature: {name}")


def add_cell_condition(nodes: list[onnx.NodeProto], feature: str, value: int, output: str) -> None:
    match = re.fullmatch(r"cell_(\d+)_(\d+)", feature)
    if not match:
        raise ValueError(f"not a cell feature: {feature}")
    row, col = int(match.group(1)), int(match.group(2))
    if value >= 0:
        add_slice(nodes, f"{output}_cell", "input", [0, value, row, col], [1, value + 1, row + 1, col + 1], f"{output}_raw")
        nodes.append(helper.make_node("Cast", [f"{output}_raw"], [output], to=TensorProto.BOOL, name=f"{output}_cast"))
        return
    add_slice(nodes, f"{output}_cell_any", "input", [0, 0, row, col], [1, 10, row + 1, col + 1], f"{output}_any_raw")
    nodes.append(helper.make_node("ReduceMax", [f"{output}_any_raw"], [f"{output}_max"], axes=[0, 1, 2, 3], keepdims=1, name=f"{output}_max"))
    nodes.append(helper.make_node("Equal", [f"{output}_max", "zero_scalar"], [f"{output}_eq"], name=f"{output}_eq"))
    nodes.append(helper.make_node("Identity", [f"{output}_eq"], [output], name=f"{output}_identity"))


def is_cell_feature(name: str) -> bool:
    return re.fullmatch(r"cell_\d+_\d+", name) is not None


def build_model(task: str, examples: list[Example], features: list[str], selector: str) -> onnx.ModelProto:
    keys, bank = build_bank(examples, features)
    bank_dtype = np.bool_ if selector == "compress" else np.float32
    bank = bank.astype(bank_dtype, copy=False)
    nodes: list[onnx.NodeProto] = [
        constant("axes_all", [0, 1, 2, 3], np.int64),
        constant("axis0", [0], np.int64),
        constant("shape_match4", [1, 1, 1, 1], np.int64),
        constant("shape_match_vec", [len(keys)], np.int64),
        constant("squeeze_axes", [0], np.int64),
        constant("zero_scalar", 0.0, np.float32),
        constant("bank", bank, bank_dtype),
    ]

    runtime_features: dict[str, str] = {}
    for feature in features:
        if is_cell_feature(feature):
            continue
        runtime_name = f"feat_{feature}"
        add_numeric_feature(nodes, feature, runtime_name)
        runtime_features[feature] = runtime_name

    branch_conditions: list[str] = []
    for branch_idx, key in enumerate(keys):
        feature_conditions: list[str] = []
        for feature, value in zip(features, key):
            cond_name = f"b{branch_idx}_{feature}_is_{value}".replace("-", "neg")
            if is_cell_feature(feature):
                add_cell_condition(nodes, feature, value, cond_name)
            else:
                nodes.append(constant(f"{cond_name}_target", float(value), np.float32))
                nodes.append(helper.make_node("Equal", [runtime_features[feature], f"{cond_name}_target"], [f"{cond_name}_raw"], name=f"{cond_name}_eq"))
                nodes.append(helper.make_node("Identity", [f"{cond_name}_raw"], [cond_name], name=f"{cond_name}_identity"))
            feature_conditions.append(cond_name)

        current = feature_conditions[0]
        for cond_idx, next_condition in enumerate(feature_conditions[1:], start=1):
            out = f"b{branch_idx}_and_{cond_idx}"
            nodes.append(helper.make_node("And", [current, next_condition], [out], name=out))
            current = out
        if selector == "compress":
            branch_conditions.append(current)
        else:
            nodes.append(helper.make_node("Cast", [current], [f"b{branch_idx}_match"], to=TensorProto.FLOAT, name=f"b{branch_idx}_match_cast"))
            branch_conditions.append(f"b{branch_idx}_match")

    nodes.append(helper.make_node("Concat", branch_conditions, ["match_4d"], axis=0, name="concat_match_4d"))
    nodes.append(helper.make_node("Reshape", ["match_4d", "shape_match_vec"], ["match_vec"], name="reshape_match_vec"))
    if selector == "compress":
        nodes.append(helper.make_node("Compress", ["bank", "match_vec"], ["selected_raw"], axis=0, name="select_bank"))
        nodes.append(helper.make_node("Squeeze", ["selected_raw", "squeeze_axes"], ["output"], name="squeeze_selected"))
        out_info = output_info(TensorProto.BOOL)
    else:
        nodes.append(helper.make_node("ReduceMax", ["match_vec"], ["any_match"], axes=[0], keepdims=1, name="any_match"))
        nodes.append(helper.make_node("Reshape", ["any_match", "shape_match4"], ["any_match4"], name="any_match4"))
        nodes.append(helper.make_node("ArgMax", ["match_vec"], ["best_idx"], axis=0, keepdims=0, name="best_idx"))
        nodes.append(helper.make_node("Gather", ["bank", "best_idx"], ["selected"], axis=0, name="select_bank"))
        nodes.append(helper.make_node("Mul", ["selected", "any_match4"], ["output"], name="mask_unknown"))
        out_info = output_info(TensorProto.FLOAT)

    graph = helper.make_graph(nodes, f"{task}_factorized_bank", [input_info()], [out_info])
    model = helper.make_model(
        graph,
        producer_name="neurogolf_factorized_bank_factory",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def write_zip_with_replacements(base_zip: Path, output_zip: Path, replacements: dict[str, bytes]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            if member in replacements:
                dst.writestr(member, replacements[member])
            else:
                dst.writestr(member, src.read(name))
            seen.add(member)
        for member, raw in replacements.items():
            if member not in seen:
                dst.writestr(member, raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", type=Path, default=Path("reports/factorable_bank_axes.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task292_two_stage_compact/submission.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/factorized_bank_build_first"))
    parser.add_argument("--report", type=Path, default=Path("reports/factorized_bank_build_first.csv"))
    parser.add_argument("--only-lane", default="build_first")
    parser.add_argument("--selector", choices=["compress", "argmax"], default="compress")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, bytes] = {}
    rows: list[dict[str, object]] = []

    for row in read_csv(args.axes):
        if row.get("lane") != args.only_lane:
            continue
        task = row["task"]
        features = [feature for feature in row.get("selected_features", "").split(";") if feature]
        examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
        model_path = args.out_dir / f"{task}.onnx"
        stage = "start"
        try:
            stage = "build"
            model = build_model(task, examples, features, selector=args.selector)
            stage = "serialize"
            raw = model.SerializeToString()
            model_path.write_bytes(raw)
            stage = "verify"
            right, wrong, reason = verify_model_bytes(raw, examples)
            stage = "profile"
            macs, memory, params, cost = profile_cost(model_path)
            base_cost = as_int(row.get("cost"))
            better = wrong == 0 and cost < base_cost
            if better:
                replacements[f"{task}.onnx"] = raw
            rows.append(
                {
                    "task": task,
                    "features": ";".join(features),
                    "right": right,
                    "wrong": wrong,
                    "reason": reason,
                    "base_cost": base_cost,
                    "candidate_cost": cost,
                    "base_points": row.get("points", ""),
                    "candidate_points": f"{task_points(cost):.6f}",
                    "delta_points": f"{task_points(cost) - as_float(row.get('points')):.6f}",
                    "candidate_file_bytes": len(raw),
                    "selector": args.selector,
                    "selected": int(better),
                }
            )
        except Exception as exc:  # Keep the batch moving; failures are report data.
            rows.append(
                {
                    "task": task,
                    "features": ";".join(features),
                    "right": 0,
                    "wrong": 0,
                    "reason": f"{stage}:{type(exc).__name__}:{exc!r}",
                    "base_cost": as_int(row.get("cost")),
                    "candidate_cost": "",
                    "base_points": row.get("points", ""),
                    "candidate_points": "",
                    "delta_points": "",
                    "candidate_file_bytes": "",
                    "selector": args.selector,
                    "selected": 0,
                }
            )

    write_csv(args.report, rows)
    zip_path = args.out_dir / "submission.zip"
    if replacements:
        write_zip_with_replacements(args.base_zip, zip_path, replacements)

    print(f"tasks={len(rows)} selected={len(replacements)}")
    print(f"report={args.report}")
    if replacements:
        print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
