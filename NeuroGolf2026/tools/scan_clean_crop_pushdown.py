#!/usr/bin/env python3
"""Scan for clean spatial crop-pushdown wins.

This pass is intentionally not a profiler exploit: it keeps the official
1x10x30x30 graph IO static, slices the real small canvas before the existing
graph body, and pads the graph output back to 30x30. Candidates are kept only
when they verify exactly on train/test/arc-gen examples and lower the local
metric-v2 cost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]


def read_current(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: row for row in csv.DictReader(f)}


def as_int(value: str | None) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return 0


def fixed_io_shape(
    examples: list[dict[str, list[list[int]]]],
    max_side: int,
    *,
    same_shape_only: bool,
) -> tuple[int, int, int, int] | None:
    shape: tuple[int, int, int, int] | None = None
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        ih, iw = len(inp), len(inp[0])
        oh, ow = len(out), len(out[0])
        if same_shape_only and (ih, iw) != (oh, ow):
            return None
        if max(ih, iw, oh, ow) > max_side:
            return None
        current = (ih, iw, oh, ow)
        if shape is None:
            shape = current
        elif shape != current:
            return None
    return shape


def default_opset(model: onnx.ModelProto) -> int:
    versions = [op.version for op in model.opset_import if op.domain in ("", "ai.onnx")]
    return max(versions) if versions else 1


def append_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def make_pad_node(model: onnx.ModelProto, source: str, output: str, task: str, oh: int, ow: int) -> onnx.NodeProto:
    pads = np.array([0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow], dtype=np.int64)
    opset = default_opset(model)
    if opset >= 11:
        append_initializer(model, f"{task}_crop_pad", pads)
        append_initializer(model, f"{task}_crop_zero", np.array([0.0], dtype=np.float32))
        return helper.make_node(
            "Pad",
            [source, f"{task}_crop_pad", f"{task}_crop_zero"],
            [output],
            name=f"{task}_pad_crop_to_static30",
            mode="constant",
        )
    return helper.make_node(
        "Pad",
        [source],
        [output],
        name=f"{task}_pad_crop_to_static30",
        mode="constant",
        pads=pads.tolist(),
        value=0.0,
    )


def build_crop_pushdown(raw: bytes, task: str, ih: int, iw: int, oh: int, ow: int) -> bytes:
    model = copy.deepcopy(onnx.load_from_string(raw))
    if default_opset(model) < 10:
        raise ValueError(f"unsupported opset for input Slice: {default_opset(model)}")

    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    cropped_name = f"{input_name}_crop_{ih}x{iw}"
    core_output = f"{output_name}_core_{oh}x{ow}"

    append_initializer(model, f"{task}_crop_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    append_initializer(model, f"{task}_crop_ends", np.array([1, 10, ih, iw], dtype=np.int64))
    append_initializer(model, f"{task}_crop_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    append_initializer(model, f"{task}_crop_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    crop_node = helper.make_node(
        "Slice",
        [
            input_name,
            f"{task}_crop_starts",
            f"{task}_crop_ends",
            f"{task}_crop_axes",
            f"{task}_crop_steps",
        ],
        [cropped_name],
        name=f"{task}_crop_input_{ih}x{iw}",
    )

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == input_name:
                node.input[idx] = cropped_name
    model.graph.node.insert(0, crop_node)

    found_output = False
    for node in model.graph.node:
        for idx, name in enumerate(node.output):
            if name == output_name:
                node.output[idx] = core_output
                found_output = True
    if not found_output:
        raise ValueError("could not locate output producer")

    model.graph.node.append(make_pad_node(model, core_output, output_name, task, oh, ow))
    del model.graph.value_info[:]
    model.producer_name = f"neurogolf_{task}_clean_crop_pushdown"
    onnx.checker.check_model(model)
    return model.SerializeToString()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/clean_crop_pushdown_scan"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/clean_crop_pushdown_scan.csv"))
    parser.add_argument("--max-side", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--same-shape-only",
        action="store_true",
        help="Restrict the scan to tasks whose fixed input and output grids have the same HxW.",
    )
    args = parser.parse_args()

    current = read_current(args.current_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        tasks = [task if task.startswith("task") else f"task{int(task):03d}" for task in args.tasks]
    else:
        tasks = sorted(current)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.base_zip) as zf:
        names = set(zf.namelist())
        for task in tasks:
            member = f"{task}.onnx"
            base_cost = as_int(current.get(task, {}).get("cost") or current.get(task, {}).get("local_cost"))
            if not base_cost or member not in names:
                continue
            task_path = args.data_root / f"{task}.json"
            if not task_path.exists():
                continue
            examples = all_examples(json.loads(task_path.read_text()))
            shape = fixed_io_shape(examples, args.max_side, same_shape_only=args.same_shape_only)
            if shape is None:
                continue
            ih, iw, oh, ow = shape
            try:
                raw = build_crop_pushdown(zf.read(member), task, ih, iw, oh, ow)
                right, wrong, reason = verify_model_bytes(raw, examples)
                if wrong:
                    rows.append(
                        {
                            "task": task,
                            "status": "verify_failed",
                            "base_cost": base_cost,
                            "base_points": f"{task_points(base_cost):.6f}",
                            "input_h": ih,
                            "input_w": iw,
                            "output_h": oh,
                            "output_w": ow,
                            "reason": reason,
                            "right": right,
                            "wrong": wrong,
                        }
                    )
                    continue
                model_path = model_dir / member
                model_path.write_bytes(raw)
                macs, memory, params, cost = profile_cost(model_path)
                base_points = task_points(base_cost)
                points = task_points(cost)
                delta = points - base_points
                if delta >= args.min_delta:
                    rows.append(
                        {
                            "task": task,
                            "status": "ok",
                            "base_cost": base_cost,
                            "base_points": f"{base_points:.6f}",
                            "candidate_cost": cost,
                            "candidate_points": f"{points:.6f}",
                            "delta_points": f"{delta:.6f}",
                            "input_h": ih,
                            "input_w": iw,
                            "output_h": oh,
                            "output_w": ow,
                            "macs": macs,
                            "memory": memory,
                            "params": params,
                            "file_bytes": len(raw),
                            "model_path": str(model_path),
                            "right": right,
                            "wrong": wrong,
                            "reason": reason,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - scan should continue.
                rows.append(
                    {
                        "task": task,
                        "status": f"error:{type(exc).__name__}",
                        "base_cost": base_cost,
                        "base_points": f"{task_points(base_cost):.6f}",
                        "input_h": ih,
                        "input_w": iw,
                        "output_h": oh,
                        "output_w": ow,
                        "reason": str(exc)[:400],
                    }
                )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: (-float(str(row["delta_points"])), str(row["task"])))
    other_rows = [row for row in rows if row.get("status") != "ok"]
    write_csv(args.report_out, ok_rows + other_rows)

    print(f"tasks considered: {len(tasks)}")
    print(f"positive candidates: {len(ok_rows)}")
    print(f"report: {args.report_out}")
    for row in ok_rows[:30]:
        print(
            f"{row['task']}: {row['base_cost']} -> {row['candidate_cost']} "
            f"delta={row['delta_points']} "
            f"io={row['input_h']}x{row['input_w']}->{row['output_h']}x{row['output_w']} "
            f"path={row['model_path']}"
        )


if __name__ == "__main__":
    main()
