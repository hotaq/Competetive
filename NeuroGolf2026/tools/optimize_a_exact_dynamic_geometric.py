#!/usr/bin/env python3
"""Build tiny dynamic-shape ONNX models for A_exact geometric tasks.

These models encode the exact geometric rule directly with Slice/Transpose and
verify on the real ARC grid shapes instead of the local 30x30 padded harness.
They are kept in a separate "dynamic" lane because some local clean-audit tools
prefer static 1x10x30x30 IO even though several accepted public models use
dynamic spatial axes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples  # noqa: E402


CHANNELS = 10
MIN_I64 = np.iinfo(np.int64).min


@dataclass(frozen=True)
class Result:
    task: str
    transform: str
    right: int
    wrong: int
    examples: int
    macs: int | str
    memory: int | str
    params: int | str
    cost: int | str
    points: float | str
    file_bytes: int
    reason: str


def const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def dynamic_grid_info(name: str) -> onnx.ValueInfoProto:
    # onnx-tool 1.0.1 accepts empty TensorShapeProto dimensions as "unknown"
    # but rejects dim_param strings.  Build with None, then clear the generated
    # dim_param so ONNX Runtime still treats the axes as dynamic.
    vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, CHANNELS, None, None])
    vi.type.tensor_type.shape.dim[2].ClearField("dim_param")
    vi.type.tensor_type.shape.dim[3].ClearField("dim_param")
    return vi


def parse_transform(notes: str) -> str:
    match = re.search(r"exact_transform=([^| ]+)", notes)
    if not match:
        raise ValueError(f"missing exact_transform note: {notes}")
    return match.group(1)


def make_slice_reverse(name: str, x: str, y: str, axes_name: str) -> onnx.NodeProto:
    return helper.make_node(
        "Slice",
        [x, "rev_starts", "rev_ends", axes_name, "rev_steps"],
        [y],
        name=name,
    )


def build_model(task: str, transform: str) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        const_i64("rev_starts", [-1]),
        const_i64("rev_ends", [MIN_I64]),
        const_i64("rev_steps", [-1]),
        const_i64("axis_h", [2]),
        const_i64("axis_w", [3]),
    ]

    if transform == "flip_h":
        nodes.append(make_slice_reverse("flip_h", "input", "output", "axis_w"))
    elif transform == "flip_v":
        nodes.append(make_slice_reverse("flip_v", "input", "output", "axis_h"))
    elif transform == "rot180":
        nodes.append(make_slice_reverse("rot180_h", "input", "hflip", "axis_h"))
        nodes.append(make_slice_reverse("rot180_w", "hflip", "output", "axis_w"))
    elif transform == "transpose":
        nodes.append(helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2], name="transpose"))
    elif transform == "rot90":
        nodes.append(helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2], name="transpose"))
        nodes.append(make_slice_reverse("rot90_w", "t", "output", "axis_w"))
    elif transform == "rot270":
        nodes.append(helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2], name="transpose"))
        nodes.append(make_slice_reverse("rot270_h", "t", "output", "axis_h"))
    else:
        raise ValueError(f"unsupported transform: {transform}")

    used = {name for node in nodes for name in node.input if name}
    inits = [init for init in inits if init.name in used]
    graph = helper.make_graph(
        nodes,
        f"{task}_dynamic_{transform}",
        [dynamic_grid_info("input")],
        [dynamic_grid_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_dynamic_geometric",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    h = len(grid)
    w = len(grid[0])
    arr = np.zeros((1, CHANNELS, h, w), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def verify_actual_shape(raw: bytes, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, str]:
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    right = 0
    wrong = 0
    reason = ""
    for idx, example in enumerate(examples):
        x = grid_to_onehot(example["input"])
        y = grid_to_onehot(example["output"])
        pred = session.run(["output"], {"input": x})[0]
        pred = (pred > 0.0).astype(np.float32)
        if pred.shape == y.shape and np.array_equal(pred, y):
            right += 1
        else:
            wrong += 1
            if not reason:
                reason = f"first mismatch example {idx}: pred_shape={pred.shape} expected_shape={y.shape}"
    return right, wrong, reason


def load_geometric_specs(triage_csv: Path, tasks: set[str] | None) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    with triage_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "A_exact":
                continue
            if "geometric_exact" not in row["families"].split(";"):
                continue
            task = row["task"]
            if tasks is not None and task not in tasks:
                continue
            specs.append((task, parse_transform(row["notes"])))
    return sorted(specs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task058_size_tail/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--triage-csv", type=Path, default=Path("reports/task_triage.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_a_exact_dynamic_geometric"))
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    task_filter = set(args.tasks) if args.tasks else None
    specs = load_geometric_specs(args.triage_csv, task_filter)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "submission.zip"
    report_path = args.out_dir / "report.csv"

    with zipfile.ZipFile(args.base_zip) as base_zf:
        members = {name: base_zf.read(name) for name in base_zf.namelist()}

    results: list[Result] = []
    for task, transform in specs:
        model = build_model(task, transform)
        raw = model.SerializeToString()
        examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
        right, wrong, reason = verify_actual_shape(raw, examples)
        model_path = model_dir / f"{task}.onnx"
        model_path.write_bytes(raw)
        if wrong:
            results.append(Result(task, transform, right, wrong, len(examples), "", "", "", "", "", len(raw), reason))
            continue
        try:
            macs, memory, params, cost = profile_cost(model_path)
            points = task_points(cost)
        except Exception as exc:
            results.append(
                Result(task, transform, right, wrong, len(examples), "", "", "", "", "", len(raw), f"profile_error:{exc}")
            )
            continue
        members[f"{task}.onnx"] = raw
        results.append(Result(task, transform, right, wrong, len(examples), macs, memory, params, cost, points, len(raw), "ok"))
        print(f"{task}: {transform} ok {right}/{len(examples)} cost={cost} points={points:.6f} bytes={len(raw)}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(members.items()):
            zf.writestr(name, raw)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Result.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    print(f"wrote: {zip_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
