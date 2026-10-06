#!/usr/bin/env python3
"""Build compact index-Gather flip models for task150/task155."""

from __future__ import annotations

import argparse
import csv
import json
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


EXPECTED_SHAPE = [1, 10, 30, 30]


@dataclass(frozen=True)
class Row:
    task: str
    transform: str
    sizes: str
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


def const_i64(name: str, values: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name),
    )


def const_f32(name: str, value: float) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name),
    )


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def onehot_30(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def verify(raw: bytes, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, str]:
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    right = wrong = 0
    first = ""
    for idx, ex in enumerate(examples):
        pred = sess.run(["output"], {"input": onehot_30(ex["input"])})[0]
        pred = (pred > 0).astype(np.float32)
        expected = onehot_30(ex["output"])
        if np.array_equal(pred, expected):
            right += 1
        else:
            wrong += 1
            if not first:
                first = f"first mismatch example {idx}"
    return right, wrong, first


def discover_sizes(examples: list[dict[str, list[list[int]]]]) -> list[int]:
    sizes: set[int] = set()
    for ex in examples:
        h = len(ex["input"])
        w = len(ex["input"][0])
        oh = len(ex["output"])
        ow = len(ex["output"][0])
        if h != w or h != oh or w != ow:
            raise ValueError(f"expected square same-shape examples, got {h}x{w}->{oh}x{ow}")
        sizes.add(h)
    return sorted(sizes)


def flip_index(size: int) -> list[int]:
    return list(reversed(range(size))) + list(range(size, 30))


def build_model(task: str, transform: str, sizes: list[int]) -> onnx.ModelProto:
    if transform == "flip_h":
        detect_axes = [1, 2]
        gather_axis = 3
    elif transform == "flip_v":
        detect_axes = [1, 3]
        gather_axis = 2
    else:
        raise ValueError(transform)

    nodes: list[onnx.NodeProto] = [
        const_i64("detect_axes", detect_axes),
        const_i64("all_axes", [0, 1, 2, 3]),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("half", 0.5),
        helper.make_node("ReduceSum", ["input", "detect_axes"], ["line_sum"], keepdims=1, name="line_sum"),
        helper.make_node("Clip", ["line_sum", "zero", "one"], ["line_has"], name="line_has"),
        helper.make_node("ReduceSum", ["line_has", "all_axes"], ["size"], keepdims=0, name="size"),
        const_i64("idx_identity", list(range(30))),
    ]

    previous = "idx_identity"
    for size in sizes:
        prefix = f"s{size}"
        nodes.extend(
            [
                const_f32(f"{prefix}_value", float(size)),
                const_i64(f"{prefix}_idx", flip_index(size)),
                helper.make_node("Sub", ["size", f"{prefix}_value"], [f"{prefix}_diff"], name=f"{prefix}_diff"),
                helper.make_node("Abs", [f"{prefix}_diff"], [f"{prefix}_abs"], name=f"{prefix}_abs"),
                helper.make_node("Less", [f"{prefix}_abs", "half"], [f"{prefix}_match"], name=f"{prefix}_match"),
                helper.make_node("Where", [f"{prefix}_match", f"{prefix}_idx", previous], [f"{prefix}_selected"], name=f"{prefix}_select_idx"),
            ]
        )
        previous = f"{prefix}_selected"

    nodes.append(helper.make_node("Gather", ["input", previous], ["output"], axis=gather_axis, name="flip_gather"))
    graph = helper.make_graph(nodes, f"{task}_{transform}_index_gather", [tensor_info("input")], [tensor_info("output")])
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task150_155_index_gather",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/task004_corner_shift_v86c/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task150_155_index_gather_v87"))
    args = parser.parse_args()

    specs = {"task150": "flip_h", "task155": "flip_v"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    rows: list[Row] = []

    with zipfile.ZipFile(args.base_zip) as base_zf:
        members = {Path(name).name: base_zf.read(name) for name in base_zf.namelist()}

    for task, transform in specs.items():
        examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
        sizes = discover_sizes(examples)
        model = build_model(task, transform, sizes)
        raw = model.SerializeToString()
        model_path = model_dir / f"{task}.onnx"
        model_path.write_bytes(raw)
        right, wrong, reason = verify(raw, examples)
        if wrong:
            rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), "", "", "", "", "", len(raw), reason))
            continue
        try:
            macs, memory, params, cost = profile_cost(model_path)
            points = task_points(cost)
        except Exception as exc:  # noqa: BLE001
            rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), "", "", "", "", "", len(raw), f"profile_error:{exc}"))
            continue
        members[f"{task}.onnx"] = raw
        rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), macs, memory, params, cost, points, len(raw), "ok"))
        print(f"{task}: ok sizes={sizes} cost={cost} points={points:.6f} bytes={len(raw)}")

    zip_path = args.out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(members.items()):
            zf.writestr(name, raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"wrote: {zip_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
