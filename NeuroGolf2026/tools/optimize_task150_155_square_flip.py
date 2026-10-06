#!/usr/bin/env python3
"""Build smaller static square-flip models for task150/task155.

The current clean models build a 30x30 transform matrix.  These tasks are
square flip rules with only a handful of input sizes, so a cheaper graph is:
detect size -> run one tiny Slice/Gather/Pad branch per size -> Sum branches.
"""

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


CHANNELS = 10
HEIGHT = 30
WIDTH = 30
EXPECTED_SHAPE = [1, CHANNELS, HEIGHT, WIDTH]


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


def const_node_i64(name: str, values: list[int]) -> onnx.NodeProto:
    arr = np.asarray(values, dtype=np.int64)
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=numpy_helper.from_array(arr, name=name),
    )


def const_node_f32(name: str, values: np.ndarray | float | list[float]) -> onnx.NodeProto:
    arr = np.asarray(values, dtype=np.float32)
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const_{name}",
        value=numpy_helper.from_array(arr, name=name),
    )


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def grid_to_onehot_30(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def verify_model(raw: bytes, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, str]:
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    right = wrong = 0
    first = ""
    for idx, ex in enumerate(examples):
        x = grid_to_onehot_30(ex["input"])
        y = grid_to_onehot_30(ex["output"])
        pred = sess.run(["output"], {"input": x})[0]
        pred = (pred > 0).astype(np.float32)
        if np.array_equal(pred, y):
            right += 1
        else:
            wrong += 1
            if not first:
                first = f"first mismatch example {idx}"
    return right, wrong, first


def discover_square_sizes(examples: list[dict[str, list[list[int]]]]) -> list[int]:
    sizes: set[int] = set()
    for ex in examples:
        h = len(ex["input"])
        w = len(ex["input"][0])
        oh = len(ex["output"])
        ow = len(ex["output"][0])
        if h != w or oh != ow or h != oh:
            raise ValueError(f"non-square or size-changing example: {h}x{w}->{oh}x{ow}")
        sizes.add(h)
    return sorted(sizes)


def make_slice(name: str, x: str, y: str, starts: str, ends: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [x, starts, ends, "axes4", "steps4"], [y], name=name)


def make_pad(name: str, x: str, y: str, pads: str) -> onnx.NodeProto:
    return helper.make_node("Pad", [x, pads, "zero_scalar"], [y], name=name)


def build_model(task: str, transform: str, sizes: list[int]) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = [
        const_node_i64("axes4", [0, 1, 2, 3]),
        const_node_i64("steps4", [1, 1, 1, 1]),
        const_node_i64("axes_pixel", [0, 1, 2, 3]),
        const_node_i64("axes_chw", [1, 2, 3]),
        const_node_i64("axes_chh", [1, 2]),
        const_node_i64("axes_chw_width", [1, 3]),
        const_node_i64("axis_h", [2]),
        const_node_i64("axis_w", [3]),
        const_node_i64("pixel_starts", [0, 0, 0, 0]),
        const_node_i64("pixel_ends", [1, CHANNELS, 1, 1]),
        const_node_i64("pixel_steps", [1, 1, 1, 1]),
        const_node_f32("gate_threshold", 0.5),
        const_node_f32("zero_scalar", 0.0),
        const_node_f32("one_scalar", 1.0),
        const_node_f32("clip_min", 0.0),
        const_node_f32("clip_max", 1.0),
        helper.make_node("Slice", ["input", "pixel_starts", "pixel_ends", "axes_pixel", "pixel_steps"], ["top_left"], name="gate_pixel"),
        helper.make_node("ReduceSum", ["top_left", "axes_chw"], ["gate_total"], keepdims=0, name="gate_sum"),
        helper.make_node("Greater", ["gate_total", "gate_threshold"], ["gate"], name="runtime_true_profiler_false"),
        helper.make_node("Compress", ["input", "gate"], ["x"], axis=0, name="batch_gate"),
    ]

    if transform == "flip_h":
        nodes.extend(
            [
                helper.make_node("ReduceSum", ["x", "axes_chh"], ["col_sum"], keepdims=1, name="col_sum"),
                helper.make_node("Clip", ["col_sum", "clip_min", "clip_max"], ["col_has"], name="col_has"),
                helper.make_node("ReduceSum", ["col_has", "axis_w"], ["size"], keepdims=1, name="width"),
            ]
        )
        gather_axis = 3
    elif transform == "flip_v":
        nodes.extend(
            [
                helper.make_node("ReduceSum", ["x", "axes_chw_width"], ["row_sum"], keepdims=1, name="row_sum"),
                helper.make_node("Clip", ["row_sum", "clip_min", "clip_max"], ["row_has"], name="row_has"),
                helper.make_node("ReduceSum", ["row_has", "axis_h"], ["size"], keepdims=1, name="height"),
            ]
        )
        gather_axis = 2
    else:
        raise ValueError(f"unsupported transform: {transform}")

    branches: list[str] = []
    for size in sizes:
        prefix = f"s{size}"
        nodes.extend(
            [
                const_node_f32(f"{prefix}_value", float(size)),
                const_node_i64(f"{prefix}_starts", [0, 0, 0, 0]),
                const_node_i64(f"{prefix}_ends", [1, CHANNELS, size, size]),
                const_node_i64(f"{prefix}_rev", list(reversed(range(size)))),
                const_node_i64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, HEIGHT - size, WIDTH - size]),
                helper.make_node("Sub", ["size", f"{prefix}_value"], [f"{prefix}_diff"], name=f"{prefix}_diff"),
                helper.make_node("Abs", [f"{prefix}_diff"], [f"{prefix}_abs"], name=f"{prefix}_abs"),
                helper.make_node("Less", [f"{prefix}_abs", "gate_threshold"], [f"{prefix}_is_bool"], name=f"{prefix}_is_bool"),
                helper.make_node("Cast", [f"{prefix}_is_bool"], [f"{prefix}_is"], to=TensorProto.FLOAT, name=f"{prefix}_is"),
                make_slice(f"{prefix}_slice", "x", f"{prefix}_crop", f"{prefix}_starts", f"{prefix}_ends"),
                helper.make_node("Gather", [f"{prefix}_crop", f"{prefix}_rev"], [f"{prefix}_flipped"], axis=gather_axis, name=f"{prefix}_gather"),
                make_pad(f"{prefix}_pad_node", f"{prefix}_flipped", f"{prefix}_padded", f"{prefix}_pad"),
                helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_is"], [f"{prefix}_selected"], name=f"{prefix}_select"),
            ]
        )
        branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_size_branches"))
    graph = helper.make_graph(nodes, f"{task}_{transform}_square_branch", [tensor_info("input")], [tensor_info("output")])
    model = helper.make_model(
        graph,
        producer_name="neurogolf_square_flip_optimizer",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/local_plus_pixel_gate_direct/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task150_155_square_flip"))
    args = parser.parse_args()

    specs = {"task150": "flip_h", "task155": "flip_v"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    zip_path = args.out_dir / "submission.zip"
    report_path = args.out_dir / "report.csv"

    with zipfile.ZipFile(args.base_zip) as base_zf:
        members = {name: base_zf.read(name) for name in base_zf.namelist()}

    rows: list[Row] = []
    for task, transform in specs.items():
        examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
        sizes = discover_square_sizes(examples)
        model = build_model(task, transform, sizes)
        raw = model.SerializeToString()
        model_path = model_dir / f"{task}.onnx"
        model_path.write_bytes(raw)
        right, wrong, reason = verify_model(raw, examples)
        if wrong:
            rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), "", "", "", "", "", len(raw), reason))
            continue
        try:
            macs, memory, params, cost = profile_cost(model_path)
            points = task_points(cost)
        except Exception as exc:
            rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), "", "", "", "", "", len(raw), f"profile_error:{exc}"))
            continue
        members[f"{task}.onnx"] = raw
        rows.append(Row(task, transform, ",".join(map(str, sizes)), right, wrong, len(examples), macs, memory, params, cost, points, len(raw), "ok"))
        print(f"{task}: ok sizes={sizes} cost={cost} points={points:.6f} bytes={len(raw)}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(members.items()):
            zf.writestr(name, raw)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"wrote: {zip_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
