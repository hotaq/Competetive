#!/usr/bin/env python3
"""Build a tiny static Slice+Pad model for task135.

Task135 is the top-right 3x3 crop of the 9x9 input.
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


EXPECTED_SHAPE = [1, 10, 30, 30]


@dataclass(frozen=True)
class Row:
    task: str
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


def grid_to_onehot_30(grid: list[list[int]]) -> np.ndarray:
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


def build_model() -> onnx.ModelProto:
    nodes = [
        const_i64("starts", [0, 0, 0, 6]),
        const_i64("ends", [1, 10, 3, 9]),
        const_i64("axes", [0, 1, 2, 3]),
        const_i64("steps", [1, 1, 1, 1]),
        const_i64("pads", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("zero", 0.0),
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["crop"], name="crop_top_right"),
        helper.make_node("Pad", ["crop", "pads", "zero"], ["output"], name="pad_to_30"),
    ]
    graph = helper.make_graph(
        nodes,
        "task135_topright_crop",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_SHAPE)],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task135_topright_crop",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/task150_155_square_flip/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task135_topright_crop"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "task135.onnx"
    zip_path = args.out_dir / "submission.zip"
    report_path = args.out_dir / "report.csv"

    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)
    examples = all_examples(json.loads((args.data_root / "task135.json").read_text()))
    right, wrong, reason = verify(raw, examples)
    if wrong:
        row = Row("task135", right, wrong, len(examples), "", "", "", "", "", len(raw), reason)
    else:
        macs, memory, params, cost = profile_cost(model_path)
        row = Row("task135", right, wrong, len(examples), macs, memory, params, cost, task_points(cost), len(raw), "ok")
        print(f"task135: ok cost={cost} points={task_points(cost):.6f} bytes={len(raw)}")

    with zipfile.ZipFile(args.base_zip) as base_zf:
        members = {name: base_zf.read(name) for name in base_zf.namelist()}
    if row.reason == "ok":
        members["task135.onnx"] = raw
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, member_raw in sorted(members.items()):
            zf.writestr(name, member_raw)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(row.__dict__)

    print(f"wrote: {zip_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
