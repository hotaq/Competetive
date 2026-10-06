#!/usr/bin/env python3
"""Build compact Resize-nearest models for exact integer repeat tasks."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import examples_fit_30  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def const_f32(name: str, values: list[float] | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def repeat_grid(grid: list[list[int]], row_scale: int, col_scale: int) -> list[list[int]]:
    out: list[list[int]] = []
    for row in grid:
        expanded: list[int] = []
        for value in row:
            expanded.extend([value] * col_scale)
        for _ in range(row_scale):
            out.append(expanded[:])
    return out


def infer_repeat(examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, int, int] | None:
    scales: tuple[int, int] | None = None
    max_h = 0
    max_w = 0
    for example in examples:
        inp = example["input"]
        out = example["output"]
        in_h, in_w = len(inp), len(inp[0])
        out_h, out_w = len(out), len(out[0])
        if out_h % in_h or out_w % in_w:
            return None
        row_scale = out_h // in_h
        col_scale = out_w // in_w
        if row_scale < 1 or col_scale < 1 or (row_scale, col_scale) == (1, 1):
            return None
        if repeat_grid(inp, row_scale, col_scale) != out:
            return None
        current = (row_scale, col_scale)
        if scales is None:
            scales = current
        elif scales != current:
            return None
        max_h = max(max_h, in_h)
        max_w = max(max_w, in_w)
    if scales is None:
        return None
    row_scale, col_scale = scales
    if max_h * row_scale > 30 or max_w * col_scale > 30:
        return None
    return row_scale, col_scale, max_h, max_w


def make_model(task: str, row_scale: int, col_scale: int, in_h: int, in_w: int) -> onnx.ModelProto:
    out_h = in_h * row_scale
    out_w = in_w * col_scale
    initializers = [
        const_i64("starts", [0, 0, 0, 0]),
        const_i64("ends", [1, 10, in_h, in_w]),
        const_i64("axes", [0, 1, 2, 3]),
        const_i64("steps", [1, 1, 1, 1]),
        const_f32("roi", []),
        const_f32("scales", [1.0, 1.0, float(row_scale), float(col_scale)]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w]),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["crop"], name=f"{task}_crop_max"),
        helper.make_node(
            "Resize",
            ["crop", "roi", "scales"],
            ["repeat"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
            name=f"{task}_nearest_repeat",
        ),
        helper.make_node("Pad", ["repeat", "pad_to_30", "zero"], ["output"], name=f"{task}_pad"),
    ]
    graph = helper.make_graph(
        nodes,
        f"{task}_repeat_resize",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_repeat_resize_tasks",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    data_root = Path("data/latest/extracted")
    out_dir = Path("generated/custom_repeat_resize_tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}

    for task_path in sorted(data_root.glob("task*.json")):
        task = task_path.stem
        examples = all_examples(json.loads(task_path.read_text()))
        if not examples_fit_30(examples):
            continue
        inferred = infer_repeat(examples)
        if inferred is None:
            continue
        row_scale, col_scale, in_h, in_w = inferred
        model = make_model(task, row_scale, col_scale, in_h, in_w)
        raw = model.SerializeToString()
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(
            f"{task} repeat=({row_scale},{col_scale}) max={in_h}x{in_w} "
            f"verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}"
        )
        if wrong:
            raise SystemExit(1)
        (out_dir / f"{task}.onnx").write_bytes(raw)
        raws[f"{task}.onnx"] = raw

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(raws.items()):
            zf.writestr(name, raw)
    print(f"wrote {out_dir / 'submission.zip'} with {len(raws)} tasks")


if __name__ == "__main__":
    main()
