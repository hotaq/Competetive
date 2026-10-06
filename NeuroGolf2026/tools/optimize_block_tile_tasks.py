#!/usr/bin/env python3
"""Build compact Tile models for exact block-repeat ARC tasks."""

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


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def tile_grid(grid: list[list[int]], row_repeats: int, col_repeats: int) -> list[list[int]]:
    out: list[list[int]] = []
    for _ in range(row_repeats):
        for row in grid:
            tiled_row: list[int] = []
            for _ in range(col_repeats):
                tiled_row.extend(row)
            out.append(tiled_row)
    return out


def infer_tile(examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, int, int] | None:
    repeats: tuple[int, int] | None = None
    max_h = 0
    max_w = 0
    for example in examples:
        inp = example["input"]
        out = example["output"]
        in_h, in_w = len(inp), len(inp[0])
        out_h, out_w = len(out), len(out[0])
        if out_h % in_h or out_w % in_w:
            return None
        row_repeats = out_h // in_h
        col_repeats = out_w // in_w
        if row_repeats < 1 or col_repeats < 1 or (row_repeats, col_repeats) == (1, 1):
            return None
        if tile_grid(inp, row_repeats, col_repeats) != out:
            return None
        current = (row_repeats, col_repeats)
        if repeats is None:
            repeats = current
        elif repeats != current:
            return None
        max_h = max(max_h, in_h)
        max_w = max(max_w, in_w)
    if repeats is None:
        return None
    row_repeats, col_repeats = repeats
    if max_h * row_repeats > 30 or max_w * col_repeats > 30:
        return None
    return row_repeats, col_repeats, max_h, max_w


def make_model(task: str, row_repeats: int, col_repeats: int, in_h: int, in_w: int) -> onnx.ModelProto:
    out_h = in_h * row_repeats
    out_w = in_w * col_repeats
    initializers = [
        const_i64("starts", [0, 0, 0, 0]),
        const_i64("ends", [1, 10, in_h, in_w]),
        const_i64("axes", [0, 1, 2, 3]),
        const_i64("steps", [1, 1, 1, 1]),
        const_i64("repeats", [1, 1, row_repeats, col_repeats]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w]),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["crop"], name=f"{task}_crop_max"),
        helper.make_node("Tile", ["crop", "repeats"], ["tiled"], name=f"{task}_tile"),
        helper.make_node("Pad", ["tiled", "pad_to_30", "zero"], ["output"], name=f"{task}_pad"),
    ]
    graph = helper.make_graph(
        nodes,
        f"{task}_block_tile",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_block_tile_tasks",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    data_root = Path("data/latest/extracted")
    out_dir = Path("generated/custom_block_tile_tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}

    for task_path in sorted(data_root.glob("task*.json")):
        task = task_path.stem
        examples = all_examples(json.loads(task_path.read_text()))
        if not examples_fit_30(examples):
            continue
        inferred = infer_tile(examples)
        if inferred is None:
            continue
        row_repeats, col_repeats, in_h, in_w = inferred
        model = make_model(task, row_repeats, col_repeats, in_h, in_w)
        raw = model.SerializeToString()
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(
            f"{task} tile=({row_repeats},{col_repeats}) max={in_h}x{in_w} "
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
