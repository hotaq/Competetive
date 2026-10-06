#!/usr/bin/env python3
"""Shrink the clean task089 graph from 30x30 internal work to 13x13.

This is a graph-preserving optimization of the current clean base.  Task089
examples are fixed 13x13 grids, while the base ONNX does its coordinate algebra
and transpose-convolutions on the full padded 30x30 canvas.  The transform crops
the input to 13x13, rewrites 30-sized coordinate constants to 13-sized ones,
retunes the transpose-conv padding, and pads the final 13x13 tensor back to the
required static 1x10x30x30 output shape.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


def rewrite_initializer(tensor: onnx.TensorProto) -> None:
    arr = numpy_helper.to_array(tensor)
    new = None
    if tensor.name == "arange30":
        new = np.arange(13, dtype=np.float32)
    elif arr.dtype == np.int64 and arr.size and np.any(arr == 30):
        new = np.where(arr == 30, 13, arr)
    elif arr.dtype == np.float32 and arr.size and np.any(arr == 14):
        new = np.where(arr == 14, 6, arr)
    elif arr.dtype == np.float32 and arr.size and np.any(arr == 28):
        new = np.where(arr == 28, 12, arr)
    elif arr.dtype == np.int64 and arr.size and np.any(arr == 28):
        new = np.where(arr == 28, 12, arr)

    if new is not None:
        tensor.CopyFrom(numpy_helper.from_array(new.astype(arr.dtype), name=tensor.name))


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    for tensor in model.graph.initializer:
        rewrite_initializer(tensor)

    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            continue
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                attr.ints[:] = [13, 13]
            elif attr.name == "pads":
                attr.ints[:] = [6, 6, 6, 6]

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == "input":
                node.input[idx] = "input13"

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name="crop13_starts"),
            numpy_helper.from_array(np.array([1, 10, 13, 13], dtype=np.int64), name="crop13_ends"),
            numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), name="crop13_axes"),
            numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="crop13_steps"),
        ]
    )
    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop13_starts", "crop13_ends", "crop13_axes", "crop13_steps"],
            ["input13"],
            name="crop_input13",
        ),
    )
    model.producer_name = "neurogolf_task089_crop13_graph"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task089_crop13_experiment"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task089.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task089.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task089.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task089.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
