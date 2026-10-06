#!/usr/bin/env python3
"""Shrink task324 lookup/scatter graph from 30x30 to its real 20x20 canvas."""

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
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def remap_flat_indices_30_to_20(values: np.ndarray) -> np.ndarray:
    old = values.astype(np.int64)
    channel = old // 900
    pos = old % 900
    row = pos // 30
    col = pos % 30
    if np.any(row >= 20) or np.any(col >= 20):
        bad = old[(row >= 20) | (col >= 20)]
        raise ValueError(f"indices outside 20x20 crop: {bad[:10]}")
    return (channel * 400 + row * 20 + col).astype(np.float32)


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))
    old_inits = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}

    old_hash = old_inits["v1"].reshape(900, 1)
    keep_positions = [r * 30 + c for r in range(20) for c in range(20)]
    replace_initializer(model, "v1", old_hash[keep_positions].astype(np.float32, copy=False))
    replace_initializer(model, "v3", remap_flat_indices_30_to_20(old_inits["v3"]))
    replace_initializer(model, "v4", np.zeros((1, 400), dtype=np.float32))
    replace_initializer(model, "v8", np.array([400], dtype=np.int64))
    replace_initializer(model, "v9", np.array([1, 1, 20, 20], dtype=np.int64))
    replace_initializer(model, "v10", np.array([1, 10, 20, 20], dtype=np.int64))

    replace_initializer(model, "crop20_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop20_ends", np.array([1, 10, 20, 20], dtype=np.int64))
    replace_initializer(model, "crop20_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop20_steps", np.array([1, 1, 1, 1], dtype=np.int64))
    replace_initializer(model, "pad20_to30", np.array([0, 0, 0, 0, 0, 0, 10, 10], dtype=np.int64))
    replace_initializer(model, "pad_zero", np.array(0.0, dtype=np.float32))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop20_starts", "crop20_ends", "crop20_axes", "crop20_steps"],
            ["input20"],
            name="crop_input20",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 0:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input20"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out20"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task324 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out20", "pad20_to30", "pad_zero"],
            ["output"],
            name="pad20_to_static30_output",
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task324_crop20_lookup"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch6/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task324_crop20_lookup"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task324.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task324.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task324.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task324.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
