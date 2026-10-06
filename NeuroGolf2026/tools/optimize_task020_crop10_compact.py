#!/usr/bin/env python3
"""Shrink task020's rotate-in-bounding-box rule to the real 10x10 canvas."""

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


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    bbox_mask = np.zeros((1, 1, 10, 10), dtype=np.float32)
    bbox_mask[:, :, :5, :5] = 1.0

    replace_initializer(model, "ar30", np.arange(10, dtype=np.int32))
    replace_initializer(model, "rev5", np.array([4, 3, 2, 1, 0, 5, 6, 7, 8, 9], dtype=np.int32))
    replace_initializer(model, "c29_i32", np.array(9, dtype=np.int32))
    replace_initializer(model, "shape_y_mask", np.array([1, 1, 10, 1], dtype=np.int64))
    replace_initializer(model, "shape_x_mask", np.array([1, 1, 1, 10], dtype=np.int64))
    replace_initializer(model, "bbox_mask", bbox_mask)
    replace_initializer(model, "bbox_mask_inv", 1.0 - bbox_mask)
    replace_initializer(model, "crop10_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop10_ends", np.array([1, 10, 10, 10], dtype=np.int64))
    replace_initializer(model, "crop10_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop10_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    compress_idx = None
    for idx, node in enumerate(model.graph.node):
        if "input_compress_batch_gate" in node.output:
            compress_idx = idx
            break
    if compress_idx is None:
        raise ValueError("could not find task020 Compress output")

    model.graph.node.insert(
        compress_idx + 1,
        helper.make_node(
            "Slice",
            ["input_compress_batch_gate", "crop10_starts", "crop10_ends", "crop10_axes", "crop10_steps"],
            ["input10"],
            name="crop_compressed_input10",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == compress_idx + 1:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input_compress_batch_gate":
                node.input[input_idx] = "input10"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out10"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task020 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out10"],
            ["output"],
            name="pad10_to_static30_output",
            pads=[0, 0, 0, 0, 0, 0, 20, 20],
            value=0.0,
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task020_crop10_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task020_crop10_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task020.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task020.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task020.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task020.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
