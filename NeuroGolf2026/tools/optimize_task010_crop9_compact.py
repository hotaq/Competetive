#!/usr/bin/env python3
"""Shrink task010's row-channel ranking graph to its real 9x9 canvas."""

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


def column_indicator(col: int, width: int = 9) -> np.ndarray:
    arr = np.zeros((1, 1, 1, width), dtype=np.float32)
    arr[:, :, :, col] = 1.0
    return arr


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_initializer(model, "_rch_ind1", column_indicator(1))
    replace_initializer(model, "_rch_ind3", column_indicator(3))
    replace_initializer(model, "_rch_ind5", column_indicator(5))
    replace_initializer(model, "_rch_ind7", column_indicator(7))
    replace_initializer(model, "crop9_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop9_ends", np.array([1, 10, 9, 9], dtype=np.int64))
    replace_initializer(model, "crop9_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop9_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    compress_idx = None
    for idx, node in enumerate(model.graph.node):
        if "input_compress_batch_gate" in node.output:
            compress_idx = idx
            break
    if compress_idx is None:
        raise ValueError("could not find task010 Compress output")

    model.graph.node.insert(
        compress_idx + 1,
        helper.make_node(
            "Slice",
            ["input_compress_batch_gate", "crop9_starts", "crop9_ends", "crop9_axes", "crop9_steps"],
            ["input9"],
            name="crop_compressed_input9",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == compress_idx + 1:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input_compress_batch_gate":
                node.input[input_idx] = "input9"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out9"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task010 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out9"],
            ["output"],
            name="pad9_to_static30_output",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=0.0,
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task010_crop9_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task010_crop9_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task010.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task010.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task010.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task010.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
