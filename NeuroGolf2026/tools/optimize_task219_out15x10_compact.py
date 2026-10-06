#!/usr/bin/env python3
"""Shrink task219's final assembly to 15x10, padding only once at the end."""

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

    replace_initializer(model, "v14", np.zeros(8, dtype=np.int64))
    replace_initializer(model, "v16", np.zeros((1, 1, 15, 10), dtype=np.float32))
    replace_initializer(model, "pad15x10_to30", np.array([0, 0, 0, 0, 0, 0, 15, 20], dtype=np.int64))
    replace_initializer(model, "pad_zero", np.array([0.0], dtype=np.float32))

    model.graph.node.insert(
        2,
        helper.make_node(
            "Slice",
            ["v18", "v2", "v3", "v4"],
            ["v18_crop"],
            name="crop_background_channel15x10",
        ),
    )

    for node in model.graph.node:
        for input_idx, name in enumerate(node.input):
            if name == "v18":
                node.input[input_idx] = "v18_crop"
            elif name == "v17":
                node.input[input_idx] = "v19"

    # Keep the original slices reading their gathered full channels.
    model.graph.node[2].input[0] = "v18"
    model.graph.node[3].input[0] = "v17"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out15x10"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task219 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out15x10", "pad15x10_to30", "pad_zero"],
            ["output"],
            name="pad15x10_to_static30_output",
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task219_out15x10_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task219_out15x10_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task219.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task219.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task219.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task219.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
