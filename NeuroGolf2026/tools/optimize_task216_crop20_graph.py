#!/usr/bin/env python3
"""Shrink task216 by doing internal component selection on the 20x20 canvas.

The clean base model already encodes the right rule: find each non-zero
rectangle component, count color 2 inside each component, and emit the component
with the largest color-2 count.  Its expensive part runs that component search
over the padded 30x30 tensor even though task216 inputs are always 20x20.

This graph surgery keeps the final 30x30 static output path intact, but routes
the component-search prefix through a 20x20 crop and swaps its coordinate
constants from 30-sized arrays to 20-sized arrays.
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
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


INTERNAL_STOP = 221


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_initializer(model, "idx20", np.arange(20, dtype=np.int64))
    replace_initializer(model, "coord_h20", np.arange(20, dtype=np.int64).reshape(1, 1, 20, 1))
    replace_initializer(model, "coord_w20", np.arange(20, dtype=np.int64).reshape(1, 1, 1, 20))
    replace_initializer(model, "pad1d20", np.array([0, 0, 0, 19], dtype=np.int64))
    replace_initializer(model, "crop20_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop20_ends", np.array([1, 10, 20, 20], dtype=np.int64))
    replace_initializer(model, "crop20_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop20_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop20_starts", "crop20_ends", "crop20_axes", "crop20_steps"],
            ["input20"],
            name="crop_input20_for_component_search",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx in (1, 2):
            for input_idx, name in enumerate(node.input):
                if name == "input":
                    node.input[input_idx] = "input20"
        if idx < INTERNAL_STOP + 1:
            for input_idx, name in enumerate(node.input):
                if name == "v2":
                    node.input[input_idx] = "idx20"
                elif name == "v3":
                    node.input[input_idx] = "coord_h20"
                elif name == "v4":
                    node.input[input_idx] = "coord_w20"
                elif name == "v7":
                    node.input[input_idx] = "pad1d20"

    # The base graph carries concrete value_info from the 30x30 version.  After
    # changing internal tensors to 20x20, those stale annotations can make
    # ONNX Runtime try to reuse buffers with the wrong shape.
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task216_crop20_graph"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task216_crop20_experiment"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task216.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task216.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task216.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task216.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
