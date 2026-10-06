#!/usr/bin/env python3
"""More aggressive task216 shrink: run the whole rule on 20x20, pad once.

Compared with ``optimize_task216_crop20_graph.py``, this version also shrinks
the final gather/mask canvas to 20x20 and appends a single Pad back to the
required 30x30 static output.  The task's input is always 20x20 and the selected
component is emitted in the top-left with zeros elsewhere, so the final pad is
behavior-preserving.
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


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_initializer(model, "v2", np.arange(20, dtype=np.int64))
    replace_initializer(model, "v3", np.arange(20, dtype=np.int64).reshape(1, 1, 20, 1))
    replace_initializer(model, "v4", np.arange(20, dtype=np.int64).reshape(1, 1, 1, 20))
    replace_initializer(model, "v7", np.array([0, 0, 0, 19], dtype=np.int64))
    replace_initializer(model, "v27", np.array([0, 0, 0, 0, 0, 0, 19, 19], dtype=np.int64))
    replace_initializer(model, "crop20_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop20_ends", np.array([1, 10, 20, 20], dtype=np.int64))
    replace_initializer(model, "crop20_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop20_steps", np.array([1, 1, 1, 1], dtype=np.int64))
    replace_initializer(model, "pad20_to30", np.array([0, 0, 0, 0, 0, 0, 10, 10], dtype=np.int64))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop20_starts", "crop20_ends", "crop20_axes", "crop20_steps"],
            ["input20"],
            name="crop_input20",
        ),
    )

    for node in model.graph.node:
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input20"

    # Keep the new Slice reading the real model input.
    model.graph.node[0].input[0] = "input"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out20"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task216 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out20", "pad20_to30", "v8"],
            ["output"],
            name="pad20_to_static30_output",
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task216_crop20_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task216_crop20_compact"))
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
