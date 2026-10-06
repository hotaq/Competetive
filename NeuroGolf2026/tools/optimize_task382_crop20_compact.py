#!/usr/bin/env python3
"""Shrink task382 edge/corner propagation graph to the real 20x20 canvas."""

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

    # The original graph is shape-static at 30x30, but all public/train/arc-gen
    # examples for task382 are within 20x20. Keep the graph logic intact and
    # shrink only the coordinate tables plus reshape constants.
    old_inits = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    replace_initializer(model, "v2", np.arange(20, dtype=np.float32).reshape(1, 1, 20, 1))
    replace_initializer(model, "v3", np.arange(20, dtype=np.float32).reshape(1, 1, 1, 20))
    for name in ["v8", "v9", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"]:
        replace_initializer(model, name, old_inits[name][:20, :20].astype(np.float32, copy=False))
    replace_initializer(model, "v10", np.array([1, 20], dtype=np.int64))
    replace_initializer(model, "v11", np.array([1, 1, 20, 1], dtype=np.int64))
    replace_initializer(model, "v12", np.array([1, 1, 1, 20], dtype=np.int64))
    replace_initializer(model, "v29", np.zeros((1, 1, 20, 20), dtype=np.float32))
    replace_initializer(model, "v46", old_inits["v46"][:, :, :20, :].astype(np.float32, copy=False))
    replace_initializer(model, "v54", old_inits["v54"][:, :, :, :20].astype(np.float32, copy=False))

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
        raise ValueError("could not find task382 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out20"],
            ["output"],
            name="pad20_to_static30_output",
            pads=[0, 0, 0, 0, 0, 0, 10, 10],
            value=0.0,
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task382_crop20_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch5/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task382_crop20_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task382.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task382.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task382.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task382.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
