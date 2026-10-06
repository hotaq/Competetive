#!/usr/bin/env python3
"""Shrink task392 by delaying the 30x30 padding until after channel assembly."""

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


def remove_initializers(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [tensor for tensor in model.graph.initializer if tensor.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_initializer(model, "v7", np.zeros((1, 1, 10, 10), dtype=np.float32))
    remove_initializers(model, {"v8", "v9"})

    del model.graph.node[:]
    base = onnx.load_from_string(base_raw)
    for idx, node in enumerate(base.graph.node):
        if idx in {19, 20, 22, 23}:
            continue
        new_node = copy.deepcopy(node)
        for input_idx, name in enumerate(new_node.input):
            if name == "v56":
                new_node.input[input_idx] = "v54"
            elif name == "v59":
                new_node.input[input_idx] = "v57"
        for output_idx, name in enumerate(new_node.output):
            if name == "output":
                new_node.output[output_idx] = "out10"
        model.graph.node.append(new_node)

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

    model.producer_name = "neurogolf_task392_out10_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task392_out10_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task392.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task392.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task392.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task392.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
