#!/usr/bin/env python3
"""Shrink task165 by running its shape-fill rule on the real 20x20 canvas."""

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


def replace_constant_tensor(model: onnx.ModelProto, output_name: str, arr: np.ndarray) -> None:
    for node in model.graph.node:
        if node.op_type != "Constant" or output_name not in node.output:
            continue
        tensor = numpy_helper.from_array(arr, name=output_name)
        for attr in node.attribute:
            if attr.name == "value":
                attr.t.CopyFrom(tensor)
                return
        node.attribute.append(helper.make_attribute("value", tensor))
        return
    raise ValueError(f"could not find Constant node for {output_name}")


def add_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_constant_tensor(model, "slc_e", np.array([1, 1, 20, 20], dtype=np.int64))
    replace_constant_tensor(model, "shape_input_flat", np.array([1, 10, 400], dtype=np.int64))
    replace_constant_tensor(model, "shape_apex_flat", np.array([1, 400, 1], dtype=np.int64))
    replace_constant_tensor(model, "slc_e2", np.array([1, 1, 18, 20], dtype=np.int64))
    replace_constant_tensor(model, "logical_mask", np.ones((1, 1, 20, 20), dtype=np.float32))

    add_initializer(model, "crop20_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    add_initializer(model, "crop20_ends", np.array([1, 10, 20, 20], dtype=np.int64))
    add_initializer(model, "crop20_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    add_initializer(model, "crop20_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    model.graph.node.insert(
        15,
        helper.make_node(
            "Slice",
            ["input", "crop20_starts", "crop20_ends", "crop20_axes", "crop20_steps"],
            ["input20"],
            name="crop_input20",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 15:
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
        raise ValueError("could not find task165 output producer")

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

    model.producer_name = "neurogolf_task165_crop20_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task165_crop20_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task165.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task165.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task165.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task165.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
