#!/usr/bin/env python3
"""Shrink task316 to 10x10 input processing and 3x3 output masks."""

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


def one_hot_mask(row: int, col: int) -> np.ndarray:
    arr = np.zeros((1, 1, 3, 3), dtype=np.float32)
    arr[:, :, row, col] = 1.0
    return arr


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_constant_tensor(model, "shape_scalar_flat", np.array([1, 1, 100], dtype=np.int64))
    replace_constant_tensor(model, "valid3", np.ones((1, 1, 3, 3), dtype=np.float32))
    replace_constant_tensor(model, "t316_mask_00", one_hot_mask(0, 0))
    replace_constant_tensor(model, "t316_mask_01", one_hot_mask(0, 1))
    replace_constant_tensor(model, "t316_mask_02", one_hot_mask(0, 2))
    replace_constant_tensor(model, "t316_mask_10", one_hot_mask(1, 0))
    replace_constant_tensor(model, "t316_mask_11", one_hot_mask(1, 1))
    replace_constant_tensor(model, "t316_mask_12", one_hot_mask(1, 2))
    replace_constant_tensor(model, "t316_mask_20", one_hot_mask(2, 0))
    replace_constant_tensor(model, "t316_mask_21", one_hot_mask(2, 1))
    replace_constant_tensor(model, "t316_mask_22", one_hot_mask(2, 2))

    add_initializer(model, "crop10_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    add_initializer(model, "crop10_ends", np.array([1, 10, 10, 10], dtype=np.int64))
    add_initializer(model, "crop10_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    add_initializer(model, "crop10_steps", np.array([1, 1, 1, 1], dtype=np.int64))
    add_initializer(model, "pad3_to30", np.array([0, 0, 0, 0, 0, 0, 27, 27], dtype=np.int64))
    add_initializer(model, "pad_zero", np.array([0.0], dtype=np.float32))

    model.graph.node.insert(
        33,
        helper.make_node(
            "Slice",
            ["input", "crop10_starts", "crop10_ends", "crop10_axes", "crop10_steps"],
            ["input10"],
            name="crop_input10",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 33:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input10"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out3"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task316 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out3", "pad3_to30", "pad_zero"],
            ["output"],
            name="pad3_to_static30_output",
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task316_crop10_out3_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task316_crop10_out3_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task316.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task316.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task316.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task316.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
