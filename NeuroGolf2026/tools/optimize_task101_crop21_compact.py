#!/usr/bin/env python3
"""Shrink task101 square-kernel graph to a 21x21 working canvas."""

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


def replace_all_constant_tensors(model: onnx.ModelProto, output_name: str, arr: np.ndarray) -> int:
    count = 0
    for node in model.graph.node:
        if node.op_type != "Constant" or output_name not in node.output:
            continue
        tensor = numpy_helper.from_array(arr, name=output_name)
        for attr in node.attribute:
            if attr.name == "value":
                attr.t.CopyFrom(tensor)
                count += 1
                break
        else:
            node.attribute.append(helper.make_attribute("value", tensor))
            count += 1
    if count == 0:
        raise ValueError(f"could not find Constant node for {output_name}")
    return count


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def first_constant_tensor(model: onnx.ModelProto, output_name: str) -> np.ndarray:
    for node in model.graph.node:
        if node.op_type != "Constant" or output_name not in node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
    raise ValueError(f"could not find Constant node for {output_name}")


def rewrite_large_kernel_attrs(model: onnx.ModelProto) -> None:
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "kernel_shape" and list(attr.ints) == [30, 30]:
                del attr.ints[:]
                attr.ints.extend([21, 21])
            if attr.name == "pads" and list(attr.ints) == [29, 29, 29, 29]:
                del attr.ints[:]
                attr.ints.extend([20, 20, 20, 20])


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))
    old_constants = {
        name: first_constant_tensor(model, name)
        for name in ["scale_S3", "scale_S_T_3", "scale_S2", "scale_S_T_2"]
    }

    replace_all_constant_tensors(model, "arange30", np.arange(21, dtype=np.float32))
    replace_all_constant_tensors(model, "sh_30_1", np.array([21, 1], dtype=np.int64))
    replace_all_constant_tensors(model, "sh_1_30", np.array([1, 21], dtype=np.int64))
    replace_all_constant_tensors(model, "sh_30_30", np.array([21, 21], dtype=np.int64))
    replace_all_constant_tensors(model, "sh_1_1_30_30", np.array([1, 1, 21, 21], dtype=np.int64))
    replace_all_constant_tensors(model, "sh_30", np.array([21], dtype=np.int64))
    replace_all_constant_tensors(model, "sh_900", np.array([441], dtype=np.int64))
    replace_all_constant_tensors(model, "zero_4d", np.zeros((1, 1, 21, 21), dtype=np.float32))

    for name, arr in old_constants.items():
        replace_all_constant_tensors(model, name, arr[:21, :21].astype(np.float32, copy=False))
    rewrite_large_kernel_attrs(model)

    replace_initializer(model, "crop21_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop21_ends", np.array([1, 10, 21, 21], dtype=np.int64))
    replace_initializer(model, "crop21_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop21_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop21_starts", "crop21_ends", "crop21_axes", "crop21_steps"],
            ["input21"],
            name="crop_input21",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 0:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input21"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out21"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task101 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out21"],
            ["output"],
            name="pad21_to_static30_output",
            pads=[0, 0, 0, 0, 0, 0, 9, 9],
            value=0.0,
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task101_crop21_compact"
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch5/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task101_crop21_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task101.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task101.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task101.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task101.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
