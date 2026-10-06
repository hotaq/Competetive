#!/usr/bin/env python3
"""Shrink task071 reflection graph to its true 16x16 canvas."""

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


EXPECTED_SHAPE = [1, 10, 30, 30]


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def set_graph_io_static(model: onnx.ModelProto) -> None:
    for value_info in list(model.graph.input) + list(model.graph.output):
        dims = value_info.type.tensor_type.shape.dim
        if len(dims) != len(EXPECTED_SHAPE):
            continue
        for dim, value in zip(dims, EXPECTED_SHAPE):
            dim.dim_param = ""
            dim.dim_value = value


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    replace_initializer(model, "rev_ends", np.array([-17], dtype=np.int64))
    replace_initializer(model, "c29_i64", np.array([[[[15]]]], dtype=np.int64))
    rr, cc = np.indices((16, 16), dtype=np.int64)
    replace_initializer(model, "ipj", rr + cc)
    replace_initializer(model, "crop16_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop16_ends", np.array([1, 10, 16, 16], dtype=np.int64))
    replace_initializer(model, "crop16_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop16_steps", np.array([1, 1, 1, 1], dtype=np.int64))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop16_starts", "crop16_ends", "crop16_axes", "crop16_steps"],
            ["input16"],
            name="crop_input16",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 0:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input16"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out16"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task071 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out16"],
            ["output"],
            name="pad16_to_static30_output",
            pads=[0, 0, 0, 0, 0, 0, 14, 14],
            value=0.0,
        )
    )
    del model.graph.value_info[:]
    set_graph_io_static(model)

    model.producer_name = "neurogolf_task071_crop16_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task071_crop16_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task071.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task071.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task071.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task071.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
