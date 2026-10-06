#!/usr/bin/env python3
"""Shrink task203 center-row remap graph to the real 18x18 canvas."""

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


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    for idx, tensor in enumerate(model.graph.initializer):
        if tensor.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))

    row_grid = np.arange(18, dtype=np.int64).reshape(18, 1).repeat(18, axis=1)
    col_grid = np.arange(18, dtype=np.int64).reshape(1, 18).repeat(18, axis=0)
    replace_constant_tensor(model, "row_grid", row_grid)
    replace_constant_tensor(model, "col_grid", col_grid)
    replace_initializer(model, "crop18_starts", np.array([0, 0, 0, 0], dtype=np.int64))
    replace_initializer(model, "crop18_ends", np.array([1, 10, 18, 18], dtype=np.int64))
    replace_initializer(model, "crop18_axes", np.array([0, 1, 2, 3], dtype=np.int64))
    replace_initializer(model, "crop18_steps", np.array([1, 1, 1, 1], dtype=np.int64))
    replace_initializer(model, "pad18_to30", np.array([0, 0, 0, 0, 0, 0, 12, 12], dtype=np.int64))
    replace_initializer(model, "pad_zero", np.array(0.0, dtype=np.float32))

    model.graph.node.insert(
        0,
        helper.make_node(
            "Slice",
            ["input", "crop18_starts", "crop18_ends", "crop18_axes", "crop18_steps"],
            ["input18"],
            name="crop_input18",
        ),
    )

    for idx, node in enumerate(model.graph.node):
        if idx == 0:
            continue
        for input_idx, name in enumerate(node.input):
            if name == "input":
                node.input[input_idx] = "input18"

    output_producer_found = False
    for node in model.graph.node:
        for output_idx, name in enumerate(node.output):
            if name == "output":
                node.output[output_idx] = "out18"
                output_producer_found = True
    if not output_producer_found:
        raise ValueError("could not find task203 output producer")

    model.graph.node.append(
        helper.make_node(
            "Pad",
            ["out18", "pad18_to30", "pad_zero"],
            ["output"],
            name="pad18_to_static30_output",
        )
    )
    del model.graph.value_info[:]

    model.producer_name = "neurogolf_task203_crop18_compact"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch5/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task203_crop18_compact"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task203.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task203.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task203.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task203.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
