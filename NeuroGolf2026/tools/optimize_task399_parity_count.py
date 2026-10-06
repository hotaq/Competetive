#!/usr/bin/env python3
"""Optimize task399 by counting one parity cell per 2x2 block.

The base model counts all color-2 cells in the 7x7 working area and selects one
of five stored 3x3 outputs.  Every object is a 2x2 block, so each block
contributes exactly one color-2 cell to any fixed parity class.  Counting the
even/even parity slice cuts the working tensor from 7x7 to 4x4.
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
from onnx import AttributeProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


TARGET_REPLACEMENTS = {
    "target4": np.asarray([1.0], dtype=np.float32),
    "target8": np.asarray([2.0], dtype=np.float32),
    "target12": np.asarray([3.0], dtype=np.float32),
    "target16": np.asarray([4.0], dtype=np.float32),
    "target20": np.asarray([5.0], dtype=np.float32),
}


def constant_value(node: onnx.NodeProto) -> np.ndarray | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
    return None


def replace_constant(model: onnx.ModelProto, output_name: str, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(value, name=f"{output_name}_value")
    for node in model.graph.node:
        if node.op_type != "Constant" or output_name not in node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                attr.t.CopyFrom(tensor)
                return
        node.attribute.append(helper.make_attribute("value", tensor))
        return
    raise ValueError(f"could not find Constant output {output_name}")


def verify_parity_invariant(examples: list[dict[str, list[list[int]]]]) -> None:
    for idx, example in enumerate(examples):
        grid = example["input"]
        total_blocks = sum(row.count(2) for row in grid) // 4
        parity_blocks = sum(
            1
            for r, row in enumerate(grid)
            for c, value in enumerate(row)
            if value == 2 and r % 2 == 0 and c % 2 == 0
        )
        if total_blocks != parity_blocks:
            raise ValueError(f"parity invariant failed at example {idx}: {total_blocks} != {parity_blocks}")


def build_optimized_model(base_raw: bytes, examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    verify_parity_invariant(examples)
    model = copy.deepcopy(onnx.load_from_string(base_raw))
    replace_constant(model, "steps", np.asarray([1, 2, 2], dtype=np.int64))
    for name, value in TARGET_REPLACEMENTS.items():
        replace_constant(model, name, value)
    model.producer_name = "neurogolf_task399_parity_count"
    onnx.checker.check_model(model)
    return model


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task399.onnx" else src.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task292_two_stage_compact/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task399_parity_count"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task399.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task399.onnx")

    model = build_optimized_model(base_raw, examples)
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task399.onnx"
    zip_path = args.out_dir / "submission.zip"
    model_path.write_bytes(raw)
    write_zip_with_replacement(args.base_zip, zip_path, raw)

    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)
    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
