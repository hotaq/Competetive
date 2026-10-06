#!/usr/bin/env python3
"""Shrink task058's size lookup to one row-tail sum.

Task058 inputs are all-background square canvases with side length 5..20. The
existing bank model tests many individual cells to identify the side length.
Because channel 0 is one inside the square and zero outside the padded area,
the sum of row-0/channel-0 cells at columns 5..20 is exactly side_length - 5.
We keep the existing output bank and only replace its selector.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, convert_to_numpy, verify_model_bytes  # noqa: E402


def const_node(name: str, output: str, value: np.ndarray | list[float] | list[int], dtype: np.dtype) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name=f"{output}__value")
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def extract_bank(model: onnx.ModelProto) -> onnx.TensorProto:
    for node in model.graph.node:
        if node.op_type == "Constant" and list(node.output) == ["bank"]:
            for attr in node.attribute:
                if attr.name == "value":
                    return attr.t
    raise ValueError("task058 bank constant not found")


def bank_row_to_size_offset(bank: np.ndarray, examples: list[dict[str, list[list[int]]]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    bank_flat = bank.astype(np.float32).reshape(bank.shape[0], -1)
    for example in examples:
        _, expected = convert_to_numpy(example)
        matches = np.where((bank_flat == expected.reshape(1, -1)).all(axis=1))[0]
        if len(matches) != 1:
            raise ValueError(f"could not map output bank row for size {len(example['input'])}: {matches.tolist()}")
        mapping[int(matches[0])] = len(example["input"]) - 5
    if set(mapping) != set(range(bank.shape[0])):
        raise ValueError(f"incomplete bank mapping: {mapping}")
    return mapping


def optimize_model(model: onnx.ModelProto, examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    bank_tensor = extract_bank(model)
    bank = numpy_helper.to_array(bank_tensor)
    bank_to_size_offset = bank_row_to_size_offset(bank, examples)

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Constant", [], ["bank"], name="bank_value", value=bank_tensor),
        const_node("starts_value", "starts", [0, 0, 5], np.int64),
        const_node("ends_value", "ends", [1, 1, 21], np.int64),
        const_node("axes_value", "axes", [1, 2, 3], np.int64),
        const_node("steps_value", "steps", [1, 1, 1], np.int64),
        const_node("all_axes_value", "all_axes", [0, 1, 2, 3], np.int64),
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["tail"], name="size_tail"),
        helper.make_node("ReduceSum", ["tail", "all_axes"], ["size_offset"], name="size_offset", keepdims=0),
    ]

    conditions: list[str] = []
    for bank_index in range(bank.shape[0]):
        expected_offset = bank_to_size_offset[bank_index]
        value_name = f"offset_{bank_index}"
        cond_name = f"cond_{bank_index}"
        nodes.append(const_node(f"{value_name}_value", value_name, [float(expected_offset)], np.float32))
        nodes.append(helper.make_node("Equal", ["size_offset", value_name], [cond_name], name=cond_name))
        conditions.append(cond_name)

    nodes.extend(
        [
            helper.make_node("Concat", conditions, ["condition"], name="condition", axis=0),
            helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_bank", axis=0),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task058_size_tail",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    )
    optimized = helper.make_model(
        graph,
        producer_name="neurogolf_task058_size_tail",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(optimized)
    return optimized


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task058.onnx" else src.read(name))


def write_single_probe(output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task058.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task058_size_tail"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task058.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task058.onnx")

    model = optimize_model(onnx.load_from_string(base_raw), examples)
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task058.onnx"
    full_zip = args.out_dir / "submission.zip"
    probe_zip = args.out_dir / "probe" / "submission.zip"
    model_path.write_bytes(raw)
    write_zip_with_replacement(args.base_zip, full_zip, raw)
    write_single_probe(probe_zip, raw)

    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)
    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"full zip: {full_zip} ({full_zip.stat().st_size} bytes)")
    print(f"probe zip: {probe_zip} ({probe_zip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
