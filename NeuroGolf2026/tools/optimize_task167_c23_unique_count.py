#!/usr/bin/env python3
"""Shrink task167's unique-color count by deriving color 4 from colors 2/3.

The task input is a 3x3 grid whose active colors are constrained to {2, 3, 4}.
The existing model slices all three color planes and checks whether each color
is present. We only need channels 2 and 3: channel 4 is present exactly when
the total count of channels 2 and 3 is less than nine.
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
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def const_node(name: str, output: str, value: np.ndarray | list[float] | list[int], dtype: np.dtype) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name=f"{output}__value")
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def extract_bank(model: onnx.ModelProto) -> onnx.TensorProto:
    for node in model.graph.node:
        if node.op_type == "Constant" and list(node.output) == ["bank"]:
            for attr in node.attribute:
                if attr.name == "value":
                    return attr.t
    raise ValueError("task167 bank constant not found")


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    bank = extract_bank(model)
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Constant", [], ["bank"], name="bank_value", value=bank),
        const_node("starts_value", "starts", [2, 0, 0], np.int64),
        const_node("ends_value", "ends", [4, 3, 3], np.int64),
        const_node("axes_value", "axes", [1, 2, 3], np.int64),
        const_node("steps_value", "steps", [1, 1, 1], np.int64),
        const_node("spatial_axes_value", "spatial_axes", [2, 3], np.int64),
        const_node("channel_axes_value", "channel_axes", [0, 1], np.int64),
        const_node("zero_value", "zero", [0.0], np.float32),
        const_node("nine_value", "nine", [9.0], np.float32),
        helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["c23_patch"], name="slice_c23"),
        helper.make_node("ReduceSum", ["c23_patch", "spatial_axes"], ["counts23"], name="counts23", keepdims=0),
        helper.make_node("Greater", ["counts23", "zero"], ["present23_b"], name="present23"),
        helper.make_node("Cast", ["present23_b"], ["present23_f"], name="present23_f", to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["counts23", "channel_axes"], ["total23"], name="total23", keepdims=0),
        helper.make_node("Less", ["total23", "nine"], ["present4_b"], name="present4"),
        helper.make_node("Cast", ["present4_b"], ["present4_f"], name="present4_f", to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["present23_f", "channel_axes"], ["n23"], name="n23", keepdims=0),
        helper.make_node("Add", ["n23", "present4_f"], ["ncolors"], name="ncolors"),
        const_node("n1_value", "n1", [1.0], np.float32),
        helper.make_node("Equal", ["ncolors", "n1"], ["cond0"], name="cond0"),
        const_node("n2_value", "n2", [2.0], np.float32),
        helper.make_node("Equal", ["ncolors", "n2"], ["cond1"], name="cond1"),
        const_node("n3_value", "n3", [3.0], np.float32),
        helper.make_node("Equal", ["ncolors", "n3"], ["cond2"], name="cond2"),
        helper.make_node("Concat", ["cond0", "cond1", "cond2"], ["condition"], name="condition", axis=0),
        helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_bank", axis=0),
    ]

    graph = helper.make_graph(
        nodes,
        "task167_c23_unique_count",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    )
    optimized = helper.make_model(
        graph,
        producer_name="neurogolf_task167_c23_unique_count",
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
            dst.writestr(member, raw if member == "task167.onnx" else src.read(name))


def write_single_probe(output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task167.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task167_c23_unique_count"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task167.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task167.onnx")

    model = optimize_model(onnx.load_from_string(base_raw))
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task167.onnx"
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
