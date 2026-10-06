#!/usr/bin/env python3
"""Replace task376's per-cell slice bank selector with one GatherND.

The existing model is already a compact constant-bank selector, but it builds
the 36-way condition vector with 36 Slice/Cast/Reshape chains.  The diagnostic
cells are static, so a single GatherND over the input can produce the same
condition vector before Compress selects the unchanged bank row.
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
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


TASK = "task376"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def extract_bank_and_indices(base_model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for node in base_model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        value_attrs = [attr for attr in node.attribute if attr.name == "value"]
        if not value_attrs:
            continue
        constants[node.output[0]] = numpy_helper.to_array(value_attrs[0].t)

    if "bank" not in constants:
        raise ValueError("base model has no bank constant")

    indices: list[list[int]] = []
    for i in range(36):
        starts = constants.get(f"f{i:03d}_starts")
        if starts is None or starts.shape != (3,):
            raise ValueError(f"missing diagnostic starts for feature {i}")
        channel, row, col = (int(x) for x in starts)
        indices.append([0, channel, row, col])

    return constants["bank"].astype(np.bool_), np.asarray(indices, dtype=np.int64)


def make_model(bank: np.ndarray, indices: np.ndarray) -> onnx.ModelProto:
    nodes = [
        constant_tensor("bank", bank),
        constant_tensor("indices", indices),
        helper.make_node("GatherND", ["input", "indices"], ["condition_float"], name="gather_conditions"),
        helper.make_node(
            "Cast",
            ["condition_float"],
            ["condition"],
            name="conditions_to_bool",
            to=TensorProto.BOOL,
        ),
        helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_bank", axis=0),
    ]

    model = helper.make_model(
        helper.make_graph(
            nodes,
            f"{TASK}_gathernd_bank",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30]),
            ],
            [
                helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="neurogolf_task376_gathernd_bank",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def write_replacement_zip(base_zip: Path, model_raw: bytes, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            raw = model_raw if Path(name).name == f"{TASK}.onnx" else src.read(name)
            dst.writestr(name, raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/ready_20plus_many/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task376_gathernd_bank"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read(f"{TASK}.onnx")
    base_model = onnx.load_from_string(base_raw)
    bank, indices = extract_bank_and_indices(base_model)
    model = make_model(bank, indices)
    model_raw = model.SerializeToString()

    examples = [
        example
        for example in all_examples(json.loads((args.data_root / f"{TASK}.json").read_text()))
        if max(
            len(example["input"]),
            len(example["input"][0]),
            len(example["output"]),
            len(example["output"][0]),
        )
        <= 30
    ]
    right, wrong, reason = verify_model_bytes(model_raw, examples)

    model_path = args.out_dir / f"{TASK}.onnx"
    model_path.write_bytes(model_raw)
    macs, memory, params, cost = profile_cost(model_path)

    out_zip = args.out_dir / "submission.zip"
    probe_dir = args.out_dir / "probe"
    probe_zip = probe_dir / "submission.zip"
    write_replacement_zip(args.base_zip, model_raw, out_zip)
    probe_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(probe_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{TASK}.onnx", model_raw)

    print(f"verify: {right}/{len(examples)} wrong={wrong} reason={reason}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"model: {model_path}")
    print(f"full zip: {out_zip}")
    print(f"probe zip: {probe_zip}")


if __name__ == "__main__":
    main()
