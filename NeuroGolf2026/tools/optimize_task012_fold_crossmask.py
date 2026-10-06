#!/usr/bin/env python3
"""Fold task012's nonzero-channel mask into the cross detector weights."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


TASK = "task012"


def constant_array(node: onnx.NodeProto) -> np.ndarray:
    value = next(attr for attr in node.attribute if attr.name == "value").t
    return numpy_helper.to_array(value)


def replace_constant_array(node: onnx.NodeProto, arr: np.ndarray) -> None:
    for attr in node.attribute:
        if attr.name == "value":
            attr.t.CopyFrom(numpy_helper.from_array(arr.astype(np.float32), name=attr.t.name))
            return
    raise ValueError(f"constant node has no value: {node.name}")


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)

    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == "k_cross":
            weights = constant_array(node).copy()
            weights[0, :, :, :] = 0.0
            replace_constant_array(node, weights)

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == "cross_count_masked":
                node.input[idx] = "cross_count"

    kept_nodes = []
    for node in model.graph.node:
        if node.output and node.output[0] == "cross_count_masked":
            continue
        kept_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task222_fold_noch0/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task012_fold_crossmask"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read(f"{TASK}.onnx")
    model = optimize_model(onnx.load_from_string(base_raw))
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
