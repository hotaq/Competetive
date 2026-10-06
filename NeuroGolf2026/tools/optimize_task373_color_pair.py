#!/usr/bin/env python3
"""Build a compact color-pair selector for task373."""

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


TASK = "task373"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_bank() -> np.ndarray:
    bank = np.zeros((10, 10, 10, 30, 30), dtype=np.bool_)
    for top_color in range(1, 10):
        for bottom_color in range(1, 10):
            if top_color == bottom_color:
                continue
            for c in range(6):
                bank[top_color, bottom_color, top_color if c % 2 == 0 else bottom_color, 0, c] = True
                bank[top_color, bottom_color, bottom_color if c % 2 == 0 else top_color, 1, c] = True
    return bank


def make_model() -> onnx.ModelProto:
    top_indices = np.asarray([[0, c, 0, 0] for c in range(10)], dtype=np.int64)
    bottom_indices = np.asarray([[0, c, 1, 0] for c in range(10)], dtype=np.int64)
    nodes = [
        constant_tensor("bank", make_bank()),
        constant_tensor("top_indices", top_indices),
        constant_tensor("bottom_indices", bottom_indices),
        constant_tensor("squeeze_axes", np.asarray([1], dtype=np.int64)),
        helper.make_node("GatherND", ["input", "top_indices"], ["top_float"], name="gather_top_color"),
        helper.make_node("GatherND", ["input", "bottom_indices"], ["bottom_float"], name="gather_bottom_color"),
        helper.make_node("Cast", ["top_float"], ["top_condition"], name="top_to_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["bottom_float"], ["bottom_condition"], name="bottom_to_bool", to=TensorProto.BOOL),
        helper.make_node("Compress", ["bank", "top_condition"], ["top_bank"], name="select_top_color", axis=0),
        helper.make_node("Compress", ["top_bank", "bottom_condition"], ["pair_bank"], name="select_bottom_color", axis=1),
        helper.make_node("Squeeze", ["pair_bank", "squeeze_axes"], ["output"], name="drop_pair_axis"),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK}_color_pair",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        value_info=[
            helper.make_tensor_value_info("top_bank", TensorProto.BOOL, [1, 10, 10, 30, 30]),
            helper.make_tensor_value_info("pair_bank", TensorProto.BOOL, [1, 1, 10, 30, 30]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task373_color_pair",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task375_nested_compress_max15/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task373_color_pair"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    model_raw = model.SerializeToString()
    model_path = args.out_dir / f"{TASK}.onnx"
    model_path.write_bytes(model_raw)

    examples = all_examples(json.loads((args.data_root / f"{TASK}.json").read_text()))
    right, wrong, reason = verify_model_bytes(model_raw, examples)
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
    print(f"model bytes: {len(model_raw)}")
    print(f"model: {model_path}")
    print(f"full zip: {out_zip}")
    print(f"probe zip: {probe_zip}")


if __name__ == "__main__":
    main()
