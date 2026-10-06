#!/usr/bin/env python3
"""Build a color/column selector for task200."""

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


TASK = "task200"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_output(color: int, start_col: int) -> np.ndarray:
    out = np.zeros((10, 30, 30), dtype=np.bool_)
    out[0, :10, :10] = True
    for c in range(start_col, 10, 2):
        out[0, :10, c] = False
        out[color, :10, c] = True
    for c in range(start_col + 1, 10, 4):
        out[0, 0, c] = False
        out[5, 0, c] = True
    for c in range(start_col + 3, 10, 4):
        out[0, 9, c] = False
        out[5, 9, c] = True
    return out


def make_bank() -> np.ndarray:
    bank = np.zeros((9, 10, 10, 30, 30), dtype=np.bool_)
    for color in range(1, 10):
        for col in range(10):
            bank[color - 1, col] = make_output(color, col)
    return bank


def make_model() -> onnx.ModelProto:
    nodes = [
        constant_tensor("bank", make_bank()),
        constant_tensor("bottom_starts", np.asarray([0, 1, 9, 0], dtype=np.int64)),
        constant_tensor("bottom_ends", np.asarray([1, 10, 10, 10], dtype=np.int64)),
        constant_tensor("slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
        constant_tensor("slice_steps", np.asarray([1, 1, 1, 1], dtype=np.int64)),
        helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends", "slice_axes", "slice_steps"], ["bottom"], name="bottom_row"),
        helper.make_node("ReduceMax", ["bottom"], ["color_float"], name="color_reduce", axes=[0, 2, 3], keepdims=0),
        helper.make_node("ReduceMax", ["bottom"], ["column_float"], name="column_reduce", axes=[0, 1, 2], keepdims=0),
        helper.make_node("Cast", ["color_float"], ["color_condition"], name="color_to_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["column_float"], ["column_condition"], name="column_to_bool", to=TensorProto.BOOL),
        helper.make_node("Compress", ["bank", "color_condition"], ["color_bank"], name="select_color", axis=0),
        helper.make_node("Compress", ["color_bank", "column_condition"], ["column_bank"], name="select_column", axis=1),
        helper.make_node("Squeeze", ["column_bank"], ["output"], name="drop_column_axis", axes=[1]),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK}_color_column",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        value_info=[
            helper.make_tensor_value_info("bottom", TensorProto.FLOAT, [1, 9, 1, 10]),
            helper.make_tensor_value_info("color_float", TensorProto.FLOAT, [9]),
            helper.make_tensor_value_info("column_float", TensorProto.FLOAT, [10]),
            helper.make_tensor_value_info("color_bank", TensorProto.BOOL, [1, 10, 10, 30, 30]),
            helper.make_tensor_value_info("column_bank", TensorProto.BOOL, [1, 1, 10, 30, 30]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task200_color_column",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task171_frame_hw_nested/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task200_color_column"))
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
