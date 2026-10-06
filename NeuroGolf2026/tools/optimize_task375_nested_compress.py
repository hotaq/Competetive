#!/usr/bin/env python3
"""Build a compact task375 selector with nested Compress.

The task is an odd-size solid square with a zero center.  The output keeps the
square's color except on both diagonals, which become color 0.  This model
selects the active color first, then selects the square size from a boundary
marker, and finally applies the same two selectors to a constant output bank.
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


TASK = "task375"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_output(size: int, color: int) -> np.ndarray:
    out = np.zeros((10, 30, 30), dtype=np.bool_)
    out[0, :size, :size] = True
    out[0, :size, :size] = False
    out[color, :size, :size] = True
    for i in range(size):
        out[color, i, i] = False
        out[0, i, i] = True
        j = size - 1 - i
        out[color, i, j] = False
        out[0, i, j] = True
    return out


def make_bank(max_size: int) -> tuple[list[int], np.ndarray]:
    sizes = list(range(3, max_size + 1, 2))
    bank = np.zeros((len(sizes), 10, 10, 30, 30), dtype=np.bool_)
    for si, size in enumerate(sizes):
        for color in range(1, 10):
            bank[si, color] = make_output(size, color)
    return sizes, bank


def make_model(max_size: int) -> onnx.ModelProto:
    sizes, bank = make_bank(max_size)
    color_indices = np.asarray([[0, c, 0, 1] for c in range(10)], dtype=np.int64)
    boundary_indices = np.asarray([[0, 0, size - 1, 0] for size in sizes], dtype=np.int64)
    after_indices = np.asarray([[0, 0, size, 0] for size in sizes], dtype=np.int64)

    nodes = [
        constant_tensor("bank", bank),
        constant_tensor("color_indices", color_indices),
        constant_tensor("boundary_indices", boundary_indices),
        constant_tensor("after_indices", after_indices),
        constant_tensor("squeeze_axes", np.asarray([1], dtype=np.int64)),
        helper.make_node("GatherND", ["input", "color_indices"], ["color_float"], name="gather_color"),
        helper.make_node("Cast", ["color_float"], ["color_condition"], name="color_to_bool", to=TensorProto.BOOL),
        helper.make_node("Compress", ["input", "color_condition"], ["selected_plane"], name="select_input_color", axis=1),
        helper.make_node("GatherND", ["selected_plane", "boundary_indices"], ["boundary_float"], name="gather_boundary"),
        helper.make_node("GatherND", ["selected_plane", "after_indices"], ["after_float"], name="gather_after"),
        helper.make_node("Cast", ["boundary_float"], ["boundary_bool"], name="boundary_to_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["after_float"], ["after_bool"], name="after_to_bool", to=TensorProto.BOOL),
        helper.make_node("Not", ["after_bool"], ["after_not"], name="after_not"),
        helper.make_node("And", ["boundary_bool", "after_not"], ["size_condition"], name="size_condition"),
        helper.make_node("Compress", ["bank", "color_condition"], ["color_bank"], name="select_bank_color", axis=1),
        helper.make_node("Compress", ["color_bank", "size_condition"], ["size_color_bank"], name="select_bank_size", axis=0),
        helper.make_node("Squeeze", ["size_color_bank", "squeeze_axes"], ["output"], name="drop_color_axis"),
    ]

    graph = helper.make_graph(
        nodes,
        f"{TASK}_nested_compress",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        value_info=[
            helper.make_tensor_value_info("selected_plane", TensorProto.FLOAT, [1, 1, 30, 30]),
            helper.make_tensor_value_info("color_bank", TensorProto.BOOL, [len(sizes), 1, 10, 30, 30]),
            helper.make_tensor_value_info("size_color_bank", TensorProto.BOOL, [1, 1, 10, 30, 30]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task375_nested_compress",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task376_nested_compress/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task375_nested_compress"))
    parser.add_argument("--max-size", type=int, default=29)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(args.max_size)
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
