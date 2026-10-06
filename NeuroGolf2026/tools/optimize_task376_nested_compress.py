#!/usr/bin/env python3
"""Build a lower-condition task376 selector with nested Compress.

The current compact model probes 36 (height, color) cells directly.  This
variant first selects the active color plane, then probes the four possible
height markers inside that plane, and finally applies the same two-stage
selection to a constant output bank.
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


TASK = "task376"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def row_sequence(height: int) -> list[int]:
    period = list(range(height)) + list(range(height - 2, 0, -1))
    return (period + period + period)[: 4 * height - 3]


def make_bank() -> np.ndarray:
    bank = np.zeros((4, 10, 10, 30, 30), dtype=np.bool_)
    width = 17
    for hi, height in enumerate(range(3, 7)):
        out_height = 4 * height - 3
        for color in range(1, 10):
            bank[hi, color, 0, :out_height, :width] = True
            rows = []
            for r in range(height):
                row = np.zeros(width, dtype=np.bool_)
                if r == 0:
                    row[2::4] = True
                elif r == height - 1:
                    row[0::4] = True
                else:
                    row[1::2] = True
                rows.append(row)
            for out_r, src_r in enumerate(row_sequence(height)):
                active = rows[src_r]
                bank[hi, color, 0, out_r, :width][active] = False
                bank[hi, color, color, out_r, :width] = active
    return bank


def make_model() -> onnx.ModelProto:
    color_indices = np.asarray([[0, c, 1, 1] for c in range(10)], dtype=np.int64)
    height_indices = np.asarray([[0, 0, r, 0] for r in range(2, 6)], dtype=np.int64)

    nodes = [
        constant_tensor("bank", make_bank()),
        constant_tensor("color_indices", color_indices),
        constant_tensor("height_indices", height_indices),
        constant_tensor("squeeze_axes", np.asarray([1], dtype=np.int64)),
        helper.make_node("GatherND", ["input", "color_indices"], ["color_float"], name="gather_color"),
        helper.make_node("Cast", ["color_float"], ["color_condition"], name="color_to_bool", to=TensorProto.BOOL),
        helper.make_node("Compress", ["input", "color_condition"], ["selected_plane"], name="select_input_color", axis=1),
        helper.make_node("GatherND", ["selected_plane", "height_indices"], ["height_float"], name="gather_height"),
        helper.make_node("Cast", ["height_float"], ["height_condition"], name="height_to_bool", to=TensorProto.BOOL),
        helper.make_node("Compress", ["bank", "color_condition"], ["color_bank"], name="select_bank_color", axis=1),
        helper.make_node("Compress", ["color_bank", "height_condition"], ["height_color_bank"], name="select_bank_height", axis=0),
        helper.make_node("Squeeze", ["height_color_bank", "squeeze_axes"], ["output"], name="drop_color_axis"),
    ]

    graph = helper.make_graph(
        nodes,
        f"{TASK}_nested_compress",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        value_info=[
            helper.make_tensor_value_info("selected_plane", TensorProto.FLOAT, [1, 1, 30, 30]),
            helper.make_tensor_value_info("color_bank", TensorProto.BOOL, [4, 1, 10, 30, 30]),
            helper.make_tensor_value_info("height_color_bank", TensorProto.BOOL, [1, 1, 10, 30, 30]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task376_nested_compress",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_batch8_task103_symmetry/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task376_nested_compress"))
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
    print(f"model: {model_path}")
    print(f"full zip: {out_zip}")
    print(f"probe zip: {probe_zip}")


if __name__ == "__main__":
    main()
