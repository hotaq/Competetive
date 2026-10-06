#!/usr/bin/env python3
"""Build a compact height/width frame selector for task171."""

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


TASK = "task171"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_bank(min_size: int, max_size: int, flat: bool) -> tuple[list[int], np.ndarray]:
    sizes = list(range(min_size, max_size + 1))
    bank = np.zeros((len(sizes), len(sizes), 10, 30, 30), dtype=np.bool_)
    for hi, h in enumerate(sizes):
        for wi, w in enumerate(sizes):
            bank[hi, wi, 0, :h, :w] = True
            bank[hi, wi, 0, 0, :w] = False
            bank[hi, wi, 0, h - 1, :w] = False
            bank[hi, wi, 0, :h, 0] = False
            bank[hi, wi, 0, :h, w - 1] = False
            bank[hi, wi, 8, 0, :w] = True
            bank[hi, wi, 8, h - 1, :w] = True
            bank[hi, wi, 8, :h, 0] = True
            bank[hi, wi, 8, :h, w - 1] = True
    if flat:
        bank = bank.reshape(len(sizes) * len(sizes), 10, 30, 30)
    return sizes, bank


def boundary_condition_nodes(axis_name: str, indices: list[list[int]], after_indices: list[list[int]]) -> list[onnx.NodeProto]:
    return [
        constant_tensor(f"{axis_name}_indices", np.asarray(indices, dtype=np.int64)),
        constant_tensor(f"{axis_name}_after_indices", np.asarray(after_indices, dtype=np.int64)),
        helper.make_node("GatherND", ["input", f"{axis_name}_indices"], [f"{axis_name}_boundary_float"], name=f"gather_{axis_name}_boundary"),
        helper.make_node("GatherND", ["input", f"{axis_name}_after_indices"], [f"{axis_name}_after_float"], name=f"gather_{axis_name}_after"),
        helper.make_node("Cast", [f"{axis_name}_boundary_float"], [f"{axis_name}_boundary_bool"], name=f"{axis_name}_boundary_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", [f"{axis_name}_after_float"], [f"{axis_name}_after_bool"], name=f"{axis_name}_after_bool", to=TensorProto.BOOL),
        helper.make_node("Not", [f"{axis_name}_after_bool"], [f"{axis_name}_after_not"], name=f"{axis_name}_after_not"),
        helper.make_node("And", [f"{axis_name}_boundary_bool", f"{axis_name}_after_not"], [f"{axis_name}_condition"], name=f"{axis_name}_condition"),
    ]


def make_model(min_size: int, max_size: int, selector: str) -> onnx.ModelProto:
    flat = selector == "flat"
    sizes, bank = make_bank(min_size, max_size, flat=flat)
    height_indices = [[0, 0, h - 1, 0] for h in sizes]
    height_after = [[0, 0, h, 0] for h in sizes]
    width_indices = [[0, 0, 0, w - 1] for w in sizes]
    width_after = [[0, 0, 0, w] for w in sizes]

    nodes = [
        constant_tensor("bank", bank),
        constant_tensor("squeeze_axes", np.asarray([1], dtype=np.int64)),
        *boundary_condition_nodes("height", height_indices, height_after),
        *boundary_condition_nodes("width", width_indices, width_after),
    ]
    value_info = []
    if selector == "nested":
        nodes.extend(
            [
                helper.make_node("Compress", ["bank", "height_condition"], ["height_bank"], name="select_height", axis=0),
                helper.make_node("Compress", ["height_bank", "width_condition"], ["hw_bank"], name="select_width", axis=1),
                helper.make_node("Squeeze", ["hw_bank", "squeeze_axes"], ["output"], name="drop_width_axis"),
            ]
        )
        value_info.extend(
            [
                helper.make_tensor_value_info("height_bank", TensorProto.BOOL, [1, len(sizes), 10, 30, 30]),
                helper.make_tensor_value_info("hw_bank", TensorProto.BOOL, [1, 1, 10, 30, 30]),
            ]
        )
    elif selector == "flat":
        nodes.extend(
            [
                constant_tensor("unsqueeze_height_axis", np.asarray([1], dtype=np.int64)),
                constant_tensor("unsqueeze_width_axis", np.asarray([0], dtype=np.int64)),
                constant_tensor("flat_shape", np.asarray([len(sizes) * len(sizes)], dtype=np.int64)),
                helper.make_node("Unsqueeze", ["height_condition", "unsqueeze_height_axis"], ["height_condition_2d"], name="height_condition_2d"),
                helper.make_node("Unsqueeze", ["width_condition", "unsqueeze_width_axis"], ["width_condition_2d"], name="width_condition_2d"),
                helper.make_node("And", ["height_condition_2d", "width_condition_2d"], ["condition_2d"], name="condition_2d"),
                helper.make_node("Reshape", ["condition_2d", "flat_shape"], ["flat_condition"], name="flat_condition"),
                helper.make_node("Compress", ["bank", "flat_condition"], ["output"], name="select_hw", axis=0),
            ]
        )
        value_info.extend(
            [
                helper.make_tensor_value_info("height_condition_2d", TensorProto.BOOL, [len(sizes), 1]),
                helper.make_tensor_value_info("width_condition_2d", TensorProto.BOOL, [1, len(sizes)]),
                helper.make_tensor_value_info("condition_2d", TensorProto.BOOL, [len(sizes), len(sizes)]),
                helper.make_tensor_value_info("flat_condition", TensorProto.BOOL, [len(sizes) * len(sizes)]),
            ]
        )
    else:
        raise ValueError(f"unsupported selector: {selector}")

    graph = helper.make_graph(
        nodes,
        f"{TASK}_frame_hw",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task171_frame_hw",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task373_color_pair/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task171_frame_hw"))
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=9)
    parser.add_argument("--selector", choices=["nested", "flat"], default="flat")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(args.min_size, args.max_size, args.selector)
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
