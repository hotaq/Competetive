#!/usr/bin/env python3
"""Build a boundary-size selector for task058."""

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


TASK = "task058"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_bank(examples: list[dict[str, list[list[int]]]]) -> tuple[list[int], np.ndarray]:
    by_size: dict[int, np.ndarray] = {}
    for example in examples:
        size = len(example["input"])
        _, output = convert_to_numpy(example)
        existing = by_size.get(size)
        if existing is not None and not np.array_equal(existing, output):
            raise ValueError(f"conflicting output for size {size}")
        by_size[size] = output.astype(np.bool_)
    sizes = sorted(by_size)
    return sizes, np.concatenate([by_size[size] for size in sizes], axis=0)


def make_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    sizes, bank = make_bank(examples)
    boundary_indices = np.asarray([[0, 0, size - 1, 0] for size in sizes], dtype=np.int64)
    after_indices = np.asarray([[0, 0, size, 0] for size in sizes], dtype=np.int64)
    nodes = [
        constant_tensor("bank", bank),
        constant_tensor("boundary_indices", boundary_indices),
        constant_tensor("after_indices", after_indices),
        helper.make_node("GatherND", ["input", "boundary_indices"], ["boundary_float"], name="gather_boundary"),
        helper.make_node("GatherND", ["input", "after_indices"], ["after_float"], name="gather_after"),
        helper.make_node("Cast", ["boundary_float"], ["boundary_bool"], name="boundary_to_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["after_float"], ["after_bool"], name="after_to_bool", to=TensorProto.BOOL),
        helper.make_node("Not", ["after_bool"], ["after_not"], name="after_not"),
        helper.make_node("And", ["boundary_bool", "after_not"], ["condition"], name="condition"),
        helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_output", axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK}_size_bank",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task058_size_bank",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task171_frame_hw_nested/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task058_size_bank"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples = all_examples(json.loads((args.data_root / f"{TASK}.json").read_text()))
    model = make_model(examples)
    model_raw = model.SerializeToString()
    model_path = args.out_dir / f"{TASK}.onnx"
    model_path.write_bytes(model_raw)

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
