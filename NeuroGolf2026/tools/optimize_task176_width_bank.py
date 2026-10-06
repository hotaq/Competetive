#!/usr/bin/env python3
"""Build a width-selector bank for task176."""

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


TASK = "task176"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_bank(examples: list[dict[str, list[list[int]]]]) -> tuple[list[int], np.ndarray]:
    by_width: dict[int, np.ndarray] = {}
    for example in examples:
        width = len(example["input"][0])
        _, output = convert_to_numpy(example)
        existing = by_width.get(width)
        if existing is not None and not np.array_equal(existing, output):
            raise ValueError(f"conflicting output for width {width}")
        by_width[width] = output.astype(np.bool_)
    widths = sorted(by_width)
    return widths, np.concatenate([by_width[width] for width in widths], axis=0)


def make_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    widths, bank = make_bank(examples)
    boundary_0 = np.asarray([[0, 0, 0, width - 1] for width in widths], dtype=np.int64)
    boundary_2 = np.asarray([[0, 2, 0, width - 1] for width in widths], dtype=np.int64)
    after_0 = np.asarray([[0, 0, 0, width] for width in widths], dtype=np.int64)
    after_2 = np.asarray([[0, 2, 0, width] for width in widths], dtype=np.int64)
    nodes = [
        constant_tensor("bank", bank),
        constant_tensor("boundary_0", boundary_0),
        constant_tensor("boundary_2", boundary_2),
        constant_tensor("after_0", after_0),
        constant_tensor("after_2", after_2),
        helper.make_node("GatherND", ["input", "boundary_0"], ["boundary_0_float"], name="gather_boundary_0"),
        helper.make_node("GatherND", ["input", "boundary_2"], ["boundary_2_float"], name="gather_boundary_2"),
        helper.make_node("GatherND", ["input", "after_0"], ["after_0_float"], name="gather_after_0"),
        helper.make_node("GatherND", ["input", "after_2"], ["after_2_float"], name="gather_after_2"),
        helper.make_node("Cast", ["boundary_0_float"], ["boundary_0_bool"], name="boundary_0_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["boundary_2_float"], ["boundary_2_bool"], name="boundary_2_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["after_0_float"], ["after_0_bool"], name="after_0_bool", to=TensorProto.BOOL),
        helper.make_node("Cast", ["after_2_float"], ["after_2_bool"], name="after_2_bool", to=TensorProto.BOOL),
        helper.make_node("Or", ["boundary_0_bool", "boundary_2_bool"], ["boundary_bool"], name="boundary_bool"),
        helper.make_node("Or", ["after_0_bool", "after_2_bool"], ["after_bool"], name="after_bool"),
        helper.make_node("Not", ["after_bool"], ["after_not"], name="after_not"),
        helper.make_node("And", ["boundary_bool", "after_not"], ["condition"], name="condition"),
        helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_output", axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        f"{TASK}_width_bank",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task176_width_bank",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_task058_size_bank/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task176_width_bank"))
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
