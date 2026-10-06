#!/usr/bin/env python3
"""Build a compact exact symmetry classifier for task103.

Rule: the 3x3 binary pattern maps to color 1 iff it is symmetric on both
vertical and horizontal axes; otherwise it maps to color 7.
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


TASK = "task103"


def constant_tensor(name: str, arr: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=f"{name}_value",
        value=numpy_helper.from_array(arr, name=f"{name}_tensor"),
    )


def make_model() -> onnx.ModelProto:
    pairs = [
        ((0, 0), (0, 2)),
        ((1, 0), (1, 2)),
        ((2, 0), (2, 2)),
        ((0, 0), (2, 0)),
        ((0, 1), (2, 1)),
        ((0, 2), (2, 2)),
    ]
    lhs = np.asarray([[0, 2, r, c] for (r, c), _ in pairs], dtype=np.int64)
    rhs = np.asarray([[0, 2, r, c] for _, (r, c) in pairs], dtype=np.int64)
    bank = np.zeros((2, 10, 30, 30), dtype=np.bool_)
    bank[0, 7, 0, 0] = True
    bank[1, 1, 0, 0] = True

    nodes = [
        constant_tensor("lhs_indices", lhs),
        constant_tensor("rhs_indices", rhs),
        constant_tensor("bank", bank),
        constant_tensor("axes0", np.asarray([0], dtype=np.int64)),
        constant_tensor("one", np.asarray(1.0, dtype=np.float32)),
        helper.make_node("GatherND", ["input", "lhs_indices"], ["lhs"], name="gather_lhs"),
        helper.make_node("GatherND", ["input", "rhs_indices"], ["rhs"], name="gather_rhs"),
        helper.make_node("Equal", ["lhs", "rhs"], ["pair_equal"], name="pair_equal"),
        helper.make_node("Cast", ["pair_equal"], ["pair_equal_float"], name="pair_equal_float", to=TensorProto.FLOAT),
        helper.make_node("ReduceMin", ["pair_equal_float", "axes0"], ["sym_float"], name="all_pairs_equal", keepdims=0),
        helper.make_node("Equal", ["sym_float", "one"], ["sym"], name="sym_bool"),
        helper.make_node("Not", ["sym"], ["not_sym"], name="not_sym_bool"),
        helper.make_node("Unsqueeze", ["not_sym", "axes0"], ["not_sym_vec"], name="not_sym_vec"),
        helper.make_node("Unsqueeze", ["sym", "axes0"], ["sym_vec"], name="sym_vec"),
        helper.make_node("Concat", ["not_sym_vec", "sym_vec"], ["condition"], name="condition", axis=0),
        helper.make_node("Compress", ["bank", "condition"], ["output"], name="select_output", axis=0),
    ]

    graph = helper.make_graph(
        nodes,
        f"{TASK}_symmetry",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task103_symmetry",
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
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_gate_const_batch7_all_wide_positive/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task103_symmetry"))
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
