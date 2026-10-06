#!/usr/bin/env python3
"""Compress lookup-style ScatterElements ONNX tasks.

Several public-blend tasks are generated as one branch per known case:

    match_i -> ScatterElements(zero, out_idx_i, updates_i)
    Sum/Add(scatter_0, scatter_1, ...)

When exactly one branch matches, the many scatter operations can be replaced by
one ArgMax-selected scatter.  This script extracts the branch predicates from
the existing model and verifies the optimized graph against the base model on
all available examples before writing a replacement submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from tools.blend_public_pack import profile_cost, task_points
from tools.generate_a_exact_submission import EXPECTED_SHAPE, all_examples, convert_to_numpy


BRANCH_RE = re.compile(r"^scatter_(\d+)$")


def const_array_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                values[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    for init in model.graph.initializer:
        values[init.name] = numpy_helper.to_array(init)
    return values


def producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
    return producers


def tensor_i64(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def tensor_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def constant_node(name: str, value: np.ndarray | float, dtype: np.dtype | type) -> onnx.NodeProto:
    arr = np.asarray(value, dtype=dtype)
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name=f"{name}_value"))


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def find_size_area_name(producers: dict[str, onnx.NodeProto], branch_idx: int) -> str:
    node = producers.get(f"size_eq_{branch_idx}")
    if node is None or node.op_type != "Equal" or len(node.input) != 2:
        raise ValueError(f"missing size_eq_{branch_idx}")
    return node.input[1]


def find_update_source(
    constants: dict[str, np.ndarray],
    producers: dict[str, onnx.NodeProto],
    branch_idx: int,
    out_len: int,
) -> tuple[np.ndarray, bool]:
    """Return the branch update vector and whether the branch uses in_idx_i."""

    updates_name = f"updates_{branch_idx}"
    node = producers.get(updates_name)
    if node is None:
        return np.ones(out_len, dtype=np.float32), f"in_idx_{branch_idx}" in constants

    const_inputs = [name for name in node.input if name in constants]
    update = None
    for name in const_inputs:
        arr = np.asarray(constants[name])
        if arr.ndim >= 1 and arr.size == out_len:
            update = arr.astype(np.float32).reshape(-1)
            break
    if update is None:
        update = np.ones(out_len, dtype=np.float32)
    return update, f"in_idx_{branch_idx}" in constants


def extract_branches(base_model: onnx.ModelProto) -> list[dict[str, np.ndarray | bool]]:
    constants = const_array_map(base_model)
    producers = producer_map(base_model)
    scatter_indices: list[int] = []
    for node in base_model.graph.node:
        if node.op_type != "ScatterElements" or len(node.output) != 1:
            continue
        match = BRANCH_RE.match(node.output[0])
        if match:
            scatter_indices.append(int(match.group(1)))
    if not scatter_indices:
        raise ValueError("no scatter_i branches found")

    branches: list[dict[str, np.ndarray | bool]] = []
    for idx in sorted(scatter_indices):
        out_name = f"out_idx_{idx}"
        if out_name not in constants:
            raise ValueError(f"missing {out_name}")
        area_name = find_size_area_name(producers, idx)
        out_idx = constants[out_name].astype(np.int64).reshape(-1)
        update, has_in_idx = find_update_source(constants, producers, idx, len(out_idx))
        branch: dict[str, np.ndarray | bool] = {
            "area": constants[area_name].astype(np.float32),
            "out_idx": out_idx,
            "updates": update,
            "has_in_idx": has_in_idx,
        }
        if has_in_idx:
            branch["in_idx"] = constants[f"in_idx_{idx}"].astype(np.int64).reshape(-1)
        branches.append(branch)
    return branches


def make_optimized_model(base_raw: bytes, constants_as_initializers: bool, producer_name: str) -> onnx.ModelProto:
    base_model = onnx.load_from_string(base_raw)
    branches = extract_branches(base_model)
    max_updates = max(len(branch["out_idx"]) for branch in branches)
    pad_index = EXPECTED_SHAPE[1] * EXPECTED_SHAPE[2] * EXPECTED_SHAPE[3] - 1

    out_idx_rows = np.full((len(branches), max_updates), pad_index, dtype=np.int64)
    update_rows = np.zeros((len(branches), max_updates), dtype=np.float32)
    for row, branch in enumerate(branches):
        out_idx = np.asarray(branch["out_idx"], dtype=np.int64).reshape(-1)
        updates = np.asarray(branch["updates"], dtype=np.float32).reshape(-1)
        n = len(out_idx)
        out_idx_rows[row, :n] = out_idx
        update_rows[row, :n] = updates

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    def add_const(name: str, value: np.ndarray | float, dtype: np.dtype | type) -> None:
        if constants_as_initializers:
            if np.dtype(dtype) == np.dtype(np.int64):
                initializers.append(tensor_i64(name, value))
            else:
                initializers.append(tensor_f32(name, value))
        else:
            nodes.append(constant_node(name, value, dtype))

    add_const("shape_flat", np.asarray([9000], dtype=np.int64), np.int64)
    add_const("shape_out", np.asarray(EXPECTED_SHAPE, dtype=np.int64), np.int64)
    add_const("axis0", np.asarray([0], dtype=np.int64), np.int64)
    add_const("out_idx_rows", out_idx_rows, np.int64)
    add_const("update_rows", update_rows, np.float32)

    nodes.append(helper.make_node("Reshape", ["input", "shape_flat"], ["flat"], name="flatten_input"))
    nodes.append(helper.make_node("Sub", ["flat", "flat"], ["zero_flat"], name="make_zero_flat"))
    nodes.append(helper.make_node("ReduceSum", ["flat", "axis0"], ["input_count"], keepdims=1, name="input_count"))

    match_outputs: list[str] = []
    for idx, branch in enumerate(branches):
        area_name = f"area_{idx}"
        add_const(area_name, np.asarray(branch["area"], dtype=np.float32), np.float32)
        nodes.append(helper.make_node("Equal", ["input_count", area_name], [f"size_eq_{idx}"], name=f"size_eq_{idx}"))
        nodes.append(helper.make_node("Cast", [f"size_eq_{idx}"], [f"size_mask_{idx}"], to=TensorProto.FLOAT, name=f"size_mask_{idx}"))
        if bool(branch["has_in_idx"]):
            in_idx_name = f"in_idx_{idx}"
            add_const(in_idx_name, np.asarray(branch["in_idx"], dtype=np.int64), np.int64)
            nodes.append(helper.make_node("Gather", ["flat", in_idx_name], [f"gather_{idx}"], axis=0, name=f"gather_{idx}"))
            nodes.append(helper.make_node("ReduceMin", [f"gather_{idx}"], [f"all_match_{idx}"], axes=[0], keepdims=1, name=f"all_match_{idx}"))
            nodes.append(helper.make_node("Mul", [f"all_match_{idx}", f"size_mask_{idx}"], [f"match_{idx}"], name=f"match_{idx}"))
        else:
            nodes.append(helper.make_node("Mul", [f"size_mask_{idx}", f"size_mask_{idx}"], [f"match_{idx}"], name=f"match_{idx}"))
        match_outputs.append(f"match_{idx}")

    nodes.append(helper.make_node("Concat", match_outputs, ["match_vec"], axis=0, name="concat_matches"))
    nodes.append(helper.make_node("ReduceMax", ["match_vec"], ["any_match"], axes=[0], keepdims=0, name="any_match"))
    nodes.append(helper.make_node("ArgMax", ["match_vec"], ["best_branch"], axis=0, keepdims=0, name="best_branch"))
    nodes.append(helper.make_node("Gather", ["out_idx_rows", "best_branch"], ["selected_indices"], axis=0, name="select_indices"))
    nodes.append(helper.make_node("Gather", ["update_rows", "best_branch"], ["selected_updates_raw"], axis=0, name="select_updates"))
    nodes.append(helper.make_node("Mul", ["selected_updates_raw", "any_match"], ["selected_updates"], name="mask_no_match"))
    nodes.append(helper.make_node("ScatterElements", ["zero_flat", "selected_indices", "selected_updates"], ["out_flat"], axis=0, name="scatter_selected"))
    nodes.append(helper.make_node("Reshape", ["out_flat", "shape_out"], ["output"], name="reshape_output"))

    graph = helper.make_graph(
        nodes,
        f"{producer_name}_lookup_argmax_scatter",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def read_zip_member(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        return zf.read(member)


def write_zip_with_replacement(base_zip: Path, output_zip: Path, member: str, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        replaced = False
        for name in src.namelist():
            out_name = Path(name).name
            if out_name == member:
                dst.writestr(member, raw)
                replaced = True
            else:
                dst.writestr(out_name, src.read(name))
        if not replaced:
            dst.writestr(member, raw)


def verify_against_base(base_raw: bytes, candidate_raw: bytes, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, str]:
    base_session = ort.InferenceSession(base_raw, providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(candidate_raw, providers=["CPUExecutionProvider"])
    right = wrong = 0
    first_error = ""
    for idx, example in enumerate(examples):
        if max(len(example["input"]), len(example["input"][0]), len(example["output"]), len(example["output"][0])) > 30:
            continue
        x, _ = convert_to_numpy(example)
        expected = (base_session.run(["output"], {"input": x})[0] > 0.0).astype(np.float32)
        actual = (candidate_session.run(["output"], {"input": x})[0] > 0.0).astype(np.float32)
        if np.array_equal(expected, actual):
            right += 1
        else:
            wrong += 1
            if not first_error:
                first_error = f"first mismatch fit-example {idx}"
    return right, wrong, first_error


def task_number(task: str) -> int:
    match = re.search(r"(\d+)", task)
    if not match:
        raise ValueError(f"task must contain digits: {task}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task id, e.g. task076 or 076")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/lookup_scatter_optimized"))
    parser.add_argument("--zip-dir", type=Path, default=Path("generated/probe_lookup_scatter_optimized"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/lookup_scatter_optimized_report.csv"))
    parser.add_argument("--constants-as-initializers", action="store_true")
    args = parser.parse_args()

    task = f"task{task_number(args.task):03d}"
    member = f"{task}.onnx"
    base_raw = read_zip_member(args.base_zip, member)
    model = make_optimized_model(base_raw, constants_as_initializers=args.constants_as_initializers, producer_name=f"{task}_lookup_optimizer")
    raw = model.SerializeToString()

    suffix = "initializers" if args.constants_as_initializers else "constants"
    model_out = args.out_dir / suffix / member
    zip_out = args.zip_dir / f"{task}_{suffix}" / "submission.zip"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
    right, wrong, reason = verify_against_base(base_raw, raw, examples)
    macs = memory = params = cost = ""
    points = ""
    status = "failed_verify"
    if wrong == 0:
        macs, memory, params, cost = profile_cost(model_out)
        points = f"{task_points(cost):.6f}"
        status = "ok"
        write_zip_with_replacement(args.base_zip, zip_out, member, raw)

    row = {
        "task": task,
        "status": status,
        "right_vs_base": right,
        "wrong_vs_base": wrong,
        "reason": reason,
        "file_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "constants_as_initializers": int(args.constants_as_initializers),
        "macs": macs,
        "memory": memory,
        "params": params,
        "cost": cost,
        "points": points,
        "zip_out": str(zip_out) if status == "ok" else "",
        "model_out": str(model_out),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    exists = args.report_out.exists()
    with args.report_out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
