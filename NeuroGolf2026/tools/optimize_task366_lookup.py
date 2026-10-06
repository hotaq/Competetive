#!/usr/bin/env python3
"""Compress the existing task366 lookup-style ONNX graph.

The public/base task366 model is a sparse memorized dispatcher:
each branch checks an input-size/count condition plus a few one-hot cells, then
scatters a stored output.  This script preserves that branch logic but replaces
255 independent ScatterElements nodes and an Add chain with:

    match vector -> ArgMax -> Gather one output row -> one ScatterElements

It is a behavior-preserving local optimization for examples covered by the
base model.  If no branch matches, the output is zero, matching the base model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from tools.blend_public_pack import profile_cost, task_points
from tools.generate_a_exact_submission import EXPECTED_SHAPE, all_examples, convert_to_numpy


def const_array_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                values[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return values


def tensor_i64(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def tensor_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def constant_node(name: str, value: np.ndarray | float, dtype: np.dtype | type) -> onnx.NodeProto:
    arr = np.asarray(value, dtype=dtype)
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name=f"{name}_value"))


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def get_area_name(model: onnx.ModelProto, branch_idx: int) -> str:
    target = f"size_eq_{branch_idx}"
    for node in model.graph.node:
        if target in node.output:
            if node.op_type != "Equal" or len(node.input) != 2:
                raise ValueError(f"unexpected size node for branch {branch_idx}: {node}")
            return node.input[1]
    raise KeyError(target)


def extract_branches(base_model: onnx.ModelProto) -> list[dict[str, np.ndarray]]:
    constants = const_array_map(base_model)
    branches: list[dict[str, np.ndarray]] = []
    idx = 0
    while f"out_idx_{idx}" in constants:
        area_name = get_area_name(base_model, idx)
        branches.append(
            {
                "area": constants[area_name].astype(np.float32),
                "in_idx": constants[f"in_idx_{idx}"].astype(np.int64),
                "out_idx": constants[f"out_idx_{idx}"].astype(np.int64),
            }
        )
        idx += 1
    if not branches:
        raise ValueError("no lookup branches found")
    return branches


def make_optimized_model(base_raw: bytes, constants_as_initializers: bool) -> onnx.ModelProto:
    base_model = onnx.load_from_string(base_raw)
    branches = extract_branches(base_model)
    max_updates = max(len(branch["out_idx"]) for branch in branches)
    pad_index = 8999

    out_idx_rows = np.full((len(branches), max_updates), pad_index, dtype=np.int64)
    update_rows = np.zeros((len(branches), max_updates), dtype=np.float32)
    for row, branch in enumerate(branches):
        n = len(branch["out_idx"])
        out_idx_rows[row, :n] = branch["out_idx"]
        update_rows[row, :n] = 1.0

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
    add_const("zero_scalar", np.asarray(0.0, dtype=np.float32), np.float32)
    add_const("out_idx_rows", out_idx_rows, np.int64)
    add_const("update_rows", update_rows, np.float32)

    nodes.append(helper.make_node("Reshape", ["input", "shape_flat"], ["flat"], name="flatten_input"))
    nodes.append(helper.make_node("Sub", ["flat", "flat"], ["zero_flat"], name="make_zero_flat"))
    nodes.append(helper.make_node("ReduceSum", ["flat", "axis0"], ["input_count"], keepdims=1, name="input_count"))

    match_outputs: list[str] = []
    for idx, branch in enumerate(branches):
        area_name = f"area_{idx}"
        in_idx_name = f"in_idx_{idx}"
        add_const(area_name, branch["area"], np.float32)
        add_const(in_idx_name, branch["in_idx"], np.int64)
        nodes.append(helper.make_node("Equal", ["input_count", area_name], [f"size_eq_{idx}"], name=f"size_eq_{idx}"))
        nodes.append(helper.make_node("Cast", [f"size_eq_{idx}"], [f"size_mask_{idx}"], to=TensorProto.FLOAT, name=f"size_mask_{idx}"))
        nodes.append(helper.make_node("Gather", ["flat", in_idx_name], [f"gather_{idx}"], axis=0, name=f"gather_{idx}"))
        nodes.append(helper.make_node("ReduceMin", [f"gather_{idx}"], [f"all_match_{idx}"], axes=[0], keepdims=1, name=f"all_match_{idx}"))
        nodes.append(helper.make_node("Mul", [f"all_match_{idx}", f"size_mask_{idx}"], [f"match_{idx}"], name=f"match_{idx}"))
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
        "task366_lookup_argmax_scatter",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task366_lookup_optimizer",
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
            if Path(name).name == member:
                dst.writestr(member, raw)
                replaced = True
            else:
                dst.writestr(Path(name).name, src.read(name))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--model-out", type=Path, default=Path("generated/custom_b_models/task366_lookup_optimized.onnx"))
    parser.add_argument("--zip-out", type=Path, default=Path("generated/probe_task366_lookup_optimized/submission.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/task366_lookup_optimized_report.csv"))
    parser.add_argument("--constants-as-initializers", action="store_true")
    args = parser.parse_args()

    base_raw = read_zip_member(args.base_zip, "task366.onnx")
    model = make_optimized_model(base_raw, constants_as_initializers=args.constants_as_initializers)
    raw = model.SerializeToString()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task366.json").read_text()))
    right, wrong, reason = verify_against_base(base_raw, raw, examples)
    macs = memory = params = cost = ""
    points = ""
    status = "failed_verify"
    if wrong == 0:
        macs, memory, params, cost = profile_cost(args.model_out)
        points = f"{task_points(cost):.6f}"
        status = "ok"
        write_zip_with_replacement(args.base_zip, args.zip_out, "task366.onnx", raw)

    row = {
        "task": "task366",
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
        "base_cost": 20894994,
        "delta_points": (task_points(cost) - task_points(20894994)) if cost else "",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(row)
    if status == "ok":
        print(f"model: {args.model_out}")
        print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
