#!/usr/bin/env python3
"""Strict metric-v2 audit and per-task selector for NeuroGolf submissions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnx_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
MAX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}


@dataclass(frozen=True)
class Candidate:
    task: str
    source: str
    raw: bytes


def task_points(cost: int | None) -> float:
    if cost is None:
        return 0.0
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def is_task_member(name: str) -> bool:
    path = Path(name)
    stem = path.stem
    if path.suffix.lower() != ".onnx":
        return False
    if not stem.startswith("task") or len(stem) != 7 or not stem[4:].isdigit():
        return False
    task_num = int(stem[4:])
    return 1 <= task_num <= 400


def load_candidates(path: Path, source_name: str | None = None) -> list[Candidate]:
    source = source_name or path.stem
    candidates: list[Candidate] = []
    if path.is_dir():
        for model_path in sorted(path.rglob("task*.onnx")):
            if is_task_member(model_path.name):
                candidates.append(Candidate(model_path.stem, source, model_path.read_bytes()))
        return candidates
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if is_task_member(name):
                    candidates.append(Candidate(Path(name).stem, source, zf.read(name)))
        return candidates
    if is_task_member(path.name):
        return [Candidate(path.stem, source, path.read_bytes())]
    raise ValueError(f"unsupported candidate source: {path}")


def example_fits_30(example: dict[str, list[list[int]]]) -> bool:
    """Match official verifier behavior by skipping >30x30 examples.

    neurogolf_utils.convert_to_numpy returns None for oversized ARC grids, and
    verify_subset skips those examples instead of rejecting the whole task.
    """
    for key in ("input", "output"):
        grid = example[key]
        if not grid or not grid[0]:
            return False
        if len(grid) > 30 or len(grid[0]) > 30:
            return False
    return True


def calculate_memory(filename: Path) -> int | None:
    model = onnx.load(filename)
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    memory = 0
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            num_elements *= dim.dim_value
        if item.name in {"input", "output"}:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        memory += num_elements * np.dtype(np_dtype).itemsize
    return int(memory)


def tensor_param_count(tensor: onnx.TensorProto) -> int:
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return int(count)


def constant_node_param_count(model: onnx.ModelProto) -> int:
    total = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                total += tensor_param_count(attr.t)
            elif attr.name == "sparse_value" and attr.HasField("sparse_tensor"):
                total += tensor_param_count(attr.sparse_tensor.values)
            elif attr.name in {"value_float", "value_int", "value_string"}:
                total += 1
            elif attr.name == "value_floats":
                total += len(attr.floats)
            elif attr.name == "value_ints":
                total += len(attr.ints)
            elif attr.name == "value_strings":
                total += len(attr.strings)
    return int(total)


def scorer_loophole_risk(model: onnx.ModelProto) -> str | None:
    if len(model.functions):
        return "model_functions_disallowed"

    custom_opsets = sorted({opset.domain for opset in model.opset_import if opset.domain not in ALLOWED_DOMAINS})
    if custom_opsets:
        return "custom_opset_domain:" + ",".join(custom_opsets)[:160]

    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ALLOWED_DOMAINS})
    if custom_domains:
        return "custom_domain_node:" + ",".join(custom_domains)[:160]

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return "graph_attribute_disallowed"

    io_names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    collisions = sorted(io_names & initializer_names)
    if collisions:
        return "io_initializer_collision:" + ",".join(collisions)[:160]

    return None


def score_network(filename: Path, *, count_constant_node_params: bool = False) -> tuple[int, int, int, int]:
    model_proto = onnx.load(filename)
    risk = scorer_loophole_risk(model_proto)
    if risk:
        raise ValueError(risk)
    model = onnx_tool.loadmodel(str(filename), {"verbose": False, "constant_folding": True})
    graph = model.graph
    graph.graph_reorder_nodes()
    graph.shape_infer(None)
    graph.profile()
    if not graph.valid_profile:
        raise ValueError("invalid_profile")
    for key in graph.nodemap.keys():
        node = graph.nodemap[key]
        if node.op_type.upper() in BANNED_OPS:
            raise ValueError(f"banned_op:{node.op_type}")
        if node.memory < 0 or node.params < 0:
            raise ValueError("negative_memory_or_params")
        if min(node.macs) < 0:
            raise ValueError("negative_macs")
    memory = calculate_memory(filename)
    if memory is None:
        raise ValueError("dynamic_or_missing_shape")
    macs = int(sum(graph.macs))
    params = int(graph.params)
    if count_constant_node_params:
        params += constant_node_param_count(model_proto)
    return macs, memory, params, macs + memory + params


def io_shapes_and_ops_from_model(model: onnx.ModelProto) -> tuple[str, str, str]:
    input_shape = "x".join(str(dim.dim_value) for dim in model.graph.input[0].type.tensor_type.shape.dim)
    output_shape = "x".join(str(dim.dim_value) for dim in model.graph.output[0].type.tensor_type.shape.dim)
    ops = ";".join(sorted({node.op_type for node in model.graph.node}))
    return input_shape, output_shape, ops


def io_shapes_and_ops(raw: bytes) -> tuple[str, str, str]:
    return io_shapes_and_ops_from_model(onnx.load_from_string(raw))


def audit_candidate(
    candidate: Candidate,
    data_root: Path,
    tmp_dir: Path,
    *,
    count_constant_node_params: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "task": candidate.task,
        "source": candidate.source,
        "ok": 0,
        "reason": "",
        "file_bytes": len(candidate.raw),
        "sha256": hashlib.sha256(candidate.raw).hexdigest(),
        "right": 0,
        "wrong": 0,
        "examples": 0,
        "input_shape": "",
        "output_shape": "",
        "ops": "",
        "macs": "",
        "memory": "",
        "params": "",
        "cost": "",
        "points": "",
    }
    if len(candidate.raw) > MAX_BYTES:
        row["reason"] = "oversized"
        return row
    try:
        model = onnx.load_from_string(candidate.raw)
        risk = scorer_loophole_risk(model)
        if risk:
            row["reason"] = risk
            return row
        input_shape, output_shape, ops = io_shapes_and_ops_from_model(model)
        row.update({"input_shape": input_shape, "output_shape": output_shape, "ops": ops})
        if input_shape != "1x10x30x30" or output_shape != "1x10x30x30":
            row["reason"] = "non_expected_io_shape"
            return row
        if {op.upper() for op in ops.split(";")} & BANNED_OPS:
            row["reason"] = "banned_op"
            return row
        task_path = data_root / f"{candidate.task}.json"
        examples = [example for example in all_examples(json.loads(task_path.read_text())) if example_fits_30(example)]
        row["examples"] = len(examples)
        right, wrong, verify_reason = verify_model_bytes(candidate.raw, examples)
        row.update({"right": right, "wrong": wrong})
        if wrong:
            row["reason"] = verify_reason or "verify_failed"
            return row
        model_path = tmp_dir / f"{candidate.source}_{candidate.task}.onnx"
        model_path.write_bytes(candidate.raw)
        macs, memory, params, cost = score_network(
            model_path,
            count_constant_node_params=count_constant_node_params,
        )
        row.update(
            {
                "ok": 1,
                "reason": "ok",
                "macs": macs,
                "memory": memory,
                "params": params,
                "cost": cost,
                "points": f"{task_points(cost):.6f}",
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001
        row["reason"] = f"error:{str(exc).replace(chr(10), ' ')[:220]}"
        return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select lowest-cost valid NeuroGolf metric-v2 candidates.")
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--zip-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--selected-report-out", type=Path, required=True)
    parser.add_argument(
        "--count-constant-node-params",
        action="store_true",
        help=(
            "Use the older conservative local scorer that adds Constant-node "
            "tensor element counts to params. By default this matches the latest "
            "official neurogolf_utils.py scorer, which uses onnx_tool graph.params."
        ),
    )
    args = parser.parse_args()

    sources = [args.base_zip] + args.candidate
    candidates: list[Candidate] = []
    seen_payloads: set[tuple[str, str]] = set()
    for source in sources:
        label = "base" if source == args.base_zip else source.parent.name if source.name == "submission.zip" else source.stem
        for candidate in load_candidates(source, label):
            payload_key = (candidate.task, hashlib.sha256(candidate.raw).hexdigest())
            if payload_key in seen_payloads:
                continue
            seen_payloads.add(payload_key)
            candidates.append(candidate)

    rows: list[dict[str, object]] = []
    raw_by_key: dict[tuple[str, str, str], bytes] = {}
    with tempfile.TemporaryDirectory(prefix="ng_metric_v2_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for candidate in candidates:
            row = audit_candidate(
                candidate,
                args.data_root,
                tmp_dir,
                count_constant_node_params=args.count_constant_node_params,
            )
            rows.append(row)
            raw_by_key[(candidate.task, candidate.source, str(row["sha256"]))] = candidate.raw

    selected_rows: dict[str, dict[str, object]] = {}
    selected_raw: dict[str, bytes] = {}
    for row in rows:
        if not row["ok"]:
            continue
        task = str(row["task"])
        key = (task, str(row["source"]), str(row["sha256"]))
        old = selected_rows.get(task)
        new_key = (int(row["cost"]), int(row["file_bytes"]), str(row["source"]))
        old_key = (
            int(old["cost"]),
            int(old["file_bytes"]),
            str(old["source"]),
        ) if old else None
        if old is None or new_key < old_key:
            selected_rows[task] = row
            selected_raw[task] = raw_by_key[key]

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task in sorted(selected_raw):
            zf.writestr(f"{task}.onnx", selected_raw[task])

    selected = [selected_rows[task] for task in sorted(selected_rows)]
    write_csv(args.report_out, rows)
    write_csv(args.selected_report_out, selected)

    total = sum(task_points(int(row["cost"])) for row in selected)
    improved = sum(1 for row in selected if row["source"] != "base")
    print(f"audited candidates: {len(rows)}")
    print(f"selected tasks: {len(selected)}")
    print(f"non-base selected: {improved}")
    print(f"local metric-v2 score: {total:.6f}")
    print(
        "score mode: "
        + ("conservative_constant_nodes" if args.count_constant_node_params else "official_current")
    )
    print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")
    print(f"selected report: {args.selected_report_out}")


if __name__ == "__main__":
    main()
