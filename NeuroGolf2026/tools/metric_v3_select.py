#!/usr/bin/env python3
"""Metric-v3 audit and per-task selector for NeuroGolf submissions.

Matches the May 6 official utility surface:
- cost is memory + params, MACs are not counted.
- memory is derived from static shape inference, then raised to the max
  shape observed by ONNX Runtime profiling across verification runs.
- Constant tensors and scalar initializers count as params.
- multi-input/output, functions, custom domains, subgraphs, sequences,
  nonpositive dims, banned ops, and tensor/initializer name collisions reject.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


EXPECTED_SHAPE = [1, 10, 30, 30]
MAX_BYTES = int(1.44 * 1024 * 1024)
BANNED_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}


Grid = list[list[int]]
Example = dict[str, Grid]


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
    return (
        path.suffix.lower() == ".onnx"
        and stem.startswith("task")
        and len(stem) == 7
        and stem[4:].isdigit()
        and 1 <= int(stem[4:]) <= 400
    )


def load_candidates(path: Path, source_name: str | None = None) -> list[Candidate]:
    source = source_name or path.stem
    if path.is_dir():
        return [
            Candidate(model_path.stem, source, model_path.read_bytes())
            for model_path in sorted(path.rglob("task*.onnx"))
            if is_task_member(model_path.name)
        ]
    if path.suffix.lower() == ".zip":
        out: list[Candidate] = []
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if is_task_member(name):
                    out.append(Candidate(Path(name).stem, source, zf.read(name)))
        return out
    if is_task_member(path.name):
        return [Candidate(path.stem, source, path.read_bytes())]
    raise ValueError(f"unsupported candidate source: {path}")


def all_examples(task: dict[str, list[Example]]) -> list[Example]:
    return task["train"] + task["test"] + task["arc-gen"]


def convert_to_numpy(example: Example) -> dict[str, np.ndarray] | None:
    benchmark: dict[str, np.ndarray] = {}
    for mode in ("input", "output"):
        grid = example[mode]
        if not grid or not grid[0] or max(len(grid), len(grid[0])) > 30:
            return None
        arr = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                arr[0, color, r, c] = 1.0
        benchmark[mode] = arr
    return benchmark


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto().FromString(model.SerializeToString())
    for node in model.graph.node:
        if not node.output or not node.output[0]:
            raise ValueError("empty_node_output")
        node.name = node.output[0]
        if "kernel_time" in node.name:
            raise ValueError("kernel_time_tensor_name")
    return model


def scorer_risk(model: onnx.ModelProto) -> str | None:
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        return "multi_input_or_output"
    if model.functions:
        return "model_functions_disallowed"
    custom_opsets = sorted({opset.domain for opset in model.opset_import if opset.domain not in ALLOWED_DOMAINS})
    if custom_opsets:
        return "custom_opset_domain:" + ",".join(custom_opsets)[:160]
    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ALLOWED_DOMAINS})
    if custom_domains:
        return "custom_domain_node:" + ",".join(custom_domains)[:160]
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS:
            return f"banned_op:{node.op_type}"
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return "graph_attribute_disallowed"
    io_names = {t.name for t in list(model.graph.input) + list(model.graph.output)}
    init_names = {init.name for init in model.graph.initializer}
    init_names.update(init.name for init in model.graph.sparse_initializer)
    collisions = sorted(io_names & init_names)
    if collisions:
        return "io_initializer_collision:" + ",".join(collisions)[:160]
    return None


def shape_text(value: onnx.ValueInfoProto) -> str:
    dims = value.type.tensor_type.shape.dim
    return "x".join(str(dim.dim_value) if dim.HasField("dim_value") else "?" for dim in dims)


def io_shapes_and_ops(model: onnx.ModelProto) -> tuple[str, str, str]:
    return (
        shape_text(model.graph.input[0]),
        shape_text(model.graph.output[0]),
        ";".join(sorted({node.op_type for node in model.graph.node})),
    )


def tensor_volume_and_dtype(item: onnx.ValueInfoProto) -> tuple[int, np.dtype] | None:
    if item.type.HasField("sequence_type"):
        raise ValueError("sequence_type")
    if not item.type.HasField("tensor_type"):
        return None
    tensor_type = item.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError("missing_shape")
    num_elements = 1
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_param"):
            raise ValueError("symbolic_dim")
        if not dim.HasField("dim_value"):
            raise ValueError("missing_dim")
        if dim.dim_value <= 0:
            raise ValueError("nonpositive_dim")
        num_elements *= int(dim.dim_value)
    return num_elements, np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))


def calculate_memory(model: onnx.ModelProto, trace_path: str) -> int:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("multi_input_or_output")

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        node_outputs[node.name] = list(node.output)
        tensor_names.update(output for output in node.output if output)

    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    tensor_names.update(tensor_map)
    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if item is None:
            raise ValueError("missing_value_info")
        info = tensor_volume_and_dtype(item)
        if info is None:
            continue
        num_elements, dtype = info
        if tensor_name not in {"input", "output"}:
            tensor_memory[tensor_name] = num_elements * dtype.itemsize
        tensor_dtypes[tensor_name] = dtype

    with open(trace_path, encoding="utf-8") as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = str(event.get("name", "")).replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes or output_name in {"input", "output"}:
                continue
            itemsize = tensor_dtypes[output_name].itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory.get(output_name, 0), mem)
    return int(sum(tensor_memory.values()))


def tensor_param_count(tensor: onnx.TensorProto) -> int:
    if any(dim <= 0 for dim in tensor.dims):
        raise ValueError("nonpositive_param_dim")
    return int(math.prod(tensor.dims))


def calculate_params(model: onnx.ModelProto) -> int:
    params = 0
    for init in model.graph.initializer:
        params += tensor_param_count(init)
    for sparse_init in model.graph.sparse_initializer:
        params += tensor_param_count(sparse_init.values)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                params += tensor_param_count(attr.t)
            elif attr.name == "sparse_value" and attr.HasField("sparse_tensor"):
                params += tensor_param_count(attr.sparse_tensor.values)
            elif attr.name in {"value_float", "value_int", "value_string"}:
                params += 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def verify_and_score(raw: bytes, examples: list[Example], tmp_dir: Path, task: str) -> tuple[int, int, int, int, str]:
    model = onnx.load_from_string(raw)
    risk = scorer_risk(model)
    if risk:
        raise ValueError(risk)
    sanitized = sanitize_model(model)
    profile_prefix = str(tmp_dir / f"{task}_profile")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = profile_prefix
    session = ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])
    right = wrong = 0
    reason = ""
    for idx, example in enumerate(examples):
        benchmark = convert_to_numpy(example)
        if benchmark is None:
            continue
        pred = session.run(["output"], {"input": benchmark["input"]})[0]
        pred = (pred > 0.0).astype(np.float32)
        if np.array_equal(pred, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if not reason:
                reason = f"mismatch_example_{idx}"
    trace_path = session.end_profiling()
    if wrong:
        raise ValueError(reason or "verify_failed")
    memory = calculate_memory(sanitized, trace_path)
    params = calculate_params(sanitized)
    cost = memory + params
    return right, wrong, memory, params, cost


def audit_candidate(candidate: Candidate, data_root: Path, tmp_dir: Path) -> dict[str, object]:
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
        risk = scorer_risk(model)
        if risk:
            row["reason"] = risk
            return row
        input_shape, output_shape, ops = io_shapes_and_ops(model)
        row.update({"input_shape": input_shape, "output_shape": output_shape, "ops": ops})
        if input_shape != "1x10x30x30" or output_shape != "1x10x30x30":
            row["reason"] = "non_expected_io_shape"
            return row
        task_path = data_root / f"{candidate.task}.json"
        examples = all_examples(json.loads(task_path.read_text()))
        row["examples"] = len(examples)
        right, wrong, memory, params, cost = verify_and_score(candidate.raw, examples, tmp_dir, candidate.task)
        row.update(
            {
                "ok": 1,
                "reason": "ok",
                "right": right,
                "wrong": wrong,
                "memory": memory,
                "params": params,
                "cost": cost,
                "points": f"{task_points(cost):.6f}",
            }
        )
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
    parser = argparse.ArgumentParser(description="Select lowest-cost valid NeuroGolf metric-v3 candidates.")
    parser.add_argument("--data-root", type=Path, default=Path("data/live/dataset_20260507T091242Z/neurogolf-2026"))
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--zip-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--selected-report-out", type=Path, required=True)
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
    with tempfile.TemporaryDirectory(prefix="ng_metric_v3_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for i, candidate in enumerate(candidates, start=1):
            if i == 1 or i == len(candidates) or i % 100 == 0:
                print(f"audit {i}/{len(candidates)} {candidate.task} {candidate.source}", flush=True)
            row = audit_candidate(candidate, args.data_root, tmp_dir)
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
        old_key = (int(old["cost"]), int(old["file_bytes"]), str(old["source"])) if old else None
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
    print(f"local metric-v3 score: {total:.6f}")
    print("score mode: official_2026_05_06_memory_plus_params")
    print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")
    print(f"selected report: {args.selected_report_out}")


if __name__ == "__main__":
    main()
