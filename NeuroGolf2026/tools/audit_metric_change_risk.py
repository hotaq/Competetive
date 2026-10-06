#!/usr/bin/env python3
"""Audit NeuroGolf submissions for likely metric-change risks.

This follows the public audit pattern from discussion 694772:
- infer shapes and flag symbolic/missing dimensions inside graph tensors,
- compare declared output shape with ONNX Runtime output on a zero tensor,
- count Constant-node tensor elements that may become counted as parameters,
- keep a per-task CSV plus a short summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnx.numpy_helper as onh


EXPECTED_SHAPE = [1, 10, 30, 30]
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
ALLOWED_DOMAINS = {"", "ai.onnx"}
RUNTIME_CHILD_CODE = r"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

EXPECTED_SHAPE = [1, 10, 30, 30]

try:
    raw = Path(sys.argv[1]).read_bytes()
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: np.zeros(EXPECTED_SHAPE, dtype=np.float32)})[0]
    shape = [int(x) for x in output.shape]
    print(json.dumps({"shape": shape, "matches": shape == EXPECTED_SHAPE, "error": ""}))
except Exception as exc:  # noqa: BLE001
    print(
        json.dumps(
            {
                "shape": [],
                "matches": False,
                "error": str(exc).replace("\n", " ")[:300],
            }
        )
    )
"""


def dims_of(value_info: onnx.ValueInfoProto) -> list[str]:
    dims: list[str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return dims


def has_dynamic_dim(value_info: onnx.ValueInfoProto) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return True
    for dim in tensor_type.shape.dim:
        if dim.dim_param or not dim.dim_value:
            return True
    return False


def value_name(value_info: onnx.ValueInfoProto) -> str:
    return value_info.name or "<unnamed>"


def constant_stats(model: onnx.ModelProto) -> tuple[int, int, int]:
    nodes = elements = bytes_ = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        node_elements = 0
        node_bytes = 0
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                arr = onh.to_array(attr.t)
                node_elements += int(arr.size)
                node_bytes += int(arr.nbytes)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    arr = onh.to_array(tensor)
                    node_elements += int(arr.size)
                    node_bytes += int(arr.nbytes)
        if node_elements:
            nodes += 1
            elements += node_elements
            bytes_ += node_bytes
    return nodes, elements, bytes_


def initializer_stats(model: onnx.ModelProto) -> tuple[int, int]:
    params = bytes_ = 0
    for init in model.graph.initializer:
        arr = onh.to_array(init)
        params += int(arr.size)
        bytes_ += int(arr.nbytes)
    return params, bytes_


def runtime_zero_check(raw: bytes, timeout: float) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        f.write(raw)
        f.flush()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", RUNTIME_CHILD_CODE, f.name],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "", f"runtime_timeout_{timeout:g}s"

    if proc.returncode != 0:
        stderr = proc.stderr.replace("\n", " ")[:220]
        return "", f"runtime_process_exit_{proc.returncode}: {stderr}"

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        stdout = proc.stdout.replace("\n", " ")[:160]
        stderr = proc.stderr.replace("\n", " ")[:160]
        return "", f"runtime_bad_json: {exc}; stdout={stdout}; stderr={stderr}"

    shape = payload.get("shape") or []
    runtime_shape = "x".join(str(x) for x in shape)
    return runtime_shape, str(payload.get("error", ""))


def audit_member(name: str, raw: bytes, runtime_timeout: float) -> dict[str, object]:
    model = onnx.load_from_string(raw)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        shape_error = ""
    except Exception as exc:  # noqa: BLE001
        inferred = model
        shape_error = str(exc).replace("\n", " ")[:300]

    all_value_infos = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    dynamic_infos = [vi for vi in all_value_infos if has_dynamic_dim(vi)]
    sample_dynamic = ";".join(
        f"{value_name(vi)}:{'x'.join(dims_of(vi))}" for vi in dynamic_infos[:8]
    )
    input_shape = dims_of(inferred.graph.input[0]) if inferred.graph.input else []
    output_shape = dims_of(inferred.graph.output[0]) if inferred.graph.output else []
    ops = {node.op_type for node in model.graph.node}
    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ALLOWED_DOMAINS})
    custom_domain_node_count = sum(node.domain not in ALLOWED_DOMAINS for node in model.graph.node)
    io_names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    io_initializer_collisions = sorted(io_names & initializer_names)
    constant_nodes, constant_elements, constant_bytes = constant_stats(model)
    initializer_params, initializer_bytes = initializer_stats(model)
    runtime_shape, runtime_error = runtime_zero_check(raw, runtime_timeout)
    runtime_matches_declared = "1" if runtime_shape.split("x") == output_shape and not runtime_error else "0"
    clean_static = (
        not dynamic_infos
        and input_shape == [str(x) for x in EXPECTED_SHAPE]
        and output_shape == [str(x) for x in EXPECTED_SHAPE]
        and runtime_matches_declared == "1"
        and not runtime_error
        and not (ops & BANNED_OPS)
        and not model.functions
        and not custom_domain_node_count
        and not io_initializer_collisions
    )
    return {
        "task": Path(name).stem,
        "file_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "input_shape": "x".join(input_shape),
        "output_shape": "x".join(output_shape),
        "runtime_zero_shape": runtime_shape,
        "runtime_matches_declared": runtime_matches_declared,
        "runtime_error": runtime_error,
        "value_info_count": len(all_value_infos),
        "dynamic_value_info_count": len(dynamic_infos),
        "sample_dynamic_value_infos": sample_dynamic,
        "shape_inference_error": shape_error,
        "function_count": len(model.functions),
        "custom_domain_node_count": custom_domain_node_count,
        "custom_domains": ";".join(custom_domains),
        "io_initializer_collision_count": len(io_initializer_collisions),
        "io_initializer_collisions": ";".join(io_initializer_collisions[:8]),
        "constant_nodes": constant_nodes,
        "constant_elements": constant_elements,
        "constant_bytes": constant_bytes,
        "initializer_params": initializer_params,
        "initializer_bytes": initializer_bytes,
        "ops": ";".join(sorted(ops)),
        "banned_ops": ";".join(sorted(ops & BANNED_OPS)),
        "clean_static_runtime": int(clean_static),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def is_submission_task(name: str) -> bool:
    path = Path(name)
    if path.suffix.lower() != ".onnx":
        return False
    stem = path.stem.lower()
    if not stem.startswith("task") or len(stem) != 7 or not stem[4:].isdigit():
        return False
    task_num = int(stem[4:])
    return 1 <= task_num <= 400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/metric_change_risk.csv"))
    parser.add_argument(
        "--safe-zip-out",
        type=Path,
        default=None,
        help="Optional zip containing only clean_static_runtime tasks.",
    )
    parser.add_argument(
        "--runtime-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for one ONNX Runtime zero-input check.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    safe_members: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(args.zip) as zf:
        for name in sorted(zf.namelist()):
            if not is_submission_task(Path(name).name):
                continue
            raw = zf.read(name)
            row = audit_member(Path(name).name, raw, args.runtime_timeout)
            rows.append(row)
            if row["clean_static_runtime"]:
                safe_members.append((Path(name).name, raw))

    write_csv(args.report_out, rows)

    if args.safe_zip_out:
        args.safe_zip_out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.safe_zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, raw in safe_members:
                zf.writestr(name, raw)

    total = len(rows)
    dynamic = sum(int(row["dynamic_value_info_count"]) > 0 for row in rows)
    runtime_bad = sum(row["runtime_matches_declared"] != "1" or bool(row["runtime_error"]) for row in rows)
    constants = sum(int(row["constant_elements"]) > 0 for row in rows)
    banned = sum(bool(row["banned_ops"]) for row in rows)
    functions = sum(int(row["function_count"]) > 0 for row in rows)
    custom_domains = sum(int(row["custom_domain_node_count"]) > 0 for row in rows)
    collisions = sum(int(row["io_initializer_collision_count"]) > 0 for row in rows)
    safe = sum(int(row["clean_static_runtime"]) for row in rows)
    total_constant_elements = sum(int(row["constant_elements"]) for row in rows)

    print(f"submission: {args.zip}")
    print(f"tasks: {total}")
    print(f"dynamic_shape_tasks: {dynamic}")
    print(f"runtime_zero_bad_tasks: {runtime_bad}")
    print(f"constant_element_tasks: {constants}")
    print(f"banned_op_tasks: {banned}")
    print(f"function_tasks: {functions}")
    print(f"custom_domain_tasks: {custom_domains}")
    print(f"io_initializer_collision_tasks: {collisions}")
    print(f"clean_static_runtime_tasks: {safe}")
    print(f"total_constant_elements: {total_constant_elements}")
    print(f"report: {args.report_out}")
    if args.safe_zip_out:
        print(f"safe_zip: {args.safe_zip_out}")


if __name__ == "__main__":
    main()
