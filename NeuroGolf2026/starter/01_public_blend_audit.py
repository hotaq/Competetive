#!/usr/bin/env python3
"""
Kaggle starter for The 2026 NeuroGolf Championship.

This script builds a conservative public-blend submission:

1. scan /kaggle/input for zip files and raw taskNNN.onnx files
2. audit every candidate for size, banned ops, static I/O shape, and ORT loadability
3. estimate local cost
4. keep the cheapest valid candidate per task
5. write /kaggle/working/submission.zip plus CSV reports

It intentionally avoids dynamic-shape and hidden-constant tricks because the
competition hosts have said those will be enforced/scored more strictly.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import zipfile
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable


# Kaggle-friendly defaults. Flip these at the top of the notebook/script if needed.
AUTO_INSTALL = True
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ZIP = WORKING_ROOT / "submission.zip"
CANDIDATE_REPORT = WORKING_ROOT / "candidate_report.csv"
SELECTED_REPORT = WORKING_ROOT / "selected_report.csv"
REJECTED_REPORT = WORKING_ROOT / "rejected_report.csv"

EXPECTED_SHAPE = [1, 10, 30, 30]
MAX_ONNX_BYTES = int(1.44 * 1024 * 1024)
TASK_RE = re.compile(r"task(\d{3})\.onnx$", re.IGNORECASE)

OFFICIAL_BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}

ALLOW_RENAME_IO = True
RUN_ORT_SMOKE = True
STRICT_IO_STATIC_SHAPE = True
STRICT_INFERRED_VALUE_SHAPES = True
INCLUDE_TASK000 = False
RUN_PUBLIC_EXAMPLE_VERIFY = False
NEUROGOLF_UTIL_DIR = Path("/kaggle/input/competitions/neurogolf-2026/neurogolf_utils")

PINNED_PACKAGES = {
    "numpy": ("numpy", "2.4.4", "numpy==2.4.4"),
    "onnx": ("onnx", "1.21.0", "onnx==1.21.0"),
    "onnxruntime": ("onnxruntime", "1.24.4", "onnxruntime==1.24.4"),
    "onnx_tool": ("onnx-tool", "1.0.1", "onnx-tool==1.0.1"),
}


@dataclass
class Candidate:
    task: str
    source: str
    raw: bytes


@dataclass
class AuditResult:
    task: str
    source: str
    status: str
    file_bytes: int
    local_cost: int | None = None
    params: int | None = None
    memory_bytes: int | None = None
    macs: int | None = None
    op_count: int | None = None
    changed_io_names: bool = False
    ops: str = ""
    reasons: list[str] = field(default_factory=list)
    fixed_raw: bytes | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def row(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "source": self.source,
            "status": self.status,
            "file_bytes": self.file_bytes,
            "local_cost": self.local_cost,
            "params": self.params,
            "memory_bytes": self.memory_bytes,
            "macs": self.macs,
            "op_count": self.op_count,
            "changed_io_names": int(self.changed_io_names),
            "ops": self.ops,
            "reasons": " | ".join(self.reasons),
        }


def ensure_deps() -> None:
    if not AUTO_INSTALL:
        return
    missing: list[str] = []
    for import_name, (dist_name, pinned_version, pip_spec) in PINNED_PACKAGES.items():
        try:
            installed_version = package_version(dist_name)
        except PackageNotFoundError:
            missing.append(pip_spec)
            continue
        if installed_version != pinned_version:
            missing.append(pip_spec)
            continue
        try:
            __import__(import_name)
        except Exception:
            missing.append(pip_spec)
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_deps()

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import numpy_helper  # noqa: E402

try:  # noqa: E402
    import onnx_tool  # type: ignore
except Exception:  # noqa: E402
    onnx_tool = None


def iter_zip_candidates(zip_path: Path) -> Iterable[Candidate]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                task_name = Path(info.filename).name
                match = TASK_RE.match(task_name)
                if not match:
                    continue
                if match.group(1) == "000" and not INCLUDE_TASK000:
                    continue
                try:
                    raw = zf.read(info)
                except Exception as exc:
                    print(f"Skip unreadable zip entry {zip_path}:{info.filename}: {exc}")
                    continue
                yield Candidate(task=task_name.lower(), source=f"zip:{zip_path}:{info.filename}", raw=raw)
    except Exception as exc:
        print(f"Skip unreadable zip {zip_path}: {exc}")


def iter_file_candidates(path: Path) -> Iterable[Candidate]:
    match = TASK_RE.match(path.name)
    if not match:
        return
    if match.group(1) == "000" and not INCLUDE_TASK000:
        return
    try:
        yield Candidate(task=path.name.lower(), source=f"file:{path}", raw=path.read_bytes())
    except Exception as exc:
        print(f"Skip unreadable ONNX {path}: {exc}")


def preview_input_tree(input_root: Path, limit: int = 120) -> None:
    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        return
    print()
    print("Input preview:")
    shown = 0
    for path in sorted(input_root.rglob("*")):
        if shown >= limit:
            print(f"... truncated after {limit} paths")
            break
        rel = path.relative_to(input_root)
        suffix = "/" if path.is_dir() else ""
        size = "" if path.is_dir() else f" ({path.stat().st_size} bytes)"
        print(f"  {rel}{suffix}{size}")
        shown += 1


def discover_candidates(input_root: Path) -> list[Candidate]:
    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        return []

    candidates: list[Candidate] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        low = path.name.lower()
        if low.endswith(".zip"):
            candidates.extend(iter_zip_candidates(path))
        elif TASK_RE.match(path.name):
            candidates.extend(iter_file_candidates(path))

    return candidates


def iter_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from iter_graphs(attr.g)
            for subgraph in attr.graphs:
                yield from iter_graphs(subgraph)


def graph_ops(model: onnx.ModelProto) -> list[str]:
    ops: list[str] = []
    for graph in iter_graphs(model.graph):
        ops.extend(node.op_type for node in graph.node)
    return ops


def tensor_numel_from_value_info(value_info: onnx.ValueInfoProto) -> int | None:
    shape = value_info.type.tensor_type.shape
    numel = 1
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            numel *= int(dim.dim_value)
        else:
            return None
    return int(numel)


def shape_from_value_info(value_info: onnx.ValueInfoProto) -> list[int] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            return None
    return dims


def unknown_static_shapes(model: onnx.ModelProto) -> list[str]:
    unknown: list[str] = []
    for graph in iter_graphs(model.graph):
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            name = value_info.name or "<unnamed>"
            tensor_type = value_info.type.tensor_type
            if not tensor_type.HasField("shape"):
                unknown.append(name)
                continue
            for dim in tensor_type.shape.dim:
                if not dim.HasField("dim_value"):
                    unknown.append(name)
                    break
    return sorted(set(unknown))


def replace_tensor_name_graph(graph: onnx.GraphProto, old: str, new: str) -> None:
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.name == old:
            value_info.name = new
    for initializer in graph.initializer:
        if initializer.name == old:
            initializer.name = new
    for sparse_initializer in graph.sparse_initializer:
        if sparse_initializer.values.name == old:
            sparse_initializer.values.name = new
        if sparse_initializer.indices.name == old:
            sparse_initializer.indices.name = new
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
        for attr in node.attribute:
            if attr.HasField("g"):
                replace_tensor_name_graph(attr.g, old, new)
            for subgraph in attr.graphs:
                replace_tensor_name_graph(subgraph, old, new)


def normalize_io_names(model: onnx.ModelProto) -> bool:
    if not model.graph.input or not model.graph.output:
        return False
    changed = False
    current_input = model.graph.input[0].name
    current_output = model.graph.output[0].name
    if current_input != "input":
        replace_tensor_name_graph(model.graph, current_input, "input")
        changed = True
    if current_output != "output":
        replace_tensor_name_graph(model.graph, current_output, "output")
        changed = True
    return changed


def infer_shapes_safe(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str | None]:
    try:
        return onnx.shape_inference.infer_shapes(model), None
    except Exception as exc:
        return model, str(exc)


def count_constant_params(model: onnx.ModelProto) -> int:
    total = 0
    for graph in iter_graphs(model.graph):
        for node in graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.HasField("t"):
                    total += int(np.prod(numpy_helper.to_array(attr.t).shape))
                for tensor in attr.tensors:
                    total += int(np.prod(numpy_helper.to_array(tensor).shape))
    return total


def fallback_cost(model: onnx.ModelProto, raw_len: int) -> tuple[int, int, int, int]:
    params = 0
    for graph in iter_graphs(model.graph):
        for initializer in graph.initializer:
            params += int(np.prod(numpy_helper.to_array(initializer).shape))
    params += count_constant_params(model)

    memory_bytes = raw_len
    inferred, _ = infer_shapes_safe(model)
    for graph in iter_graphs(inferred.graph):
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            numel = tensor_numel_from_value_info(value_info)
            if numel is not None:
                memory_bytes += 4 * numel

    macs = 0
    value_shapes = {}
    for graph in iter_graphs(inferred.graph):
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            shape = shape_from_value_info(value_info)
            if shape:
                value_shapes[value_info.name] = shape
        for initializer in graph.initializer:
            value_shapes[initializer.name] = list(numpy_helper.to_array(initializer).shape)

        for node in graph.node:
            if node.op_type == "Conv" and len(node.input) >= 2:
                x_shape = value_shapes.get(node.input[0])
                w_shape = value_shapes.get(node.input[1])
                y_shape = value_shapes.get(node.output[0]) if node.output else None
                if x_shape and w_shape and y_shape and len(y_shape) == 4 and len(w_shape) == 4:
                    out_n, out_c, out_h, out_w = y_shape
                    _, in_per_group, k_h, k_w = w_shape
                    macs += int(out_n * out_c * out_h * out_w * in_per_group * k_h * k_w)
            elif node.op_type in {"Gemm", "MatMul"} and len(node.input) >= 2:
                a_shape = value_shapes.get(node.input[0])
                b_shape = value_shapes.get(node.input[1])
                y_shape = value_shapes.get(node.output[0]) if node.output else None
                if a_shape and b_shape and y_shape:
                    k_dim = a_shape[-1]
                    macs += int(np.prod(y_shape) * k_dim)

    local_cost = int(params + memory_bytes + macs)
    return local_cost, int(params), int(memory_bytes), int(macs)


def onnx_tool_cost(raw: bytes, model: onnx.ModelProto) -> tuple[int, int, int, int] | None:
    if onnx_tool is None:
        return None

    try:
        temp_path = WORKING_ROOT / "__neurogolf_cost_tmp.onnx"
        temp_path.write_bytes(raw)
        graph = onnx_tool.loadmodel(str(temp_path), {"verbose": False})
        graph.shape_infer(None)
        graph.profile()
        params = int(sum(graph.params)) if isinstance(graph.params, (list, tuple)) else int(graph.params)
        memory = int(sum(graph.memory)) if isinstance(graph.memory, (list, tuple)) else int(graph.memory)
        macs = int(sum(graph.macs)) if isinstance(graph.macs, (list, tuple)) else int(graph.macs)
        if memory < 0 or params < 0 or macs < 0:
            return None
        params += count_constant_params(model)
        return int(params + memory + macs), params, memory, macs
    except Exception:
        return None
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


def ort_smoke(raw: bytes) -> str | None:
    try:
        session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            return f"ORT expected 1 input, got {len(inputs)}"
        if len(outputs) != 1:
            return f"ORT expected 1 output, got {len(outputs)}"
        input_shape = [dim if isinstance(dim, int) else expected for dim, expected in zip(inputs[0].shape, EXPECTED_SHAPE)]
        x = np.zeros(input_shape, dtype=np.float32)
        y = session.run(None, {inputs[0].name: x})[0]
        if list(y.shape) != EXPECTED_SHAPE:
            return f"ORT output shape {list(y.shape)} != {EXPECTED_SHAPE}"
        return None
    except Exception as exc:
        return f"ORT load/run failed: {exc}"


def verify_public_examples(model: onnx.ModelProto, task_name: str) -> str | None:
    if not RUN_PUBLIC_EXAMPLE_VERIFY:
        return None
    if not NEUROGOLF_UTIL_DIR.exists():
        return f"neurogolf_utils not found: {NEUROGOLF_UTIL_DIR}"

    try:
        if str(NEUROGOLF_UTIL_DIR) not in sys.path:
            sys.path.append(str(NEUROGOLF_UTIL_DIR))
        from neurogolf_utils import load_examples, verify_network  # type: ignore

        task_num = int(TASK_RE.match(task_name).group(1))  # type: ignore[union-attr]
        examples = load_examples(task_num)
        verdict = verify_network(model, task_num, examples)
        if verdict is False:
            return "official verify_network returned False"
        return None
    except Exception as exc:
        return f"official verify_network failed: {exc}"


def audit_candidate(candidate: Candidate) -> AuditResult:
    result = AuditResult(
        task=candidate.task,
        source=candidate.source,
        status="reject",
        file_bytes=len(candidate.raw),
    )

    if result.file_bytes > MAX_ONNX_BYTES:
        result.reasons.append(f"file too large: {result.file_bytes} > {MAX_ONNX_BYTES}")
        return result

    try:
        model = onnx.load_model_from_string(candidate.raw)
    except Exception as exc:
        result.reasons.append(f"ONNX parse failed: {exc}")
        return result

    if not model.graph.input:
        result.reasons.append("missing graph input")
        return result
    if not model.graph.output:
        result.reasons.append("missing graph output")
        return result

    ops = graph_ops(model)
    result.ops = ",".join(sorted(set(ops)))
    result.op_count = len(ops)
    banned = sorted(set(ops).intersection(OFFICIAL_BANNED_OPS))
    if banned:
        result.reasons.append("banned ops: " + ",".join(banned))
        return result

    if ALLOW_RENAME_IO:
        result.changed_io_names = normalize_io_names(model)
    elif model.graph.input[0].name != "input" or model.graph.output[0].name != "output":
        result.reasons.append("input/output names are not input/output")
        return result

    inferred, infer_error = infer_shapes_safe(model)
    input_shape = shape_from_value_info(inferred.graph.input[0])
    output_shape = shape_from_value_info(inferred.graph.output[0])
    if STRICT_IO_STATIC_SHAPE:
        if input_shape != EXPECTED_SHAPE:
            result.reasons.append(f"input shape {input_shape} != {EXPECTED_SHAPE}")
            return result
        if output_shape != EXPECTED_SHAPE:
            result.reasons.append(f"output shape {output_shape} != {EXPECTED_SHAPE}")
            return result
    elif infer_error:
        result.reasons.append(f"shape inference warning: {infer_error}")

    if STRICT_INFERRED_VALUE_SHAPES:
        unknown_shapes = unknown_static_shapes(inferred)
        if unknown_shapes:
            preview = ",".join(unknown_shapes[:8])
            more = "" if len(unknown_shapes) <= 8 else f",+{len(unknown_shapes) - 8} more"
            result.reasons.append(f"unknown/static-missing shapes: {preview}{more}")
            return result

    fixed_raw = candidate.raw
    if result.changed_io_names:
        fixed_raw = model.SerializeToString()
        result.file_bytes = len(fixed_raw)
        if result.file_bytes > MAX_ONNX_BYTES:
            result.reasons.append(f"renamed model too large: {result.file_bytes} > {MAX_ONNX_BYTES}")
            return result

    if RUN_ORT_SMOKE:
        ort_error = ort_smoke(fixed_raw)
        if ort_error:
            result.reasons.append(ort_error)
            return result

    public_verify_error = verify_public_examples(model, candidate.task)
    if public_verify_error:
        result.reasons.append(public_verify_error)
        return result

    cost = onnx_tool_cost(fixed_raw, model)
    if cost is None:
        cost = fallback_cost(inferred, len(fixed_raw))
    result.local_cost, result.params, result.memory_bytes, result.macs = cost
    result.fixed_raw = fixed_raw
    result.status = "ok"
    if infer_error:
        result.reasons.append(f"shape inference warning: {infer_error}")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimated_task_score(local_cost: int | None) -> float:
    if not local_cost or local_cost <= 0:
        return 1.0
    return max(1.0, 25.0 - math.log(local_cost))


def select_best(results: list[AuditResult]) -> dict[str, AuditResult]:
    selected: dict[str, AuditResult] = {}
    for result in results:
        if not result.ok:
            continue
        if result.task not in selected:
            selected[result.task] = result
            continue
        old = selected[result.task]
        old_cost = old.local_cost if old.local_cost is not None else 10**30
        new_cost = result.local_cost if result.local_cost is not None else 10**30
        if (new_cost, result.file_bytes, result.source) < (old_cost, old.file_bytes, old.source):
            selected[result.task] = result
    return dict(sorted(selected.items()))


def write_submission(selected: dict[str, AuditResult], output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task, result in selected.items():
            if result.fixed_raw is None:
                raise RuntimeError(f"selected result has no bytes: {task}")
            zf.writestr(task, result.fixed_raw)


def main() -> None:
    print("NeuroGolf conservative public-blend starter")
    print(f"input root: {INPUT_ROOT}")
    print(f"working root: {WORKING_ROOT}")
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = discover_candidates(INPUT_ROOT)
    print(f"discovered candidates: {len(candidates)}")
    if not candidates:
        preview_input_tree(INPUT_ROOT)
        print()
        print("No ONNX candidates found.")
        print("Attach public notebook outputs/datasets that contain taskNNN.onnx files or any zip with taskNNN.onnx inside.")
        print("The official competition data only contains task JSON files, not candidate ONNX networks.")

    seen_payloads: set[tuple[str, int, int]] = set()
    unique_candidates: list[Candidate] = []
    for candidate in candidates:
        payload_key = (candidate.task, len(candidate.raw), hash(candidate.raw))
        if payload_key in seen_payloads:
            continue
        seen_payloads.add(payload_key)
        unique_candidates.append(candidate)
    print(f"unique candidates: {len(unique_candidates)}")

    results: list[AuditResult] = []
    for i, candidate in enumerate(unique_candidates, start=1):
        if i % 100 == 0 or i == 1 or i == len(unique_candidates):
            print(f"audit {i}/{len(unique_candidates)}")
        results.append(audit_candidate(candidate))

    selected = select_best(results)
    if selected:
        write_submission(selected, OUTPUT_ZIP)
    else:
        print("No valid models selected, so no submission.zip was written.")

    all_rows = [result.row() for result in results]
    ok_rows = [result.row() | {"estimated_score": estimated_task_score(result.local_cost)} for result in selected.values()]
    rejected_rows = [result.row() for result in results if not result.ok]

    write_csv(CANDIDATE_REPORT, all_rows)
    write_csv(SELECTED_REPORT, ok_rows)
    write_csv(REJECTED_REPORT, rejected_rows)

    total_estimated_score = sum(estimated_task_score(result.local_cost) for result in selected.values())
    missing = [f"task{i:03d}.onnx" for i in range(1, 401) if f"task{i:03d}.onnx" not in selected]

    print()
    print("Done")
    print(f"valid candidates: {sum(result.ok for result in results)}")
    print(f"rejected candidates: {len(rejected_rows)}")
    print(f"selected tasks: {len(selected)}")
    print(f"missing tasks: {len(missing)}")
    print(f"local estimated score: {total_estimated_score:.3f}")
    if selected:
        print(f"submission: {OUTPUT_ZIP}")
    print(f"candidate report: {CANDIDATE_REPORT}")
    print(f"selected report: {SELECTED_REPORT}")
    print(f"rejected report: {REJECTED_REPORT}")
    if missing:
        print("first missing tasks:", ", ".join(missing[:30]))


if __name__ == "__main__":
    main()
