#!/usr/bin/env python3
"""Replace one-hot Compress selectors with allowed ArgMax+Gather selectors.

The April metric bans Compress.  Many strong public/local NeuroGolf graphs use
Compress only as a one-hot selector (pick one batch/color/size/output-bank row).
For those cases,

    Compress(data, condition, axis=A)

is equivalent to:

    idx = ArgMax(Cast(condition, INT64), keepdims=1)
    Gather(data, idx, axis=A)

as long as exactly one condition element is true for every scored example.  This
tool applies that mechanical rewrite to candidate zips/files, verifies each
task against the local examples (with the official >30 skip behavior), profiles
with the current metric, and writes a partial submission zip containing only
valid transformed models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import (  # noqa: E402
    BANNED_OPS,
    calculate_memory,
    example_fits_30,
    score_network,
    task_points,
)


@dataclass(frozen=True)
class SourceModel:
    task: str
    source: str
    raw: bytes


def is_task_name(name: str) -> bool:
    stem = Path(name).stem
    return Path(name).suffix.lower() == ".onnx" and stem.startswith("task") and len(stem) == 7 and stem[4:].isdigit()


def load_source_models(path: Path) -> list[SourceModel]:
    label = path.parent.name if path.name == "submission.zip" else path.stem
    out: list[SourceModel] = []
    if path.is_dir():
        for p in sorted(path.rglob("task*.onnx")):
            if is_task_name(p.name):
                out.append(SourceModel(p.stem, label, p.read_bytes()))
        return out
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if is_task_name(name):
                    out.append(SourceModel(Path(name).stem, label, zf.read(name)))
        return out
    if is_task_name(path.name):
        return [SourceModel(path.stem, label, path.read_bytes())]
    raise ValueError(f"unsupported source: {path}")


def unique_name(model: onnx.ModelProto, prefix: str) -> str:
    used = {init.name for init in model.graph.initializer}
    used.update(v.name for v in model.graph.input)
    used.update(v.name for v in model.graph.output)
    used.update(v.name for v in model.graph.value_info)
    for node in model.graph.node:
        used.update(x for x in node.input if x)
        used.update(x for x in node.output if x)
        if node.name:
            used.add(node.name)
    if prefix not in used:
        return prefix
    i = 1
    while f"{prefix}_{i}" in used:
        i += 1
    return f"{prefix}_{i}"


def get_attr_int(node: onnx.NodeProto, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(helper.get_attribute_value(attr))
    return None


def default_opset(model: onnx.ModelProto) -> int:
    return max((int(op.version) for op in model.opset_import if op.domain in {"", "ai.onnx"}), default=13)


def normalize_opset(model: onnx.ModelProto, min_version: int = 13) -> None:
    found = False
    for op in model.opset_import:
        if op.domain in {"", "ai.onnx"}:
            found = True
            if op.version < min_version:
                op.version = min_version
    if not found:
        model.opset_import.append(helper.make_opsetid("", min_version))


def infer_shapes_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        t = value.type.tensor_type
        if not t.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in t.shape.dim:
            if not dim.HasField("dim_value"):
                ok = False
                break
            dims.append(int(dim.dim_value))
        if ok:
            shapes[value.name] = dims
    for init in inferred.graph.initializer:
        shapes[init.name] = [int(d) for d in init.dims]
    return shapes


def transform_model(raw: bytes) -> tuple[bytes, int, str]:
    model = onnx.load_from_string(raw)
    normalize_opset(model, 13)
    shapes = infer_shapes_map(model)
    new_nodes: list[onnx.NodeProto] = []
    added_initializers: list[onnx.TensorProto] = []
    changed = 0
    skipped: list[str] = []

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Compress":
            new_nodes.append(node)
            continue
        if len(node.input) < 2 or not node.output:
            skipped.append(f"{node.name or node_index}:bad_arity")
            new_nodes.append(node)
            continue

        data_name, cond_name = node.input[0], node.input[1]
        out_name = node.output[0]
        axis = get_attr_int(node, "axis")
        prefix = unique_name(model, f"cg_{changed}")

        cond_i64 = f"{prefix}_cond_i64"
        idx = f"{prefix}_idx"
        new_nodes.append(
            helper.make_node(
                "Cast",
                [cond_name],
                [cond_i64],
                name=f"{prefix}_condition_to_i64",
                to=TensorProto.INT64,
            )
        )
        new_nodes.append(
            helper.make_node(
                "ArgMax",
                [cond_i64],
                [idx],
                name=f"{prefix}_argmax",
                axis=0,
                keepdims=1,
            )
        )

        if axis is None:
            data_shape = shapes.get(data_name)
            if not data_shape:
                skipped.append(f"{node.name or node_index}:no_axis_unknown_shape")
                # Keep the original node after the added nodes; the candidate
                # will be rejected as banned, but we preserve graph validity.
                new_nodes.append(node)
                continue
            total = 1
            for dim in data_shape:
                total *= dim
            shape_name = f"{prefix}_flat_shape"
            flat = f"{prefix}_flat_data"
            added_initializers.append(numpy_helper.from_array(__import__("numpy").array([total], dtype="int64"), shape_name))
            new_nodes.append(helper.make_node("Reshape", [data_name, shape_name], [flat], name=f"{prefix}_flatten_data"))
            new_nodes.append(helper.make_node("Gather", [flat, idx], [out_name], name=f"{prefix}_gather_flat", axis=0))
        else:
            new_nodes.append(helper.make_node("Gather", [data_name, idx], [out_name], name=f"{prefix}_gather_axis{axis}", axis=axis))
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        model.graph.initializer.extend(added_initializers)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        return model.SerializeToString(), changed, f"checker:{str(exc)[:160]}"
    return model.SerializeToString(), changed, ";".join(skipped)


def write_zip(models: dict[str, bytes], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task, raw in sorted(models.items()):
            zf.writestr(f"{task}.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/compress_to_gather_v118"))
    parser.add_argument("--min-points", type=float, default=0.0, help="Keep only transformed models scoring at least this many points.")
    parser.add_argument("--tasks", nargs="*", default=[])
    args = parser.parse_args()

    wanted = {t if t.startswith("task") else f"task{int(t):03d}" for t in args.tasks}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    models: dict[str, bytes] = {}
    best_cost: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    for source in args.source:
        for sm in load_source_models(source):
            if wanted and sm.task not in wanted:
                continue
            key = (sm.task, hashlib.sha256(sm.raw).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            row: dict[str, object] = {
                "task": sm.task,
                "source": sm.source,
                "input_sha256": key[1],
                "changed": 0,
                "ok": 0,
                "reason": "",
                "points": "",
                "cost": "",
                "macs": "",
                "memory": "",
                "params": "",
                "file_bytes": "",
            }
            try:
                raw, changed, note = transform_model(sm.raw)
                row["changed"] = changed
                row["note"] = note
                if changed == 0:
                    row["reason"] = "no_compress"
                    rows.append(row)
                    continue
                model = onnx.load_from_string(raw)
                ops = {node.op_type.upper() for node in model.graph.node}
                row["ops"] = ";".join(sorted({node.op_type for node in model.graph.node}))
                if ops & BANNED_OPS:
                    row["reason"] = "still_banned:" + ",".join(sorted(ops & BANNED_OPS))
                    rows.append(row)
                    continue
                examples = [
                    example
                    for example in all_examples(json.loads((args.data_root / f"{sm.task}.json").read_text()))
                    if example_fits_30(example)
                ]
                right, wrong, verify_reason = verify_model_bytes(raw, examples)
                row.update({"right": right, "wrong": wrong, "examples": len(examples)})
                if wrong:
                    row["reason"] = verify_reason or "verify_failed"
                    rows.append(row)
                    continue
                out_path = work / f"{sm.source}_{sm.task}_{key[1][:10]}.onnx"
                out_path.write_bytes(raw)
                macs, memory, params, cost = score_network(out_path)
                points = task_points(cost)
                row.update(
                    {
                        "ok": 1,
                        "reason": "ok",
                        "macs": macs,
                        "memory": memory,
                        "params": params,
                        "cost": cost,
                        "points": f"{points:.6f}",
                        "file_bytes": len(raw),
                    }
                )
                if points >= args.min_points and (sm.task not in best_cost or cost < best_cost[sm.task]):
                    best_cost[sm.task] = cost
                    models[sm.task] = raw
            except Exception as exc:  # noqa: BLE001
                row["reason"] = f"error:{type(exc).__name__}:{str(exc)[:220]}"
            rows.append(row)
            if row.get("ok"):
                print(f"{sm.task} {sm.source}: ok cost={row['cost']} points={row['points']} changed={row['changed']}")

    rows.sort(key=lambda r: (str(r.get("task", "")), int(r.get("cost") or 10**18), str(r.get("source", ""))))
    report = args.out_dir / "report.csv"
    if rows:
        fields = sorted({k for row in rows for k in row})
        with report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        report.write_text("", encoding="utf-8")
    zip_path = args.out_dir / "submission.zip"
    write_zip(models, zip_path)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"audited transformed models: {len(rows)}")
    print(f"valid transformed models: {ok}")
    print(f"kept best-per-task models: {len(models)}")
    print(f"zip: {zip_path}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
