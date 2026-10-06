#!/usr/bin/env python3
"""Audit NeuroGolf ONNX zips for scorer-loophole risk markers.

This is defensive tooling: it does not build or modify models. It reports
features that current discussion/organizer notes identify as unstable or
explicitly disallowed, such as model-local functions and custom domains.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import onnx


def audit_model(raw: bytes, name: str, source: Path) -> dict[str, str]:
    row: dict[str, str] = {
        "source": str(source),
        "model": name,
        "ok": "0",
        "error": "",
        "function_count": "",
        "custom_domain_node_count": "",
        "io_initializer_collision_count": "",
        "custom_domains": "",
        "ops": "",
    }
    try:
        model = onnx.load_model_from_string(raw)
    except Exception as exc:  # noqa: BLE001
        row["error"] = type(exc).__name__ + ": " + str(exc)
        return row

    custom_domains = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    ops = sorted({(node.domain or "ai.onnx") + "::" + node.op_type for node in model.graph.node})
    io_names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    collisions = sorted(io_names & initializer_names)

    row.update(
        {
            "ok": "1",
            "function_count": str(len(model.functions)),
            "custom_domain_node_count": str(sum(node.domain not in ("", "ai.onnx") for node in model.graph.node)),
            "io_initializer_collision_count": str(len(collisions)),
            "custom_domains": ";".join(custom_domains),
            "ops": ";".join(ops),
        }
    )
    return row


def audit_path(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".onnx":
        return [audit_model(path.read_bytes(), path.name, path)]
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".onnx"):
                rows.append(audit_model(zf.read(name), name, path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.paths:
        rows.extend(audit_path(path))

    fields = [
        "source",
        "model",
        "ok",
        "error",
        "function_count",
        "custom_domain_node_count",
        "io_initializer_collision_count",
        "custom_domains",
        "ops",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    text = out.getvalue()

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        args.csv_out.write_text(text, encoding="utf-8")

    risk_rows = [
        row
        for row in rows
        if row["ok"] != "1"
        or int(row["function_count"] or 0)
        or int(row["custom_domain_node_count"] or 0)
        or int(row["io_initializer_collision_count"] or 0)
    ]
    print(f"models: {len(rows)}")
    print(f"risk rows: {len(risk_rows)}")
    print(f"with functions: {sum(int(row['function_count'] or 0) > 0 for row in rows)}")
    print(f"with custom-domain nodes: {sum(int(row['custom_domain_node_count'] or 0) > 0 for row in rows)}")
    print(f"with IO/initializer collisions: {sum(int(row['io_initializer_collision_count'] or 0) > 0 for row in rows)}")
    if args.csv_out:
        print(f"csv: {args.csv_out}")
    for row in risk_rows[:20]:
        print(
            row["model"],
            "functions=" + row["function_count"],
            "custom_nodes=" + row["custom_domain_node_count"],
            "collisions=" + row["io_initializer_collision_count"],
            "source=" + row["source"],
        )


if __name__ == "__main__":
    main()
