#!/usr/bin/env python3
"""Strip dead ONNX graph payload from a NeuroGolf submission zip.

This is intentionally conservative:
- keep only nodes reachable from graph outputs;
- keep only initializers consumed by the kept nodes;
- keep value_info only for kept node outputs, preserving static-shape patches;
- do not fold constants or rewrite math.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import onnx


def is_task_file(name: str) -> bool:
    path = Path(name)
    return (
        path.suffix.lower() == ".onnx"
        and path.stem.startswith("task")
        and len(path.stem) == 7
        and path.stem[4:].isdigit()
    )


def strip_model(raw: bytes) -> tuple[bytes, dict[str, int]]:
    model = onnx.load_from_string(raw)
    graph = model.graph

    producer: dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for output in node.output:
            if output:
                producer[output] = i

    needed_nodes: set[int] = set()
    stack = [output.name for output in graph.output if output.name]
    while stack:
        name = stack.pop()
        node_index = producer.get(name)
        if node_index is None or node_index in needed_nodes:
            continue
        needed_nodes.add(node_index)
        stack.extend(inp for inp in graph.node[node_index].input if inp)

    old_nodes = len(graph.node)
    if len(needed_nodes) != old_nodes:
        kept_nodes = [node for i, node in enumerate(graph.node) if i in needed_nodes]
        del graph.node[:]
        graph.node.extend(kept_nodes)

    kept_node_outputs = {output for node in graph.node for output in node.output if output}
    consumed = {inp for node in graph.node for inp in node.input if inp}

    old_initializers = len(graph.initializer)
    kept_initializers = [tensor for tensor in graph.initializer if tensor.name in consumed]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    old_sparse_initializers = len(graph.sparse_initializer)
    kept_sparse = [
        tensor
        for tensor in graph.sparse_initializer
        if tensor.values.name in consumed or tensor.indices.name in consumed
    ]
    del graph.sparse_initializer[:]
    graph.sparse_initializer.extend(kept_sparse)

    keep_value_info = kept_node_outputs | {item.name for item in graph.input} | {item.name for item in graph.output}
    old_value_info = len(graph.value_info)
    kept_value_info = [item for item in graph.value_info if item.name in keep_value_info]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)

    stats = {
        "nodes_removed": old_nodes - len(graph.node),
        "initializers_removed": old_initializers - len(graph.initializer),
        "sparse_initializers_removed": old_sparse_initializers - len(graph.sparse_initializer),
        "value_info_removed": old_value_info - len(graph.value_info),
        "bytes_delta": len(raw) - len(model.SerializeToString()),
    }
    return model.SerializeToString(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip dead graph payload from NeuroGolf ONNX zip.")
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "models": 0,
        "nodes_removed": 0,
        "initializers_removed": 0,
        "sparse_initializers_removed": 0,
        "value_info_removed": 0,
        "bytes_delta": 0,
    }
    with zipfile.ZipFile(args.input_zip) as src, zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in sorted(src.namelist()):
            data = src.read(name)
            if not is_task_file(name):
                dst.writestr(name, data)
                continue
            stripped, stats = strip_model(data)
            dst.writestr(Path(name).name, stripped)
            totals["models"] += 1
            for key, value in stats.items():
                totals[key] += value

    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"input_zip: {args.input_zip}")
    print(f"output_zip: {args.output_zip}")


if __name__ == "__main__":
    main()
