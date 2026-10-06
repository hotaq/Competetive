#!/usr/bin/env python3
"""Inline ONNX FunctionProto bodies into ordinary graph nodes.

Some public NeuroGolf models wrap a standard ONNX subgraph in a custom-domain
FunctionProto call.  The competition now rejects FunctionProto/custom domains,
but the body can still be evaluated honestly if expanded into normal nodes.
This tool performs that expansion without trying to optimize the result.
"""

from __future__ import annotations

import argparse
import zipfile
from copy import deepcopy
from pathlib import Path

import onnx


ALLOWED_DOMAINS = {"", "ai.onnx"}


def _unique(name: str, prefix: str) -> str:
    return f"{prefix}__{name}" if name else name


def _inline_function_node(
    node: onnx.NodeProto,
    function: onnx.FunctionProto,
    call_index: int,
) -> list[onnx.NodeProto]:
    prefix = f"inlined_{call_index}_{node.name or function.name}"
    mapping: dict[str, str] = {}
    for formal, actual in zip(function.input, node.input):
        mapping[formal] = actual
    for formal, actual in zip(function.output, node.output):
        mapping[formal] = actual

    expanded: list[onnx.NodeProto] = []
    for inner_index, inner in enumerate(function.node):
        new_node = deepcopy(inner)
        # Function bodies from public notebooks frequently reuse generic node
        # names such as "Slice" multiple times inside the same FunctionProto.
        # ONNXRuntime rejects duplicate node names after inlining, so include
        # the inner-node index in the generated name.
        inner_name = new_node.name or inner.op_type
        new_node.name = _unique(f"{inner_index}_{inner_name}", prefix)
        for idx, name in enumerate(new_node.input):
            if name:
                new_node.input[idx] = mapping.get(name, _unique(name, prefix))
        for idx, name in enumerate(new_node.output):
            if name:
                mapped = mapping.get(name)
                if mapped is None:
                    mapped = _unique(name, prefix)
                    mapping[name] = mapped
                new_node.output[idx] = mapped
        expanded.append(new_node)
    return expanded


def inline_functions(raw: bytes) -> bytes:
    model = onnx.load_from_string(raw)
    functions = {(fn.domain, fn.name): fn for fn in model.functions}
    if not functions:
        return raw

    new_nodes: list[onnx.NodeProto] = []
    changed = False
    for idx, node in enumerate(model.graph.node):
        function = functions.get((node.domain, node.op_type))
        if function is None:
            new_nodes.append(deepcopy(node))
            continue
        new_nodes.extend(_inline_function_node(node, function, idx))
        changed = True

    if not changed:
        return raw

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.functions[:]

    used_domains = {node.domain for node in model.graph.node}
    merged_opsets: dict[str, int] = {}
    for opset in model.opset_import:
        if opset.domain in ALLOWED_DOMAINS or opset.domain in used_domains:
            merged_opsets[opset.domain] = max(merged_opsets.get(opset.domain, 0), opset.version)
    for function in functions.values():
        for opset in function.opset_import:
            if opset.domain in ALLOWED_DOMAINS or opset.domain in used_domains:
                merged_opsets[opset.domain] = max(merged_opsets.get(opset.domain, 0), opset.version)

    del model.opset_import[:]
    for domain, version in sorted(merged_opsets.items()):
        if domain in ALLOWED_DOMAINS or domain in used_domains:
            model.opset_import.add(domain=domain, version=version)

    onnx.checker.check_model(model)
    return model.SerializeToString()


def is_task_onnx(name: str) -> bool:
    path = Path(name)
    return path.suffix == ".onnx" and path.stem.startswith("task") and path.stem[4:].isdigit()


def convert_path(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix == ".zip":
        with zipfile.ZipFile(input_path) as src, zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for name in src.namelist():
                raw = src.read(name)
                if is_task_onnx(name):
                    raw = inline_functions(raw)
                dst.writestr(Path(name).name, raw)
        return
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for path in sorted(input_path.rglob("task*.onnx")):
            (output_path / path.name).write_bytes(inline_functions(path.read_bytes()))
        return
    output_path.write_bytes(inline_functions(input_path.read_bytes()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_path(args.input, args.output)
    print(f"inlined: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
