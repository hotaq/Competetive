#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import onnx


def initializer_key(tensor: onnx.TensorProto) -> tuple[tuple[int, ...], int, str]:
    payload = tensor.raw_data if tensor.raw_data else tensor.SerializeToString()
    return tuple(tensor.dims), tensor.data_type, hashlib.sha256(payload).hexdigest()


def rewrite_model(raw: bytes) -> tuple[bytes, int]:
    model = onnx.load_from_string(raw)
    canonical_by_key: dict[tuple[tuple[int, ...], int, str], str] = {}
    replacements: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []

    for initializer in model.graph.initializer:
        key = initializer_key(initializer)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = initializer.name
            kept.append(initializer)
            continue
        replacements[initializer.name] = canonical

    if not replacements:
        return raw, 0

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in replacements:
                node.input[i] = replacements[name]

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    return model.SerializeToString(), len(replacements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate identical ONNX initializers inside a submission zip.")
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    args = parser.parse_args()

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    total_removed = 0
    changed_tasks = 0

    with zipfile.ZipFile(args.input_zip, "r") as zin, zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            raw = zin.read(info.filename)
            if info.filename.endswith(".onnx"):
                raw, removed = rewrite_model(raw)
                if removed:
                    changed_tasks += 1
                    total_removed += removed
            zout.writestr(info.filename, raw)

    print(f"changed_tasks={changed_tasks}")
    print(f"removed_initializers={total_removed}")
    print(f"output_zip={args.output_zip}")


if __name__ == "__main__":
    main()
