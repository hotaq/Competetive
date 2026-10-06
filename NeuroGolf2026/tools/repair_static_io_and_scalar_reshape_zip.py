#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx import numpy_helper
import numpy as np


EXPECTED_SHAPE = [1, 10, 30, 30]


def is_task_file(name: str) -> bool:
    path = Path(name)
    return path.suffix.lower() == ".onnx" and path.stem.startswith("task") and path.stem[4:].isdigit()


def tensor_shape(model: onnx.ModelProto, name: str) -> list[int | None] | None:
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name != name or not item.type.HasField("tensor_type"):
            continue
        dims: list[int | None] = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
        return dims
    for init in model.graph.initializer:
        if init.name == name:
            return [int(dim) for dim in init.dims]
    return None


def set_static_io(model: onnx.ModelProto) -> int:
    changed = 0
    for value in list(model.graph.input) + list(model.graph.output):
        old = [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]
        if old == EXPECTED_SHAPE:
            continue
        value.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for dim_value in EXPECTED_SHAPE:
            dims.add().dim_value = dim_value
        changed += 1
    return changed


def repair_scalar_reshapes(model: onnx.ModelProto) -> int:
    empty_shape_names = {
        init.name
        for init in model.graph.initializer
        if list(init.dims) == [0] and len(init.raw_data) == 0
    }
    if not empty_shape_names:
        return 0

    opset = 0
    for opset_import in model.opset_import:
        if opset_import.domain in ("", "ai.onnx"):
            opset = int(opset_import.version)
            break

    repaired = 0
    for index, node in enumerate(list(model.graph.node)):
        if node.op_type != "Reshape" or len(node.input) < 2 or node.input[1] not in empty_shape_names:
            continue
        input_shape = tensor_shape(model, node.input[0])
        output_shape = tensor_shape(model, node.output[0]) if node.output else None
        if not input_shape or any(dim != 1 for dim in input_shape):
            continue
        if output_shape not in ([], None):
            continue
        axes = list(range(len(input_shape)))
        if opset >= 13:
            axes_name = f"{node.output[0]}_axes"
            model.graph.initializer.append(numpy_helper.from_array(np.array(axes, dtype=np.int64), axes_name))
            squeeze = helper.make_node(
                "Squeeze",
                [node.input[0], axes_name],
                list(node.output),
                name=node.name or (node.output[0] if node.output else ""),
            )
        else:
            squeeze = helper.make_node(
                "Squeeze",
                [node.input[0]],
                list(node.output),
                name=node.name or (node.output[0] if node.output else ""),
                axes=axes,
            )
        model.graph.node.remove(node)
        model.graph.node.insert(index, squeeze)
        repaired += 1

    if repaired:
        consumed = {inp for node in model.graph.node for inp in node.input if inp}
        kept = [init for init in model.graph.initializer if init.name in consumed]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return repaired


def repair_model(raw: bytes) -> tuple[bytes, dict[str, int]]:
    model = onnx.load_from_string(raw)
    stats = {
        "static_io": set_static_io(model),
        "scalar_reshape": repair_scalar_reshapes(model),
    }
    if not any(stats.values()):
        return raw, stats
    onnx.checker.check_model(model)
    return model.SerializeToString(), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair dynamic IO and zero-length scalar reshape patterns in ONNX zip.")
    parser.add_argument("--input-zip", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    args = parser.parse_args()

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    totals = {"models": 0, "static_io": 0, "scalar_reshape": 0, "changed": 0}
    with zipfile.ZipFile(args.input_zip, "r") as zin, zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for name in sorted(zin.namelist()):
            raw = zin.read(name)
            if is_task_file(name):
                raw, stats = repair_model(raw)
                totals["models"] += 1
                totals["static_io"] += stats["static_io"]
                totals["scalar_reshape"] += stats["scalar_reshape"]
                if any(stats.values()):
                    totals["changed"] += 1
            zout.writestr(Path(name).name if is_task_file(name) else name, raw)
    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"output_zip: {args.output_zip}")


if __name__ == "__main__":
    main()
