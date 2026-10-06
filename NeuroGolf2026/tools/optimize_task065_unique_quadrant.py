#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task065."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def make_size_gate(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], side: int) -> str:
    prefix = f"s{side}"
    inits.extend(slice_4d(f"{prefix}_last_cell", [0, 0, side - 1, side - 1], [1, 1, side, side]))
    inits.extend(slice_4d(f"{prefix}_next_cell", [0, 0, side, side], [1, 1, side + 1, side + 1]))
    nodes.extend(
        [
            make_slice("inside", f"{prefix}_last", f"{prefix}_last_cell"),
            make_slice("inside", f"{prefix}_next", f"{prefix}_next_cell"),
            helper.make_node("Sub", ["one_scalar", f"{prefix}_next"], [f"{prefix}_not_next"], name=f"{prefix}_not_next"),
            helper.make_node("Mul", [f"{prefix}_last", f"{prefix}_not_next"], [f"{prefix}_size_gate"], name=f"{prefix}_size_gate"),
        ]
    )
    return f"{prefix}_size_gate"


def make_quadrant_nodes(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], side: int, n: int, size_gate: str) -> str:
    quads = {
        "tl": (0, 0),
        "tr": (0, n + 1),
        "bl": (n + 1, 0),
        "br": (n + 1, n + 1),
    }
    selected: list[str] = []
    for name, (r0, c0) in quads.items():
        prefix = f"s{side}_{name}"
        inits.extend(slice_4d(prefix, [0, 0, r0, c0], [1, 10, r0 + n, c0 + n]))
        nodes.append(make_slice("input", prefix, prefix))
        if n == 1:
            continue
        nodes.extend(
            [
                helper.make_node("ReduceSum", [prefix, "axes_hw"], [f"{prefix}_counts"], keepdims=1, name=f"{prefix}_counts"),
                helper.make_node("ReduceMax", [f"{prefix}_counts"], [f"{prefix}_max_count"], axes=[1], keepdims=1, name=f"{prefix}_max_count"),
                helper.make_node("Less", [f"{prefix}_max_count", f"s{side}_area_limit"], [f"{prefix}_nonuniform_bool"], name=f"{prefix}_nonuniform"),
                helper.make_node("Cast", [f"{prefix}_nonuniform_bool"], [f"{prefix}_select"], to=TensorProto.FLOAT, name=f"{prefix}_select"),
                helper.make_node("Mul", [f"{prefix}_select", size_gate], [f"{prefix}_gate"], name=f"{prefix}_gate"),
                helper.make_node("Mul", [prefix, f"{prefix}_gate"], [f"{prefix}_selected"], name=f"{prefix}_selected"),
            ]
        )
        selected.append(f"{prefix}_selected")

    if n == 1:
        quad_names = [f"s{side}_{name}" for name in quads]
        nodes.append(helper.make_node("Sum", quad_names, [f"s{side}_channel_count"], name=f"s{side}_channel_count"))
        for prefix in quad_names:
            nodes.extend(
                [
                    helper.make_node("Mul", [prefix, f"s{side}_channel_count"], [f"{prefix}_same_color_planes"], name=f"{prefix}_same_color_planes"),
                    helper.make_node("ReduceSum", [f"{prefix}_same_color_planes", "axis_channel"], [f"{prefix}_same_color_count"], keepdims=1, name=f"{prefix}_same_color_count"),
                    helper.make_node("Less", [f"{prefix}_same_color_count", "one_half"], [f"{prefix}_unique_bool"], name=f"{prefix}_unique"),
                    helper.make_node("Cast", [f"{prefix}_unique_bool"], [f"{prefix}_select"], to=TensorProto.FLOAT, name=f"{prefix}_select"),
                    helper.make_node("Mul", [f"{prefix}_select", size_gate], [f"{prefix}_gate"], name=f"{prefix}_gate"),
                    helper.make_node("Mul", [prefix, f"{prefix}_gate"], [f"{prefix}_selected"], name=f"{prefix}_selected"),
                ]
            )
            selected.append(f"{prefix}_selected")

    small = f"s{side}_small"
    padded = f"s{side}_padded"
    inits.append(const_i64(f"s{side}_pad", [0, 0, 0, 0, 0, 0, 30 - n, 30 - n]))
    if len(selected) == 1:
        nodes.append(helper.make_node("Identity", [selected[0]], [small], name=f"s{side}_small_identity"))
    else:
        nodes.append(helper.make_node("Sum", selected, [small], name=f"s{side}_small_sum"))
    nodes.append(helper.make_node("Pad", [small, f"s{side}_pad", "zero_scalar"], [padded], name=f"s{side}_pad_output"))
    return padded


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axes_hw", [2, 3]),
        const_f32("zero_scalar", 0.0),
        const_f32("one_scalar", 1.0),
        const_f32("one_half", 1.5),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["inside"], keepdims=1, name="inside_shape"),
    ]
    padded_branches: list[str] = []
    for side in (3, 5, 7, 9, 11, 13, 15):
        n = (side - 1) // 2
        if n > 1:
            inits.append(const_f32(f"s{side}_area_limit", float(n * n) - 0.5))
        size_gate = make_size_gate(nodes, inits, side)
        padded_branches.append(make_quadrant_nodes(nodes, inits, side, n, size_gate))

    nodes.append(helper.make_node("Sum", padded_branches, ["output"], name="sum_size_branches"))
    graph = helper.make_graph(
        nodes,
        "task065_unique_quadrant",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task065_unique_quadrant",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task065_unique_quadrant")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task065.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task065 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task065.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task065.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
