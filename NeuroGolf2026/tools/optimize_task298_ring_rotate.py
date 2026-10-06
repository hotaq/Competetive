#!/usr/bin/env python3
"""Build a compact clean ONNX model for NeuroGolf task298.

The task contains nested square rings of size 6x6 or 8x8.  Reading the three
meaningful colors from outer to inner as [a, b, c], the output paints rings as
[c, a, b] (and the 8x8 center as c).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
SMALL = 8


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def ring_mask(size: int, level: int) -> np.ndarray:
    mask = np.zeros((1, 1, SMALL, SMALL), dtype=np.float32)
    lo = level
    hi = size - 1 - level
    for r in range(lo, hi + 1):
        for c in range(lo, hi + 1):
            if r in (lo, hi) or c in (lo, hi):
                mask[0, 0, r, c] = 1.0
    return mask


def center_mask8() -> np.ndarray:
    mask = np.zeros((1, 1, SMALL, SMALL), dtype=np.float32)
    mask[:, :, 3:5, 3:5] = 1.0
    return mask


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("crop_starts", [0, 0, 0, 0]),
        const_i64("crop_ends", [1, 10, SMALL, SMALL]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axis_width", [3]),
        const_i64("axis_height_width", [2, 3]),
        const_f32("zero_scalar", 0.0),
        const_f32("six", 6.0),
        const_f32("eight", 8.0),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 22, 22]),
    ]
    for size in (6, 8):
        for level in (0, 1, 2):
            inits.append(const_f32(f"mask_{size}_{level}", ring_mask(size, level)))
    inits.append(const_f32("mask_8_center", center_mask8()))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["small"], name="slice_small"),
        helper.make_node("ReduceSum", ["small", "axis_channel"], ["valid"], keepdims=1, name="valid_mask"),
        helper.make_node("ReduceMax", ["valid"], ["valid_rows"], axes=[3], keepdims=1, name="valid_rows"),
        helper.make_node("ReduceSum", ["valid_rows", "axis_height_width"], ["height"], keepdims=1, name="grid_height"),
        helper.make_node("Equal", ["height", "six"], ["is6_bool"], name="is6_bool"),
        helper.make_node("Equal", ["height", "eight"], ["is8_bool"], name="is8_bool"),
        helper.make_node("Cast", ["is6_bool"], ["is6"], to=TensorProto.FLOAT, name="is6_cast"),
        helper.make_node("Cast", ["is8_bool"], ["is8"], to=TensorProto.FLOAT, name="is8_cast"),
    ]

    size_terms: list[str] = []
    for size in (6, 8):
        color_names: list[str] = []
        for level in (0, 1, 2):
            masked = f"masked_{size}_{level}"
            color = f"color_{size}_{level}"
            nodes.extend(
                [
                    helper.make_node("Mul", ["small", f"mask_{size}_{level}"], [masked], name=masked),
                    helper.make_node("ReduceMax", [masked], [color], axes=[2, 3], keepdims=1, name=color),
                ]
            )
            color_names.append(color)

        # Output ring colors: [c, a, b], plus c again in the 8x8 center.
        paint_specs = [
            (color_names[2], f"mask_{size}_0", f"paint_{size}_0"),
            (color_names[0], f"mask_{size}_1", f"paint_{size}_1"),
            (color_names[1], f"mask_{size}_2", f"paint_{size}_2"),
        ]
        if size == 8:
            paint_specs.append((color_names[2], "mask_8_center", "paint_8_center"))
        paints: list[str] = []
        for color, mask, out in paint_specs:
            nodes.append(helper.make_node("Mul", [color, mask], [out], name=out))
            paints.append(out)
        nodes.append(helper.make_node("Sum", paints, [f"out_{size}_raw"], name=f"sum_out_{size}"))
        nodes.append(helper.make_node("Mul", [f"out_{size}_raw", f"is{size}"], [f"out_{size}"], name=f"gate_out_{size}"))
        size_terms.append(f"out_{size}")

    nodes.extend(
        [
            helper.make_node("Sum", size_terms, ["out8"], name="sum_size_outputs"),
            helper.make_node("Pad", ["out8", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task298_ring_rotate",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task298_ring_rotate",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task298_ring_rotate_v104"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task298.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task298 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task298.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task298.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task298 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
