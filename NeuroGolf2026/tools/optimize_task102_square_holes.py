#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task102.

The task draws square frames in color 5.  Only frames whose missing interior is
a solid square block are repaired, and the repaired cells become color 2.
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
TASK_H = 12
TASK_W = 12


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_init(prefix: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{prefix}_starts", starts), const_i64(f"{prefix}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def square_hole_patterns(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Conv weights, thresholds, and ConvTranspose paint kernels."""
    specs: list[tuple[int, int, int]] = []
    for block in (size - 3, size - 2):
        for row in range(1, size - block):
            for col in range(1, size - block):
                specs.append((block, row, col))

    conv = np.zeros((len(specs), 1, size, size), dtype=np.float32)
    paint = np.zeros((len(specs), 1, size, size), dtype=np.float32)
    thresholds = np.zeros((1, len(specs), 1, 1), dtype=np.float32)
    for i, (block, row, col) in enumerate(specs):
        required = np.ones((size, size), dtype=np.float32)
        required[row : row + block, col : col + block] = 0
        fill = 1 - required
        # Large negative weights ensure any existing color-5 cell in the hole
        # makes the pattern miss without needing a separate zero test.
        conv[i, 0] = required - (fill * 100.0)
        paint[i, 0] = fill
        thresholds[0, i, 0, 0] = required.sum() - 0.5
    return conv, thresholds, paint


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - TASK_H, 30 - TASK_W]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        *slice_init("ch0_full", [0, 0, 0, 0], [1, 1, 30, 30]),
        *slice_init("ch5_full", [0, 5, 0, 0], [1, 6, 30, 30]),
        *slice_init("ch5_crop", [0, 5, 0, 0], [1, 6, TASK_H, TASK_W]),
    ]

    paint_terms: list[str] = []
    for size in (4, 5, 6):
        conv, thresholds, paint = square_hole_patterns(size)
        inits.extend(
            [
                const_f32(f"match_w{size}", conv),
                const_f32(f"match_t{size}", thresholds),
                const_f32(f"paint_w{size}", paint),
            ]
        )
        paint_terms.append(f"paint{size}")

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "ch0_in", "ch0_full"),
        make_slice("input", "ch5_in", "ch5_full"),
        make_slice("input", "ch5_crop", "ch5_crop"),
    ]

    for size in (4, 5, 6):
        nodes.extend(
            [
                helper.make_node("Conv", ["ch5_crop", f"match_w{size}"], [f"match_score{size}"], name=f"match_score{size}"),
                helper.make_node("Greater", [f"match_score{size}", f"match_t{size}"], [f"hit_bool{size}"], name=f"hit_bool{size}"),
                helper.make_node("Cast", [f"hit_bool{size}"], [f"hit{size}"], to=TensorProto.FLOAT, name=f"hit_cast{size}"),
                helper.make_node("ConvTranspose", [f"hit{size}", f"paint_w{size}"], [f"paint{size}"], name=f"paint{size}"),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Sum", paint_terms, ["paint12_raw"], name="sum_paint12"),
            helper.make_node("Greater", ["paint12_raw", "half"], ["paint12_bool"], name="paint12_bool"),
            helper.make_node("Cast", ["paint12_bool"], ["paint12"], to=TensorProto.FLOAT, name="paint12_cast"),
            helper.make_node("Pad", ["paint12", "pad_to_30", "zero_scalar"], ["paint"], name="paint_to_30"),
            helper.make_node("Sub", ["ch0_in", "paint"], ["ch0_out"], name="background_out"),
            helper.make_node("Sub", ["paint", "paint"], ["zero"], name="zero_channel"),
        ]
    )
    zero_names = [f"z{i}" for i in range(7)]
    for zero_name in zero_names:
        nodes.append(helper.make_node("Identity", ["zero"], [zero_name], name=zero_name))
    nodes.append(
        helper.make_node(
            "Concat",
            ["ch0_out", "z0", "paint", "z1", "z2", "ch5_in", "z3", "z4", "z5", "z6"],
            ["output"],
            axis=1,
            name="assemble_output",
        )
    )

    graph = helper.make_graph(
        nodes,
        "task102_square_holes",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task102_square_holes",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task102_square_holes_v97"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task102.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task102 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task102.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task102.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task102 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
