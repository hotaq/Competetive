#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task168.

The task contains one-color 2x2 L shapes.  For each L, the missing corner is
extended outward along the diagonal from the opposite filled corner.
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
SMALL_SHAPE = [1, 9, 10, 10]
DIRECTIONS = ("n", "s", "e", "w", "nw", "ne", "sw", "se")


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


def shift_spec(direction: str, k: int) -> tuple[list[int], list[int], list[int]]:
    if direction == "n":
        return [0, 0, k, 0], [1, 9, 10, 10], [0, 0, 0, 0, 0, 0, k, 0]
    if direction == "s":
        return [0, 0, 0, 0], [1, 9, 10 - k, 10], [0, 0, k, 0, 0, 0, 0, 0]
    if direction == "e":
        return [0, 0, 0, 0], [1, 9, 10, 10 - k], [0, 0, 0, k, 0, 0, 0, 0]
    if direction == "w":
        return [0, 0, 0, k], [1, 9, 10, 10], [0, 0, 0, 0, 0, 0, 0, k]
    if direction == "ne":
        return [0, 0, k, 0], [1, 9, 10, 10 - k], [0, 0, 0, k, 0, 0, k, 0]
    if direction == "nw":
        return [0, 0, k, k], [1, 9, 10, 10], [0, 0, 0, 0, 0, 0, k, k]
    if direction == "se":
        return [0, 0, 0, 0], [1, 9, 10 - k, 10 - k], [0, 0, k, k, 0, 0, 0, 0]
    if direction == "sw":
        return [0, 0, 0, k], [1, 9, 10 - k, 10], [0, 0, k, 0, 0, 0, 0, k]
    raise ValueError(direction)


def shift_nodes(src: str, dst: str, direction: str, k: int) -> list[onnx.NodeProto]:
    prefix = f"shift_{direction}_{k}"
    return [
        make_slice(src, f"{dst}_part", prefix),
        helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
    ]


def mul_chain(nodes: list[onnx.NodeProto], names: list[str], out: str, prefix: str) -> None:
    cur = names[0]
    for idx, name in enumerate(names[1:], start=1):
        nxt = out if idx == len(names) - 1 else f"{prefix}_mul{idx}"
        nodes.append(helper.make_node("Mul", [cur, name], [nxt], name=f"{prefix}_mul_{idx}"))
        cur = nxt


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones10", np.ones((1, 1, 10, 10), dtype=np.float32)),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        *slice_4d("small_color", [0, 1, 0, 0], [1, 10, 10, 10]),
    ]
    for direction in DIRECTIONS:
        for k in range(1, 10):
            starts, ends, pads = shift_spec(direction, k)
            inits.extend(slice_4d(f"shift_{direction}_{k}", starts, ends))
            inits.append(const_i64(f"shift_{direction}_{k}_pad", pads))

    nodes: list[onnx.NodeProto] = [make_slice("input", "small", "small_color")]

    # Missing corner -> detection sample directions -> ray direction.
    patterns = [
        ("tl", ("w", "n", "nw"), "nw"),
        ("tr", ("e", "n", "ne"), "ne"),
        ("bl", ("w", "s", "sw"), "sw"),
        ("br", ("e", "s", "se"), "se"),
    ]

    ray_terms = ["small"]
    for corner, sample_dirs, ray_dir in patterns:
        samples: list[str] = []
        for sample_dir in sample_dirs:
            sample = f"{corner}_sample_{sample_dir}"
            nodes.extend(shift_nodes("small", sample, sample_dir, 1))
            samples.append(sample)
        anchor = f"{corner}_anchor"
        mul_chain(nodes, samples, anchor, anchor)
        for k in range(1, 10):
            shifted = f"{corner}_{ray_dir}_{k}"
            nodes.extend(shift_nodes(anchor, shifted, ray_dir, k))
            ray_terms.append(shifted)

    nodes.extend(
        [
            helper.make_node("Sum", ray_terms, ["raw_colors"], name="sum_input_and_rays"),
            helper.make_node("Greater", ["raw_colors", "half"], ["colors_bool"], name="colors_bool"),
            helper.make_node("Cast", ["colors_bool"], ["colors"], to=TensorProto.FLOAT, name="colors_cast"),
            helper.make_node("ReduceSum", ["colors", "axis_channel"], ["occupied_count"], keepdims=1, name="occupied_count"),
            helper.make_node("Greater", ["occupied_count", "half"], ["occupied_bool"], name="occupied_bool"),
            helper.make_node("Cast", ["occupied_bool"], ["occupied"], to=TensorProto.FLOAT, name="occupied_cast"),
            helper.make_node("Sub", ["ones10", "occupied"], ["ch0"], name="background_channel"),
            helper.make_node("Concat", ["ch0", "colors"], ["small_out"], axis=1, name="assemble_10x10"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task168_l_diagonal_rays",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task168_l_diagonal_rays",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task168_l_diagonal_rays_v94"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task168.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task168 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task168.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task168.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task168 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
