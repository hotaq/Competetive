#!/usr/bin/env python3
"""
Generate static ONNX models for the current A_exact NeuroGolf triage lane.

This targets four simple families:
- color_remap
- geometric_exact
- split_binary_relation
- crop_exact_same_offset

Outputs:
- generated/a_exact_models/taskNNN.onnx
- generated/a_exact_submission.zip
- reports/a_exact_generation_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


EXPECTED_SHAPE = [1, 10, 30, 30]
CHANNELS = 10
HEIGHT = 30
WIDTH = 30


Grid = list[list[int]]
Example = dict[str, Grid]


@dataclass(frozen=True)
class TaskSpec:
    task: str
    task_num: int
    family: str
    note: str
    row: dict[str, str]


def const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def const_f32(name: str, value: np.ndarray | float | list[float]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def grid_to_tuple(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def transpose_grid(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def flip_h_grid(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def flip_v_grid(grid: Grid) -> Grid:
    return list(reversed([row[:] for row in grid]))


def rot90_grid(grid: Grid) -> Grid:
    return flip_h_grid(transpose_grid(grid))


def rot180_grid(grid: Grid) -> Grid:
    return flip_v_grid(flip_h_grid(grid))


def rot270_grid(grid: Grid) -> Grid:
    return flip_v_grid(transpose_grid(grid))


GRID_TRANSFORMS: dict[str, Callable[[Grid], Grid]] = {
    "rot90": rot90_grid,
    "rot180": rot180_grid,
    "rot270": rot270_grid,
    "flip_h": flip_h_grid,
    "flip_v": flip_v_grid,
    "transpose": transpose_grid,
}


def all_examples(task: dict[str, list[Example]]) -> list[Example]:
    return task["train"] + task["test"] + task["arc-gen"]


def parse_color_map(note: str) -> dict[int, int]:
    match = re.search(r"color_map=([^|]+)", note)
    if not match:
        raise ValueError(f"missing color_map note: {note}")
    mapping: dict[int, int] = {}
    for item in match.group(1).strip().split(","):
        src, dst = item.split("->")
        mapping[int(src)] = int(dst)
    return mapping


def parse_exact_transform(note: str) -> str:
    match = re.search(r"exact_transform=([^| ]+)", note)
    if not match:
        raise ValueError(f"missing exact_transform note: {note}")
    transform = match.group(1)
    if transform not in GRID_TRANSFORMS:
        raise ValueError(f"unsupported transform: {transform}")
    return transform


def parse_split_relation(note: str) -> tuple[str, int, str, int, int, int, int]:
    pattern = r"split_(vertical|horizontal)_sep(\d+)_(and|or|xor)_(\d+)_(\d+)_to_(\d+)_else_(\d+)"
    match = re.search(pattern, note)
    if not match:
        raise ValueError(f"missing split relation note: {note}")
    orientation, sep, mode, a, b, out_color, else_color = match.groups()
    return orientation, int(sep), mode, int(a), int(b), int(out_color), int(else_color)


def infer_common_crop_offset(examples: list[Example]) -> tuple[int, int]:
    common: set[tuple[int, int]] | None = None
    for ex in examples:
        inp = ex["input"]
        out = ex["output"]
        in_h, in_w = shape(inp)
        out_h, out_w = shape(out)
        target = grid_to_tuple(out)
        positions: set[tuple[int, int]] = set()
        for r in range(in_h - out_h + 1):
            for c in range(in_w - out_w + 1):
                candidate = tuple(tuple(inp[r + rr][c : c + out_w]) for rr in range(out_h))
                if candidate == target:
                    positions.add((r, c))
        common = positions if common is None else common & positions
    if not common:
        raise ValueError("no common crop offset")
    return sorted(common)[0]


def make_slice(name: str, x: str, y: str, starts: str, ends: str, axes: str = "axes4", steps: str = "steps4") -> onnx.NodeProto:
    return helper.make_node("Slice", [x, starts, ends, axes, steps], [y], name=name)


def make_pad(name: str, x: str, y: str, pads: str = "pad_to_30", value: str = "zero_scalar") -> onnx.NodeProto:
    return helper.make_node("Pad", [x, pads, value], [y], name=name)


def make_gather(name: str, x: str, y: str, indices: str, axis: int) -> onnx.NodeProto:
    return helper.make_node("Gather", [x, indices], [y], axis=axis, name=name)


def make_unsqueeze_scalar(name: str, x: str, y: str) -> list[onnx.NodeProto]:
    return [
        helper.make_node("Unsqueeze", [x, "scalar_axes"], [y], name=name),
    ]


def base_initializers() -> list[onnx.TensorProto]:
    return [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("axis_channel", [1]),
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("scalar_axes", [0, 1, 2, 3]),
        const_f32("zero_scalar", 0.0),
        const_f32("one_scalar", 1.0),
    ]


def make_model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto], name: str) -> onnx.ModelProto:
    used_initializer_names = {name for node in nodes for name in node.input if name}
    filtered_initializers = [
        initializer
        for initializer in base_initializers() + initializers
        if initializer.name in used_initializer_names
    ]
    graph = helper.make_graph(
        nodes,
        name,
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=filtered_initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_a_exact_generator",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def output_size_pairs(examples: list[Example]) -> list[tuple[int, int]]:
    return sorted(set(shape(ex["output"]) for ex in examples))


def input_size_pairs(examples: list[Example]) -> list[tuple[int, int]]:
    return sorted(set(shape(ex["input"]) for ex in examples))


def size_mask_initializer(name: str, h: int, w: int) -> onnx.TensorProto:
    mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask[:, :, :h, :w] = 1.0
    return const_f32(name, mask)


def make_size_mask_nodes(prefix: str, h: int, w: int, mask_name: str) -> list[onnx.NodeProto]:
    return [
        helper.make_node("ReduceSum", ["input", "axis_channel"], [f"{prefix}_occ"], keepdims=1, name=f"{prefix}_occ"),
        helper.make_node("Sub", [f"{prefix}_occ", mask_name], [f"{prefix}_diff"], name=f"{prefix}_diff"),
        helper.make_node("Abs", [f"{prefix}_diff"], [f"{prefix}_abs"], name=f"{prefix}_abs"),
        helper.make_node("ReduceSum", [f"{prefix}_abs", "axes_all"], [f"{prefix}_err"], keepdims=0, name=f"{prefix}_err"),
        helper.make_node("Equal", [f"{prefix}_err", "zero_scalar"], [f"{prefix}_is_size_bool"], name=f"{prefix}_is_size_bool"),
        helper.make_node("Cast", [f"{prefix}_is_size_bool"], [f"{prefix}_is_size"], to=TensorProto.FLOAT, name=f"{prefix}_is_size"),
        *make_unsqueeze_scalar(f"{prefix}_unsqueeze_mask", f"{prefix}_is_size", f"{prefix}_is_size_4d"),
    ]


def pad_for_shape(name: str, h: int, w: int) -> onnx.TensorProto:
    return const_i64(name, [0, 0, 0, 0, 0, 0, HEIGHT - h, WIDTH - w])


def slice_initializers(prefix: str, h0: int, w0: int, h1: int, w1: int) -> list[onnx.TensorProto]:
    return [
        const_i64(f"{prefix}_starts", [0, 0, h0, w0]),
        const_i64(f"{prefix}_ends", [1, CHANNELS, h1, w1]),
    ]


def slice_channel_initializers(prefix: str, color: int, h0: int, w0: int, h1: int, w1: int) -> list[onnx.TensorProto]:
    return [
        const_i64(f"{prefix}_starts", [0, color, h0, w0]),
        const_i64(f"{prefix}_ends", [1, color + 1, h1, w1]),
    ]


def make_color_remap_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    mapping = parse_color_map(spec.note)
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    channel_outputs: list[str] = []

    for out_color in range(CHANNELS):
        source_colors = [src for src, dst in sorted(mapping.items()) if dst == out_color]
        if not source_colors:
            zero_name = f"zero_ch{out_color}"
            nodes.append(helper.make_node("Sub", ["input", "input"], [f"zero_full_{out_color}"], name=f"zero_full_{out_color}"))
            initializers.extend(slice_channel_initializers(f"zero_slice_{out_color}", 0, 0, 0, HEIGHT, WIDTH))
            nodes.append(make_slice(f"zero_slice_{out_color}", f"zero_full_{out_color}", zero_name, f"zero_slice_{out_color}_starts", f"zero_slice_{out_color}_ends"))
            channel_outputs.append(zero_name)
            continue
        slices: list[str] = []
        for src in source_colors:
            slice_name = f"out{out_color}_from{src}"
            initializers.extend(slice_channel_initializers(slice_name, src, 0, 0, HEIGHT, WIDTH))
            nodes.append(make_slice(f"slice_{slice_name}", "input", slice_name, f"{slice_name}_starts", f"{slice_name}_ends"))
            slices.append(slice_name)
        if len(slices) == 1:
            channel_outputs.append(slices[0])
        else:
            sum_name = f"out{out_color}_sum"
            nodes.append(helper.make_node("Sum", slices, [sum_name], name=f"sum_out{out_color}"))
            channel_outputs.append(sum_name)

    nodes.append(helper.make_node("Concat", channel_outputs, ["output"], axis=1, name="assemble_remap"))
    return make_model(nodes, initializers, f"{spec.task}_color_remap")


def transform_slice_nodes(prefix: str, transform: str, h: int, w: int) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str, int, int]:
    initializers = slice_initializers(prefix, 0, 0, h, w)
    nodes = [make_slice(f"{prefix}_slice", "input", f"{prefix}_crop", f"{prefix}_starts", f"{prefix}_ends")]
    current = f"{prefix}_crop"
    out_h, out_w = h, w

    if transform == "flip_h":
        initializers.append(const_i64(f"{prefix}_wrev", list(reversed(range(w)))))
        nodes.append(make_gather(f"{prefix}_flip_h", current, f"{prefix}_transformed", f"{prefix}_wrev", axis=3))
        current = f"{prefix}_transformed"
    elif transform == "flip_v":
        initializers.append(const_i64(f"{prefix}_hrev", list(reversed(range(h)))))
        nodes.append(make_gather(f"{prefix}_flip_v", current, f"{prefix}_transformed", f"{prefix}_hrev", axis=2))
        current = f"{prefix}_transformed"
    elif transform == "rot180":
        initializers.extend(
            [
                const_i64(f"{prefix}_hrev", list(reversed(range(h)))),
                const_i64(f"{prefix}_wrev", list(reversed(range(w)))),
            ]
        )
        nodes.append(make_gather(f"{prefix}_rot180_h", current, f"{prefix}_hflip", f"{prefix}_hrev", axis=2))
        nodes.append(make_gather(f"{prefix}_rot180_w", f"{prefix}_hflip", f"{prefix}_transformed", f"{prefix}_wrev", axis=3))
        current = f"{prefix}_transformed"
    elif transform == "transpose":
        nodes.append(helper.make_node("Transpose", [current], [f"{prefix}_transformed"], perm=[0, 1, 3, 2], name=f"{prefix}_transpose"))
        current = f"{prefix}_transformed"
        out_h, out_w = w, h
    elif transform == "rot90":
        initializers.append(const_i64(f"{prefix}_hrev", list(reversed(range(h)))))
        nodes.append(helper.make_node("Transpose", [current], [f"{prefix}_t"], perm=[0, 1, 3, 2], name=f"{prefix}_transpose"))
        nodes.append(make_gather(f"{prefix}_rot90", f"{prefix}_t", f"{prefix}_transformed", f"{prefix}_hrev", axis=3))
        current = f"{prefix}_transformed"
        out_h, out_w = w, h
    elif transform == "rot270":
        initializers.append(const_i64(f"{prefix}_wrev", list(reversed(range(w)))))
        nodes.append(helper.make_node("Transpose", [current], [f"{prefix}_t"], perm=[0, 1, 3, 2], name=f"{prefix}_transpose"))
        nodes.append(make_gather(f"{prefix}_rot270", f"{prefix}_t", f"{prefix}_transformed", f"{prefix}_wrev", axis=2))
        current = f"{prefix}_transformed"
        out_h, out_w = w, h
    else:
        raise ValueError(f"unsupported transform: {transform}")

    return nodes, initializers, current, out_h, out_w


def make_geometric_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    transform = parse_exact_transform(spec.note)
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []
    branches: list[str] = []
    input_shapes = input_size_pairs(examples)

    for idx, (h, w) in enumerate(input_shapes):
        prefix = f"b{idx}_{h}x{w}"
        initializers.append(size_mask_initializer(f"{prefix}_size_mask", h, w))
        nodes.extend(make_size_mask_nodes(prefix, h, w, f"{prefix}_size_mask"))
        t_nodes, t_inits, transformed, out_h, out_w = transform_slice_nodes(prefix, transform, h, w)
        nodes.extend(t_nodes)
        initializers.extend(t_inits)
        initializers.append(pad_for_shape(f"{prefix}_pad", out_h, out_w))
        nodes.append(make_pad(f"{prefix}_pad_node", transformed, f"{prefix}_padded", f"{prefix}_pad"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_is_size_4d"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        branches.append(f"{prefix}_selected")

    if len(branches) == 1:
        nodes.append(helper.make_node("Identity", [branches[0]], ["output"], name="output_identity"))
    else:
        nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_shape_branches"))
    return make_model(nodes, initializers, f"{spec.task}_geometric_{transform}")


def make_split_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    orientation, sep_color, mode, color_a, color_b, out_color, else_color = parse_split_relation(spec.note)
    # These A_exact split tasks have one fixed input/output shape across examples.
    in_h, in_w = shape(examples[0]["input"])
    out_h, out_w = shape(examples[0]["output"])
    initializers: list[onnx.TensorProto] = [
        pad_for_shape("pad_out", out_h, out_w),
        const_f32("ones", np.ones((1, 1, out_h, out_w), dtype=np.float32)),
    ]

    if orientation == "vertical":
        sep_index = (in_w - 1) // 2
        initializers.extend(
            slice_channel_initializers("left", color_a, 0, 0, out_h, sep_index)
            + slice_channel_initializers("right", color_b, 0, sep_index + 1, out_h, in_w)
        )
    else:
        sep_index = (in_h - 1) // 2
        initializers.extend(
            slice_channel_initializers("left", color_a, 0, 0, sep_index, out_w)
            + slice_channel_initializers("right", color_b, sep_index + 1, 0, in_h, out_w)
        )

    nodes: list[onnx.NodeProto] = [
        make_slice("left_slice", "input", "left", "left_starts", "left_ends"),
        make_slice("right_slice", "input", "right", "right_starts", "right_ends"),
    ]
    if mode == "and":
        nodes.append(helper.make_node("Mul", ["left", "right"], ["hit"], name="and_hit"))
    elif mode == "or":
        nodes.extend(
            [
                helper.make_node("Mul", ["left", "right"], ["both"], name="or_both"),
                helper.make_node("Add", ["left", "right"], ["either_raw"], name="or_sum"),
                helper.make_node("Sub", ["either_raw", "both"], ["hit"], name="or_hit"),
            ]
        )
    elif mode == "xor":
        nodes.extend(
            [
                helper.make_node("Mul", ["left", "right"], ["both"], name="xor_both"),
                helper.make_node("Add", ["both", "both"], ["two_both"], name="xor_two_both"),
                helper.make_node("Add", ["left", "right"], ["either_raw"], name="xor_sum"),
                helper.make_node("Sub", ["either_raw", "two_both"], ["hit"], name="xor_hit"),
            ]
        )
    else:
        raise ValueError(f"unsupported split mode: {mode}")
    nodes.append(helper.make_node("Sub", ["ones", "hit"], ["miss"], name="miss"))

    channel_inputs: list[str] = []
    for color in range(CHANNELS):
        src = "zero"
        if color == out_color:
            src = "hit_pad"
        elif color == else_color:
            src = "miss_pad"
        if color == 0:
            nodes.append(helper.make_node("Sub", ["hit", "hit"], ["zero"], name="zero_small"))
            nodes.append(make_pad("hit_pad_node", "hit", "hit_pad", "pad_out"))
            nodes.append(make_pad("miss_pad_node", "miss", "miss_pad", "pad_out"))
            nodes.append(make_pad("zero_pad_node", "zero", "zero_pad", "pad_out"))
        channel_inputs.append(src if src != "zero" else "zero_pad")

    nodes.append(helper.make_node("Concat", channel_inputs, ["output"], axis=1, name="assemble_output"))
    _ = sep_color  # Kept in spec for readability; slicing by position is enough.
    return make_model(nodes, initializers, f"{spec.task}_split_relation")


def make_crop_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    r0, c0 = infer_common_crop_offset(examples)
    branches: list[str] = []
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    pairs = sorted(set((shape(ex["input"]), shape(ex["output"])) for ex in examples))

    for idx, ((in_h, in_w), (out_h, out_w)) in enumerate(pairs):
        prefix = f"crop{idx}_{in_h}x{in_w}_{out_h}x{out_w}"
        initializers.append(size_mask_initializer(f"{prefix}_size_mask", in_h, in_w))
        nodes.extend(make_size_mask_nodes(prefix, in_h, in_w, f"{prefix}_size_mask"))
        initializers.extend(slice_initializers(prefix, r0, c0, r0 + out_h, c0 + out_w))
        initializers.append(pad_for_shape(f"{prefix}_pad", out_h, out_w))
        nodes.append(make_slice(f"{prefix}_slice", "input", f"{prefix}_crop", f"{prefix}_starts", f"{prefix}_ends"))
        nodes.append(make_pad(f"{prefix}_pad_node", f"{prefix}_crop", f"{prefix}_padded", f"{prefix}_pad"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_is_size_4d"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        branches.append(f"{prefix}_selected")

    if len(branches) == 1:
        nodes.append(helper.make_node("Identity", [branches[0]], ["output"], name="output_identity"))
    else:
        nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_crop_branches"))
    return make_model(nodes, initializers, f"{spec.task}_crop_offset_{r0}_{c0}")


def make_repeated_tile_crop_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    branches: list[str] = []
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    pairs = sorted(set((shape(ex["input"]), shape(ex["output"])) for ex in examples))

    for idx, ((in_h, in_w), (out_h, out_w)) in enumerate(pairs):
        prefix = f"tile{idx}_{in_h}x{in_w}_{out_h}x{out_w}"
        if out_h == in_h and out_w * 2 == in_w:
            orientation = "horizontal"
            cmp_a = (0, 0, out_h, out_w)
            cmp_b = (0, out_w, out_h, in_w)
        elif out_h * 2 == in_h and out_w == in_w:
            orientation = "vertical"
            cmp_a = (0, 0, out_h, out_w)
            cmp_b = (out_h, 0, in_h, out_w)
        else:
            raise ValueError(f"unsupported repeated-tile shape: {(in_h, in_w)} -> {(out_h, out_w)}")

        initializers.append(size_mask_initializer(f"{prefix}_size_mask", in_h, in_w))
        nodes.extend(make_size_mask_nodes(prefix, in_h, in_w, f"{prefix}_size_mask"))

        initializers.extend(slice_initializers(f"{prefix}_crop", 0, 0, out_h, out_w))
        initializers.extend(slice_initializers(f"{prefix}_cmp_a", *cmp_a))
        initializers.extend(slice_initializers(f"{prefix}_cmp_b", *cmp_b))
        initializers.append(pad_for_shape(f"{prefix}_pad", out_h, out_w))

        nodes.append(make_slice(f"{prefix}_crop_slice", "input", f"{prefix}_crop", f"{prefix}_crop_starts", f"{prefix}_crop_ends"))
        nodes.append(make_slice(f"{prefix}_{orientation}_a", "input", f"{prefix}_a", f"{prefix}_cmp_a_starts", f"{prefix}_cmp_a_ends"))
        nodes.append(make_slice(f"{prefix}_{orientation}_b", "input", f"{prefix}_b", f"{prefix}_cmp_b_starts", f"{prefix}_cmp_b_ends"))
        nodes.append(helper.make_node("Sub", [f"{prefix}_a", f"{prefix}_b"], [f"{prefix}_rep_diff"], name=f"{prefix}_rep_diff"))
        nodes.append(helper.make_node("Abs", [f"{prefix}_rep_diff"], [f"{prefix}_rep_abs"], name=f"{prefix}_rep_abs"))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_rep_abs", "axes_all"], [f"{prefix}_rep_err"], keepdims=0, name=f"{prefix}_rep_err"))
        nodes.append(helper.make_node("Equal", [f"{prefix}_rep_err", "zero_scalar"], [f"{prefix}_rep_bool"], name=f"{prefix}_rep_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_rep_bool"], [f"{prefix}_rep"], to=TensorProto.FLOAT, name=f"{prefix}_rep_cast"))
        nodes.extend(make_unsqueeze_scalar(f"{prefix}_rep_unsqueeze", f"{prefix}_rep", f"{prefix}_rep_4d"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_is_size_4d", f"{prefix}_rep_4d"], [f"{prefix}_branch_mask"], name=f"{prefix}_branch_mask"))
        nodes.append(make_pad(f"{prefix}_pad_node", f"{prefix}_crop", f"{prefix}_padded", f"{prefix}_pad"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_branch_mask"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_repeated_tile_branches"))
    return make_model(nodes, initializers, f"{spec.task}_repeated_tile_crop")


def convert_to_numpy(example: Example) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    out = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
    for mode, arr in (("input", inp), ("output", out)):
        grid = example[mode]
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                arr[0, color, r, c] = 1.0
    return inp, out


def verify_model_bytes(raw: bytes, examples: list[Example]) -> tuple[int, int, str]:
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    right = 0
    wrong = 0
    first_error = ""
    for idx, example in enumerate(examples):
        x, y = convert_to_numpy(example)
        pred = session.run(["output"], {"input": x})[0]
        pred = (pred > 0.0).astype(np.float32)
        if np.array_equal(pred, y):
            right += 1
        else:
            wrong += 1
            if not first_error:
                first_error = f"first mismatch example {idx}"
    return right, wrong, first_error


def load_a_exact_specs(triage_csv: Path) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    with triage_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "A_exact":
                continue
            families = row["families"].split(";")
            if "color_remap" in families:
                family = "color_remap"
            elif "geometric_exact" in families:
                family = "geometric_exact"
            elif "split_binary_relation" in families:
                family = "split_binary_relation"
            elif "crop_exact_same_offset" in families:
                family = "crop_exact_same_offset"
            else:
                continue
            specs.append(TaskSpec(row["task"], int(row["task_num"]), family, row["notes"], row))
    return sorted(specs, key=lambda spec: spec.task_num)


def build_model(spec: TaskSpec, examples: list[Example]) -> onnx.ModelProto:
    if spec.family == "color_remap":
        return make_color_remap_model(spec, examples)
    if spec.family == "geometric_exact":
        return make_geometric_model(spec, examples)
    if spec.family == "split_binary_relation":
        return make_split_model(spec, examples)
    if spec.family == "crop_exact_same_offset":
        if spec.task_num == 188:
            return make_repeated_tile_crop_model(spec, examples)
        return make_crop_model(spec, examples)
    raise ValueError(f"unsupported family: {spec.family}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A_exact NeuroGolf ONNX submission.")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--triage-csv", type=Path, default=Path("reports/task_triage.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("generated/a_exact_models"))
    parser.add_argument("--zip-out", type=Path, default=Path("generated/a_exact_submission.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/a_exact_generation_report.csv"))
    parser.add_argument("--include-failed", action="store_true", help="Include failed local-verification models in the zip.")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    specs = load_a_exact_specs(args.triage_csv)
    rows: list[dict[str, str | int]] = []
    selected_paths: list[Path] = []

    for spec in specs:
        task_path = args.data_root / f"{spec.task}.json"
        task = json.loads(task_path.read_text())
        examples = all_examples(task)
        model_path = args.model_dir / f"{spec.task}.onnx"
        status = "ok"
        reason = ""
        right = wrong = 0
        try:
            model = build_model(spec, examples)
            raw = model.SerializeToString()
            model_path.write_bytes(raw)
            right, wrong, reason = verify_model_bytes(raw, examples)
            if wrong:
                status = "failed_verify"
            if status == "ok" or args.include_failed:
                selected_paths.append(model_path)
        except Exception as exc:
            status = "error"
            reason = str(exc)
        rows.append(
            {
                "task": spec.task,
                "family": spec.family,
                "status": status,
                "right": right,
                "wrong": wrong,
                "examples": len(examples),
                "file_bytes": model_path.stat().st_size if model_path.exists() else 0,
                "reason": reason,
                "notes": spec.note,
            }
        )
        print(f"{spec.task}: {status} {right}/{len(examples)} {reason}")

    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(selected_paths):
            zf.write(path, path.name)

    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print()
    print(f"generated models: {len(rows)}")
    print(f"verified ok: {ok_count}")
    print(f"zip entries: {len(selected_paths)}")
    print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")
    if ok_count != len(rows):
        print("Some A_exact models failed local verification; inspect the report before submitting.")


if __name__ == "__main__":
    main()
