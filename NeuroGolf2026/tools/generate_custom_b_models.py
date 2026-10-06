#!/usr/bin/env python3
"""Generate clean custom NeuroGolf models for selected B-template tasks.

This is the hand-built lane for high-cost tasks where public blending has
stopped helping.  It keeps the output format identical to the current clean
base zip and only replaces a task when local verification and profiling pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from tools.blend_public_pack import profile_cost, task_points
from tools.generate_a_exact_submission import (
    CHANNELS,
    EXPECTED_SHAPE,
    HEIGHT,
    WIDTH,
    all_examples,
    const_f32,
    const_i64,
    convert_to_numpy,
    make_model,
    make_pad,
    make_slice,
    make_size_mask_nodes,
    pad_for_shape,
    shape,
    slice_channel_initializers,
    slice_initializers,
    size_mask_initializer,
    verify_model_bytes,
)


def make_task111_marker_crop_model() -> onnx.ModelProto:
    """Crop the 3x3 window starting at marker(5) + (1, -1)."""

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        pad_for_shape("pad_3x3", 3, 3),
    ]
    branches: list[str] = []

    # The task has fixed 10x10 inputs and a single marker color 5.  Valid
    # marker positions keep the 3x3 target window inside the 10x10 grid.
    for idx, r in enumerate(range(0, 7)):
        for c in range(1, 9):
            prefix = f"p{idx}_{r}_{c}"
            initializers.extend(
                [
                    const_i64(f"{prefix}_marker_starts", [0, 5, r, c]),
                    const_i64(f"{prefix}_marker_ends", [1, 6, r + 1, c + 1]),
                    *slice_initializers(prefix, r + 1, c - 1, r + 4, c + 2),
                ]
            )
            nodes.append(
                make_slice(
                    f"{prefix}_marker_slice",
                    "input",
                    f"{prefix}_marker",
                    f"{prefix}_marker_starts",
                    f"{prefix}_marker_ends",
                )
            )
            nodes.append(make_slice(f"{prefix}_crop_slice", "input", f"{prefix}_crop", f"{prefix}_starts", f"{prefix}_ends"))
            nodes.append(helper.make_node("Mul", [f"{prefix}_crop", f"{prefix}_marker"], [f"{prefix}_selected"], name=f"{prefix}_select"))
            branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", branches, ["selected_crop"], name="sum_marker_selected_crops"))
    nodes.append(make_pad("pad_selected_crop", "selected_crop", "output", "pad_3x3"))
    return make_model(nodes, initializers, "task111_marker_relative_crop")


def make_top_left_shape_crop_model(task: str, examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    branches: list[str] = []
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    pairs = sorted(set((shape(ex["input"]), shape(ex["output"])) for ex in examples))

    for idx, ((in_h, in_w), (out_h, out_w)) in enumerate(pairs):
        prefix = f"tl{idx}_{in_h}x{in_w}_{out_h}x{out_w}"
        initializers.append(size_mask_initializer(f"{prefix}_size_mask", in_h, in_w))
        nodes.extend(make_size_mask_nodes(prefix, in_h, in_w, f"{prefix}_size_mask"))
        initializers.extend(slice_initializers(prefix, 0, 0, out_h, out_w))
        initializers.append(pad_for_shape(f"{prefix}_pad", out_h, out_w))
        nodes.append(make_slice(f"{prefix}_slice", "input", f"{prefix}_crop", f"{prefix}_starts", f"{prefix}_ends"))
        nodes.append(make_pad(f"{prefix}_pad_node", f"{prefix}_crop", f"{prefix}_padded", f"{prefix}_pad"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_is_size_4d"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", branches, ["output"], name="sum_top_left_shape_crops"))
    return make_model(nodes, initializers, f"{task}_top_left_shape_crop")


def make_task188_repeated_tile_crop_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    """Return the repeated half of task188, grouping pads by output shape."""

    pairs = sorted(set((shape(ex["input"]), shape(ex["output"])) for ex in examples))
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    grouped: dict[tuple[int, int], list[str]] = {}

    for idx, ((in_h, in_w), (out_h, out_w)) in enumerate(pairs):
        prefix = f"rep{idx}_{in_h}x{in_w}_{out_h}x{out_w}"
        if out_h == in_h and out_w * 2 == in_w:
            cmp_a = (0, 0, out_h, out_w)
            cmp_b = (0, out_w, out_h, in_w)
        elif out_h * 2 == in_h and out_w == in_w:
            cmp_a = (0, 0, out_h, out_w)
            cmp_b = (out_h, 0, in_h, out_w)
        else:
            raise ValueError(f"unsupported repeated-tile shape: {(in_h, in_w)} -> {(out_h, out_w)}")

        initializers.append(size_mask_initializer(f"{prefix}_size_mask", in_h, in_w))
        nodes.extend(make_size_mask_nodes(prefix, in_h, in_w, f"{prefix}_size_mask"))
        initializers.extend(slice_initializers(f"{prefix}_crop", 0, 0, out_h, out_w))
        initializers.extend(slice_initializers(f"{prefix}_cmp_a", *cmp_a))
        initializers.extend(slice_initializers(f"{prefix}_cmp_b", *cmp_b))

        nodes.append(make_slice(f"{prefix}_crop_slice", "input", f"{prefix}_crop", f"{prefix}_crop_starts", f"{prefix}_crop_ends"))
        nodes.append(make_slice(f"{prefix}_cmp_a_slice", "input", f"{prefix}_a", f"{prefix}_cmp_a_starts", f"{prefix}_cmp_a_ends"))
        nodes.append(make_slice(f"{prefix}_cmp_b_slice", "input", f"{prefix}_b", f"{prefix}_cmp_b_starts", f"{prefix}_cmp_b_ends"))
        nodes.append(helper.make_node("Sub", [f"{prefix}_a", f"{prefix}_b"], [f"{prefix}_rep_diff"], name=f"{prefix}_rep_diff"))
        nodes.append(helper.make_node("Abs", [f"{prefix}_rep_diff"], [f"{prefix}_rep_abs"], name=f"{prefix}_rep_abs"))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_rep_abs", "axes_all"], [f"{prefix}_rep_err"], keepdims=0, name=f"{prefix}_rep_err"))
        nodes.append(helper.make_node("Equal", [f"{prefix}_rep_err", "zero_scalar"], [f"{prefix}_rep_bool"], name=f"{prefix}_rep_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_rep_bool"], [f"{prefix}_rep"], to=TensorProto.FLOAT, name=f"{prefix}_rep_cast"))
        nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_rep", "scalar_axes"], [f"{prefix}_rep_4d"], name=f"{prefix}_rep_unsqueeze"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_is_size_4d", f"{prefix}_rep_4d"], [f"{prefix}_mask"], name=f"{prefix}_mask"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_crop", f"{prefix}_mask"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        grouped.setdefault((out_h, out_w), []).append(f"{prefix}_selected")

    padded_groups: list[str] = []
    for idx, ((out_h, out_w), branches) in enumerate(sorted(grouped.items())):
        group = f"group{idx}_{out_h}x{out_w}"
        if len(branches) == 1:
            selected = branches[0]
        else:
            selected = f"{group}_sum"
            nodes.append(helper.make_node("Sum", branches, [selected], name=f"{group}_sum"))
        initializers.append(pad_for_shape(f"{group}_pad", out_h, out_w))
        nodes.append(make_pad(f"{group}_pad_node", selected, f"{group}_padded", f"{group}_pad"))
        padded_groups.append(f"{group}_padded")

    nodes.append(helper.make_node("Sum", padded_groups, ["output"], name="sum_repeated_tile_groups"))
    return make_model(nodes, initializers, "task188_repeated_tile_grouped")


def make_task113_top_band_mirror_model() -> onnx.ModelProto:
    """Mirror the contiguous non-zero top band into the bottom of the 10-row grid."""

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    selected: list[str] = []

    for h in (2, 3, 4):
        prefix = f"mirror_h{h}"
        top_pad = 10 - h
        bottom_pad = HEIGHT - 10
        initializers.extend(
            [
                *slice_initializers(f"{prefix}_keep", 0, 0, 10 - h, WIDTH),
                const_i64(f"{prefix}_keep_pad", [0, 0, 0, 0, 0, 0, HEIGHT - (10 - h), 0]),
                *slice_initializers(f"{prefix}_top", 0, 0, h, WIDTH),
                const_i64(f"{prefix}_rev_idx", list(range(h - 1, -1, -1))),
                const_i64(f"{prefix}_pad", [0, 0, top_pad, 0, 0, 0, bottom_pad, 0]),
                const_i64(f"{prefix}_prev_row_starts", [0, 1, h - 1, 0]),
                const_i64(f"{prefix}_prev_row_ends", [1, CHANNELS, h, WIDTH]),
                const_i64(f"{prefix}_next_row_starts", [0, 1, h, 0]),
                const_i64(f"{prefix}_next_row_ends", [1, CHANNELS, h + 1, WIDTH]),
            ]
        )
        nodes.append(make_slice(f"{prefix}_keep_slice", "input", f"{prefix}_keep", f"{prefix}_keep_starts", f"{prefix}_keep_ends"))
        nodes.append(make_pad(f"{prefix}_keep_pad_node", f"{prefix}_keep", f"{prefix}_kept", f"{prefix}_keep_pad"))
        nodes.append(make_slice(f"{prefix}_top_slice", "input", f"{prefix}_top", f"{prefix}_top_starts", f"{prefix}_top_ends"))
        nodes.append(helper.make_node("Gather", [f"{prefix}_top", f"{prefix}_rev_idx"], [f"{prefix}_rev"], axis=2, name=f"{prefix}_reverse"))
        nodes.append(make_pad(f"{prefix}_pad_node", f"{prefix}_rev", f"{prefix}_padded", f"{prefix}_pad"))
        nodes.append(helper.make_node("Add", [f"{prefix}_kept", f"{prefix}_padded"], [f"{prefix}_branch"], name=f"{prefix}_branch"))

        nodes.append(
            make_slice(
                f"{prefix}_prev_row_slice",
                "input",
                f"{prefix}_prev_row",
                f"{prefix}_prev_row_starts",
                f"{prefix}_prev_row_ends",
            )
        )
        nodes.append(
            make_slice(
                f"{prefix}_next_row_slice",
                "input",
                f"{prefix}_next_row",
                f"{prefix}_next_row_starts",
                f"{prefix}_next_row_ends",
            )
        )
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_prev_row", "axes_all"], [f"{prefix}_prev_sum"], keepdims=0, name=f"{prefix}_prev_sum"))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_next_row", "axes_all"], [f"{prefix}_next_sum"], keepdims=0, name=f"{prefix}_next_sum"))
        nodes.append(helper.make_node("Greater", [f"{prefix}_prev_sum", "zero_scalar"], [f"{prefix}_prev_nonzero"], name=f"{prefix}_prev_nonzero"))
        nodes.append(helper.make_node("Equal", [f"{prefix}_next_sum", "zero_scalar"], [f"{prefix}_next_zero"], name=f"{prefix}_next_zero"))
        nodes.append(helper.make_node("And", [f"{prefix}_prev_nonzero", f"{prefix}_next_zero"], [f"{prefix}_is_h_bool"], name=f"{prefix}_is_h_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_is_h_bool"], [f"{prefix}_is_h"], to=TensorProto.FLOAT, name=f"{prefix}_is_h_cast"))
        nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_is_h", "scalar_axes"], [f"{prefix}_is_h_4d"], name=f"{prefix}_is_h_unsqueeze"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_branch", f"{prefix}_is_h_4d"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        selected.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", selected, ["output"], name="sum_selected_mirror"))
    return make_model(nodes, initializers, "task113_top_band_mirror")


def ring_mask_initializer(name: str, size: int, ring: int) -> onnx.TensorProto:
    mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    last = size - 1 - ring
    mask[:, :, ring : last + 1, ring] = 1.0
    mask[:, :, ring : last + 1, last] = 1.0
    mask[:, :, ring, ring : last + 1] = 1.0
    mask[:, :, last, ring : last + 1] = 1.0
    return const_f32(name, mask)


def make_task203_ring_reverse_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    """Reverse the order of concentric square ring colors."""

    sizes = sorted({shape(example["input"])[0] for example in examples})
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    shape_branches: list[str] = []

    for size in sizes:
        prefix = f"ringrev_{size}"
        ring_parts: list[str] = []
        initializers.append(size_mask_initializer(f"{prefix}_size_mask", size, size))
        nodes.extend(make_size_mask_nodes(prefix, size, size, f"{prefix}_size_mask"))

        ring_count = size // 2
        for dst_ring in range(ring_count):
            src_ring = ring_count - 1 - dst_ring
            part = f"{prefix}_r{dst_ring}_from{src_ring}"
            initializers.extend(
                [
                    *slice_initializers(part, src_ring, src_ring, src_ring + 1, src_ring + 1),
                    ring_mask_initializer(f"{part}_dst_mask", size, dst_ring),
                ]
            )
            nodes.append(make_slice(f"{part}_pixel_slice", "input", f"{part}_color", f"{part}_starts", f"{part}_ends"))
            nodes.append(helper.make_node("Mul", [f"{part}_color", f"{part}_dst_mask"], [f"{part}_paint"], name=f"{part}_paint"))
            ring_parts.append(f"{part}_paint")

        nodes.append(helper.make_node("Sum", ring_parts, [f"{prefix}_rings"], name=f"{prefix}_rings"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_rings", f"{prefix}_is_size_4d"], [f"{prefix}_selected"], name=f"{prefix}_select"))
        shape_branches.append(f"{prefix}_selected")

    nodes.append(helper.make_node("Sum", shape_branches, ["output"], name="sum_shape_ring_reversals"))
    return make_model(nodes, initializers, "task203_ring_reverse")


def parity_column_mask_initializer(name: str, width: int) -> onnx.TensorProto:
    mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    for col in range(width):
        if (col % 2) != (width % 2):
            mask[:, :, :3, col] = 1.0
    return const_f32(name, mask)


def make_task332_alternating_5_to_3_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    """Recolor alternating column-wise 5s to 3; parity depends on row width."""

    widths = sorted({shape(example["input"])[1] for example in examples})
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        *slice_channel_initializers("ch5", 5, 0, 0, HEIGHT, WIDTH),
        const_i64("pad_ch3", [0, 3, 0, 0, 0, 6, 0, 0]),
        const_i64("pad_ch5", [0, 5, 0, 0, 0, 4, 0, 0]),
    ]
    changed_candidates: list[str] = []

    nodes.append(make_slice("slice_ch5", "input", "ch5", "ch5_starts", "ch5_ends"))
    for width in widths:
        prefix = f"w{width}"
        initializers.extend(
            [
                size_mask_initializer(f"{prefix}_size_mask", 3, width),
                parity_column_mask_initializer(f"{prefix}_parity_mask", width),
            ]
        )
        nodes.extend(make_size_mask_nodes(prefix, 3, width, f"{prefix}_size_mask"))
        nodes.append(helper.make_node("Mul", ["ch5", f"{prefix}_parity_mask"], [f"{prefix}_changed_unselected"], name=f"{prefix}_changed_unselected"))
        nodes.append(helper.make_node("Mul", [f"{prefix}_changed_unselected", f"{prefix}_is_size_4d"], [f"{prefix}_changed"], name=f"{prefix}_select"))
        changed_candidates.append(f"{prefix}_changed")

    nodes.append(helper.make_node("Sum", changed_candidates, ["changed_ch5"], name="sum_changed_by_width"))
    nodes.append(make_pad("pad_changed_to_ch5", "changed_ch5", "changed_as_ch5", "pad_ch5"))
    nodes.append(make_pad("pad_changed_to_ch3", "changed_ch5", "changed_as_ch3", "pad_ch3"))
    nodes.append(helper.make_node("Sub", ["input", "changed_as_ch5"], ["without_changed_5"], name="remove_changed_5"))
    nodes.append(helper.make_node("Add", ["without_changed_5", "changed_as_ch3"], ["output"], name="add_changed_3"))
    return make_model(nodes, initializers, "task332_alternating_5_to_3")


def rectangle_exact_kernel(name: str, height: int, width: int) -> onnx.TensorProto:
    kernel = np.full((1, 1, height + 2, width + 2), -float(height * width + 1), dtype=np.float32)
    kernel[:, :, 1 : height + 1, 1 : width + 1] = 1.0
    return const_f32(name, kernel)


def rectangle_inner_kernel(name: str, height: int, width: int) -> onnx.TensorProto:
    kernel = np.zeros((1, 1, height, width), dtype=np.float32)
    kernel[:, :, 1 : height - 1, 1 : width - 1] = 1.0
    return const_f32(name, kernel)


def make_task156_rect_interior_rank_model(examples: list[dict[str, list[list[int]]]]) -> onnx.ModelProto:
    """Fill interiors of the two solid 4-rectangles: smaller -> 1, larger -> 2."""

    sizes: list[tuple[int, int]] = []
    for example in examples:
        grid = example["input"]
        seen: set[tuple[int, int]] = set()
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value != 4 or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                cells: list[tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if (
                            0 <= nr < len(grid)
                            and 0 <= nc < len(grid[0])
                            and grid[nr][nc] == 4
                            and (nr, nc) not in seen
                        ):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                rows = [cell[0] for cell in cells]
                cols = [cell[1] for cell in cells]
                sizes.append((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1))

    rect_sizes = sorted(set(sizes))
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        *slice_channel_initializers("ch4_10", 4, 0, 0, 10, 10),
        const_i64("ch4_10_pad1", [0, 0, 1, 1, 0, 0, 1, 1]),
        const_i64("pad_inner_ch1", [0, 1, 0, 0, 0, 8, 20, 20]),
        const_i64("pad_inner_ch2", [0, 2, 0, 0, 0, 7, 20, 20]),
        const_i64("pad_inner_ch4", [0, 4, 0, 0, 0, 5, 20, 20]),
    ]
    interiors: dict[tuple[int, int], str] = {}
    presents: dict[tuple[int, int], str] = {}
    big_masks: list[str] = []

    nodes.append(make_slice("slice_ch4_10", "input", "ch4_10", "ch4_10_starts", "ch4_10_ends"))
    nodes.append(make_pad("pad_ch4_10_by1", "ch4_10", "ch4_pad1", "ch4_10_pad1"))

    for height, width in rect_sizes:
        prefix = f"rect_{height}x{width}"
        initializers.extend(
            [
                rectangle_exact_kernel(f"{prefix}_exact_kernel", height, width),
                rectangle_inner_kernel(f"{prefix}_inner_kernel", height, width),
                const_f32(f"{prefix}_target", float(height * width)),
            ]
        )
        nodes.append(
            helper.make_node(
                "Conv",
                ["ch4_pad1", f"{prefix}_exact_kernel"],
                [f"{prefix}_score"],
                name=f"{prefix}_detect_exact",
            )
        )
        nodes.append(helper.make_node("Equal", [f"{prefix}_score", f"{prefix}_target"], [f"{prefix}_hits_bool"], name=f"{prefix}_hits_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_hits_bool"], [f"{prefix}_hits"], to=TensorProto.FLOAT, name=f"{prefix}_hits_cast"))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_hits", "axes_all"], [f"{prefix}_hit_count"], keepdims=0, name=f"{prefix}_hit_count"))
        nodes.append(helper.make_node("Greater", [f"{prefix}_hit_count", "zero_scalar"], [f"{prefix}_present_bool"], name=f"{prefix}_present_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_present_bool"], [f"{prefix}_present"], to=TensorProto.FLOAT, name=f"{prefix}_present_cast"))
        nodes.append(
            helper.make_node(
                "ConvTranspose",
                [f"{prefix}_hits", f"{prefix}_inner_kernel"],
                [f"{prefix}_inner"],
                name=f"{prefix}_paint_inner",
            )
        )
        interiors[(height, width)] = f"{prefix}_inner"
        presents[(height, width)] = f"{prefix}_present"

    for height, width in rect_sizes:
        prefix = f"rect_{height}x{width}"
        larger_h = [present for (h, _), present in presents.items() if h > height]
        larger_w = [present for (_, w), present in presents.items() if w > width]
        if larger_h:
            nodes.append(helper.make_node("Sum", larger_h, [f"{prefix}_larger_h_count"], name=f"{prefix}_larger_h_count"))
            larger_h_count = f"{prefix}_larger_h_count"
        else:
            larger_h_count = "zero_scalar"
        if larger_w:
            nodes.append(helper.make_node("Sum", larger_w, [f"{prefix}_larger_w_count"], name=f"{prefix}_larger_w_count"))
            larger_w_count = f"{prefix}_larger_w_count"
        else:
            larger_w_count = "zero_scalar"
        nodes.append(helper.make_node("Equal", [larger_h_count, "zero_scalar"], [f"{prefix}_max_h_bool"], name=f"{prefix}_max_h_bool"))
        nodes.append(helper.make_node("Equal", [larger_w_count, "zero_scalar"], [f"{prefix}_max_w_bool"], name=f"{prefix}_max_w_bool"))
        nodes.append(helper.make_node("And", [f"{prefix}_present_bool", f"{prefix}_max_h_bool"], [f"{prefix}_present_max_h"], name=f"{prefix}_present_max_h"))
        nodes.append(helper.make_node("And", [f"{prefix}_present_max_h", f"{prefix}_max_w_bool"], [f"{prefix}_big_bool"], name=f"{prefix}_big_bool"))
        nodes.append(helper.make_node("Cast", [f"{prefix}_big_bool"], [f"{prefix}_big"], to=TensorProto.FLOAT, name=f"{prefix}_big_cast"))
        nodes.append(helper.make_node("Unsqueeze", [f"{prefix}_big", "scalar_axes"], [f"{prefix}_big_4d"], name=f"{prefix}_big_unsqueeze"))
        nodes.append(helper.make_node("Mul", [interiors[(height, width)], f"{prefix}_big_4d"], [f"{prefix}_big_inner"], name=f"{prefix}_big_inner"))
        big_masks.append(f"{prefix}_big_inner")

    nodes.append(helper.make_node("Sum", list(interiors.values()), ["inner_all"], name="sum_all_rectangle_interiors"))
    nodes.append(helper.make_node("Sum", big_masks, ["inner_big"], name="sum_big_rectangle_interiors"))
    nodes.append(helper.make_node("Sub", ["inner_all", "inner_big"], ["inner_small"], name="inner_small"))
    nodes.append(make_pad("pad_small_to_ch1", "inner_small", "small_as_ch1", "pad_inner_ch1"))
    nodes.append(make_pad("pad_big_to_ch2", "inner_big", "big_as_ch2", "pad_inner_ch2"))
    nodes.append(make_pad("pad_all_to_ch4", "inner_all", "inner_as_ch4", "pad_inner_ch4"))
    nodes.append(helper.make_node("Sub", ["input", "inner_as_ch4"], ["without_inner_4"], name="remove_rect_interiors_from_ch4"))
    nodes.append(helper.make_node("Add", ["without_inner_4", "small_as_ch1"], ["with_small_inner"], name="add_small_inner_1"))
    nodes.append(helper.make_node("Add", ["with_small_inner", "big_as_ch2"], ["output"], name="add_big_inner_2"))
    return make_model(nodes, initializers, "task156_rect_interior_rank")


def write_submission_with_replacements(base_zip: Path, replacements: dict[str, bytes], output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as out:
        seen: set[str] = set()
        for name in base.namelist():
            member = Path(name).name
            if member in replacements:
                out.writestr(member, replacements[member])
            else:
                out.writestr(member, base.read(name))
            seen.add(member)
        for member, raw in sorted(replacements.items()):
            if member not in seen:
                out.writestr(member, raw)


def read_base_member(base_zip: Path, member: str) -> bytes | None:
    with zipfile.ZipFile(base_zip) as zf:
        for name in zf.namelist():
            if Path(name).name == member:
                return zf.read(name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean custom B-template NeuroGolf models.")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--model-dir", type=Path, default=Path("generated/custom_b_models"))
    parser.add_argument("--zip-out", type=Path, default=Path("generated/custom_b_submission.zip"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/custom_b_generation_report.csv"))
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    replacements: dict[str, bytes] = {}

    builders = {
        "task111": lambda examples: make_task111_marker_crop_model(),
        "task113": lambda examples: make_task113_top_band_mirror_model(),
        "task156": lambda examples: make_task156_rect_interior_rank_model(examples),
        "task203": lambda examples: make_task203_ring_reverse_model(examples),
        "task332": lambda examples: make_task332_alternating_5_to_3_model(examples),
        "task188": lambda examples: make_task188_repeated_tile_crop_model(examples),
    }

    for task, builder in builders.items():
        examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
        model = builder(examples)
        raw = model.SerializeToString()
        model_path = args.model_dir / f"{task}.onnx"
        model_path.write_bytes(raw)
        right, wrong, reason = verify_model_bytes(raw, examples)
        status = "ok" if wrong == 0 else "failed_verify"
        macs = memory = params = cost = ""
        points = ""
        if status == "ok":
            macs, memory, params, cost = profile_cost(model_path)
            points = f"{task_points(cost):.6f}"
            base_raw = read_base_member(args.base_zip, f"{task}.onnx")
            base_cost = ""
            action = "no_base"
            if base_raw is not None:
                base_path = args.model_dir / "base" / f"{task}.onnx"
                base_path.parent.mkdir(parents=True, exist_ok=True)
                base_path.write_bytes(base_raw)
                _, _, _, base_cost = profile_cost(base_path)
                action = "replace" if int(cost) < int(base_cost) else "keep_base"
            if action in {"replace", "no_base"}:
                replacements[f"{task}.onnx"] = raw
        else:
            base_cost = ""
            action = "keep_base"

        rows.append(
            {
                "task": task,
                "action": action,
                "status": status,
                "right": right,
                "wrong": wrong,
                "examples": len(examples),
                "reason": reason,
                "base_cost": base_cost,
                "file_bytes": len(raw),
                "macs": macs,
                "memory": memory,
                "params": params,
                "cost": cost,
                "points": points,
            }
        )

    if replacements:
        write_submission_with_replacements(args.base_zip, replacements, args.zip_out)

    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    if replacements:
        print(f"zip: {args.zip_out}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
