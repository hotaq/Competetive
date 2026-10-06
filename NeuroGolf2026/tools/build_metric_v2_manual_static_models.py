#!/usr/bin/env python3
"""Build hand-written static ONNX models for metric-v2-safe NeuroGolf tasks."""

from __future__ import annotations

import argparse
import csv
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


def arr(name: str, value: np.ndarray | list[int] | float, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model(name: str, nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, [tensor_info("input")], [tensor_info("output")], initializer=initializers)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_metric_v2_manual_static",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def onehot_channel(channel: int) -> np.ndarray:
    value = np.zeros((1, 10, 1, 1), dtype=np.float32)
    value[:, channel, :, :] = 1.0
    return value


def build_task022() -> onnx.ModelProto:
    """Overlay the 3x3 neighborhoods around every color-5 marker."""

    nonzero_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_keep[:, 0, :, :] = 0.0

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 11, 11]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 11, 11]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("nonzero_keep", nonzero_keep),
        const_f32("ch0", onehot_channel(0)),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_11x11"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["markers"], name="slice_markers"),
    ]

    paint_terms: list[str] = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            out_r = dr + 1
            out_c = dc + 1
            prefix = f"cell_{out_r}_{out_c}"
            row_start = max(0, -dr)
            row_end = 11 - max(0, dr)
            col_start = max(0, -dc)
            col_end = 11 - max(0, dc)
            pad_top = max(0, dr)
            pad_bottom = max(0, -dr)
            pad_left = max(0, dc)
            pad_right = max(0, -dc)
            cell_mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
            cell_mask[:, :, out_r, out_c] = 1.0
            inits.extend(
                [
                    const_i64(f"{prefix}_starts", [0, 0, row_start, col_start]),
                    const_i64(f"{prefix}_ends", [1, 1, row_end, col_end]),
                    const_i64(f"{prefix}_pad", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]),
                    const_f32(f"{prefix}_mask", cell_mask),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["markers", f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"],
                        [f"{prefix}_marker_slice"],
                        name=f"{prefix}_slice_shift_source",
                    ),
                    helper.make_node(
                        "Pad",
                        [f"{prefix}_marker_slice", f"{prefix}_pad", "zero"],
                        [f"{prefix}_shifted_markers"],
                        name=f"{prefix}_shift_markers",
                    ),
                    helper.make_node("Mul", ["small", f"{prefix}_shifted_markers"], [f"{prefix}_samples"], name=f"{prefix}_sample_neighbors"),
                    helper.make_node("ReduceMax", [f"{prefix}_samples"], [f"{prefix}_raw_vec"], axes=[2, 3], keepdims=1, name=f"{prefix}_color_vector"),
                    helper.make_node("Mul", [f"{prefix}_raw_vec", "nonzero_keep"], [f"{prefix}_nonzero_vec"], name=f"{prefix}_drop_zero_candidate"),
                    helper.make_node(
                        "ReduceMax",
                        [f"{prefix}_nonzero_vec"],
                        [f"{prefix}_nonzero_present"],
                        axes=[1],
                        keepdims=1,
                        name=f"{prefix}_has_nonzero",
                    ),
                    helper.make_node("Sub", ["one", f"{prefix}_nonzero_present"], [f"{prefix}_keep_zero"], name=f"{prefix}_keep_zero_if_empty"),
                    helper.make_node("Mul", [f"{prefix}_keep_zero", "ch0"], [f"{prefix}_zero_vec"], name=f"{prefix}_zero_vector"),
                    helper.make_node("Add", [f"{prefix}_nonzero_vec", f"{prefix}_zero_vec"], [f"{prefix}_vec"], name=f"{prefix}_fixed_vector"),
                    helper.make_node("Mul", [f"{prefix}_vec", f"{prefix}_mask"], [f"{prefix}_paint"], name=f"{prefix}_paint_cell"),
                ]
            )
            paint_terms.append(f"{prefix}_paint")

    nodes.extend(
        [
            helper.make_node("Sum", paint_terms, ["small_out"], name="sum_output_cells"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task022_overlay_marker_neighborhoods", nodes, inits)


def build_task037() -> onnx.ModelProto:
    """Connect same-color endpoints that lie on a diagonal."""

    nonzero_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_keep[:, 0, :, :] = 0.0
    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_axis", [1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("nonzero_keep", nonzero_keep),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Mul", ["small", "nonzero_keep"], ["object"], name="drop_background"),
    ]

    def add_shift(source: str, prefix: str, delta_r: int, delta_c: int, terms: list[str]) -> None:
        row_start = max(0, -delta_r)
        row_end = min(10, 10 - delta_r)
        col_start = max(0, -delta_c)
        col_end = min(10, 10 - delta_c)
        if row_end <= row_start or col_end <= col_start:
            return
        pad_top = row_start + delta_r
        pad_left = col_start + delta_c
        height = row_end - row_start
        width = col_end - col_start
        pad_bottom = 10 - pad_top - height
        pad_right = 10 - pad_left - width
        inits.extend(
            [
                const_i64(f"{prefix}_starts", [0, 0, row_start, col_start]),
                const_i64(f"{prefix}_ends", [1, 10, row_end, col_end]),
                const_i64(f"{prefix}_pad", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [source, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"],
                    [f"{prefix}_slice"],
                    name=f"{prefix}_slice_source",
                ),
                helper.make_node("Pad", [f"{prefix}_slice", f"{prefix}_pad", "zero"], [f"{prefix}_shift"], name=f"{prefix}_shift"),
            ]
        )
        terms.append(f"{prefix}_shift")

    direction_terms: dict[str, list[str]] = {name: [] for name in ("nw", "se", "ne", "sw")}
    for k in range(1, 10):
        add_shift("object", f"nw_{k}", k, k, direction_terms["nw"])
        add_shift("object", f"se_{k}", -k, -k, direction_terms["se"])
        add_shift("object", f"ne_{k}", k, -k, direction_terms["ne"])
        add_shift("object", f"sw_{k}", -k, k, direction_terms["sw"])

    for name, terms in direction_terms.items():
        nodes.extend(
            [
                helper.make_node("Sum", terms, [f"{name}_raw"], name=f"{name}_sum"),
                helper.make_node("Greater", [f"{name}_raw", "zero"], [f"{name}_bool"], name=f"{name}_seen_bool"),
                helper.make_node("Cast", [f"{name}_bool"], [f"{name}_seen"], to=TensorProto.FLOAT, name=f"{name}_seen_float"),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Mul", ["nw_seen", "se_seen"], ["diag_down"], name="same_color_on_nw_se"),
            helper.make_node("Mul", ["ne_seen", "sw_seen"], ["diag_up"], name="same_color_on_ne_sw"),
            helper.make_node("Add", ["diag_down", "diag_up"], ["fill_raw"], name="combined_diagonal_fill"),
            helper.make_node("Greater", ["fill_raw", "zero"], ["fill_bool"], name="fill_bool"),
            helper.make_node("Cast", ["fill_bool"], ["fill"], to=TensorProto.FLOAT, name="fill_float"),
            helper.make_node("ReduceMax", ["fill"], ["fill_occ"], axes=[1], keepdims=1, name="fill_occupancy"),
            helper.make_node("Sub", ["one", "fill_occ"], ["keep_mask"], name="keep_non_fill"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_existing"),
            helper.make_node("Add", ["kept", "fill"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task037_connect_diagonal_pairs", nodes, inits)


def build_task045() -> onnx.ModelProto:
    """Fill rows whose left and right endpoint colors match."""

    nonzero_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_keep[:, 0, :, :] = 0.0
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("left_starts", [0, 0, 0, 0]),
        const_i64("left_ends", [1, 10, 10, 1]),
        const_i64("right_starts", [0, 0, 0, 9]),
        const_i64("right_ends", [1, 10, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("shape_row", [1, 10, 10, 10]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("nonzero_keep", nonzero_keep),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["small", "left_starts", "left_ends", "axes4", "steps4"], ["left_raw"], name="left_endpoint"),
        helper.make_node("Slice", ["small", "right_starts", "right_ends", "axes4", "steps4"], ["right_raw"], name="right_endpoint"),
        helper.make_node("Mul", ["left_raw", "nonzero_keep"], ["left_color"], name="left_nonzero"),
        helper.make_node("Mul", ["right_raw", "nonzero_keep"], ["right_color"], name="right_nonzero"),
        helper.make_node("Mul", ["left_color", "right_color"], ["match_color"], name="matching_endpoint_color"),
        helper.make_node("ReduceMax", ["match_color"], ["match_occ"], axes=[1], keepdims=1, name="matching_rows"),
        helper.make_node("Expand", ["match_color", "shape_row"], ["filled_rows"], name="expand_matching_rows"),
        helper.make_node("Sub", ["one", "match_occ"], ["keep_mask"], name="keep_nonmatching_rows"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
        helper.make_node("Add", ["kept", "filled_rows"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task045_fill_matching_endpoint_rows", nodes, inits)


def build_task053() -> onnx.ModelProto:
    """Shift the 3x3 grid down by one row, filling the top row as color 0."""

    inits = [
        const_i64("body_starts", [0, 0, 0, 0]),
        const_i64("body_ends", [1, 10, 2, 3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("top_ones", np.ones((1, 1, 1, 3), dtype=np.float32)),
        const_f32("ch0", onehot_channel(0)),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "body_starts", "body_ends", "axes4", "steps4"], ["body"], name="slice_top_two_rows"),
        helper.make_node("Mul", ["top_ones", "ch0"], ["top_zero_row"], name="make_top_zero_row"),
        helper.make_node("Concat", ["top_zero_row", "body"], ["small_shifted"], axis=2, name="concat_shifted_3x3"),
        helper.make_node("Pad", ["small_shifted", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task053_shift_down_one", nodes, inits)


def build_task060() -> onnx.ModelProto:
    """Fill rows between two endpoint colors with a color-5 divider."""

    nonzero_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_keep[:, 0, :, :] = 0.0
    inits = [
        const_i64("left_starts", [0, 0, 0, 0]),
        const_i64("left_ends", [1, 10, 5, 1]),
        const_i64("right_starts", [0, 0, 0, 10]),
        const_i64("right_ends", [1, 10, 5, 11]),
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 5, 11]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("shape_half", [1, 10, 5, 5]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 25, 19]),
        const_f32("nonzero_keep", nonzero_keep),
        const_f32("ch5", onehot_channel(5)),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_5x11"),
        helper.make_node("Slice", ["small", "left_starts", "left_ends", "axes4", "steps4"], ["left_raw"], name="left_endpoint"),
        helper.make_node("Slice", ["small", "right_starts", "right_ends", "axes4", "steps4"], ["right_raw"], name="right_endpoint"),
        helper.make_node("Mul", ["left_raw", "nonzero_keep"], ["left_color"], name="left_nonzero"),
        helper.make_node("Mul", ["right_raw", "nonzero_keep"], ["right_color"], name="right_nonzero"),
        helper.make_node("Add", ["left_color", "right_color"], ["endpoints"], name="endpoints"),
        helper.make_node("ReduceMax", ["endpoints"], ["active"], axes=[1], keepdims=1, name="active_rows"),
        helper.make_node("Expand", ["left_color", "shape_half"], ["left_block"], name="left_block"),
        helper.make_node("Expand", ["right_color", "shape_half"], ["right_block"], name="right_block"),
        helper.make_node("Mul", ["active", "ch5"], ["middle"], name="middle_five"),
        helper.make_node("Concat", ["left_block", "middle", "right_block"], ["filled"], axis=3, name="filled_rows"),
        helper.make_node("Sub", ["one", "active"], ["inactive"], name="inactive_rows"),
        helper.make_node("Mul", ["small", "inactive"], ["kept"], name="keep_inactive_rows"),
        helper.make_node("Add", ["kept", "filled"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task060_fill_endpoint_rows", nodes, inits)


def build_task073() -> onnx.ModelProto:
    """Move color-1 cells down two rows inside the 5x5 grid."""

    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 5, 5]),
        const_i64("ch1_starts", [0, 1, 0, 0]),
        const_i64("ch1_ends", [1, 2, 5, 5]),
        const_i64("move_starts", [0, 0, 0, 0]),
        const_i64("move_ends", [1, 1, 3, 5]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_down_two", [0, 0, 2, 0, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 25, 25]),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch1", onehot_channel(1)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_5x5"),
        helper.make_node("Slice", ["input", "ch1_starts", "ch1_ends", "axes4", "steps4"], ["orig_ch1"], name="slice_color_one"),
        helper.make_node("Slice", ["orig_ch1", "move_starts", "move_ends", "axes4", "steps4"], ["move_body"], name="slice_movable_rows"),
        helper.make_node("Pad", ["move_body", "pad_down_two", "zero"], ["shifted_ch1"], name="shift_down_two"),
        helper.make_node("Add", ["orig_ch1", "shifted_ch1"], ["changed_raw"], name="changed_raw"),
        helper.make_node("Sub", ["one", "changed_raw"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
        helper.make_node("Mul", ["orig_ch1", "ch0"], ["old_background"], name="paint_old_background"),
        helper.make_node("Mul", ["shifted_ch1", "ch1"], ["new_ones"], name="paint_shifted_ones"),
        helper.make_node("Add", ["kept", "old_background"], ["with_background"], name="with_background"),
        helper.make_node("Add", ["with_background", "new_ones"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task073_move_ones_down_two", nodes, inits)


def build_task074() -> onnx.ModelProto:
    """Complete the symmetric canvas while ignoring color-9 placeholders."""

    keep_mask = np.ones((1, 10, 1, 1), dtype=np.bool_)
    keep_mask[:, 9, :, :] = False
    inits = [
        arr("keep_mask", keep_mask, np.bool_),
        arr("false", np.array(False, dtype=np.bool_), np.bool_),
        const_i64("reverse_start", [29]),
        const_i64("reverse_end", [-31]),
        const_i64("axis_height", [2]),
        const_i64("axis_width", [3]),
        const_i64("reverse_step", [-1]),
        const_i64("full_starts", [0, 0, 0, 0]),
        const_i64("full_ends", [1, 10, 30, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_left_two", [0, 0, 0, 2, 0, 0, 0, 0]),
        const_i64("pad_top_two", [0, 0, 2, 0, 0, 0, 0, 0]),
    ]
    nodes = [
        helper.make_node("Cast", ["input"], ["input_bool"], to=TensorProto.BOOL, name="cast_input_bool"),
        helper.make_node("And", ["input_bool", "keep_mask"], ["object"], name="drop_placeholders"),
        helper.make_node("Transpose", ["object"], ["main_transpose"], perm=[0, 1, 3, 2], name="main_transpose"),
        helper.make_node(
            "Slice",
            ["object", "reverse_start", "reverse_end", "axis_width", "reverse_step"],
            ["reverse_width"],
            name="reverse_width",
        ),
        helper.make_node("Pad", ["reverse_width", "pad_left_two", "false"], ["reverse_width_padded"], name="pad_reverse_width"),
        helper.make_node(
            "Slice",
            ["reverse_width_padded", "full_starts", "full_ends", "axes4", "steps4"],
            ["shift_width"],
            name="crop_shift_width",
        ),
        helper.make_node("Transpose", ["shift_width"], ["shift_width_transpose"], perm=[0, 1, 3, 2], name="shift_width_transpose"),
        helper.make_node(
            "Slice",
            ["object", "reverse_start", "reverse_end", "axis_height", "reverse_step"],
            ["reverse_height"],
            name="reverse_height",
        ),
        helper.make_node("Pad", ["reverse_height", "pad_top_two", "false"], ["reverse_height_padded"], name="pad_reverse_height"),
        helper.make_node(
            "Slice",
            ["reverse_height_padded", "full_starts", "full_ends", "axes4", "steps4"],
            ["shift_height"],
            name="crop_shift_height",
        ),
        helper.make_node("Transpose", ["shift_height"], ["shift_height_transpose"], perm=[0, 1, 3, 2], name="shift_height_transpose"),
        helper.make_node(
            "Slice",
            ["shift_width", "reverse_start", "reverse_end", "axis_height", "reverse_step"],
            ["reverse_shift_width_height"],
            name="reverse_shift_width_height",
        ),
        helper.make_node(
            "Pad",
            ["reverse_shift_width_height", "pad_top_two", "false"],
            ["reverse_shift_width_height_padded"],
            name="pad_reverse_shift_width_height",
        ),
        helper.make_node(
            "Slice",
            ["reverse_shift_width_height_padded", "full_starts", "full_ends", "axes4", "steps4"],
            ["shift_width_height"],
            name="crop_shift_width_height",
        ),
        helper.make_node(
            "Transpose",
            ["shift_width_height"],
            ["shift_width_height_transpose"],
            perm=[0, 1, 3, 2],
            name="shift_width_height_transpose",
        ),
        helper.make_node("Or", ["object", "main_transpose"], ["or_main"], name="or_main"),
        helper.make_node("Or", ["shift_width", "shift_width_transpose"], ["or_width"], name="or_width"),
        helper.make_node("Or", ["shift_height", "shift_height_transpose"], ["or_height"], name="or_height"),
        helper.make_node("Or", ["shift_width_height", "shift_width_height_transpose"], ["or_corner"], name="or_corner"),
        helper.make_node("Or", ["or_main", "or_width"], ["or_first_half"], name="or_first_half"),
        helper.make_node("Or", ["or_height", "or_corner"], ["or_second_half"], name="or_second_half"),
        helper.make_node("Or", ["or_first_half", "or_second_half"], ["output_bool"], name="or_all"),
        helper.make_node("Cast", ["output_bool"], ["output"], to=TensorProto.FLOAT, name="cast_output_float"),
    ]
    return make_model("task074_bool_symmetric_completion", nodes, inits)


def build_task082() -> onnx.ModelProto:
    """Repeat the top row, alternating with its left/right neighbor expansion."""

    object_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    object_keep[:, 0, :, :] = 0.0
    side_kernel = np.zeros((10, 1, 1, 3), dtype=np.float32)
    for ch in range(1, 10):
        side_kernel[ch, 0, 0, 0] = 1.0
        side_kernel[ch, 0, 0, 2] = 1.0

    inits = [
        const_i64("top_starts", [0, 0, 0, 0]),
        const_i64("top_ends", [1, 10, 1, 15]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 24, 15]),
        const_f32("object_keep", object_keep),
        const_f32("side_kernel", side_kernel),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("ch0", onehot_channel(0)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "top_starts", "top_ends", "axes4", "steps4"], ["top"], name="slice_top_row"),
        helper.make_node("ReduceMax", ["top"], ["active_width"], axes=[1], keepdims=1, name="active_width"),
        helper.make_node("Mul", ["top", "object_keep"], ["top_object"], name="drop_background"),
        helper.make_node("Conv", ["top_object", "side_kernel"], ["odd_raw"], group=10, pads=[0, 1, 0, 1], name="expand_left_right"),
        helper.make_node("Greater", ["odd_raw", "zero"], ["odd_bool"], name="odd_bool"),
        helper.make_node("Cast", ["odd_bool"], ["odd_object"], to=TensorProto.FLOAT, name="odd_object_float"),
        helper.make_node("ReduceMax", ["odd_object"], ["odd_occ"], axes=[1], keepdims=1, name="odd_occupancy"),
        helper.make_node("Sub", ["active_width", "odd_occ"], ["odd_bg_mask"], name="odd_background_mask"),
        helper.make_node("Mul", ["odd_bg_mask", "ch0"], ["odd_bg"], name="odd_background"),
        helper.make_node("Add", ["odd_object", "odd_bg"], ["odd"], name="odd_row"),
        helper.make_node("Concat", ["top", "odd", "top", "odd", "top", "odd"], ["small_out"], axis=2, name="six_rows"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task082_alternating_top_row_neighbors", nodes, inits)


def build_task115() -> onnx.ModelProto:
    """Extract nonzero color runs from the top row or left column."""

    object_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    object_keep[:, 0, :, :] = 0.0

    inits: list[onnx.TensorProto] = [
        const_i64("row_starts", [0, 0, 0, 0]),
        const_i64("row_ends", [1, 10, 1, 30]),
        const_i64("col_starts", [0, 0, 0, 0]),
        const_i64("col_ends", [1, 10, 30, 1]),
        const_i64("row_prev_starts", [0, 0, 0, 0]),
        const_i64("row_prev_ends", [1, 10, 1, 29]),
        const_i64("col_prev_starts", [0, 0, 0, 0]),
        const_i64("col_prev_ends", [1, 10, 29, 1]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_axis", [1]),
        const_i64("spatial_axes", [2, 3]),
        arr("width_axis_scalar", np.array(3, dtype=np.int64), np.int64),
        arr("height_axis_scalar", np.array(2, dtype=np.int64), np.int64),
        const_i64("row_prev_pad", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("col_prev_pad", [0, 0, 1, 0, 0, 0, 0, 0]),
        const_i64("row_pad_to_30", [0, 0, 0, 0, 0, 0, 29, 0]),
        const_i64("col_pad_to_30", [0, 0, 0, 0, 0, 0, 0, 29]),
        const_f32("object_keep", object_keep),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("one_half", 1.5),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "row_starts", "row_ends", "axes4", "steps4"], ["first_row"], name="slice_first_row"),
        helper.make_node("Slice", ["input", "col_starts", "col_ends", "axes4", "steps4"], ["first_col"], name="slice_first_col"),
        helper.make_node("Mul", ["first_row", "object_keep"], ["row_obj"], name="drop_row_background"),
        helper.make_node("Mul", ["first_col", "object_keep"], ["col_obj"], name="drop_col_background"),
        helper.make_node(
            "Slice",
            ["row_obj", "row_prev_starts", "row_prev_ends", "axes4", "steps4"],
            ["row_prev_body"],
            name="row_previous_body",
        ),
        helper.make_node("Pad", ["row_prev_body", "row_prev_pad", "zero"], ["row_prev"], name="row_previous"),
        helper.make_node(
            "Slice",
            ["col_obj", "col_prev_starts", "col_prev_ends", "axes4", "steps4"],
            ["col_prev_body"],
            name="col_previous_body",
        ),
        helper.make_node("Pad", ["col_prev_body", "col_prev_pad", "zero"], ["col_prev"], name="col_previous"),
        helper.make_node("ReduceSum", ["row_obj", "channel_axis"], ["row_active"], keepdims=1, name="row_active"),
        helper.make_node("ReduceSum", ["col_obj", "channel_axis"], ["col_active"], keepdims=1, name="col_active"),
        helper.make_node("Mul", ["row_obj", "row_prev"], ["row_same_color_raw"], name="row_same_color_raw"),
        helper.make_node("Mul", ["col_obj", "col_prev"], ["col_same_color_raw"], name="col_same_color_raw"),
        helper.make_node("ReduceSum", ["row_same_color_raw", "channel_axis"], ["row_same_color"], keepdims=1, name="row_same_color"),
        helper.make_node("ReduceSum", ["col_same_color_raw", "channel_axis"], ["col_same_color"], keepdims=1, name="col_same_color"),
        helper.make_node("Sub", ["one", "row_same_color"], ["row_new_run_gate"], name="row_new_run_gate"),
        helper.make_node("Sub", ["one", "col_same_color"], ["col_new_run_gate"], name="col_new_run_gate"),
        helper.make_node("Mul", ["row_active", "row_new_run_gate"], ["row_start"], name="row_run_starts"),
        helper.make_node("Mul", ["col_active", "col_new_run_gate"], ["col_start"], name="col_run_starts"),
        helper.make_node("CumSum", ["row_start", "width_axis_scalar"], ["row_rank"], name="row_start_rank"),
        helper.make_node("CumSum", ["col_start", "height_axis_scalar"], ["col_rank"], name="col_start_rank"),
    ]

    row_terms: list[str] = []
    col_terms: list[str] = []
    for idx in range(4):
        slot = idx + 1
        row_mask = np.zeros((1, 1, 1, 30), dtype=np.float32)
        row_mask[:, :, :, idx] = 1.0
        col_mask = np.zeros((1, 1, 30, 1), dtype=np.float32)
        col_mask[:, :, idx, :] = 1.0
        inits.extend(
            [
                const_f32(f"slot_{slot}", float(slot)),
                const_f32(f"row_slot_mask_{slot}", row_mask),
                const_f32(f"col_slot_mask_{slot}", col_mask),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Equal", ["row_rank", f"slot_{slot}"], [f"row_slot_{slot}_bool"], name=f"row_slot_{slot}_bool"),
                helper.make_node("Cast", [f"row_slot_{slot}_bool"], [f"row_slot_{slot}_rank"], to=TensorProto.FLOAT, name=f"row_slot_{slot}_rank"),
                helper.make_node("Mul", ["row_start", f"row_slot_{slot}_rank"], [f"row_slot_{slot}_start"], name=f"row_slot_{slot}_start"),
                helper.make_node("Mul", ["row_obj", f"row_slot_{slot}_start"], [f"row_slot_{slot}_cells"], name=f"row_slot_{slot}_cells"),
                helper.make_node(
                    "ReduceSum",
                    [f"row_slot_{slot}_cells", "spatial_axes"],
                    [f"row_slot_{slot}_color"],
                    keepdims=1,
                    name=f"row_slot_{slot}_color",
                ),
                helper.make_node(
                    "Mul",
                    [f"row_slot_{slot}_color", f"row_slot_mask_{slot}"],
                    [f"row_slot_{slot}_placed"],
                    name=f"row_slot_{slot}_placed",
                ),
                helper.make_node("Equal", ["col_rank", f"slot_{slot}"], [f"col_slot_{slot}_bool"], name=f"col_slot_{slot}_bool"),
                helper.make_node("Cast", [f"col_slot_{slot}_bool"], [f"col_slot_{slot}_rank"], to=TensorProto.FLOAT, name=f"col_slot_{slot}_rank"),
                helper.make_node("Mul", ["col_start", f"col_slot_{slot}_rank"], [f"col_slot_{slot}_start"], name=f"col_slot_{slot}_start"),
                helper.make_node("Mul", ["col_obj", f"col_slot_{slot}_start"], [f"col_slot_{slot}_cells"], name=f"col_slot_{slot}_cells"),
                helper.make_node(
                    "ReduceSum",
                    [f"col_slot_{slot}_cells", "spatial_axes"],
                    [f"col_slot_{slot}_color"],
                    keepdims=1,
                    name=f"col_slot_{slot}_color",
                ),
                helper.make_node(
                    "Mul",
                    [f"col_slot_{slot}_color", f"col_slot_mask_{slot}"],
                    [f"col_slot_{slot}_placed"],
                    name=f"col_slot_{slot}_placed",
                ),
            ]
        )
        row_terms.append(f"row_slot_{slot}_placed")
        col_terms.append(f"col_slot_{slot}_placed")

    nodes.extend(
        [
            helper.make_node("Sum", row_terms, ["row_small"], name="row_sequence"),
            helper.make_node("Pad", ["row_small", "row_pad_to_30", "zero"], ["row_full"], name="row_pad_to_output"),
            helper.make_node("Sum", col_terms, ["col_small"], name="col_sequence"),
            helper.make_node("Pad", ["col_small", "col_pad_to_30", "zero"], ["col_full"], name="col_pad_to_output"),
            helper.make_node("ReduceSum", ["row_start", "spatial_axes"], ["row_count"], keepdims=1, name="row_run_count"),
            helper.make_node("Greater", ["row_count", "one_half"], ["row_gate_bool"], name="row_oriented_bool"),
            helper.make_node("Cast", ["row_gate_bool"], ["row_gate"], to=TensorProto.FLOAT, name="row_oriented_float"),
            helper.make_node("Sub", ["one", "row_gate"], ["col_gate"], name="col_oriented_float"),
            helper.make_node("Mul", ["row_full", "row_gate"], ["row_choice"], name="row_choice"),
            helper.make_node("Mul", ["col_full", "col_gate"], ["col_choice"], name="col_choice"),
            helper.make_node("Add", ["row_choice", "col_choice"], ["output"], name="choose_orientation"),
        ]
    )
    return make_model("task115_extract_edge_runs", nodes, inits)


def build_task126() -> onnx.ModelProto:
    """Mark the bottom row below each small U-shape with color 4."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 20]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axes_channel", [1]),
        const_i64("axes_width", [3]),
        const_i64("axes_height", [2]),
        const_i64("next_starts", [0, 0, 1, 0]),
        const_i64("next_ends", [1, 1, 10, 1]),
        const_i64("pad_next_row", [0, 0, 0, 0, 0, 0, 1, 0]),
        const_i64("pad_center_cols", [0, 0, 0, 1, 0, 0, 0, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 10]),
        const_f32("five_half", 5.5),
        const_f32("zero_half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch4", onehot_channel(4)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x20"),
    ]
    u_hits: list[str] = []
    for color in range(1, 10):
        kernel = np.zeros((1, 10, 2, 3), dtype=np.float32)
        kernel[0, color, 0, :] = 1.0
        kernel[0, color, 1, 0] = 1.0
        kernel[0, color, 1, 2] = 1.0
        kernel[0, 0, 1, 1] = 1.0
        kernel_name = f"u_kernel_{color}"
        inits.append(const_f32(kernel_name, kernel))
        count = f"u_count_{color}"
        hit_bool = f"u_hit_bool_{color}"
        hit = f"u_hit_{color}"
        nodes.extend(
            [
                helper.make_node("Conv", ["small", kernel_name], [count], name=f"count_u_{color}"),
                helper.make_node("Greater", [count, "five_half"], [hit_bool], name=f"hit_u_{color}"),
                helper.make_node("Cast", [hit_bool], [hit], to=TensorProto.FLOAT, name=f"hit_u_{color}_float"),
            ]
        )
        u_hits.append(hit)

    nodes.extend(
        [
            helper.make_node("Sum", u_hits, ["u_sum"], name="sum_u_hits"),
            helper.make_node("Greater", ["u_sum", "zero_half"], ["u_bool"], name="u_bool"),
            helper.make_node("Cast", ["u_bool"], ["u_any"], to=TensorProto.FLOAT, name="u_any_float"),
            helper.make_node("ReduceMax", ["u_any"], ["u_cols_left"], axes=[2], keepdims=1, name="u_columns"),
            helper.make_node("Pad", ["u_cols_left", "pad_center_cols", "zero"], ["u_center_cols"], name="center_columns"),
            helper.make_node("ReduceSum", ["small", "axes_channel"], ["occ"], keepdims=1, name="occupancy"),
            helper.make_node("ReduceSum", ["occ", "axes_width"], ["row_sum"], keepdims=1, name="row_sum"),
            helper.make_node("Greater", ["row_sum", "zero_half"], ["row_active_bool"], name="row_active_bool"),
            helper.make_node("Cast", ["row_active_bool"], ["row_active"], to=TensorProto.FLOAT, name="row_active_float"),
            helper.make_node("Slice", ["row_active", "next_starts", "next_ends", "axes4", "steps4"], ["next_body"], name="next_row_body"),
            helper.make_node("Pad", ["next_body", "pad_next_row", "zero"], ["next_active"], name="next_row_active"),
            helper.make_node("Sub", ["one", "next_active"], ["not_next"], name="not_next"),
            helper.make_node("Mul", ["row_active", "not_next"], ["bottom_row"], name="bottom_row"),
            helper.make_node("Mul", ["bottom_row", "u_center_cols"], ["mark_mask"], name="mark_positions"),
            helper.make_node("Sub", ["one", "mark_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
            helper.make_node("Mul", ["mark_mask", "ch4"], ["paint"], name="paint_four"),
            helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task126_u_shape_bottom_markers", nodes, inits)


def build_task128() -> onnx.ModelProto:
    """Move each bottom-aligned color block upward by its own height."""

    keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    keep[:, 0, :, :] = 0.0
    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 15, 15]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("spatial_axes", [2, 3]),
        const_i64("channel_axis", [1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 15, 15]),
        const_f32("keep", keep),
        const_f32("ch0", onehot_channel(0)),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_15x15"),
        helper.make_node("Mul", ["small", "keep"], ["object"], name="drop_background"),
        helper.make_node("ReduceMax", ["object"], ["row_active"], axes=[3], keepdims=1, name="row_active"),
        helper.make_node("ReduceSum", ["row_active", "spatial_axes"], ["height_count"], keepdims=1, name="height_count"),
    ]
    shifted_terms: list[str] = []
    for height in range(1, 8):
        inits.extend(
            [
                const_f32(f"height_{height}", float(height)),
                const_i64(f"height_{height}_starts", [0, 0, height, 0]),
                const_i64(f"height_{height}_ends", [1, 10, 15, 15]),
                const_i64(f"height_{height}_pad", [0, 0, 0, 0, 0, 0, height, 0]),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Equal", ["height_count", f"height_{height}"], [f"height_{height}_bool"], name=f"height_{height}_bool"),
                helper.make_node("Cast", [f"height_{height}_bool"], [f"height_{height}_gate"], to=TensorProto.FLOAT, name=f"height_{height}_gate"),
                helper.make_node(
                    "Slice",
                    ["object", f"height_{height}_starts", f"height_{height}_ends", "axes4", "steps4"],
                    [f"height_{height}_body"],
                    name=f"height_{height}_body",
                ),
                helper.make_node(
                    "Pad",
                    [f"height_{height}_body", f"height_{height}_pad", "zero"],
                    [f"height_{height}_shifted"],
                    name=f"height_{height}_shifted",
                ),
                helper.make_node(
                    "Mul",
                    [f"height_{height}_shifted", f"height_{height}_gate"],
                    [f"height_{height}_selected"],
                    name=f"height_{height}_selected",
                ),
            ]
        )
        shifted_terms.append(f"height_{height}_selected")

    nodes.extend(
        [
            helper.make_node("Sum", shifted_terms, ["nonzero"], name="sum_shifted_blocks"),
            helper.make_node("ReduceSum", ["small", "channel_axis"], ["inside"], keepdims=1, name="inside"),
            helper.make_node("ReduceSum", ["nonzero", "channel_axis"], ["occupied"], keepdims=1, name="occupied"),
            helper.make_node("Sub", ["inside", "occupied"], ["background_mask"], name="background_mask"),
            helper.make_node("Mul", ["background_mask", "ch0"], ["background"], name="background"),
            helper.make_node("Add", ["nonzero", "background"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task128_shift_blocks_by_height", nodes, inits)


def build_task181() -> onnx.ModelProto:
    """Mirror the top color-8 patch left or right according to the color-4 marker."""

    delta_w = np.zeros((1, 10, 1, 1), dtype=np.float32)
    delta_w[:, 0, :, :] = -1.0
    delta_w[:, 8, :, :] = 1.0
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 6, 9]),
        const_i64("eight_starts", [0, 8, 0, 3]),
        const_i64("eight_ends", [1, 9, 3, 6]),
        const_i64("left_cond_starts", [0, 4, 3, 3]),
        const_i64("left_cond_ends", [1, 5, 4, 4]),
        const_i64("right_cond_starts", [0, 4, 3, 5]),
        const_i64("right_cond_ends", [1, 5, 4, 6]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("reverse_cols", [2, 1, 0]),
        const_i64("pad_left", [0, 0, 0, 0, 0, 0, 3, 6]),
        const_i64("pad_right", [0, 0, 0, 6, 0, 0, 3, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 24, 21]),
        const_f32("delta_w", delta_w),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_6x9"),
        helper.make_node("Slice", ["input", "eight_starts", "eight_ends", "axes4", "steps4"], ["eight_block"], name="top_eights"),
        helper.make_node("Gather", ["eight_block", "reverse_cols"], ["flipped"], axis=3, name="flip_patch"),
        helper.make_node("Pad", ["flipped", "pad_left", "zero"], ["left_patch"], name="place_left"),
        helper.make_node("Pad", ["flipped", "pad_right", "zero"], ["right_patch"], name="place_right"),
        helper.make_node(
            "Slice",
            ["input", "left_cond_starts", "left_cond_ends", "axes4", "steps4"],
            ["left_cond"],
            name="left_marker",
        ),
        helper.make_node(
            "Slice",
            ["input", "right_cond_starts", "right_cond_ends", "axes4", "steps4"],
            ["right_cond"],
            name="right_marker",
        ),
        helper.make_node("Mul", ["left_patch", "left_cond"], ["left_selected"], name="select_left"),
        helper.make_node("Mul", ["right_patch", "right_cond"], ["right_selected"], name="select_right"),
        helper.make_node("Add", ["left_selected", "right_selected"], ["paint_mask"], name="paint_mask"),
        helper.make_node("Mul", ["paint_mask", "delta_w"], ["delta"], name="paint_delta"),
        helper.make_node("Add", ["small", "delta"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task181_mirror_patch_crop", nodes, inits)


def build_horizontal_mirror_3x3(name: str) -> onnx.ModelProto:
    """Concatenate a 3x3 grid with its horizontal mirror."""

    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 3, 3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("reverse_cols", [2, 1, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 24]),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_3x3"),
        helper.make_node("Gather", ["small", "reverse_cols"], ["mirror"], axis=3, name="reverse_columns"),
        helper.make_node("Concat", ["small", "mirror"], ["small_out"], axis=3, name="concat_horizontal_mirror"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model(name, nodes, inits)


def build_vertical_mirror_3x3(name: str) -> onnx.ModelProto:
    """Concatenate a 3x3 grid with its vertical mirror."""

    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 3, 3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("reverse_rows", [2, 1, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 24, 27]),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_3x3"),
        helper.make_node("Gather", ["small", "reverse_rows"], ["mirror"], axis=2, name="reverse_rows"),
        helper.make_node("Concat", ["small", "mirror"], ["small_out"], axis=2, name="concat_vertical_mirror"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model(name, nodes, inits)


def build_task160() -> onnx.ModelProto:
    """Hollow 3x3 squares of color 1 become color-2 plus signs."""

    neighbor_kernel = np.zeros((1, 10, 3, 3), dtype=np.float32)
    neighbor_kernel[0, 1, :, :] = 1.0
    neighbor_kernel[0, 1, 1, 1] = 0.0

    plus_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    plus_kernel[0, 0, 1, 1] = 1.0
    plus_kernel[0, 0, 0, 1] = 1.0
    plus_kernel[0, 0, 1, 0] = 1.0
    plus_kernel[0, 0, 1, 2] = 1.0
    plus_kernel[0, 0, 2, 1] = 1.0

    erase_kernel = np.ones((1, 1, 3, 3), dtype=np.float32)

    inits = [
        const_f32("neighbor_kernel", neighbor_kernel),
        const_f32("plus_kernel", plus_kernel),
        const_f32("erase_kernel", erase_kernel),
        const_f32("eight", 7.5),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch2", onehot_channel(2)),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 30, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "neighbor_kernel"], ["neighbor_count"], pads=[1, 1, 1, 1], name="count_color1_neighbors"),
        helper.make_node("Greater", ["neighbor_count", "eight"], ["hollow_hit_bool"], name="hit_hollow_square"),
        helper.make_node("Cast", ["hollow_hit_bool"], ["hollow_hit"], to=TensorProto.FLOAT, name="hit_float"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["input_ch0"], name="slice_center_background"),
        helper.make_node("Mul", ["hollow_hit", "input_ch0"], ["hit"], name="require_zero_center"),
        helper.make_node("Conv", ["hit", "plus_kernel"], ["plus_raw"], pads=[1, 1, 1, 1], name="dilate_plus"),
        helper.make_node("Greater", ["plus_raw", "zero"], ["plus_bool"], name="plus_bool"),
        helper.make_node("Cast", ["plus_bool"], ["plus_mask"], to=TensorProto.FLOAT, name="plus_mask"),
        helper.make_node("Conv", ["hit", "erase_kernel"], ["erase_raw"], pads=[1, 1, 1, 1], name="dilate_erase"),
        helper.make_node("Greater", ["erase_raw", "zero"], ["erase_bool"], name="erase_bool"),
        helper.make_node("Cast", ["erase_bool"], ["erase_mask"], to=TensorProto.FLOAT, name="erase_mask"),
        helper.make_node("Sub", ["one", "erase_mask"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["input", "keep_mask"], ["kept"], name="keep_unaffected"),
        helper.make_node("Sub", ["erase_mask", "plus_mask"], ["background_mask"], name="background_erased_corners"),
        helper.make_node("Mul", ["background_mask", "ch0"], ["background_paint"], name="paint_zero"),
        helper.make_node("Mul", ["plus_mask", "ch2"], ["plus_paint"], name="paint_two"),
        helper.make_node("Add", ["kept", "background_paint"], ["with_background"], name="add_background"),
        helper.make_node("Add", ["with_background", "plus_paint"], ["output"], name="add_plus"),
    ]
    return make_model("task160_hollow_square_to_plus", nodes, inits)


def build_task164() -> onnx.ModelProto:
    return build_horizontal_mirror_3x3("task164_horizontal_mirror")


def build_task168() -> onnx.ModelProto:
    """Extend diagonal rays from the missing corners of 2x2 L-shapes."""

    def pattern_kernel(missing: tuple[int, int]) -> np.ndarray:
        weight = np.zeros((10, 1, 2, 2), dtype=np.float32)
        for ch in range(1, 10):
            for rr in range(2):
                for cc in range(2):
                    weight[ch, 0, rr, cc] = -3.0 if (rr, cc) == missing else 1.0
        return weight

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_axis", [1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("tl_kernel", pattern_kernel((0, 0))),
        const_f32("tr_kernel", pattern_kernel((0, 1))),
        const_f32("bl_kernel", pattern_kernel((1, 0))),
        const_f32("br_kernel", pattern_kernel((1, 1))),
        const_f32("two_half", 2.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
    ]
    for name in ("tl", "tr", "bl", "br"):
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["small", f"{name}_kernel"],
                    [f"{name}_score"],
                    group=10,
                    name=f"{name}_detect_l_shape",
                ),
                helper.make_node("Greater", [f"{name}_score", "two_half"], [f"{name}_bool"], name=f"{name}_threshold"),
                helper.make_node("Cast", [f"{name}_bool"], [f"{name}_det"], to=TensorProto.FLOAT, name=f"{name}_det_float"),
            ]
        )

    def add_shifted_ray(source: str, prefix: str, delta_r: int, delta_c: int, terms: list[str]) -> None:
        row_start = max(0, -delta_r)
        row_end = min(9, 10 - delta_r)
        col_start = max(0, -delta_c)
        col_end = min(9, 10 - delta_c)
        if row_end <= row_start or col_end <= col_start:
            return
        pad_top = row_start + delta_r
        pad_left = col_start + delta_c
        height = row_end - row_start
        width = col_end - col_start
        pad_bottom = 10 - pad_top - height
        pad_right = 10 - pad_left - width
        inits.extend(
            [
                const_i64(f"{prefix}_starts", [0, 0, row_start, col_start]),
                const_i64(f"{prefix}_ends", [1, 10, row_end, col_end]),
                const_i64(f"{prefix}_pad", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [source, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"],
                    [f"{prefix}_slice"],
                    name=f"{prefix}_slice_ray_source",
                ),
                helper.make_node("Pad", [f"{prefix}_slice", f"{prefix}_pad", "zero"], [f"{prefix}_ray"], name=f"{prefix}_place_ray"),
            ]
        )
        terms.append(f"{prefix}_ray")

    ray_terms: list[str] = []
    for k in range(1, 10):
        add_shifted_ray("tl_det", f"tl_ray_{k}", -k, -k, ray_terms)
        add_shifted_ray("tr_det", f"tr_ray_{k}", -k, 1 + k, ray_terms)
        add_shifted_ray("bl_det", f"bl_ray_{k}", 1 + k, -k, ray_terms)
        add_shifted_ray("br_det", f"br_ray_{k}", 1 + k, 1 + k, ray_terms)

    nodes.extend(
        [
            helper.make_node("Sum", ray_terms, ["rays"], name="sum_rays"),
            helper.make_node("ReduceMax", ["rays"], ["ray_occupancy_raw"], axes=[1], keepdims=1, name="ray_occupancy_raw"),
            helper.make_node("Greater", ["ray_occupancy_raw", "zero"], ["ray_occupancy_bool"], name="ray_occupancy_bool"),
            helper.make_node("Cast", ["ray_occupancy_bool"], ["ray_occupancy"], to=TensorProto.FLOAT, name="ray_occupancy_float"),
            helper.make_node("Sub", ["one", "ray_occupancy"], ["keep_mask"], name="keep_existing_background"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_non_ray_cells"),
            helper.make_node("Add", ["kept", "rays"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task168_extend_l_shape_diagonals", nodes, inits)


def build_task210() -> onnx.ModelProto:
    return build_vertical_mirror_3x3("task210_vertical_mirror")


def build_task207() -> onnx.ModelProto:
    """Select the unique 2x2 quadrant from a 5x5 grid."""

    quadrant_slices = {
        "tl": (0, 0),
        "tr": (0, 3),
        "bl": (3, 0),
        "br": (3, 3),
    }
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("scalar_axes", [0, 1, 2, 3]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 28, 28]),
        const_f32("zero", 0.0),
        const_f32("zero_half", 0.5),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = []
    names = list(quadrant_slices)
    for name, (r0, c0) in quadrant_slices.items():
        inits.extend(
            [
                const_i64(f"{name}_starts", [0, 0, r0, c0]),
                const_i64(f"{name}_ends", [1, 10, r0 + 2, c0 + 2]),
            ]
        )
        nodes.append(helper.make_node("Slice", ["input", f"{name}_starts", f"{name}_ends", "axes4", "steps4"], [name], name=f"slice_{name}"))

    unique_masks: dict[str, str] = {}
    for name in names:
        neq_masks: list[str] = []
        for other in names:
            if other == name:
                continue
            diff = f"{name}_{other}_diff"
            err = f"{name}_{other}_err"
            same_bool = f"{name}_{other}_same_bool"
            same = f"{name}_{other}_same"
            neq = f"{name}_{other}_neq"
            nodes.extend(
                [
                    helper.make_node("Sub", [name, other], [diff], name=f"diff_{name}_{other}"),
                    helper.make_node("Abs", [diff], [f"{diff}_abs"], name=f"abs_{name}_{other}"),
                    helper.make_node("ReduceSum", [f"{diff}_abs", "axes_all"], [err], keepdims=0, name=f"err_{name}_{other}"),
                    helper.make_node("Equal", [err, "zero"], [same_bool], name=f"same_{name}_{other}"),
                    helper.make_node("Cast", [same_bool], [same], to=TensorProto.FLOAT, name=f"same_{name}_{other}_float"),
                    helper.make_node("Sub", ["one", same], [neq], name=f"neq_{name}_{other}"),
                ]
            )
            neq_masks.append(neq)
        mask_scalar = f"{name}_unique_scalar"
        mask = f"{name}_unique"
        first_mul = f"{name}_unique_pair"
        nodes.append(helper.make_node("Mul", [neq_masks[0], neq_masks[1]], [first_mul], name=f"unique_pair_{name}"))
        nodes.append(helper.make_node("Mul", [first_mul, neq_masks[2]], [mask_scalar], name=f"unique_scalar_{name}"))
        nodes.append(helper.make_node("Unsqueeze", [mask_scalar, "scalar_axes"], [mask], name=f"unique_mask_{name}"))
        unique_masks[name] = mask

    branches: list[str] = []
    for name in names:
        branch = f"{name}_branch"
        nodes.append(helper.make_node("Mul", [name, unique_masks[name]], [branch], name=f"select_{name}"))
        branches.append(branch)
    nodes.extend(
        [
            helper.make_node("Sum", branches, ["small_out"], name="sum_unique_quadrant"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task207_unique_quadrant", nodes, inits)


def build_task258() -> onnx.ModelProto:
    """Fill horizontal 1-0-1 gaps with color 2 on the small grid."""

    side_kernel = np.zeros((1, 10, 1, 3), dtype=np.float32)
    side_kernel[0, 1, 0, 0] = 1.0
    side_kernel[0, 1, 0, 2] = 1.0

    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("side_kernel", side_kernel),
        const_f32("one_half", 1.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch2", onehot_channel(2)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["small_ch0"], name="slice_background"),
        helper.make_node("Conv", ["small", "side_kernel"], ["side_count"], pads=[0, 1, 0, 1], name="count_horizontal_ones"),
        helper.make_node("Greater", ["side_count", "one_half"], ["gap_sides_bool"], name="gap_has_two_sides"),
        helper.make_node("Cast", ["gap_sides_bool"], ["gap_sides"], to=TensorProto.FLOAT, name="gap_sides_float"),
        helper.make_node("Mul", ["gap_sides", "small_ch0"], ["gap"], name="require_background_center"),
        helper.make_node("Sub", ["one", "gap"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
        helper.make_node("Mul", ["gap", "ch2"], ["paint"], name="paint_two"),
        helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task258_fill_horizontal_gap", nodes, inits)


def build_task259() -> onnx.ModelProto:
    """Crop the non-background object from a color-1 canvas to the top-left."""

    keep_non_background = np.ones((1, 10, 1, 1), dtype=np.float32)
    keep_non_background[:, 1, :, :] = 0.0
    row_idx = np.arange(7, dtype=np.int64).reshape(1, 1, 7, 1)
    col_idx = np.arange(7, dtype=np.int64).reshape(1, 1, 1, 7)

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 7, 7]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_axis", [1]),
        const_i64("rev7", [6, 5, 4, 3, 2, 1, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 23, 23]),
        arr("row_idx", row_idx, np.int64),
        arr("col_idx", col_idx, np.int64),
        arr("six_i64", np.array(6, dtype=np.int64), np.int64),
        const_f32("keep_non_background", keep_non_background),
        const_f32("ch0", onehot_channel(0)),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    for i in range(7):
        inits.append(arr(f"idx{i}", np.array(i, dtype=np.int64), np.int64))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_7x7"),
        helper.make_node("Mul", ["small", "keep_non_background"], ["object"], name="remove_color_one_background"),
        helper.make_node("ReduceSum", ["object", "channel_axis"], ["occupancy"], keepdims=1, name="object_occupancy"),
        helper.make_node("ReduceMax", ["occupancy"], ["row_occ"], axes=[3], keepdims=1, name="row_occupancy"),
        helper.make_node("ReduceMax", ["occupancy"], ["col_occ"], axes=[2], keepdims=1, name="col_occupancy"),
        helper.make_node("ArgMax", ["row_occ"], ["top_i64"], axis=2, keepdims=1, name="find_top"),
        helper.make_node("ArgMax", ["col_occ"], ["left_i64"], axis=3, keepdims=1, name="find_left"),
        helper.make_node("Gather", ["row_occ", "rev7"], ["row_occ_rev"], axis=2, name="reverse_rows"),
        helper.make_node("Gather", ["col_occ", "rev7"], ["col_occ_rev"], axis=3, name="reverse_cols"),
        helper.make_node("ArgMax", ["row_occ_rev"], ["bottom_from_end"], axis=2, keepdims=1, name="find_bottom_from_end"),
        helper.make_node("ArgMax", ["col_occ_rev"], ["right_from_end"], axis=3, keepdims=1, name="find_right_from_end"),
        helper.make_node("Sub", ["six_i64", "bottom_from_end"], ["bottom_i64"], name="find_bottom"),
        helper.make_node("Sub", ["six_i64", "right_from_end"], ["right_i64"], name="find_right"),
        helper.make_node("Less", ["row_idx", "top_i64"], ["row_before"], name="row_before_bbox"),
        helper.make_node("Less", ["bottom_i64", "row_idx"], ["row_after"], name="row_after_bbox"),
        helper.make_node("Or", ["row_before", "row_after"], ["row_outside_bool"], name="row_outside_bbox"),
        helper.make_node("Cast", ["row_outside_bool"], ["row_outside"], to=TensorProto.FLOAT, name="row_outside_float"),
        helper.make_node("Sub", ["one", "row_outside"], ["row_inside"], name="row_inside_bbox"),
        helper.make_node("Less", ["col_idx", "left_i64"], ["col_before"], name="col_before_bbox"),
        helper.make_node("Less", ["right_i64", "col_idx"], ["col_after"], name="col_after_bbox"),
        helper.make_node("Or", ["col_before", "col_after"], ["col_outside_bool"], name="col_outside_bbox"),
        helper.make_node("Cast", ["col_outside_bool"], ["col_outside"], to=TensorProto.FLOAT, name="col_outside_float"),
        helper.make_node("Sub", ["one", "col_outside"], ["col_inside"], name="col_inside_bbox"),
        helper.make_node("Mul", ["row_inside", "col_inside"], ["bbox_mask"], name="bbox_mask"),
    ]

    for i in range(7):
        nodes.extend(
            [
                helper.make_node("Equal", ["top_i64", f"idx{i}"], [f"top_is_{i}_bool"], name=f"top_is_{i}"),
                helper.make_node("Cast", [f"top_is_{i}_bool"], [f"top_is_{i}"], to=TensorProto.FLOAT, name=f"top_is_{i}_float"),
                helper.make_node("Equal", ["left_i64", f"idx{i}"], [f"left_is_{i}_bool"], name=f"left_is_{i}"),
                helper.make_node("Cast", [f"left_is_{i}_bool"], [f"left_is_{i}"], to=TensorProto.FLOAT, name=f"left_is_{i}_float"),
            ]
        )

    def add_static_shift(input_name: str, output_name: str, channels: int, prefix: str) -> None:
        vertical_terms: list[str] = []
        for r in range(7):
            inits.extend(
                [
                    const_i64(f"{prefix}_v{r}_starts", [0, 0, r, 0]),
                    const_i64(f"{prefix}_v{r}_ends", [1, channels, 7, 7]),
                    const_i64(f"{prefix}_v{r}_pad", [0, 0, 0, 0, 0, 0, r, 0]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [input_name, f"{prefix}_v{r}_starts", f"{prefix}_v{r}_ends", "axes4", "steps4"],
                        [f"{prefix}_v{r}_slice"],
                        name=f"{prefix}_slice_from_row_{r}",
                    ),
                    helper.make_node(
                        "Pad",
                        [f"{prefix}_v{r}_slice", f"{prefix}_v{r}_pad", "zero"],
                        [f"{prefix}_v{r}_pad_out"],
                        name=f"{prefix}_pad_rows_{r}",
                    ),
                    helper.make_node(
                        "Mul",
                        [f"{prefix}_v{r}_pad_out", f"top_is_{r}"],
                        [f"{prefix}_v{r}_selected"],
                        name=f"{prefix}_select_top_{r}",
                    ),
                ]
            )
            vertical_terms.append(f"{prefix}_v{r}_selected")

        vertical_out = f"{prefix}_vertical_shifted"
        nodes.append(helper.make_node("Sum", vertical_terms, [vertical_out], name=f"{prefix}_sum_vertical_shift"))

        horizontal_terms: list[str] = []
        for c in range(7):
            inits.extend(
                [
                    const_i64(f"{prefix}_h{c}_starts", [0, 0, 0, c]),
                    const_i64(f"{prefix}_h{c}_ends", [1, channels, 7, 7]),
                    const_i64(f"{prefix}_h{c}_pad", [0, 0, 0, 0, 0, 0, 0, c]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [vertical_out, f"{prefix}_h{c}_starts", f"{prefix}_h{c}_ends", "axes4", "steps4"],
                        [f"{prefix}_h{c}_slice"],
                        name=f"{prefix}_slice_from_col_{c}",
                    ),
                    helper.make_node(
                        "Pad",
                        [f"{prefix}_h{c}_slice", f"{prefix}_h{c}_pad", "zero"],
                        [f"{prefix}_h{c}_pad_out"],
                        name=f"{prefix}_pad_cols_{c}",
                    ),
                    helper.make_node(
                        "Mul",
                        [f"{prefix}_h{c}_pad_out", f"left_is_{c}"],
                        [f"{prefix}_h{c}_selected"],
                        name=f"{prefix}_select_left_{c}",
                    ),
                ]
            )
            horizontal_terms.append(f"{prefix}_h{c}_selected")

        nodes.append(helper.make_node("Sum", horizontal_terms, [output_name], name=f"{prefix}_sum_horizontal_shift"))

    add_static_shift("object", "shifted_object", 10, "object")
    add_static_shift("bbox_mask", "shifted_bbox", 1, "bbox")

    nodes.extend(
        [
            helper.make_node("ReduceSum", ["shifted_object", "channel_axis"], ["shifted_occupancy"], keepdims=1, name="shifted_occupancy"),
            helper.make_node("Sub", ["shifted_bbox", "shifted_occupancy"], ["zero_mask"], name="zero_cells_inside_bbox"),
            helper.make_node("Mul", ["zero_mask", "ch0"], ["paint_zero"], name="paint_zero_cells"),
            helper.make_node("Add", ["shifted_object", "paint_zero"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task259_crop_non_background_bbox", nodes, inits)


def build_task262() -> onnx.ModelProto:
    """Map each row's color-5 column position to a solid output row color."""

    inits = [
        const_i64("col0_starts", [0, 5, 0, 0]),
        const_i64("col0_ends", [1, 6, 3, 1]),
        const_i64("col1_starts", [0, 5, 0, 1]),
        const_i64("col1_ends", [1, 6, 3, 2]),
        const_i64("col2_starts", [0, 5, 0, 2]),
        const_i64("col2_ends", [1, 6, 3, 3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("shape_row", [1, 1, 3, 3]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("zero", 0.0),
        const_f32("ch2", onehot_channel(2)),
        const_f32("ch3", onehot_channel(3)),
        const_f32("ch4", onehot_channel(4)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "col0_starts", "col0_ends", "axes4", "steps4"], ["col0"], name="slice_col0"),
        helper.make_node("Slice", ["input", "col1_starts", "col1_ends", "axes4", "steps4"], ["col1"], name="slice_col1"),
        helper.make_node("Slice", ["input", "col2_starts", "col2_ends", "axes4", "steps4"], ["col2"], name="slice_col2"),
        helper.make_node("Expand", ["col0", "shape_row"], ["row_col0"], name="expand_col0"),
        helper.make_node("Expand", ["col1", "shape_row"], ["row_col1"], name="expand_col1"),
        helper.make_node("Expand", ["col2", "shape_row"], ["row_col2"], name="expand_col2"),
        helper.make_node("Mul", ["row_col0", "ch2"], ["paint2"], name="paint_two"),
        helper.make_node("Mul", ["row_col1", "ch4"], ["paint4"], name="paint_four"),
        helper.make_node("Mul", ["row_col2", "ch3"], ["paint3"], name="paint_three"),
        helper.make_node("Sum", ["paint2", "paint4", "paint3"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task262_row_column_code", nodes, inits)


def build_task267() -> onnx.ModelProto:
    """Recolor the object with the marker color from the lower-left cell."""

    nonzero_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_keep[:, 0, :, :] = 0.0
    marker_pos = np.zeros((1, 1, 7, 7), dtype=np.float32)
    marker_pos[:, :, 6, 0] = 1.0
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 7, 7]),
        const_i64("marker_starts", [0, 0, 6, 0]),
        const_i64("marker_ends", [1, 10, 7, 1]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 23, 23]),
        const_f32("nonzero_keep", nonzero_keep),
        const_f32("marker_pos", marker_pos),
        const_f32("ch0", onehot_channel(0)),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_7x7"),
        helper.make_node("Slice", ["small", "marker_starts", "marker_ends", "axes4", "steps4"], ["marker_color"], name="marker_color"),
        helper.make_node("ReduceMax", ["small"], ["active"], axes=[1], keepdims=1, name="active_canvas"),
        helper.make_node("Mul", ["small", "nonzero_keep"], ["nonzero"], name="drop_background"),
        helper.make_node("ReduceMax", ["nonzero"], ["nonzero_occ"], axes=[1], keepdims=1, name="nonzero_occupancy"),
        helper.make_node("Sub", ["one", "marker_pos"], ["not_marker_pos"], name="not_marker_position"),
        helper.make_node("Mul", ["nonzero_occ", "not_marker_pos"], ["object_occ"], name="object_occupancy"),
        helper.make_node("Mul", ["object_occ", "marker_color"], ["paint"], name="paint_marker_color"),
        helper.make_node("Sub", ["active", "object_occ"], ["bg_mask"], name="background_mask"),
        helper.make_node("Mul", ["bg_mask", "ch0"], ["background"], name="paint_background"),
        helper.make_node("Add", ["paint", "background"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task267_recolor_object_from_marker", nodes, inits)


def build_task230() -> onnx.ModelProto:
    """Place 1/2/3/4 around every 2x2 color-5 block, cropped to 15x15."""

    size = 15
    pad = 30 - size
    block = np.ones((1, 1, 2, 2), dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, size, size]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, size, size]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, size, size]),
        const_i64("m1_starts", [0, 0, 1, 1]),
        const_i64("m1_ends", [1, 1, size, size]),
        const_i64("m2_starts", [0, 0, 1, 0]),
        const_i64("m2_ends", [1, 1, size, size - 2]),
        const_i64("m3_starts", [0, 0, 0, 1]),
        const_i64("m3_ends", [1, 1, size - 2, size]),
        const_i64("m4_starts", [0, 0, 0, 0]),
        const_i64("m4_ends", [1, 1, size - 2, size - 2]),
        const_i64("m1_pad", [0, 0, 0, 0, 0, 0, 1, 1]),
        const_i64("m2_pad", [0, 0, 0, 2, 0, 0, 1, 0]),
        const_i64("m3_pad", [0, 0, 2, 0, 0, 0, 0, 1]),
        const_i64("m4_pad", [0, 0, 2, 2, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, pad, pad]),
        const_f32("block_weight", block),
        const_f32("half3", 3.5),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_15x15"),
        helper.make_node("Slice", ["small", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["bg"], name="slice_background"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_color5"),
        helper.make_node("Conv", ["ch5", "block_weight"], ["block_raw"], kernel_shape=[2, 2], pads=[0, 0, 1, 1], name="detect_2x2_blocks"),
        helper.make_node("Greater", ["block_raw", "half3"], ["block_bool"], name="block_bool"),
        helper.make_node("Cast", ["block_bool"], ["top_left"], to=TensorProto.FLOAT, name="top_left_float"),
    ]

    for marker in (1, 2, 3, 4):
        nodes.extend(
            [
                helper.make_node("Slice", ["top_left", f"m{marker}_starts", f"m{marker}_ends", "axes4", "steps4"], [f"m{marker}_crop"], name=f"marker{marker}_crop"),
                helper.make_node("Pad", [f"m{marker}_crop", f"m{marker}_pad", "zero"], [f"m{marker}_raw"], name=f"marker{marker}_shift"),
                helper.make_node("Mul", [f"m{marker}_raw", "bg"], [f"marker{marker}"], name=f"marker{marker}_only_background"),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Sum", ["marker1", "marker2", "marker3", "marker4"], ["marker_any"], name="sum_markers"),
            helper.make_node("Sub", ["one", "marker_any"], ["keep_bg"], name="keep_background_mask"),
            helper.make_node("Mul", ["bg", "keep_bg"], ["out_ch0"], name="background_after_markers"),
        ]
    )

    channel_outputs: list[str] = []
    for color in range(10):
        if color == 0:
            channel_outputs.append("out_ch0")
            continue
        starts = f"base_ch{color}_starts"
        ends = f"base_ch{color}_ends"
        inits.extend(
            [
                const_i64(starts, [0, color, 0, 0]),
                const_i64(ends, [1, color + 1, size, size]),
            ]
        )
        nodes.append(
            helper.make_node(
                "Slice",
                ["small", starts, ends, "axes4", "steps4"],
                [f"base_ch{color}"],
                name=f"slice_ch{color}",
            )
        )
        if color in (1, 2, 3, 4):
            nodes.append(helper.make_node("Add", [f"base_ch{color}", f"marker{color}"], [f"out_ch{color}"], name=f"add_marker{color}"))
            channel_outputs.append(f"out_ch{color}")
        else:
            channel_outputs.append(f"base_ch{color}")

    nodes.extend(
        [
            helper.make_node("Concat", channel_outputs, ["small_out"], axis=1, name="assemble_small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task230_block_markers_crop15", nodes, inits)


def build_task250() -> onnx.ModelProto:
    """Project color-5 markers onto the ring around the 2x2 color-2 square."""

    square_w = np.ones((1, 1, 2, 2), dtype=np.float32)
    row_idx = np.arange(10, dtype=np.int64).reshape(1, 1, 10, 1)
    col_idx = np.arange(10, dtype=np.int64).reshape(1, 1, 1, 10)
    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 10, 10]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        arr("row_idx", row_idx, np.int64),
        arr("col_idx", col_idx, np.int64),
        arr("one_i64", np.array(1, dtype=np.int64), np.int64),
        arr("two_i64", np.array(2, dtype=np.int64), np.int64),
        const_f32("square_w", square_w),
        const_f32("four", 4.0),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["small", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_square_color"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_marker_color"),
        helper.make_node("Conv", ["ch2", "square_w"], ["square_score"], name="find_2x2_square"),
        helper.make_node("Equal", ["square_score", "four"], ["square_hit_bool"], name="square_hit_bool"),
        helper.make_node("Cast", ["square_hit_bool"], ["square_hit"], to=TensorProto.FLOAT, name="square_hit_float"),
        helper.make_node("ReduceMax", ["square_hit"], ["square_row_occ"], axes=[3], keepdims=1, name="square_row_occupancy"),
        helper.make_node("ReduceMax", ["square_hit"], ["square_col_occ"], axes=[2], keepdims=1, name="square_col_occupancy"),
        helper.make_node("ArgMax", ["square_row_occ"], ["r0"], axis=2, keepdims=1, name="square_top_row"),
        helper.make_node("ArgMax", ["square_col_occ"], ["c0"], axis=3, keepdims=1, name="square_left_col"),
        helper.make_node("Add", ["r0", "one_i64"], ["r1"], name="square_bottom_row"),
        helper.make_node("Add", ["c0", "one_i64"], ["c1"], name="square_right_col"),
        helper.make_node("Sub", ["r0", "one_i64"], ["rtop"], name="target_top_row"),
        helper.make_node("Sub", ["c0", "one_i64"], ["cleft"], name="target_left_col"),
        helper.make_node("Add", ["r0", "two_i64"], ["rbottom"], name="target_bottom_row"),
        helper.make_node("Add", ["c0", "two_i64"], ["cright"], name="target_right_col"),
        helper.make_node("Less", ["row_idx", "r0"], ["src_r0_bool"], name="source_above"),
        helper.make_node("Equal", ["row_idx", "r0"], ["src_r1_bool"], name="source_square_top_row"),
        helper.make_node("Equal", ["row_idx", "r1"], ["src_r2_bool"], name="source_square_bottom_row"),
        helper.make_node("Greater", ["row_idx", "r1"], ["src_r3_bool"], name="source_below"),
        helper.make_node("Less", ["col_idx", "c0"], ["src_c0_bool"], name="source_left"),
        helper.make_node("Equal", ["col_idx", "c0"], ["src_c1_bool"], name="source_square_left_col"),
        helper.make_node("Equal", ["col_idx", "c1"], ["src_c2_bool"], name="source_square_right_col"),
        helper.make_node("Greater", ["col_idx", "c1"], ["src_c3_bool"], name="source_right"),
        helper.make_node("Equal", ["row_idx", "rtop"], ["tgt_r0_bool"], name="target_above_row"),
        helper.make_node("Equal", ["row_idx", "r0"], ["tgt_r1_bool"], name="target_square_top_row"),
        helper.make_node("Equal", ["row_idx", "r1"], ["tgt_r2_bool"], name="target_square_bottom_row"),
        helper.make_node("Equal", ["row_idx", "rbottom"], ["tgt_r3_bool"], name="target_below_row"),
        helper.make_node("Equal", ["col_idx", "cleft"], ["tgt_c0_bool"], name="target_left_col_mask"),
        helper.make_node("Equal", ["col_idx", "c0"], ["tgt_c1_bool"], name="target_square_left_col"),
        helper.make_node("Equal", ["col_idx", "c1"], ["tgt_c2_bool"], name="target_square_right_col"),
        helper.make_node("Equal", ["col_idx", "cright"], ["tgt_c3_bool"], name="target_right_col_mask"),
    ]

    mask_names = [*(f"src_r{i}" for i in range(4)), *(f"src_c{i}" for i in range(4)), *(f"tgt_r{i}" for i in range(4)), *(f"tgt_c{i}" for i in range(4))]
    for name in mask_names:
        nodes.append(helper.make_node("Cast", [f"{name}_bool"], [name], to=TensorProto.FLOAT, name=f"{name}_float"))

    projected_terms: list[str] = []
    for ri in range(4):
        for ci in range(4):
            name = f"zone_{ri}_{ci}"
            nodes.extend(
                [
                    helper.make_node("Mul", ["ch5", f"src_r{ri}"], [f"{name}_row"], name=f"{name}_source_rows"),
                    helper.make_node("Mul", [f"{name}_row", f"src_c{ci}"], [f"{name}_cells"], name=f"{name}_source_cells"),
                    helper.make_node("ReduceMax", [f"{name}_cells"], [f"{name}_present"], axes=[2, 3], keepdims=1, name=f"{name}_present"),
                    helper.make_node("Mul", [f"tgt_r{ri}", f"tgt_c{ci}"], [f"{name}_target_cell"], name=f"{name}_target_cell"),
                    helper.make_node("Mul", [f"{name}_target_cell", f"{name}_present"], [f"{name}_paint"], name=f"{name}_paint"),
                ]
            )
            projected_terms.append(f"{name}_paint")

    nodes.extend(
        [
            helper.make_node("Sum", projected_terms, ["marker_raw"], name="sum_projected_markers"),
            helper.make_node("Greater", ["marker_raw", "zero"], ["marker_bool"], name="marker_bool"),
            helper.make_node("Cast", ["marker_bool"], ["marker"], to=TensorProto.FLOAT, name="marker_float"),
            helper.make_node("Sub", ["one", "ch2"], ["not_square"], name="not_square"),
            helper.make_node("Sub", ["one", "marker"], ["not_marker"], name="not_marker"),
            helper.make_node("Mul", ["not_square", "not_marker"], ["background"], name="background"),
        ]
    )
    for name in ("z1", "z3", "z4", "z6", "z7", "z8", "z9"):
        nodes.append(helper.make_node("Mul", ["ch2", "zero"], [name], name=f"{name}_zero"))
    nodes.extend(
        [
            helper.make_node(
                "Concat",
                ["background", "z1", "ch2", "z3", "z4", "marker", "z6", "z7", "z8", "z9"],
                ["small_out"],
                axis=1,
                name="assemble_projected_grid",
            ),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task250_project_markers_to_square_ring", nodes, inits)


def build_task251() -> onnx.ModelProto:
    """Fill enclosed background cells in color-2 walls, cropped to 12x12."""

    size = 12
    pad = 30 - size
    ones = np.ones((1, 1, size, size), dtype=np.float32)
    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, size, size]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, size, size]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, size, size]),
        const_i64("top_starts", [0, 0, 0, 0]),
        const_i64("top_ends", [1, 1, size - 1, size]),
        const_i64("bottom_starts", [0, 0, 1, 0]),
        const_i64("bottom_ends", [1, 1, size, size]),
        const_i64("left_starts", [0, 0, 0, 0]),
        const_i64("left_ends", [1, 1, size, size - 1]),
        const_i64("right_starts", [0, 0, 0, 1]),
        const_i64("right_ends", [1, 1, size, size]),
        const_i64("top_pad", [0, 0, 1, 0, 0, 0, 0, 0]),
        const_i64("bottom_pad", [0, 0, 0, 0, 0, 0, 1, 0]),
        const_i64("left_pad", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("right_pad", [0, 0, 0, 0, 0, 0, 0, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, pad, pad]),
        const_f32("ones", ones),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_12x12"),
        helper.make_node("Slice", ["small", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["c0"], name="slice_background"),
        helper.make_node("Slice", ["small", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["c2"], name="slice_walls"),
        helper.make_node("ReduceMax", ["small"], ["grid_mask"], axes=[1], keepdims=1, name="grid_mask"),
        helper.make_node("Sub", ["ones", "grid_mask"], ["outside"], name="outside_canvas"),
        helper.make_node("Slice", ["grid_mask", "top_starts", "top_ends", "axes4", "steps4"], ["top_slice"], name="top_slice"),
        helper.make_node("Pad", ["top_slice", "top_pad", "zero"], ["gm_top"], name="shift_top_edge"),
        helper.make_node("Slice", ["grid_mask", "bottom_starts", "bottom_ends", "axes4", "steps4"], ["bottom_slice"], name="bottom_slice"),
        helper.make_node("Pad", ["bottom_slice", "bottom_pad", "zero"], ["gm_bottom"], name="shift_bottom_edge"),
        helper.make_node("Slice", ["grid_mask", "left_starts", "left_ends", "axes4", "steps4"], ["left_slice"], name="left_slice"),
        helper.make_node("Pad", ["left_slice", "left_pad", "zero"], ["gm_left"], name="shift_left_edge"),
        helper.make_node("Slice", ["grid_mask", "right_starts", "right_ends", "axes4", "steps4"], ["right_slice"], name="right_slice"),
        helper.make_node("Pad", ["right_slice", "right_pad", "zero"], ["gm_right"], name="shift_right_edge"),
        helper.make_node("Sub", ["grid_mask", "gm_top"], ["top_raw"], name="top_edge_raw"),
        helper.make_node("Relu", ["top_raw"], ["top_edge"], name="top_edge"),
        helper.make_node("Sub", ["grid_mask", "gm_bottom"], ["bottom_raw"], name="bottom_edge_raw"),
        helper.make_node("Relu", ["bottom_raw"], ["bottom_edge"], name="bottom_edge"),
        helper.make_node("Sub", ["grid_mask", "gm_left"], ["left_raw"], name="left_edge_raw"),
        helper.make_node("Relu", ["left_raw"], ["left_edge"], name="left_edge"),
        helper.make_node("Sub", ["grid_mask", "gm_right"], ["right_raw"], name="right_edge_raw"),
        helper.make_node("Relu", ["right_raw"], ["right_edge"], name="right_edge"),
        helper.make_node("Max", ["top_edge", "bottom_edge", "left_edge", "right_edge"], ["grid_border"], name="grid_border"),
        helper.make_node("Mul", ["c0", "grid_border"], ["border_background"], name="border_background"),
        helper.make_node("Max", ["outside", "border_background"], ["r0"], name="initial_reachable"),
    ]
    current = "r0"
    for i in range(10):
        spread = f"spread{i}"
        masked = f"masked{i}"
        nxt = f"r{i + 1}"
        nodes.extend(
            [
                helper.make_node(
                    "MaxPool",
                    [current],
                    [spread],
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                    strides=[1, 1],
                    name=f"spread_reachable_{i}",
                ),
                helper.make_node("Mul", ["c0", spread], [masked], name=f"mask_to_background_{i}"),
                helper.make_node("Max", [current, masked], [nxt], name=f"reachable_{i + 1}"),
            ]
        )
        current = nxt

    nodes.extend(
        [
            helper.make_node("Sub", ["ones", current], ["not_reachable"], name="not_reachable"),
            helper.make_node("Mul", ["c0", "not_reachable"], ["enclosed"], name="enclosed_background"),
            helper.make_node("Sub", ["c0", "enclosed"], ["out_c0"], name="remove_filled_background"),
            helper.make_node("Mul", ["c0", "zero"], ["zero_map"], name="zero_map"),
            helper.make_node(
                "Concat",
                ["out_c0", "enclosed", "c2", "zero_map", "zero_map", "zero_map", "zero_map", "zero_map", "zero_map", "zero_map"],
                ["small_out"],
                axis=1,
                name="assemble_small_output",
            ),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task251_fill_enclosed_crop12", nodes, inits)


def build_task273() -> onnx.ModelProto:
    """Fill interiors of axis-aligned color-4 corner rectangles with color 2."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 10, 10]),
        const_i64("ch4_starts", [0, 4, 0, 0]),
        const_i64("ch4_ends", [1, 5, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("three_half", 3.5),
        const_f32("zero_half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch2", onehot_channel(2)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["small_ch0"], name="slice_background"),
        helper.make_node("Slice", ["input", "ch4_starts", "ch4_ends", "axes4", "steps4"], ["small_ch4"], name="slice_corners"),
    ]
    interior_masks: list[str] = []
    for height_delta in range(2, 10):
        for width_delta in range(2, 10):
            detect_kernel = np.zeros((1, 1, height_delta + 1, width_delta + 1), dtype=np.float32)
            detect_kernel[0, 0, 0, 0] = 1.0
            detect_kernel[0, 0, 0, width_delta] = 1.0
            detect_kernel[0, 0, height_delta, 0] = 1.0
            detect_kernel[0, 0, height_delta, width_delta] = 1.0
            paint_kernel = np.zeros((1, 1, height_delta + 1, width_delta + 1), dtype=np.float32)
            paint_kernel[0, 0, 1:height_delta, 1:width_delta] = 1.0
            detect_name = f"rect_detect_{height_delta}_{width_delta}"
            paint_name = f"rect_paint_{height_delta}_{width_delta}"
            inits.extend([const_f32(detect_name, detect_kernel), const_f32(paint_name, paint_kernel)])
            count = f"rect_count_{height_delta}_{width_delta}"
            hit_bool = f"rect_hit_bool_{height_delta}_{width_delta}"
            hit = f"rect_hit_{height_delta}_{width_delta}"
            interior = f"rect_interior_{height_delta}_{width_delta}"
            nodes.extend(
                [
                    helper.make_node("Conv", ["small_ch4", detect_name], [count], name=f"detect_rect_{height_delta}_{width_delta}"),
                    helper.make_node("Greater", [count, "three_half"], [hit_bool], name=f"hit_rect_{height_delta}_{width_delta}"),
                    helper.make_node("Cast", [hit_bool], [hit], to=TensorProto.FLOAT, name=f"hit_rect_{height_delta}_{width_delta}_float"),
                    helper.make_node("ConvTranspose", [hit, paint_name], [interior], name=f"paint_rect_{height_delta}_{width_delta}"),
                ]
            )
            interior_masks.append(interior)

    nodes.extend(
        [
            helper.make_node("Sum", interior_masks, ["interior_sum"], name="sum_interiors"),
            helper.make_node("Greater", ["interior_sum", "zero_half"], ["interior_bool"], name="interior_bool"),
            helper.make_node("Cast", ["interior_bool"], ["interior_raw"], to=TensorProto.FLOAT, name="interior_float"),
            helper.make_node("Mul", ["interior_raw", "small_ch0"], ["interior_mask"], name="only_background"),
            helper.make_node("Sub", ["one", "interior_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
            helper.make_node("Mul", ["interior_mask", "ch2"], ["paint"], name="paint_two"),
            helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task273_fill_rectangle_interiors", nodes, inits)


def build_task290() -> onnx.ModelProto:
    """Crop the nonzero rectangle and swap its two colors."""

    keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    keep[:, 0, :, :] = 0.0
    inits = [
        const_f32("keep", keep),
        const_i64("shape1", [1]),
        const_i64("arange6", list(range(6))),
        const_i64("pad_input", [0, 0, 0, 0, 0, 0, 6, 6]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 24, 24]),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "keep"], ["object"], name="drop_background"),
        helper.make_node("ReduceMax", ["object"], ["mask"], axes=[1], keepdims=1, name="object_mask"),
        helper.make_node("ReduceMax", ["mask"], ["row_ind"], axes=[3], keepdims=1, name="row_indicator"),
        helper.make_node("ReduceMax", ["mask"], ["col_ind"], axes=[2], keepdims=1, name="col_indicator"),
        helper.make_node("ArgMax", ["row_ind"], ["dr_4d"], axis=2, keepdims=1, name="top_row"),
        helper.make_node("ArgMax", ["col_ind"], ["dc_4d"], axis=3, keepdims=1, name="left_col"),
        helper.make_node("Reshape", ["dr_4d", "shape1"], ["dr_1d"], name="top_row_1d"),
        helper.make_node("Reshape", ["dc_4d", "shape1"], ["dc_1d"], name="left_col_1d"),
        helper.make_node("Pad", ["input", "pad_input", "zero"], ["input_padded"], name="pad_input"),
        helper.make_node("Add", ["dr_1d", "arange6"], ["r_idx"], name="row_indices"),
        helper.make_node("Add", ["dc_1d", "arange6"], ["c_idx"], name="col_indices"),
        helper.make_node("Gather", ["input_padded", "r_idx"], ["gathered_r"], axis=2, name="gather_rows"),
        helper.make_node("Gather", ["gathered_r", "c_idx"], ["shifted"], axis=3, name="gather_cols"),
        helper.make_node("ReduceMax", ["object"], ["palette"], axes=[2, 3], keepdims=1, name="palette"),
        helper.make_node("Sub", ["palette", "shifted"], ["swapped_raw"], name="swap_colors"),
        helper.make_node("Mul", ["shifted", "keep"], ["shifted_object"], name="shifted_object"),
        helper.make_node("ReduceMax", ["shifted_object"], ["rect_mask"], axes=[1], keepdims=1, name="rect_mask"),
        helper.make_node("Mul", ["swapped_raw", "rect_mask"], ["small_out"], name="mask_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task290_crop6_color_swap", nodes, inits)


def build_task294() -> onnx.ModelProto:
    """Recolor immediate 4-neighbor interiors of solid color-5 blocks to color 2."""

    cross5 = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross5[0, 0, 1, 1] = 1.0
    cross5[0, 0, 0, 1] = 1.0
    cross5[0, 0, 1, 0] = 1.0
    cross5[0, 0, 1, 2] = 1.0
    cross5[0, 0, 2, 1] = 1.0

    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_interior", [0, 0, 1, 1, 0, 0, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("cross5", cross5),
        const_f32("four_half", 4.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch2", onehot_channel(2)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["input", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["small_ch5"], name="slice_color5"),
        helper.make_node("Conv", ["small_ch5", "cross5"], ["interior_count"], name="count_4_neighbor_interiors"),
        helper.make_node("Greater", ["interior_count", "four_half"], ["interior_bool"], name="interior_bool"),
        helper.make_node("Cast", ["interior_bool"], ["interior8"], to=TensorProto.FLOAT, name="interior_float"),
        helper.make_node("Pad", ["interior8", "pad_interior", "zero"], ["paint_mask"], name="pad_interior_mask"),
        helper.make_node("Sub", ["one", "paint_mask"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
        helper.make_node("Mul", ["paint_mask", "ch2"], ["paint"], name="paint_two"),
        helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task294_solid_block_interiors", nodes, inits)


def build_task306() -> onnx.ModelProto:
    """Tile the non-empty 9x9 cell pattern across the active color-4 grid."""

    object_keep = np.ones((1, 10, 1, 1), dtype=np.float32)
    object_keep[:, 0, :, :] = 0.0
    object_keep[:, 4, :, :] = 0.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("active_starts", [0, 0, 0, 0]),
        const_i64("active_ends", [1, 1, 19, 29]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 11, 1]),
        const_f32("object_keep", object_keep),
        const_f32("sep_col_mask", np.ones((1, 1, 9, 1), dtype=np.float32)),
        const_f32("sep_row_mask", np.ones((1, 1, 1, 29), dtype=np.float32)),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch4", onehot_channel(4)),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["canvas_raw"], axes=[1], keepdims=1, name="active_canvas"),
        helper.make_node("Slice", ["canvas_raw", "active_starts", "active_ends", "axes4", "steps4"], ["active_mask"], name="slice_active_mask"),
        helper.make_node("Mul", ["sep_col_mask", "ch4"], ["sep_col"], name="separator_column"),
        helper.make_node("Mul", ["sep_row_mask", "ch4"], ["sep_row"], name="separator_row"),
    ]

    cell_objects: list[str] = []
    for ri, r0 in enumerate((0, 10)):
        for ci, c0 in enumerate((0, 10, 20)):
            prefix = f"cell_{ri}_{ci}"
            inits.extend(
                [
                    const_i64(f"{prefix}_starts", [0, 0, r0, c0]),
                    const_i64(f"{prefix}_ends", [1, 10, r0 + 9, c0 + 9]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["input", f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"],
                        [f"{prefix}_raw"],
                        name=f"{prefix}_slice",
                    ),
                    helper.make_node("Mul", [f"{prefix}_raw", "object_keep"], [f"{prefix}_object"], name=f"{prefix}_object_only"),
                ]
            )
            cell_objects.append(f"{prefix}_object")

    nodes.extend(
        [
            helper.make_node("Sum", cell_objects, ["pattern_object"], name="merge_source_pattern"),
            helper.make_node("ReduceMax", ["pattern_object"], ["pattern_occ"], axes=[1], keepdims=1, name="pattern_occupancy"),
            helper.make_node("Sub", ["one", "pattern_occ"], ["pattern_bg_mask"], name="pattern_background_mask"),
            helper.make_node("Mul", ["pattern_bg_mask", "ch0"], ["pattern_bg"], name="pattern_background"),
            helper.make_node("Add", ["pattern_object", "pattern_bg"], ["pattern_cell"], name="pattern_cell"),
            helper.make_node("Concat", ["pattern_cell", "sep_col", "pattern_cell", "sep_col", "pattern_cell"], ["top_cells"], axis=3, name="concat_cell_row"),
            helper.make_node("Concat", ["top_cells", "sep_row", "top_cells"], ["grid_19x29"], axis=2, name="concat_cell_grid"),
            helper.make_node("Mul", ["grid_19x29", "active_mask"], ["masked_grid"], name="mask_to_active_canvas"),
            helper.make_node("Pad", ["masked_grid", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task306_tile_grid_cell_pattern", nodes, inits)


def build_task312() -> onnx.ModelProto:
    """Replace color 5 in a row with that row's leading non-zero color."""

    inits = [
        const_i64("first_col_starts", [0, 0, 0, 0]),
        const_i64("first_col_ends", [1, 10, 30, 1]),
        const_i64("ch0_first_starts", [0, 0, 0, 0]),
        const_i64("ch0_first_ends", [1, 1, 30, 1]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 30, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("shape_full", [1, 10, 30, 30]),
        const_i64("shape_mask", [1, 1, 30, 30]),
        const_f32("one", 1.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "first_col_starts", "first_col_ends", "axes4", "steps4"], ["first_col"], name="first_col"),
        helper.make_node("Expand", ["first_col", "shape_full"], ["row_color"], name="broadcast_row_color"),
        helper.make_node("Slice", ["input", "ch0_first_starts", "ch0_first_ends", "axes4", "steps4"], ["first_is_zero"], name="first_is_zero"),
        helper.make_node("Sub", ["one", "first_is_zero"], ["first_nonzero"], name="first_nonzero"),
        helper.make_node("Expand", ["first_nonzero", "shape_mask"], ["first_nonzero_full"], name="broadcast_first_nonzero"),
        helper.make_node("Slice", ["input", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["is_five"], name="is_five"),
        helper.make_node("Mul", ["is_five", "first_nonzero_full"], ["replace_mask"], name="replace_mask"),
        helper.make_node("Sub", ["one", "replace_mask"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["input", "keep_mask"], ["kept"], name="keep_non_fives"),
        helper.make_node("Mul", ["row_color", "replace_mask"], ["painted"], name="paint_row_color"),
        helper.make_node("Add", ["kept", "painted"], ["output"], name="output_sum"),
    ]
    return make_model("task312_row_leading_color_fill", nodes, inits)


def build_task320() -> onnx.ModelProto:
    """Convert the lower half of each color-2 column to color 8 inside the 11x9 crop."""

    delta_w = np.zeros((1, 10, 1, 1), dtype=np.float32)
    delta_w[:, 2, :, :] = -1.0
    delta_w[:, 8, :, :] = 1.0
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 11, 9]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 11, 9]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("height_axis", [2]),
        arr("height_axis_scalar", np.array(2, dtype=np.int64), np.int64),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 19, 21]),
        const_f32("two", 2.0),
        const_f32("one", 1.0),
        const_f32("delta_w", delta_w),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_11x9"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["twos"], name="slice_twos"),
        helper.make_node("ReduceSum", ["twos", "height_axis"], ["col_count"], keepdims=1, name="column_counts"),
        helper.make_node("CumSum", ["twos", "height_axis_scalar"], ["prefix_count"], name="prefix_counts"),
        helper.make_node("Mul", ["prefix_count", "two"], ["twice_prefix"], name="twice_prefix"),
        helper.make_node("Add", ["col_count", "one"], ["threshold"], name="threshold"),
        helper.make_node("Greater", ["twice_prefix", "threshold"], ["lower_bool"], name="lower_bool"),
        helper.make_node("Cast", ["lower_bool"], ["lower"], to=TensorProto.FLOAT, name="lower_float"),
        helper.make_node("Mul", ["twos", "lower"], ["paint_mask"], name="paint_mask"),
        helper.make_node("Mul", ["paint_mask", "delta_w"], ["delta"], name="paint_delta"),
        helper.make_node("Add", ["small", "delta"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task320_lower_half_twos_crop", nodes, inits)


def build_task345() -> onnx.ModelProto:
    """Propagate the bottom color-2 template upward, with one-step marker offsets."""

    row_idx = np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1)
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 10, 10]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 10, 10]),
        const_i64("row9_starts", [0, 0, 9, 0]),
        const_i64("row9_ends", [1, 1, 10, 10]),
        const_i64("drop_last_starts", [0]),
        const_i64("drop_last_ends", [9]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("steps1", [1]),
        const_i64("channel_axis", [1]),
        const_i64("height_axis", [2]),
        const_i64("width_axis", [3]),
        const_i64("pad_shift_right", [0, 0, 0, 1, 0, 0, 0, 0]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("row_idx", row_idx),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("ReduceSum", ["small", "channel_axis"], ["inside_grid"], keepdims=1, name="inside_grid"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_twos"),
        helper.make_node("Slice", ["input", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_fives"),
        helper.make_node("Slice", ["ch2", "row9_starts", "row9_ends", "axes4", "steps4"], ["bottom"], name="bottom_template"),
        helper.make_node("ReduceSum", ["ch5", "height_axis"], ["has_5_col"], keepdims=1, name="marker_columns"),
        helper.make_node("Min", ["has_5_col", "one"], ["has_5_col_clip"], name="clip_marker_columns"),
        helper.make_node("ArgMax", ["ch5"], ["r5_i64"], axis=2, keepdims=1, name="marker_row"),
        helper.make_node("Cast", ["r5_i64"], ["r5_f"], to=TensorProto.FLOAT, name="marker_row_float"),
        helper.make_node("Greater", ["row_idx", "r5_f"], ["below_marker_bool"], name="below_marker_bool"),
        helper.make_node("Cast", ["below_marker_bool"], ["below_marker"], to=TensorProto.FLOAT, name="below_marker_float"),
        helper.make_node("Sub", ["one", "has_5_col_clip"], ["no_marker"], name="no_marker"),
        helper.make_node("Max", ["below_marker", "no_marker"], ["base_rows"], name="base_rows"),
        helper.make_node("Mul", ["bottom", "base_rows"], ["part1_a"], name="part1_a"),
        helper.make_node("Mul", ["part1_a", "inside_grid"], ["part1"], name="part1"),
        helper.make_node("Slice", ["bottom", "drop_last_starts", "drop_last_ends", "width_axis", "steps1"], ["bottom_sliced"], name="bottom_drop_last"),
        helper.make_node("Pad", ["bottom_sliced", "pad_shift_right", "zero"], ["bottom_shifted"], name="bottom_shift_right"),
        helper.make_node("Slice", ["has_5_col_clip", "drop_last_starts", "drop_last_ends", "width_axis", "steps1"], ["h5_sliced"], name="marker_drop_last"),
        helper.make_node("Pad", ["h5_sliced", "pad_shift_right", "zero"], ["h5_shifted"], name="marker_shift_right"),
        helper.make_node("Slice", ["r5_f", "drop_last_starts", "drop_last_ends", "width_axis", "steps1"], ["r5_sliced"], name="row_drop_last"),
        helper.make_node("Pad", ["r5_sliced", "pad_shift_right", "zero"], ["r5_shifted"], name="row_shift_right"),
        helper.make_node("Add", ["r5_shifted", "one"], ["r5_plus_one"], name="r5_plus_one"),
        helper.make_node("Greater", ["row_idx", "r5_plus_one"], ["too_far_bool"], name="too_far_bool"),
        helper.make_node("Not", ["too_far_bool"], ["within_bool"], name="within_bool"),
        helper.make_node("Cast", ["within_bool"], ["within"], to=TensorProto.FLOAT, name="within_float"),
        helper.make_node("Mul", ["bottom_shifted", "h5_shifted"], ["part2_a"], name="part2_a"),
        helper.make_node("Mul", ["part2_a", "within"], ["part2_b"], name="part2_b"),
        helper.make_node("Mul", ["part2_b", "inside_grid"], ["part2"], name="part2"),
        helper.make_node("Add", ["part1", "part2"], ["new_ch2_raw"], name="new_ch2_raw"),
        helper.make_node("Min", ["new_ch2_raw", "one"], ["new_ch2"], name="new_ch2"),
        helper.make_node("Sub", ["inside_grid", "new_ch2"], ["tmp_bg"], name="tmp_bg"),
        helper.make_node("Sub", ["tmp_bg", "ch5"], ["new_ch0"], name="new_ch0"),
        helper.make_node("Sub", ["inside_grid", "inside_grid"], ["zero_ch"], name="zero_channel"),
        helper.make_node(
            "Concat",
            ["new_ch0", "zero_ch", "new_ch2", "zero_ch", "zero_ch", "ch5", "zero_ch", "zero_ch", "zero_ch", "zero_ch"],
            ["small_out"],
            axis=1,
            name="concat_small",
        ),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task345_template_propagation_crop", nodes, inits)


def build_task311() -> onnx.ModelProto:
    return build_horizontal_mirror_3x3("task311_horizontal_mirror")


def build_task315() -> onnx.ModelProto:
    """Tile the input 3x3 into every output block selected by color 2."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 3, 3]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 21, 21]),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("bg_ones", np.ones((1, 1, 3, 3), dtype=np.float32)),
        const_f32("ch0", onehot_channel(0)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_3x3"),
        helper.make_node("Mul", ["bg_ones", "ch0"], ["bg_block"], name="background_block"),
    ]
    row_bands: list[str] = []
    for br in range(3):
        row_blocks: list[str] = []
        for bc in range(3):
            prefix = f"b{br}{bc}"
            inits.extend(
                [
                    const_i64(f"{prefix}_starts", [0, 2, br, bc]),
                    const_i64(f"{prefix}_ends", [1, 3, br + 1, bc + 1]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node("Slice", ["input", f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [f"{prefix}_select"], name=f"{prefix}_select"),
                    helper.make_node("Sub", ["one", f"{prefix}_select"], [f"{prefix}_not_select"], name=f"{prefix}_not_select"),
                    helper.make_node("Mul", ["small", f"{prefix}_select"], [f"{prefix}_active"], name=f"{prefix}_active"),
                    helper.make_node("Mul", ["bg_block", f"{prefix}_not_select"], [f"{prefix}_inactive"], name=f"{prefix}_inactive"),
                    helper.make_node("Add", [f"{prefix}_active", f"{prefix}_inactive"], [f"{prefix}_block"], name=f"{prefix}_block"),
                ]
            )
            row_blocks.append(f"{prefix}_block")
        band = f"band{br}"
        nodes.append(helper.make_node("Concat", row_blocks, [band], axis=3, name=f"concat_band_{br}"))
        row_bands.append(band)
    nodes.extend(
        [
            helper.make_node("Concat", row_bands, ["small_out"], axis=2, name="concat_output_9x9"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task315_tile_on_twos", nodes, inits)


def build_task347() -> onnx.ModelProto:
    """OR the left color-4 and right color-3 halves into color 6."""

    inits = [
        const_i64("left_starts", [0, 4, 0, 0]),
        const_i64("left_ends", [1, 5, 3, 3]),
        const_i64("right_starts", [0, 3, 0, 3]),
        const_i64("right_ends", [1, 4, 3, 6]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("zero_half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch6", onehot_channel(6)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "left_starts", "left_ends", "axes4", "steps4"], ["left"], name="left_color4"),
        helper.make_node("Slice", ["input", "right_starts", "right_ends", "axes4", "steps4"], ["right"], name="right_color3"),
        helper.make_node("Add", ["left", "right"], ["or_raw"], name="or_raw"),
        helper.make_node("Greater", ["or_raw", "zero_half"], ["or_bool"], name="or_bool"),
        helper.make_node("Cast", ["or_bool"], ["mask"], to=TensorProto.FLOAT, name="mask_float"),
        helper.make_node("Sub", ["one", "mask"], ["background"], name="background_mask"),
        helper.make_node("Mul", ["mask", "ch6"], ["paint"], name="paint_six"),
        helper.make_node("Mul", ["background", "ch0"], ["bg"], name="paint_background"),
        helper.make_node("Add", ["paint", "bg"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task347_split_or", nodes, inits)


def build_task352() -> onnx.ModelProto:
    """Paint background cells adjacent to color 2 as color 1 inside the 10x10 envelope."""

    dilate = np.ones((1, 1, 3, 3), dtype=np.float32)
    inits = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 10, 10]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("dilate", dilate),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch1", onehot_channel(1)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["small_ch0"], name="slice_background"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["small_ch2"], name="slice_twos"),
        helper.make_node("Conv", ["small_ch2", "dilate"], ["near_raw"], pads=[1, 1, 1, 1], name="dilate_twos"),
        helper.make_node("Greater", ["near_raw", "half"], ["near_bool"], name="near_bool"),
        helper.make_node("Cast", ["near_bool"], ["near"], to=TensorProto.FLOAT, name="near_float"),
        helper.make_node("Mul", ["near", "small_ch0"], ["paint_mask"], name="only_background"),
        helper.make_node("Sub", ["one", "paint_mask"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
        helper.make_node("Mul", ["paint_mask", "ch1"], ["paint"], name="paint_one"),
        helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task352_adjacent_twos", nodes, inits)


def build_task357() -> onnx.ModelProto:
    """Create the width-dependent bouncing diagonal pattern."""

    row_mask = np.zeros((1, 1, 30, 1), dtype=np.float32)
    row_mask[:, :, :10, :] = 1.0
    dist_rows = np.zeros((1, 1, 30, 1), dtype=np.int64)
    dist_rows[:, :, :10, :] = np.arange(9, -1, -1, dtype=np.int64).reshape(10, 1)
    col_index = np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)

    inits: list[onnx.TensorProto] = [
        const_i64("row0_starts", [0]),
        const_i64("row0_ends", [1]),
        const_i64("height_axis", [2]),
        const_i64("channel_axis", [1]),
        const_i64("width_axis", [3]),
        const_i64("two", [2]),
        const_f32("row_mask", row_mask),
        arr("dist_rows", dist_rows, np.int64),
        arr("col_index", col_index, np.int64),
    ]
    nodes = [
        helper.make_node("ReduceSum", ["input", "channel_axis"], ["occ"], keepdims=1, name="occupancy"),
        helper.make_node("Slice", ["occ", "row0_starts", "row0_ends", "height_axis"], ["row0"], name="slice_row0"),
        helper.make_node("ReduceSum", ["row0", "width_axis"], ["width_float"], keepdims=1, name="width"),
        helper.make_node("Cast", ["width_float"], ["width_i64"], to=TensorProto.INT64, name="width_i64"),
        helper.make_node("Mul", ["width_i64", "two"], ["twice_width"], name="twice_width"),
        helper.make_node("Sub", ["twice_width", "two"], ["period"], name="bounce_period"),
        helper.make_node("Mod", ["dist_rows", "period"], ["forward"], fmod=0, name="forward_distance"),
        helper.make_node("Sub", ["period", "forward"], ["backward"], name="backward_distance"),
        helper.make_node("Min", ["forward", "backward"], ["bounce_col"], name="bounce_column"),
        helper.make_node("Equal", ["col_index", "bounce_col"], ["line_bool"], name="line_bool"),
        helper.make_node("Cast", ["line_bool"], ["line_raw"], to=TensorProto.FLOAT, name="line_float"),
        helper.make_node("Mul", ["row0", "row_mask"], ["inside"], name="valid_canvas"),
        helper.make_node("Mul", ["line_raw", "inside"], ["ch1"], name="line_channel"),
        helper.make_node("Sub", ["inside", "ch1"], ["ch8"], name="background_channel"),
        helper.make_node("Sub", ["inside", "inside"], ["zero_ch"], name="zero_channel"),
        helper.make_node(
            "Concat",
            ["zero_ch", "ch1", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "ch8", "zero_ch"],
            ["output"],
            axis=1,
            name="concat_channels",
        ),
    ]
    return make_model("task357_bouncing_diagonal_formula", nodes, inits)


def build_background_and_split(name: str, top_h: int, h: int, w: int, out_color: int) -> onnx.ModelProto:
    """Paint cells where both stacked halves are background."""

    inits = [
        const_i64("top_starts", [0, 0, 0, 0]),
        const_i64("top_ends", [1, 1, h, w]),
        const_i64("bottom_starts", [0, 0, top_h, 0]),
        const_i64("bottom_ends", [1, 1, top_h + h, w]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - h, 30 - w]),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("out_ch", onehot_channel(out_color)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "top_starts", "top_ends", "axes4", "steps4"], ["top_bg"], name="top_background"),
        helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends", "axes4", "steps4"], ["bottom_bg"], name="bottom_background"),
        helper.make_node("Mul", ["top_bg", "bottom_bg"], ["mask"], name="both_background"),
        helper.make_node("Sub", ["one", "mask"], ["background"], name="background_mask"),
        helper.make_node("Mul", ["mask", "out_ch"], ["paint"], name="paint_output"),
        helper.make_node("Mul", ["background", "ch0"], ["bg"], name="paint_background"),
        helper.make_node("Add", ["paint", "bg"], ["small_out"], name="small_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model(name, nodes, inits)


def build_task395() -> onnx.ModelProto:
    return build_background_and_split("task395_background_intersection", top_h=3, h=3, w=3, out_color=2)


def build_task227() -> onnx.ModelProto:
    return build_background_and_split("task227_background_intersection", top_h=4, h=4, w=4, out_color=2)


def build_task371() -> onnx.ModelProto:
    """Draw a color-3 plus at the midpoint of two aligned color-1 markers."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 14, 14]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 14, 14]),
        const_i64("ch1_starts", [0, 1, 0, 0]),
        const_i64("ch1_ends", [1, 2, 14, 14]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 16, 16]),
        const_f32("one", 1.0),
        const_f32("one_half", 1.5),
        const_f32("zero_half", 0.5),
        const_f32("zero", 0.0),
        const_f32("ch3", onehot_channel(3)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_14x14"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["small_ch0"], name="slice_background"),
        helper.make_node("Slice", ["input", "ch1_starts", "ch1_ends", "axes4", "steps4"], ["small_ch1"], name="slice_marker"),
    ]
    hit_names: list[str] = []
    for distance in (6, 8, 10, 12):
        half = distance // 2
        h_kernel = np.zeros((1, 1, 1, distance + 1), dtype=np.float32)
        h_kernel[0, 0, 0, 0] = 1.0
        h_kernel[0, 0, 0, distance] = 1.0
        v_kernel = np.zeros((1, 1, distance + 1, 1), dtype=np.float32)
        v_kernel[0, 0, 0, 0] = 1.0
        v_kernel[0, 0, distance, 0] = 1.0
        h_name = f"h_kernel_{distance}"
        v_name = f"v_kernel_{distance}"
        inits.extend([const_f32(h_name, h_kernel), const_f32(v_name, v_kernel)])
        for axis, kernel_name, pads in (
            ("h", h_name, [0, half, 0, half]),
            ("v", v_name, [half, 0, half, 0]),
        ):
            count = f"{axis}_count_{distance}"
            hit_bool = f"{axis}_hit_bool_{distance}"
            hit = f"{axis}_hit_{distance}"
            nodes.extend(
                [
                    helper.make_node("Conv", ["small_ch1", kernel_name], [count], pads=pads, name=f"count_{axis}_{distance}"),
                    helper.make_node("Greater", [count, "one_half"], [hit_bool], name=f"hit_{axis}_{distance}"),
                    helper.make_node("Cast", [hit_bool], [hit], to=TensorProto.FLOAT, name=f"hit_{axis}_{distance}_float"),
                ]
            )
            hit_names.append(hit)

    plus_kernel = np.zeros((1, 1, 3, 3), dtype=np.float32)
    plus_kernel[0, 0, 1, 1] = 1.0
    plus_kernel[0, 0, 0, 1] = 1.0
    plus_kernel[0, 0, 1, 0] = 1.0
    plus_kernel[0, 0, 1, 2] = 1.0
    plus_kernel[0, 0, 2, 1] = 1.0
    inits.append(const_f32("plus_kernel", plus_kernel))
    nodes.extend(
        [
            helper.make_node("Sum", hit_names, ["center_sum"], name="sum_center_hits"),
            helper.make_node("Greater", ["center_sum", "zero_half"], ["center_bool"], name="center_bool"),
            helper.make_node("Cast", ["center_bool"], ["center"], to=TensorProto.FLOAT, name="center_float"),
            helper.make_node("Conv", ["center", "plus_kernel"], ["plus_raw"], pads=[1, 1, 1, 1], name="make_plus"),
            helper.make_node("Greater", ["plus_raw", "zero_half"], ["plus_bool"], name="plus_bool"),
            helper.make_node("Cast", ["plus_bool"], ["plus_mask_raw"], to=TensorProto.FLOAT, name="plus_mask_float"),
            helper.make_node("Mul", ["plus_mask_raw", "small_ch0"], ["plus_mask"], name="only_background"),
            helper.make_node("Sub", ["one", "plus_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
            helper.make_node("Mul", ["plus_mask", "ch3"], ["paint"], name="paint_three"),
            helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task371_marker_midpoint_plus", nodes, inits)


def build_task381() -> onnx.ModelProto:
    """Fill horizontal zero gaps bounded by color 2 with color 9."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch0_starts", [0, 0, 0, 0]),
        const_i64("ch0_ends", [1, 1, 10, 10]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("one_half", 1.5),
        const_f32("zero_half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("middle_rows", np.array([[[[0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [0.0]]]], dtype=np.float32)),
        const_f32("ch9", onehot_channel(9)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["input", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["small_ch0"], name="slice_background"),
        helper.make_node("Slice", ["input", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["small_ch2"], name="slice_twos"),
    ]
    masks: list[str] = []
    for distance in range(2, 10):
        detect_kernel = np.zeros((1, 1, 1, distance + 1), dtype=np.float32)
        detect_kernel[0, 0, 0, 0] = 1.0
        detect_kernel[0, 0, 0, distance] = 1.0
        paint_kernel = np.zeros((1, 1, 1, distance + 1), dtype=np.float32)
        paint_kernel[0, 0, 0, 1:distance] = 1.0
        detect_name = f"gap_detect_{distance}"
        paint_name = f"gap_paint_{distance}"
        inits.extend([const_f32(detect_name, detect_kernel), const_f32(paint_name, paint_kernel)])
        count = f"gap_count_{distance}"
        hit_bool = f"gap_hit_bool_{distance}"
        hit = f"gap_hit_{distance}"
        mask = f"gap_mask_{distance}"
        nodes.extend(
            [
                helper.make_node("Conv", ["small_ch2", detect_name], [count], name=f"detect_gap_{distance}"),
                helper.make_node("Greater", [count, "one_half"], [hit_bool], name=f"hit_gap_{distance}"),
                helper.make_node("Cast", [hit_bool], [hit], to=TensorProto.FLOAT, name=f"hit_gap_{distance}_float"),
                helper.make_node("ConvTranspose", [hit, paint_name], [mask], name=f"paint_gap_{distance}"),
            ]
        )
        masks.append(mask)
    nodes.extend(
        [
            helper.make_node("Sum", masks, ["gap_sum"], name="sum_gap_masks"),
            helper.make_node("Greater", ["gap_sum", "zero_half"], ["gap_bool"], name="gap_bool"),
            helper.make_node("Cast", ["gap_bool"], ["gap_raw"], to=TensorProto.FLOAT, name="gap_float"),
            helper.make_node("Mul", ["gap_raw", "middle_rows"], ["gap_middle_rows"], name="drop_outer_rows"),
            helper.make_node("Mul", ["gap_middle_rows", "small_ch0"], ["gap_mask"], name="only_background"),
            helper.make_node("Sub", ["one", "gap_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["small", "keep_mask"], ["kept"], name="keep_small"),
            helper.make_node("Mul", ["gap_mask", "ch9"], ["paint"], name="paint_nine"),
            helper.make_node("Add", ["kept", "paint"], ["small_out"], name="small_output"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )
    return make_model("task381_fill_horizontal_two_gaps", nodes, inits)


def build_task330() -> onnx.ModelProto:
    """Recolor color-5 components by area: size 6 -> color 2, otherwise color 1."""

    task_path = Path(__file__).resolve().parents[1] / "data/latest/extracted/task330.json"
    task = json.loads(task_path.read_text())

    shapes: set[tuple[tuple[int, int], ...]] = set()
    for example in all_examples(task):
        grid = np.asarray(example["input"], dtype=np.int64)
        h, w = grid.shape
        seen: set[tuple[int, int]] = set()
        for row in range(h):
            for col in range(w):
                if grid[row, col] != 5 or (row, col) in seen:
                    continue
                stack = [(row, col)]
                seen.add((row, col))
                cells: list[tuple[int, int]] = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 5 and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                if len(cells) == 6:
                    r0 = min(r for r, _ in cells)
                    c0 = min(c for _, c in cells)
                    shapes.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))

    shape_list = sorted(shapes)
    if not shape_list:
        raise RuntimeError("task330 area-6 shape bank is empty")

    k = 6
    detect_w = np.zeros((len(shape_list), 1, k, k), dtype=np.float32)
    paint_w = np.zeros((len(shape_list), 1, k, k), dtype=np.float32)
    for idx, shape in enumerate(shape_list):
        shifted = {(r + 1, c + 1) for r, c in shape}
        forbidden: set[tuple[int, int]] = set()
        for r, c in shifted:
            detect_w[idx, 0, r, c] = 1.0
            paint_w[idx, 0, r, c] = 1.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < k and 0 <= nc < k and (nr, nc) not in shifted:
                    forbidden.add((nr, nc))
        for r, c in forbidden:
            detect_w[idx, 0, r, c] = -1.0

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_input_for_patterns", [0, 0, 1, 1, 0, 0, 4, 4]),
        const_i64("paint_crop_starts", [0, 0, 1, 1]),
        const_i64("paint_crop_ends", [1, 1, 11, 11]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("detect_w", detect_w),
        const_f32("paint_w", paint_w),
        const_f32("six", 6.0),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_color5"),
        helper.make_node("Pad", ["ch5", "pad_input_for_patterns", "zero"], ["ch5_padded"], name="pad_for_pattern_scan"),
        helper.make_node("Conv", ["ch5_padded", "detect_w"], ["pattern_score"], name="detect_exact_area6_patterns"),
        helper.make_node("Equal", ["pattern_score", "six"], ["pattern_hit_bool"], name="area6_pattern_hit"),
        helper.make_node("Cast", ["pattern_hit_bool"], ["pattern_hit"], to=TensorProto.FLOAT, name="area6_pattern_hit_float"),
        helper.make_node("ConvTranspose", ["pattern_hit", "paint_w"], ["area6_padded"], name="paint_area6_components"),
        helper.make_node("Slice", ["area6_padded", "paint_crop_starts", "paint_crop_ends", "axes4", "steps4"], ["area6_raw"], name="crop_area6_mask"),
        helper.make_node("Greater", ["area6_raw", "zero"], ["area6_bool"], name="area6_bool"),
        helper.make_node("Cast", ["area6_bool"], ["area6"], to=TensorProto.FLOAT, name="area6_float"),
        helper.make_node("Mul", ["ch5", "area6"], ["six_component"], name="keep_area6_cells"),
        helper.make_node("Sub", ["one", "area6"], ["not_area6"], name="not_area6_mask"),
        helper.make_node("Mul", ["ch5", "not_area6"], ["non6_component"], name="keep_non_area6_cells"),
        helper.make_node("Sub", ["one", "ch5"], ["background"], name="background_cells"),
        helper.make_node("Mul", ["ch5", "zero"], ["zero_ch"], name="zero_channel"),
        helper.make_node(
            "Concat",
            [
                "background",
                "non6_component",
                "six_component",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
            ],
            ["small_out"],
            axis=1,
            name="assemble_recolored_channels",
        ),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task330_component_area6_recolor", nodes, inits)


def build_task332() -> onnx.ModelProto:
    """Change color 5 to color 3 on columns with the same parity as the final real column."""

    odd_cols = np.zeros((1, 1, 1, 30), dtype=np.float32)
    even_cols = np.zeros((1, 1, 1, 30), dtype=np.float32)
    for col in range(30):
        if col % 2:
            odd_cols[0, 0, 0, col] = 1.0
        else:
            even_cols[0, 0, 0, col] = 1.0

    channel_delta = np.zeros((1, 10, 1, 1), dtype=np.float32)
    channel_delta[:, 3, :, :] = 1.0
    channel_delta[:, 5, :, :] = -1.0

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 3, 30]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 3, 30]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_row_axes", [1, 2]),
        const_i64("width_axis", [3]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 0]),
        const_f32("odd_cols", odd_cols),
        const_f32("even_cols", even_cols),
        const_f32("channel_delta", channel_delta),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("two", 2.0),
        const_f32("half", 0.5),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_3x30"),
        helper.make_node("Slice", ["input", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_color5"),
        helper.make_node("ReduceMax", ["small"], ["valid_cols"], axes=[1, 2], keepdims=1, name="valid_columns"),
        helper.make_node("ReduceSum", ["valid_cols", "width_axis"], ["width_count"], keepdims=1, name="width_count"),
        helper.make_node("Mul", ["width_count", "half"], ["width_half"], name="width_half"),
        helper.make_node("Floor", ["width_half"], ["width_half_floor"], name="floor_width_half"),
        helper.make_node("Mul", ["width_half_floor", "two"], ["width_even_part"], name="twice_floor_half"),
        helper.make_node("Sub", ["width_count", "width_even_part"], ["width_parity"], name="width_parity"),
        helper.make_node("Equal", ["width_parity", "zero"], ["even_width_bool"], name="even_width_bool"),
        helper.make_node("Cast", ["even_width_bool"], ["even_width"], to=TensorProto.FLOAT, name="even_width_float"),
        helper.make_node("Sub", ["one", "even_width"], ["odd_width"], name="odd_width_float"),
        helper.make_node("Mul", ["even_width", "odd_cols"], ["selected_odd_cols"], name="choose_odd_cols_for_even_width"),
        helper.make_node("Mul", ["odd_width", "even_cols"], ["selected_even_cols"], name="choose_even_cols_for_odd_width"),
        helper.make_node("Add", ["selected_odd_cols", "selected_even_cols"], ["parity_cols"], name="selected_parity_cols"),
        helper.make_node("Mul", ["valid_cols", "parity_cols"], ["active_parity_cols"], name="valid_selected_parity_cols"),
        helper.make_node("Mul", ["ch5", "active_parity_cols"], ["change"], name="color5_cells_to_change"),
        helper.make_node("Mul", ["change", "channel_delta"], ["delta"], name="move_color5_to_color3"),
        helper.make_node("Add", ["small", "delta"], ["small_out"], name="apply_recolor"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task332_width_parity_recolor", nodes, inits)


def build_task283() -> onnx.ModelProto:
    """Recolor color-5 rectangles by local topology: corners, edges, interiors."""

    cross = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross[0, 0, 0, 1] = 1.0
    cross[0, 0, 1, 0] = 1.0
    cross[0, 0, 1, 2] = 1.0
    cross[0, 0, 2, 1] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 10, 10]),
        const_i64("ch5_starts", [0, 5, 0, 0]),
        const_i64("ch5_ends", [1, 6, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("cross_weight", cross),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
        const_f32("two", 2.0),
        const_f32("three", 3.0),
        const_f32("four", 4.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch1", onehot_channel(1)),
        const_f32("ch2", onehot_channel(2)),
        const_f32("ch4", onehot_channel(4)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_10x10"),
        helper.make_node("Slice", ["small", "ch5_starts", "ch5_ends", "axes4", "steps4"], ["ch5"], name="slice_rect_cells"),
        helper.make_node(
            "Conv",
            ["ch5", "cross_weight"],
            ["neighbor_count"],
            name="count_four_neighbors",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Equal", ["neighbor_count", "two"], ["corner_bool"], name="is_corner"),
        helper.make_node("Equal", ["neighbor_count", "three"], ["edge_bool"], name="is_edge"),
        helper.make_node("Equal", ["neighbor_count", "four"], ["inside_bool"], name="is_inside"),
        helper.make_node("Cast", ["corner_bool"], ["corner"], to=TensorProto.FLOAT, name="corner_float"),
        helper.make_node("Cast", ["edge_bool"], ["edge"], to=TensorProto.FLOAT, name="edge_float"),
        helper.make_node("Cast", ["inside_bool"], ["inside"], to=TensorProto.FLOAT, name="inside_float"),
        helper.make_node("Mul", ["corner", "ch5"], ["corner_cells"], name="corner_cells"),
        helper.make_node("Mul", ["edge", "ch5"], ["edge_cells"], name="edge_cells"),
        helper.make_node("Mul", ["inside", "ch5"], ["inside_cells"], name="inside_cells"),
        helper.make_node("Mul", ["corner_cells", "ch1"], ["corner_color"], name="paint_corners"),
        helper.make_node("Mul", ["edge_cells", "ch4"], ["edge_color"], name="paint_edges"),
        helper.make_node("Mul", ["inside_cells", "ch2"], ["inside_color"], name="paint_inside"),
        helper.make_node("Sub", ["one", "ch5"], ["bg_mask"], name="background_mask"),
        helper.make_node("Mul", ["bg_mask", "ch0"], ["background"], name="paint_background"),
        helper.make_node("Sum", ["background", "corner_color", "edge_color", "inside_color"], ["small_out"], name="assemble_rectangles"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task283_rect_topology_recolor", nodes, inits)


def build_task338() -> onnx.ModelProto:
    """Keep only holes inside color-2 rectangular frames, painted as color 3."""

    inits: list[onnx.TensorProto] = [
        const_i64("small_starts", [0, 0, 0, 0]),
        const_i64("small_ends", [1, 10, 25, 25]),
        const_i64("ch2_starts", [0, 2, 0, 0]),
        const_i64("ch2_ends", [1, 3, 25, 25]),
        const_i64("channel_axis", [1]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_h", 2),
        const_i64("axis_w", 3),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 5, 5]),
        const_f32("half", 0.5),
        const_f32("one", 1.0),
        const_f32("zero", 0.0),
        const_f32("ch0", onehot_channel(0)),
        const_f32("ch3", onehot_channel(3)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "small_starts", "small_ends", "axes4", "steps4"], ["small"], name="slice_25x25"),
        helper.make_node("ReduceMax", ["small"], ["grid_area"], axes=[1], keepdims=1, name="grid_area"),
        helper.make_node("Slice", ["small", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_frame_color"),
        helper.make_node("Less", ["ch2", "half"], ["zero_cell"], name="is_empty_cell"),
        helper.make_node("CumSum", ["ch2", "axis_w"], ["left_count"], name="count_left"),
        helper.make_node("CumSum", ["ch2", "axis_w"], ["right_count"], name="count_right", reverse=1),
        helper.make_node("CumSum", ["ch2", "axis_h"], ["up_count"], name="count_up"),
        helper.make_node("CumSum", ["ch2", "axis_h"], ["down_count"], name="count_down", reverse=1),
        helper.make_node("Greater", ["left_count", "half"], ["left_seen"], name="left_seen"),
        helper.make_node("Greater", ["right_count", "half"], ["right_seen"], name="right_seen"),
        helper.make_node("Greater", ["up_count", "half"], ["up_seen"], name="up_seen"),
        helper.make_node("Greater", ["down_count", "half"], ["down_seen"], name="down_seen"),
        helper.make_node("And", ["left_seen", "right_seen"], ["inside_h"], name="inside_horizontal_walls"),
        helper.make_node("And", ["up_seen", "down_seen"], ["inside_v"], name="inside_vertical_walls"),
        helper.make_node("And", ["inside_h", "inside_v"], ["inside_box"], name="inside_box"),
        helper.make_node("And", ["inside_box", "zero_cell"], ["hole_bool"], name="hole_cells"),
        helper.make_node("Cast", ["hole_bool"], ["holes"], to=TensorProto.FLOAT, name="hole_float"),
        helper.make_node("Sub", ["one", "holes"], ["not_hole"], name="not_hole"),
        helper.make_node("Mul", ["grid_area", "not_hole"], ["background_cells"], name="background_cells"),
        helper.make_node("Mul", ["background_cells", "ch0"], ["background"], name="paint_background"),
        helper.make_node("Mul", ["holes", "ch3"], ["hole_color"], name="paint_holes"),
        helper.make_node("Sum", ["background", "hole_color"], ["small_out"], name="assemble_hole_output"),
        helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
    ]
    return make_model("task338_frame_holes", nodes, inits)


BUILDERS = {
    "task022": build_task022,
    "task037": build_task037,
    "task045": build_task045,
    "task053": build_task053,
    "task060": build_task060,
    "task073": build_task073,
    "task074": build_task074,
    "task082": build_task082,
    "task115": build_task115,
    "task126": build_task126,
    "task128": build_task128,
    "task160": build_task160,
    "task164": build_task164,
    "task168": build_task168,
    "task181": build_task181,
    "task207": build_task207,
    "task210": build_task210,
    "task227": build_task227,
    "task230": build_task230,
    "task250": build_task250,
    "task251": build_task251,
    "task258": build_task258,
    "task259": build_task259,
    "task262": build_task262,
    "task267": build_task267,
    "task273": build_task273,
    "task283": build_task283,
    "task290": build_task290,
    "task294": build_task294,
    "task306": build_task306,
    "task311": build_task311,
    "task312": build_task312,
    "task315": build_task315,
    "task320": build_task320,
    "task330": build_task330,
    "task332": build_task332,
    "task345": build_task345,
    "task347": build_task347,
    "task352": build_task352,
    "task357": build_task357,
    "task371": build_task371,
    "task381": build_task381,
    "task395": build_task395,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/metric_v2_manual_static"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/metric_v2_manual_static_generation.csv"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected: list[Path] = []

    for task, builder in sorted(BUILDERS.items()):
        path = model_dir / f"{task}.onnx"
        reason = ""
        status = "ok"
        right = wrong = examples_n = 0
        try:
            raw = builder().SerializeToString()
            path.write_bytes(raw)
            examples = all_examples(json.loads((args.data_root / f"{task}.json").read_text()))
            examples_n = len(examples)
            right, wrong, reason = verify_model_bytes(raw, examples)
            if wrong:
                status = "failed_verify"
            else:
                selected.append(path)
        except Exception as exc:  # noqa: BLE001 - generation report should continue.
            status = "error"
            reason = str(exc)
        rows.append(
            {
                "task": task,
                "status": status,
                "right": right,
                "wrong": wrong,
                "examples": examples_n,
                "file_bytes": path.stat().st_size if path.exists() else 0,
                "reason": reason,
            }
        )
        print(f"{task}: {status} {right}/{examples_n} {reason}")

    zip_path = args.out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in selected:
            zf.write(path, path.name)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"zip: {zip_path}")
    print(f"report: {args.report_out}")


if __name__ == "__main__":
    main()
