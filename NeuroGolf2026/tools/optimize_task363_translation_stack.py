#!/usr/bin/env python3
"""Build a compact clean solver for task363 template-copy logic.

Task363 has a fixed 10x10 canvas.  The color-2 cells form a template with a
bounding-box context; the output copies that template to every translation
where the source bounding-box pattern fits, while preserving the original grid.
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
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
H = 10
W = 10
FLAT = H * W
ZERO_SENTINEL = FLAT
TRANSLATIONS = [(dr, dc) for dr in range(-(H - 1), H) for dc in range(-(W - 1), W)]


def i64(name: str, arr: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def f32(name: str, arr: np.ndarray | list[float] | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def make_indices() -> tuple[np.ndarray, np.ndarray]:
    fit = np.full((len(TRANSLATIONS), FLAT), ZERO_SENTINEL, dtype=np.int64)
    shifted = np.full((len(TRANSLATIONS), FLAT), ZERO_SENTINEL, dtype=np.int64)

    for t_idx, (dr, dc) in enumerate(TRANSLATIONS):
        for r in range(H):
            for c in range(W):
                p = r * W + c

                tr = r + dr
                tc = c + dc
                if 0 <= tr < H and 0 <= tc < W:
                    fit[t_idx, p] = tr * W + tc

                sr = r - dr
                sc = c - dc
                if 0 <= sr < H and 0 <= sc < W:
                    shifted[t_idx, p] = sr * W + sc

    return fit, shifted


def make_model() -> onnx.ModelProto:
    fit_idx, shift_idx = make_indices()

    initializers = [
        i64("crop_starts", [0, 0, 0, 0]),
        i64("crop_ends", [1, 10, H, W]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("ch0_st", [0]),
        i64("ch0_en", [1]),
        i64("ch2_st", [2]),
        i64("ch2_en", [3]),
        i64("ch3_st", [3]),
        i64("ch3_en", [10]),
        i64("ch5_st", [5]),
        i64("ch5_en", [6]),
        i64("axis_ch", [1]),
        i64("axis_h", [2]),
        i64("axis_w", [3]),
        i64("rev_starts", [-1]),
        i64("rev_ends", [-11]),
        i64("rev_steps", [-1]),
        i64("nine_i64", [[[[9]]]]),
        i64("row_idx", np.arange(H, dtype=np.int64).reshape(1, 1, H, 1)),
        i64("col_idx", np.arange(W, dtype=np.int64).reshape(1, 1, 1, W)),
        i64("flat_shape", [1, FLAT]),
        i64("mask_shape", [1, 1, H, W]),
        i64("score_shape", [1, 1, 1]),
        i64("pad30", [0, 0, 0, 0, 0, 0, 20, 20]),
        i64("fit_idx", fit_idx),
        i64("shift_idx", shift_idx),
        f32("zero_col", [[0.0]]),
        f32("zero_scalar", [0.0]),
        f32("one_scalar", [1.0]),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "crop_starts", "crop_ends", "axes4", "steps4"], ["input10"], name="crop_input10"),
        helper.make_node("Slice", ["input10", "ch0_st", "ch0_en", "axis_ch"], ["x0"], name="slice_zero_channel"),
        helper.make_node("Slice", ["input10", "ch2_st", "ch2_en", "axis_ch"], ["x2"], name="slice_template_channel"),
        helper.make_node("Slice", ["input10", "ch5_st", "ch5_en", "axis_ch"], ["x5"], name="slice_barrier_channel"),
        helper.make_node("Slice", ["input10", "ch3_st", "ch3_en", "axis_ch"], ["x3_to_x9"], name="slice_tail_channels"),
        helper.make_node("ReduceMax", ["x2"], ["row_has_template"], name="row_has_template", axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["x2"], ["col_has_template"], name="col_has_template", axes=[2], keepdims=1),
        helper.make_node("ArgMax", ["row_has_template"], ["r0_i64"], name="first_template_row", axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_has_template"], ["c0_i64"], name="first_template_col", axis=3, keepdims=1),
        helper.make_node("Slice", ["row_has_template", "rev_starts", "rev_ends", "axis_h", "rev_steps"], ["row_has_rev"], name="reverse_template_rows"),
        helper.make_node("Slice", ["col_has_template", "rev_starts", "rev_ends", "axis_w", "rev_steps"], ["col_has_rev"], name="reverse_template_cols"),
        helper.make_node("ArgMax", ["row_has_rev"], ["r1_rev_i64"], name="last_template_row_reverse", axis=2, keepdims=1),
        helper.make_node("ArgMax", ["col_has_rev"], ["c1_rev_i64"], name="last_template_col_reverse", axis=3, keepdims=1),
        helper.make_node("Sub", ["nine_i64", "r1_rev_i64"], ["r1_i64"], name="last_template_row"),
        helper.make_node("Sub", ["nine_i64", "c1_rev_i64"], ["c1_i64"], name="last_template_col"),
        helper.make_node("Less", ["row_idx", "r0_i64"], ["row_below"], name="row_below_bbox"),
        helper.make_node("Less", ["r1_i64", "row_idx"], ["row_above"], name="row_above_bbox"),
        helper.make_node("Or", ["row_below", "row_above"], ["row_outside"], name="row_outside_bbox"),
        helper.make_node("Not", ["row_outside"], ["row_inside_bool"], name="row_inside_bbox"),
        helper.make_node("Less", ["col_idx", "c0_i64"], ["col_left"], name="col_left_bbox"),
        helper.make_node("Less", ["c1_i64", "col_idx"], ["col_right"], name="col_right_bbox"),
        helper.make_node("Or", ["col_left", "col_right"], ["col_outside"], name="col_outside_bbox"),
        helper.make_node("Not", ["col_outside"], ["col_inside_bool"], name="col_inside_bbox"),
        helper.make_node("Cast", ["row_inside_bool"], ["row_inside"], name="cast_row_bbox", to=TensorProto.FLOAT),
        helper.make_node("Cast", ["col_inside_bool"], ["col_inside"], name="cast_col_bbox", to=TensorProto.FLOAT),
        helper.make_node("Mul", ["row_inside", "col_inside"], ["bbox_mask"], name="template_bbox_mask"),
        helper.make_node("Add", ["x0", "x2"], ["zero_or_template"], name="zero_or_template_cells"),
        helper.make_node("Mul", ["bbox_mask", "zero_or_template"], ["pattern_zero"], name="bbox_zero_pattern"),
        helper.make_node("Mul", ["bbox_mask", "x5"], ["pattern_five"], name="bbox_five_pattern"),
        helper.make_node("ReduceSum", ["bbox_mask"], ["bbox_area_4d"], name="count_bbox_area", axes=[2, 3], keepdims=1),
        helper.make_node("Reshape", ["bbox_area_4d", "score_shape"], ["bbox_area"], name="reshape_bbox_area"),
        helper.make_node("Reshape", ["x0", "flat_shape"], ["x0_flat"], name="flatten_zero_mask"),
        helper.make_node("Reshape", ["x5", "flat_shape"], ["x5_flat"], name="flatten_five_mask"),
        helper.make_node("Reshape", ["x2", "flat_shape"], ["x2_flat"], name="flatten_template_mask"),
        helper.make_node("Reshape", ["pattern_zero", "flat_shape"], ["pattern_zero_flat"], name="flatten_zero_pattern"),
        helper.make_node("Reshape", ["pattern_five", "flat_shape"], ["pattern_five_flat"], name="flatten_five_pattern"),
        helper.make_node("Concat", ["x0_flat", "zero_col"], ["x0_flat101"], name="append_zero_sentinel_to_zero_mask", axis=1),
        helper.make_node("Concat", ["x5_flat", "zero_col"], ["x5_flat101"], name="append_zero_sentinel_to_five_mask", axis=1),
        helper.make_node("Concat", ["x2_flat", "zero_col"], ["x2_flat101"], name="append_zero_sentinel_to_template", axis=1),
        helper.make_node("Gather", ["x0_flat101", "fit_idx"], ["zero_stack"], name="stack_zero_masks_for_all_translations", axis=1),
        helper.make_node("Gather", ["x5_flat101", "fit_idx"], ["five_stack"], name="stack_five_masks_for_all_translations", axis=1),
        helper.make_node("Unsqueeze", ["pattern_zero_flat"], ["pattern_zero_u"], name="broadcast_zero_pattern", axes=[1]),
        helper.make_node("Unsqueeze", ["pattern_five_flat"], ["pattern_five_u"], name="broadcast_five_pattern", axes=[1]),
        helper.make_node("Mul", ["zero_stack", "pattern_zero_u"], ["zero_fit_cells"], name="zero_pattern_cells_that_fit"),
        helper.make_node("Mul", ["five_stack", "pattern_five_u"], ["five_fit_cells"], name="five_pattern_cells_that_fit"),
        helper.make_node("ReduceSum", ["zero_fit_cells"], ["zero_fit_scores"], name="score_zero_pattern", axes=[2], keepdims=1),
        helper.make_node("ReduceSum", ["five_fit_cells"], ["five_fit_scores"], name="score_five_pattern", axes=[2], keepdims=1),
        helper.make_node("Add", ["zero_fit_scores", "five_fit_scores"], ["fit_scores"], name="score_bbox_pattern"),
        helper.make_node("Equal", ["fit_scores", "bbox_area"], ["fit_bool"], name="all_bbox_cells_fit"),
        helper.make_node("Cast", ["fit_bool"], ["fit_f"], name="cast_fit_mask", to=TensorProto.FLOAT),
        helper.make_node("Gather", ["x2_flat101", "shift_idx"], ["shifted_templates"], name="stack_shifted_templates", axis=1),
        helper.make_node("Mul", ["shifted_templates", "fit_f"], ["accepted_templates"], name="keep_accepted_templates"),
        helper.make_node("ReduceMax", ["accepted_templates"], ["add_flat"], name="union_template_copies", axes=[1], keepdims=0),
        helper.make_node("Greater", ["add_flat", "zero_scalar"], ["add_bool_flat"], name="threshold_copied_cells"),
        helper.make_node("Cast", ["add_bool_flat"], ["add_f_flat"], name="cast_copied_cells", to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["add_f_flat", "mask_shape"], ["add_f"], name="reshape_copied_cells"),
        helper.make_node("Sub", ["one_scalar", "add_f"], ["not_add_f"], name="invert_copied_cells"),
        helper.make_node("Mul", ["x0", "not_add_f"], ["out_ch0"], name="clear_zero_channel_where_copied"),
        helper.make_node("Max", ["x2", "add_f"], ["out_ch2"], name="add_copies_to_color2"),
        helper.make_node("Concat", ["out_ch0", "input10", "out_ch2", "x3_to_x9"], ["bad_out10"], name="unused_bad_concat", axis=1),
    ]

    # Replace the placeholder concat with a correct channel layout:
    # ch0, ch1, ch2, ch3..ch9.  Keeping ch1 separate avoids duplicating ch0/ch2.
    nodes.pop()
    initializers.extend([i64("ch1_st", [1]), i64("ch1_en", [2])])
    nodes.extend(
        [
            helper.make_node("Slice", ["input10", "ch1_st", "ch1_en", "axis_ch"], ["x1"], name="slice_channel1"),
            helper.make_node("Concat", ["out_ch0", "x1", "out_ch2", "x3_to_x9"], ["out10"], name="assemble_output10", axis=1),
            helper.make_node("Pad", ["out10", "pad30", "zero_scalar"], ["output"], name="pad_output_to_30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task363_translation_stack",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task363_translation_stack",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task363_translation_stack"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task363.onnx"
    zip_path = args.out_dir / "submission.zip"

    model = make_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task363.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task363.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
