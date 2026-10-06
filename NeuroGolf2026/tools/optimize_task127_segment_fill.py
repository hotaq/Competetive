#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task127."""

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


def arr(name: str, value: np.ndarray | list[int] | float, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | list[float] | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    """Task rule: map each colored segment marker 1..4 to filled colors 6..9."""

    top_rows = np.zeros((1, 1, 30, 1), dtype=np.float32)
    top_rows[:, :, :3, :] = 1.0
    bottom_rows = np.zeros((1, 1, 30, 1), dtype=np.float32)
    bottom_rows[:, :, 4:7, :] = 1.0
    separator_row = np.zeros((1, 1, 30, 1), dtype=np.float32)
    separator_row[:, :, 3:4, :] = 1.0
    separator_pattern = np.zeros((1, 10, 1, 11), dtype=np.float32)
    separator_pattern[:, 5, :, :] = 1.0

    inits: list[onnx.TensorProto] = [
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("tile3", [1, 1, 1, 3]),
        const_i64("plus6", 6),
        const_i64("sep5", [[[[5]]]]),
        const_i64("depth10", 10),
        const_i64("pattern_shape", [1, 10, 1, 11]),
        const_i64("pad_to_30w", [0, 0, 0, 0, 0, 0, 0, 19]),
        const_i64("bottom_starts", [0, 0, 4, 0]),
        const_i64("bottom_ends", [1, 10, 7, 11]),
        const_f32("onehot_values", [0.0, 1.0]),
        const_f32("top_rows", top_rows),
        const_f32("bottom_rows", bottom_rows),
        const_f32("separator_row", separator_row),
        const_f32("separator_pattern", separator_pattern),
        const_f32("zero", 0.0),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends", "axes_all", "steps4"], ["bottom_slice"], name="slice_optional_bottom_panel"),
        helper.make_node("ReduceMax", ["bottom_slice"], ["bottom_active"], axes=[0, 1, 2, 3], keepdims=1, name="bottom_panel_present"),
    ]

    panel_outputs: list[str] = []
    for panel_name, r0, r1, row_mask in (("top", 0, 3, "top_rows"), ("bottom", 4, 7, "bottom_rows")):
        filled_blocks: list[str] = []
        for block_idx, (c0, c1) in enumerate(((0, 3), (4, 7), (8, 11))):
            starts_name = f"{panel_name}_seg{block_idx}_starts"
            ends_name = f"{panel_name}_seg{block_idx}_ends"
            inits.extend(
                [
                    const_i64(starts_name, [0, 1, r0, c0]),
                    const_i64(ends_name, [1, 5, r1, c1]),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["input", starts_name, ends_name, "axes_all", "steps4"],
                        [f"{panel_name}_seg{block_idx}_colors"],
                        name=f"{panel_name}_slice_segment_{block_idx}_colors_1_to_4",
                    ),
                    helper.make_node(
                        "ReduceMax",
                        [f"{panel_name}_seg{block_idx}_colors"],
                        [f"{panel_name}_seg{block_idx}_present"],
                        axes=[2, 3],
                        keepdims=1,
                        name=f"{panel_name}_segment_{block_idx}_color_presence",
                    ),
                    helper.make_node(
                        "ArgMax",
                        [f"{panel_name}_seg{block_idx}_present"],
                        [f"{panel_name}_seg{block_idx}_arg"],
                        axis=1,
                        keepdims=1,
                        name=f"{panel_name}_segment_{block_idx}_source_color_index",
                    ),
                    helper.make_node(
                        "Add",
                        [f"{panel_name}_seg{block_idx}_arg", "plus6"],
                        [f"{panel_name}_seg{block_idx}_out_color"],
                        name=f"{panel_name}_segment_{block_idx}_map_to_output_color",
                    ),
                    helper.make_node(
                        "Tile",
                        [f"{panel_name}_seg{block_idx}_out_color", "tile3"],
                        [f"{panel_name}_seg{block_idx}_cols"],
                        name=f"{panel_name}_segment_{block_idx}_repeat_three_columns",
                    ),
                ]
            )
            filled_blocks.append(f"{panel_name}_seg{block_idx}_cols")

        nodes.extend(
            [
                helper.make_node(
                    "Concat",
                    [filled_blocks[0], "sep5", filled_blocks[1], "sep5", filled_blocks[2]],
                    [f"{panel_name}_color_indices"],
                    axis=3,
                    name=f"{panel_name}_assemble_11_column_indices",
                ),
                helper.make_node(
                    "OneHot",
                    [f"{panel_name}_color_indices", "depth10", "onehot_values"],
                    [f"{panel_name}_onehot_5d"],
                    axis=1,
                    name=f"{panel_name}_indices_to_onehot",
                ),
                helper.make_node(
                    "Reshape",
                    [f"{panel_name}_onehot_5d", "pattern_shape"],
                    [f"{panel_name}_pattern_1x11"],
                    name=f"{panel_name}_drop_singleton_axis",
                ),
                helper.make_node(
                    "Mul",
                    [f"{panel_name}_pattern_1x11", row_mask],
                    [f"{panel_name}_row_pattern"],
                    name=f"{panel_name}_repeat_pattern_over_rows",
                ),
            ]
        )
        if panel_name == "bottom":
            nodes.append(
                helper.make_node(
                    "Mul",
                    [f"{panel_name}_row_pattern", "bottom_active"],
                    [f"{panel_name}_active_pattern"],
                    name="gate_bottom_panel",
                )
            )
            panel_outputs.append(f"{panel_name}_active_pattern")
        else:
            panel_outputs.append(f"{panel_name}_row_pattern")

    nodes.extend(
        [
            helper.make_node("Mul", ["separator_pattern", "separator_row"], ["separator_row_pattern"], name="make_horizontal_separator"),
            helper.make_node("Mul", ["separator_row_pattern", "bottom_active"], ["separator_active"], name="gate_horizontal_separator"),
            helper.make_node("Sum", [panel_outputs[0], "separator_active", panel_outputs[1]], ["active_pattern"], name="sum_panels"),
            helper.make_node("Pad", ["active_pattern", "pad_to_30w", "zero"], ["output"], name="pad_width_to_static_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task127_segment_fill",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task127_segment_fill",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task127_segment_fill")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()

    data = json.loads(Path("data/latest/extracted/task127.json").read_text())
    examples = all_examples(data)
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task127 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = out_dir / "task127.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task127.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
