#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task388."""

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


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("axes_all", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("nonzero_starts", [0, 1, 0, 0]),
        const_i64("nonzero_ends", [1, 10, 30, 30]),
        const_i64("pad_12x12_to_30", [0, 0, 0, 0, 0, 0, 18, 18]),
        const_f32("zero", 0.0),
        const_f32("one", 1.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceMax", ["input"], ["active_rows"], axes=[1, 3], keepdims=1, name="active_rows"),
        helper.make_node("ReduceMax", ["input"], ["active_cols"], axes=[1, 2], keepdims=1, name="active_cols"),
        helper.make_node("ReduceSum", ["active_rows", "axes_all"], ["row_count"], keepdims=1, name="row_count"),
        helper.make_node("ReduceSum", ["active_cols", "axes_all"], ["col_count"], keepdims=1, name="col_count"),
        helper.make_node("Slice", ["input", "nonzero_starts", "nonzero_ends", "axes4", "steps4"], ["nonzero_channels"], name="slice_nonzero_colors"),
        helper.make_node("ReduceMax", ["nonzero_channels"], ["nonzero_cols_full"], axes=[1, 2], keepdims=1, name="nonzero_cols_full"),
    ]

    branches: list[str] = []
    for n in (2, 3, 4, 5, 6):
        prefix = f"b{n}"
        inits.extend(
            [
                const_f32(f"{prefix}_n", np.ones((1, 1, 1, 1), dtype=np.float32) * n),
                const_i64(f"{prefix}_crop_starts", [0, 0, 0, 0]),
                const_i64(f"{prefix}_crop_ends", [1, 10, n, n]),
                const_i64(f"{prefix}_col_starts", [0, 0, 0, 0]),
                const_i64(f"{prefix}_col_ends", [1, 1, 1, n]),
                const_i64(f"{prefix}_pad_to_12", [0, 0, 0, 0, 0, 0, 12 - (2 * n), 12 - (2 * n)]),
                const_f32(f"{prefix}_row_ones", np.ones((1, 1, n, 1), dtype=np.float32)),
            ]
        )

        nodes.extend(
            [
                helper.make_node("Equal", ["row_count", f"{prefix}_n"], [f"{prefix}_h_ok"], name=f"{prefix}_h_ok"),
                helper.make_node("Equal", ["col_count", f"{prefix}_n"], [f"{prefix}_w_ok"], name=f"{prefix}_w_ok"),
                helper.make_node("And", [f"{prefix}_h_ok", f"{prefix}_w_ok"], [f"{prefix}_size_ok_bool"], name=f"{prefix}_size_ok_bool"),
                helper.make_node("Cast", [f"{prefix}_size_ok_bool"], [f"{prefix}_size_ok"], to=TensorProto.FLOAT, name=f"{prefix}_size_ok"),
                helper.make_node("Slice", ["input", f"{prefix}_crop_starts", f"{prefix}_crop_ends", "axes4", "steps4"], [f"{prefix}_crop"], name=f"{prefix}_crop"),
                helper.make_node("Slice", ["nonzero_cols_full", f"{prefix}_col_starts", f"{prefix}_col_ends", "axes4", "steps4"], [f"{prefix}_cols"], name=f"{prefix}_cols"),
                helper.make_node("Mul", [f"{prefix}_cols", f"{prefix}_row_ones"], [f"{prefix}_col_grid"], name=f"{prefix}_broadcast_cols"),
                helper.make_node("Sub", ["one", f"{prefix}_col_grid"], [f"{prefix}_inactive_cols"], name=f"{prefix}_inactive_cols"),
            ]
        )

        channel_outputs: list[str] = []
        for color in range(10):
            starts = f"{prefix}_ch{color}_starts"
            ends = f"{prefix}_ch{color}_ends"
            inits.extend(
                [
                    const_i64(starts, [0, color, 0, 0]),
                    const_i64(ends, [1, color + 1, n, n]),
                ]
            )
            base = f"{prefix}_base_ch{color}"
            nodes.append(helper.make_node("Slice", [f"{prefix}_crop", starts, ends, "axes4", "steps4"], [base], name=f"{prefix}_slice_ch{color}"))
            if color == 0:
                nodes.append(helper.make_node("Mul", [base, f"{prefix}_inactive_cols"], [f"{prefix}_out_ch0"], name=f"{prefix}_keep_inactive_background"))
                channel_outputs.append(f"{prefix}_out_ch0")
            elif color == 8:
                nodes.append(helper.make_node("Mul", [f"{prefix}_base_ch0", f"{prefix}_col_grid"], [f"{prefix}_paint8"], name=f"{prefix}_paint_background_8"))
                nodes.append(helper.make_node("Add", [base, f"{prefix}_paint8"], [f"{prefix}_out_ch8"], name=f"{prefix}_add_color8"))
                channel_outputs.append(f"{prefix}_out_ch8")
            else:
                channel_outputs.append(base)

        nodes.extend(
            [
                helper.make_node("Concat", channel_outputs, [f"{prefix}_tile"], axis=1, name=f"{prefix}_assemble_tile"),
                helper.make_node("Concat", [f"{prefix}_tile", f"{prefix}_tile"], [f"{prefix}_wide"], axis=3, name=f"{prefix}_tile_width"),
                helper.make_node("Concat", [f"{prefix}_wide", f"{prefix}_wide"], [f"{prefix}_twice"], axis=2, name=f"{prefix}_tile_height"),
                helper.make_node("Pad", [f"{prefix}_twice", f"{prefix}_pad_to_12", "zero"], [f"{prefix}_padded"], name=f"{prefix}_pad_to_12"),
                helper.make_node("Mul", [f"{prefix}_padded", f"{prefix}_size_ok"], [f"{prefix}_selected"], name=f"{prefix}_select"),
            ]
        )
        branches.append(f"{prefix}_selected")

    nodes.extend(
        [
            helper.make_node("Sum", branches, ["small_out"], name="sum_size_branches"),
            helper.make_node("Pad", ["small_out", "pad_12x12_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task388_col_tile",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task388_col_tile",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task388_col_tile")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task388.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task388 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task388.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task388.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
