#!/usr/bin/env python3
"""Build clean static ONNX line-projection models for tasks 025 and 340."""

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
TASKS = ("task025", "task340")


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int] | int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def shift_col_left(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], src: str, dst: str, prefix: str) -> None:
    inits.extend(slice_4d(prefix, [0, 0, 0, 1], [1, 1, 1, 30]))
    inits.append(const_i64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, 0, 1]))
    nodes.extend(
        [
            make_slice(src, f"{dst}_part", prefix),
            helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
        ]
    )


def shift_col_right(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], src: str, dst: str, prefix: str) -> None:
    inits.extend(slice_4d(prefix, [0, 0, 0, 0], [1, 1, 1, 29]))
    inits.append(const_i64(f"{prefix}_pad", [0, 0, 0, 1, 0, 0, 0, 0]))
    nodes.extend(
        [
            make_slice(src, f"{dst}_part", prefix),
            helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
        ]
    )


def shift_row_up(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], src: str, dst: str, prefix: str) -> None:
    inits.extend(slice_4d(prefix, [0, 0, 1, 0], [1, 1, 30, 1]))
    inits.append(const_i64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, 1, 0]))
    nodes.extend(
        [
            make_slice(src, f"{dst}_part", prefix),
            helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
        ]
    )


def shift_row_down(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], src: str, dst: str, prefix: str) -> None:
    inits.extend(slice_4d(prefix, [0, 0, 0, 0], [1, 1, 29, 1]))
    inits.append(const_i64(f"{prefix}_pad", [0, 0, 1, 0, 0, 0, 0, 0]))
    nodes.extend(
        [
            make_slice(src, f"{dst}_part", prefix),
            helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
        ]
    )


def build_color(nodes: list[onnx.NodeProto], inits: list[onnx.TensorProto], color: int) -> str:
    p = f"c{color}"
    inits.extend(slice_4d(p, [0, color, 0, 0], [1, color + 1, 30, 30]))
    nodes.extend(
        [
            make_slice("input", p, p),
            helper.make_node("ReduceSum", [p, "axis_h"], [f"{p}_col_count"], keepdims=1, name=f"{p}_col_count"),
            helper.make_node("Greater", [f"{p}_col_count", "line_threshold"], [f"{p}_col_line_bool"], name=f"{p}_col_line_bool"),
            helper.make_node("Cast", [f"{p}_col_line_bool"], [f"{p}_col_line"], to=TensorProto.FLOAT, name=f"{p}_col_line"),
            helper.make_node("ReduceSum", [p, "axis_w"], [f"{p}_row_count"], keepdims=1, name=f"{p}_row_count"),
            helper.make_node("Greater", [f"{p}_row_count", "line_threshold"], [f"{p}_row_line_bool"], name=f"{p}_row_line_bool"),
            helper.make_node("Cast", [f"{p}_row_line_bool"], [f"{p}_row_line"], to=TensorProto.FLOAT, name=f"{p}_row_line"),
            helper.make_node("Add", [f"{p}_col_line", f"{p}_row_line"], [f"{p}_line_area_raw"], name=f"{p}_line_area_raw"),
            helper.make_node("Greater", [f"{p}_line_area_raw", "half"], [f"{p}_line_area_bool"], name=f"{p}_line_area_bool"),
            helper.make_node("Cast", [f"{p}_line_area_bool"], [f"{p}_line_area"], to=TensorProto.FLOAT, name=f"{p}_line_area"),
            helper.make_node("Mul", [p, f"{p}_line_area"], [f"{p}_line_keep"], name=f"{p}_line_keep"),
            helper.make_node("Sub", ["one_scalar", f"{p}_line_area"], [f"{p}_not_line"], name=f"{p}_not_line"),
            helper.make_node("Mul", [p, f"{p}_not_line"], [f"{p}_marker"], name=f"{p}_marker"),
        ]
    )

    shift_col_left(nodes, inits, f"{p}_col_line", f"{p}_col_before", f"{p}_col_before")
    shift_col_right(nodes, inits, f"{p}_col_line", f"{p}_col_after", f"{p}_col_after")
    shift_row_up(nodes, inits, f"{p}_row_line", f"{p}_row_above", f"{p}_row_above")
    shift_row_down(nodes, inits, f"{p}_row_line", f"{p}_row_below", f"{p}_row_below")

    nodes.extend(
        [
            helper.make_node("CumSum", [f"{p}_col_line", "axis_w_scalar"], [f"{p}_col_cum"], name=f"{p}_col_cum"),
            helper.make_node("Less", [f"{p}_col_cum", "half"], [f"{p}_before_col_bool"], name=f"{p}_before_col_bool"),
            helper.make_node("Cast", [f"{p}_before_col_bool"], [f"{p}_before_col"], to=TensorProto.FLOAT, name=f"{p}_before_col"),
            helper.make_node("Sub", [f"{p}_col_cum", f"{p}_col_line"], [f"{p}_after_col_raw"], name=f"{p}_after_col_raw"),
            helper.make_node("Greater", [f"{p}_after_col_raw", "half"], [f"{p}_after_col_bool"], name=f"{p}_after_col_bool"),
            helper.make_node("Cast", [f"{p}_after_col_bool"], [f"{p}_after_col"], to=TensorProto.FLOAT, name=f"{p}_after_col"),
            helper.make_node("Mul", [f"{p}_marker", f"{p}_before_col"], [f"{p}_markers_left"], name=f"{p}_markers_left"),
            helper.make_node("ReduceMax", [f"{p}_markers_left"], [f"{p}_rows_left"], axes=[3], keepdims=1, name=f"{p}_rows_left"),
            helper.make_node("Mul", [f"{p}_rows_left", f"{p}_col_before"], [f"{p}_move_left"], name=f"{p}_move_left"),
            helper.make_node("Mul", [f"{p}_marker", f"{p}_after_col"], [f"{p}_markers_right"], name=f"{p}_markers_right"),
            helper.make_node("ReduceMax", [f"{p}_markers_right"], [f"{p}_rows_right"], axes=[3], keepdims=1, name=f"{p}_rows_right"),
            helper.make_node("Mul", [f"{p}_rows_right", f"{p}_col_after"], [f"{p}_move_right"], name=f"{p}_move_right"),
            helper.make_node("CumSum", [f"{p}_row_line", "axis_h_scalar"], [f"{p}_row_cum"], name=f"{p}_row_cum"),
            helper.make_node("Less", [f"{p}_row_cum", "half"], [f"{p}_above_row_bool"], name=f"{p}_above_row_bool"),
            helper.make_node("Cast", [f"{p}_above_row_bool"], [f"{p}_above_row"], to=TensorProto.FLOAT, name=f"{p}_above_row"),
            helper.make_node("Sub", [f"{p}_row_cum", f"{p}_row_line"], [f"{p}_below_row_raw"], name=f"{p}_below_row_raw"),
            helper.make_node("Greater", [f"{p}_below_row_raw", "half"], [f"{p}_below_row_bool"], name=f"{p}_below_row_bool"),
            helper.make_node("Cast", [f"{p}_below_row_bool"], [f"{p}_below_row"], to=TensorProto.FLOAT, name=f"{p}_below_row"),
            helper.make_node("Mul", [f"{p}_marker", f"{p}_above_row"], [f"{p}_markers_above"], name=f"{p}_markers_above"),
            helper.make_node("ReduceMax", [f"{p}_markers_above"], [f"{p}_cols_above"], axes=[2], keepdims=1, name=f"{p}_cols_above"),
            helper.make_node("Mul", [f"{p}_row_above", f"{p}_cols_above"], [f"{p}_move_above"], name=f"{p}_move_above"),
            helper.make_node("Mul", [f"{p}_marker", f"{p}_below_row"], [f"{p}_markers_below"], name=f"{p}_markers_below"),
            helper.make_node("ReduceMax", [f"{p}_markers_below"], [f"{p}_cols_below"], axes=[2], keepdims=1, name=f"{p}_cols_below"),
            helper.make_node("Mul", [f"{p}_row_below", f"{p}_cols_below"], [f"{p}_move_below"], name=f"{p}_move_below"),
            helper.make_node(
                "Sum",
                [f"{p}_line_keep", f"{p}_move_left", f"{p}_move_right", f"{p}_move_above", f"{p}_move_below"],
                [f"{p}_raw"],
                name=f"{p}_raw",
            ),
            helper.make_node("Greater", [f"{p}_raw", "half"], [f"{p}_bool"], name=f"{p}_bool"),
            helper.make_node("Cast", [f"{p}_bool"], [f"{p}_out"], to=TensorProto.FLOAT, name=f"{p}_out"),
        ]
    )
    return f"{p}_out"


def make_model(task: str) -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("axis_h", [2]),
        const_i64("axis_w", [3]),
        const_i64("axis_h_scalar", 2),
        const_i64("axis_w_scalar", 3),
        const_f32("zero_scalar", 0.0),
        const_f32("one_scalar", 1.0),
        const_f32("half", 0.5),
        const_f32("line_threshold", 6.0),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["inside"], keepdims=1, name="inside_shape"),
    ]
    color_outputs = [build_color(nodes, inits, color) for color in range(1, 10)]
    nodes.extend(
        [
            helper.make_node("Sum", color_outputs, ["occupied"], name="occupied"),
            helper.make_node("Sub", ["inside", "occupied"], ["ch0"], name="background"),
            helper.make_node("Concat", ["ch0", *color_outputs], ["output"], axis=1, name="assemble_output"),
        ]
    )
    graph = helper.make_graph(nodes, f"{task}_line_projection", [tensor_info("input")], [tensor_info("output")], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="neurogolf_line_projection",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task025_340_line_projection")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}
    for task in TASKS:
        model = make_model(task)
        raw = model.SerializeToString()
        examples = all_examples(json.loads(Path(f"data/latest/extracted/{task}.json").read_text()))
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(f"{task} verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
        if wrong:
            raise SystemExit(1)
        (out_dir / f"{task}.onnx").write_bytes(raw)
        raws[f"{task}.onnx"] = raw

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(raws.items()):
            zf.writestr(name, raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
