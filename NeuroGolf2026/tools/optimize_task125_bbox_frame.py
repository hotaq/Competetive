#!/usr/bin/env python3
"""Build a clean static ONNX model for NeuroGolf task125."""

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
SMALL = 15


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_i64_scalar(name: str, value: int) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def shift_spec(dr: int, dc: int) -> tuple[list[int], list[int], list[int]]:
    rsrc0 = max(0, -dr)
    rsrc1 = min(SMALL, SMALL - dr)
    csrc0 = max(0, -dc)
    csrc1 = min(SMALL, SMALL - dc)
    top = max(0, dr)
    left = max(0, dc)
    bottom = max(0, -dr)
    right = max(0, -dc)
    return [0, 0, rsrc0, csrc0], [1, 1, rsrc1, csrc1], [0, 0, top, left, 0, 0, bottom, right]


def shift_nodes(src: str, dst: str, prefix: str) -> list[onnx.NodeProto]:
    return [
        make_slice(src, f"{dst}_part", prefix),
        helper.make_node("Pad", [f"{dst}_part", f"{prefix}_pad", "zero_scalar"], [dst], name=f"pad_{dst}"),
    ]


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64_scalar("axis_h", 2),
        const_i64_scalar("axis_w", 3),
        const_i64("rev15", list(reversed(range(SMALL)))),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 15, 15]),
        const_f32("half", 0.5),
        const_f32("zero_scalar", 0.0),
        const_f32("ones15", np.ones((1, 1, SMALL, SMALL), dtype=np.float32)),
        *slice_4d("ch6", [0, 6, 0, 0], [1, 7, SMALL, SMALL]),
    ]
    shift_names: list[str] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            name = f"shift_{dr + 1}_{dc + 1}"
            starts, ends, pads = shift_spec(dr, dc)
            inits.extend(slice_4d(name, starts, ends))
            inits.append(const_i64(f"{name}_pad", pads))
            shift_names.append(name)

    nodes: list[onnx.NodeProto] = [
        make_slice("input", "ch6", "ch6"),
        helper.make_node("CumSum", ["ch6", "axis_w"], ["left_count"], name="left_count"),
        helper.make_node("Greater", ["left_count", "half"], ["left_bool"], name="left_bool"),
        helper.make_node("Cast", ["left_bool"], ["left"], to=TensorProto.FLOAT, name="left"),
        helper.make_node("Gather", ["ch6", "rev15"], ["ch6_rev_w"], axis=3, name="ch6_rev_w"),
        helper.make_node("CumSum", ["ch6_rev_w", "axis_w"], ["right_count_rev"], name="right_count_rev"),
        helper.make_node("Gather", ["right_count_rev", "rev15"], ["right_count"], axis=3, name="right_unrev"),
        helper.make_node("Greater", ["right_count", "half"], ["right_bool"], name="right_bool"),
        helper.make_node("Cast", ["right_bool"], ["right"], to=TensorProto.FLOAT, name="right"),
        helper.make_node("CumSum", ["ch6", "axis_h"], ["up_count"], name="up_count"),
        helper.make_node("Greater", ["up_count", "half"], ["up_bool"], name="up_bool"),
        helper.make_node("Cast", ["up_bool"], ["up"], to=TensorProto.FLOAT, name="up"),
        helper.make_node("Gather", ["ch6", "rev15"], ["ch6_rev_h"], axis=2, name="ch6_rev_h"),
        helper.make_node("CumSum", ["ch6_rev_h", "axis_h"], ["down_count_rev"], name="down_count_rev"),
        helper.make_node("Gather", ["down_count_rev", "rev15"], ["down_count"], axis=2, name="down_unrev"),
        helper.make_node("Greater", ["down_count", "half"], ["down_bool"], name="down_bool"),
        helper.make_node("Cast", ["down_bool"], ["down"], to=TensorProto.FLOAT, name="down"),
        helper.make_node("Mul", ["left", "right"], ["hspan"], name="hspan"),
        helper.make_node("Mul", ["up", "down"], ["vspan"], name="vspan"),
        helper.make_node("Mul", ["hspan", "vspan"], ["bbox"], name="bbox"),
    ]
    dil_terms = ["bbox"]
    for name in shift_names:
        dst = f"{name}_bbox"
        nodes.extend(shift_nodes("bbox", dst, name))
        dil_terms.append(dst)
    nodes.extend(
        [
            helper.make_node("Sum", dil_terms, ["dilated_raw"], name="dilated_raw"),
            helper.make_node("Greater", ["dilated_raw", "half"], ["dilated_bool"], name="dilated_bool"),
            helper.make_node("Cast", ["dilated_bool"], ["dilated"], to=TensorProto.FLOAT, name="dilated"),
            helper.make_node("Sub", ["ones15", "bbox"], ["not_bbox"], name="not_bbox"),
            helper.make_node("Mul", ["dilated", "not_bbox"], ["ch3"], name="frame_color3"),
            helper.make_node("Sub", ["ones15", "ch6"], ["not_six"], name="not_six"),
            helper.make_node("Mul", ["bbox", "not_six"], ["ch4"], name="inside_color4"),
            helper.make_node("Sum", ["ch3", "ch4", "ch6"], ["occupied"], name="occupied"),
            helper.make_node("Sub", ["ones15", "occupied"], ["ch8"], name="background8"),
            helper.make_node("Sub", ["ones15", "ones15"], ["zero15"], name="zero15"),
            helper.make_node(
                "Concat",
                ["zero15", "zero15", "zero15", "ch3", "ch4", "zero15", "ch6", "zero15", "ch8", "zero15"],
                ["small_out"],
                axis=1,
                name="assemble_15x15",
            ),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task125_bbox_frame",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task125_bbox_frame",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task125_bbox_frame")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads(Path("data/latest/extracted/task125.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task125 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    onnx_path = out_dir / "task125.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task125.onnx", raw)
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
