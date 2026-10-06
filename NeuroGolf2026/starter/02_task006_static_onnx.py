#!/usr/bin/env python3
"""
Task006 static ONNX solver for The 2026 NeuroGolf Championship.

Rule:
  input is a 3x7 grid: left 3x3, separator color 5, right 3x3.
  output is 3x3.
  output cell is color 2 where both left and right cells are color 1.
  all other output cells are color 0.

Kaggle usage:
  run this file/cell, then submit /kaggle/working/submission.zip.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


WORKING_ROOT = Path("/kaggle/working")
TASK_PATH = WORKING_ROOT / "task006.onnx"
SUBMISSION_PATH = WORKING_ROOT / "submission.zip"
EXPECTED_SHAPE = [1, 10, 30, 30]


def const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)


def const_f32(name: str, values: np.ndarray | list[float]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def make_slice(name: str, x: str, y: str, starts: str, ends: str) -> onnx.NodeProto:
    return helper.make_node(
        "Slice",
        inputs=[x, starts, ends, "slice_axes", "slice_steps"],
        outputs=[y],
        name=name,
    )


def build_task006_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_SHAPE)

    initializers = [
        const_i64("left_starts", [0, 1, 0, 0]),
        const_i64("left_ends", [1, 2, 3, 3]),
        const_i64("right_starts", [0, 1, 0, 4]),
        const_i64("right_ends", [1, 2, 3, 7]),
        const_i64("slice_axes", [0, 1, 2, 3]),
        const_i64("slice_steps", [1, 1, 1, 1]),
        const_i64("pad_3_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        const_f32("zero_scalar", np.array(0.0, dtype=np.float32)),
        const_f32("ones_3x3", np.ones((1, 1, 3, 3), dtype=np.float32)),
    ]

    nodes = [
        make_slice("left_color1", "input", "left", "left_starts", "left_ends"),
        make_slice("right_color1", "input", "right", "right_starts", "right_ends"),
        helper.make_node("Mul", inputs=["left", "right"], outputs=["hit"], name="intersection"),
        helper.make_node("Sub", inputs=["ones_3x3", "hit"], outputs=["background"], name="background_color0"),
        helper.make_node("Pad", inputs=["background", "pad_3_to_30", "zero_scalar"], outputs=["background_pad"], name="pad_background"),
        helper.make_node("Pad", inputs=["hit", "pad_3_to_30", "zero_scalar"], outputs=["hit_pad"], name="pad_hit"),
        helper.make_node("Sub", inputs=["hit_pad", "hit_pad"], outputs=["zero_channel"], name="zero_channel"),
        helper.make_node(
            "Concat",
            inputs=[
                "background_pad",
                "zero_channel",
                "hit_pad",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
                "zero_channel",
            ],
            outputs=["output"],
            axis=1,
            name="assemble_color_channels",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task006_intersection_solver",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task006_static",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_submission(model: onnx.ModelProto) -> None:
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    TASK_PATH.write_bytes(model.SerializeToString())
    with zipfile.ZipFile(SUBMISSION_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(TASK_PATH, "task006.onnx")
    print(f"wrote {TASK_PATH} ({TASK_PATH.stat().st_size} bytes)")
    print(f"wrote {SUBMISSION_PATH}")


def try_official_verify(model: onnx.ModelProto) -> None:
    util_dir = Path("/kaggle/input/competitions/neurogolf-2026/neurogolf_utils")
    if not util_dir.exists():
        print("skip official verify: neurogolf_utils not found")
        return
    import sys

    sys.path.append(str(util_dir))
    from neurogolf_utils import load_examples, verify_network  # type: ignore

    examples = load_examples(6)
    verify_network(model, 6, examples)


if __name__ == "__main__":
    network = build_task006_model()
    try_official_verify(network)
    write_submission(network)
