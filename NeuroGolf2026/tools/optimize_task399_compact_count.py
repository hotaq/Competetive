#!/usr/bin/env python3
"""Build a compact clean model for task399.

The input contains 1..5 non-overlapping 2x2 blocks of color 2.  The output is a
3x3 code: positions (0,0), (0,2), (1,1), (2,0), (2,2) are enabled as the block
count reaches 1..5.
"""

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
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
CODE_POSITIONS = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]


def vi(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def i64(name: str, value: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def make_model() -> onnx.ModelProto:
    inside = np.ones((1, 1, 3, 3), dtype=np.float32)
    masks = []
    for row, col in CODE_POSITIONS:
        mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
        mask[0, 0, row, col] = 1.0
        masks.append(mask)

    inits: list[onnx.TensorProto] = [
        i64("starts", [0, 2, 0, 0]),
        i64("ends", [1, 3, 7, 7]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps_even_even", [1, 1, 2, 2]),
        i64("axes_all", [0, 1, 2, 3]),
        i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
        f32("zero", 0.0),
        f32("inside", inside),
    ]
    for idx, mask in enumerate(masks, start=1):
        inits.append(f32(f"mask{idx}", mask))
        inits.append(f32(f"threshold{idx}", float(idx) - 0.5))

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes4", "steps_even_even"], ["parity2"], name="slice_even_even_color2"),
        helper.make_node("ReduceSum", ["parity2", "axes_all"], ["block_count"], keepdims=1, name="count_blocks"),
    ]
    ch1_terms: list[str] = []
    for idx in range(1, 6):
        nodes.extend(
            [
                helper.make_node("Greater", ["block_count", f"threshold{idx}"], [f"hit{idx}_bool"], name=f"hit{idx}_bool"),
                helper.make_node("Cast", [f"hit{idx}_bool"], [f"hit{idx}"], to=TensorProto.FLOAT, name=f"hit{idx}"),
                helper.make_node("Mul", [f"hit{idx}", f"mask{idx}"], [f"dot{idx}"], name=f"dot{idx}"),
            ]
        )
        ch1_terms.append(f"dot{idx}")

    zero_channels = []
    for color in range(2, 10):
        nodes.append(helper.make_node("Mul", ["inside", "zero"], [f"ch{color}"], name=f"zero_ch{color}"))
        zero_channels.append(f"ch{color}")

    nodes.extend(
        [
            helper.make_node("Sum", ch1_terms, ["ch1"], name="assemble_code_channel"),
            helper.make_node("Sub", ["inside", "ch1"], ["ch0"], name="background_channel"),
            helper.make_node("Concat", ["ch0", "ch1", *zero_channels], ["small_out"], axis=1, name="assemble_3x3"),
            helper.make_node("Pad", ["small_out", "pad_to_30", "zero"], ["output"], name="pad_to_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task399_compact_count",
        [vi("input")],
        [vi("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task399_compact_count",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_task399_compact_count")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "task399.onnx"
    zip_path = out_dir / "submission.zip"

    raw = make_model().SerializeToString()
    model_path.write_bytes(raw)
    examples = all_examples(json.loads(Path("data/latest/extracted/task399.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task399 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)
    macs, memory, params, cost = score_network(model_path)
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task399.onnx", raw)
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
