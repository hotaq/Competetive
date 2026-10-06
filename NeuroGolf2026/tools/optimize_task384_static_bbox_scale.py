#!/usr/bin/env python3
"""Build a strict-static bbox-crop + 2x scale model for task384."""

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
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]


def f32(name: str, arr: np.ndarray | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def i64(name: str, arr: np.ndarray | list[int] | int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def vi(name: str, dtype: int = TensorProto.FLOAT, shape: list[int] | None = None) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape or EXPECTED_SHAPE)


def slice_init(prefix: str, starts: list[int], ends: list[int], axes: list[int], steps: list[int]) -> list[onnx.TensorProto]:
    return [
        i64(f"{prefix}_starts", starts),
        i64(f"{prefix}_ends", ends),
        i64(f"{prefix}_axes", axes),
        i64(f"{prefix}_steps", steps),
    ]


def build_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        f32("scale2_weight", np.ones((1, 1, 2, 2), dtype=np.float32)),
        f32("scale2_row_weight", np.ones((1, 1, 2, 1), dtype=np.float32)),
        f32("scale2_col_weight", np.ones((1, 1, 1, 2), dtype=np.float32)),
        i64("crop9_starts", [0, 0, 0, 0]),
        i64("crop9_ends", [1, 10, 9, 9]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("ch4_starts", [0, 4, 0, 0]),
        i64("ch4_ends", [1, 5, 9, 9]),
        i64("axes_width", [3]),
        i64("axes_height", [2]),
        i64("pad18_to30", [0, 0, 0, 0, 0, 0, 12, 12]),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop9_starts", "crop9_ends", "axes4", "steps4"], ["x9"], name="crop_9x9"),
        helper.make_node("Slice", ["x9", "ch4_starts", "ch4_ends", "axes4", "steps4"], ["ch4"], name="slice_ch4"),
        helper.make_node("ReduceMax", ["ch4", "axes_width"], ["row_has"], keepdims=1, name="row_has"),
        helper.make_node("ReduceMax", ["ch4", "axes_height"], ["col_has"], keepdims=1, name="col_has"),
        helper.make_node("ArgMax", ["row_has"], ["top"], axis=2, keepdims=1, name="top_row"),
        helper.make_node("ArgMax", ["col_has"], ["left"], axis=3, keepdims=1, name="left_col"),
    ]

    selected: list[str] = []
    for top in range(9):
        for left in range(9):
            p = f"shift_t{top}_l{left}"
            inits.extend(slice_init(p, [0, 0, top, left], [1, 1, 9, 9], [0, 1, 2, 3], [1, 1, 1, 1]))
            inits.append(i64(f"{p}_pads", [0, 0, 0, 0, 0, 0, top, left]))
            inits.append(i64(f"{p}_top", np.array([[[[top]]]], dtype=np.int64)))
            inits.append(i64(f"{p}_left", np.array([[[[left]]]], dtype=np.int64)))
            nodes.extend(
                [
                    helper.make_node("Equal", ["top", f"{p}_top"], [f"{p}_top_match"], name=f"{p}_top_match"),
                    helper.make_node("Equal", ["left", f"{p}_left"], [f"{p}_left_match"], name=f"{p}_left_match"),
                    helper.make_node("And", [f"{p}_top_match", f"{p}_left_match"], [f"{p}_match_bool"], name=f"{p}_match_and"),
                    helper.make_node("Cast", [f"{p}_match_bool"], [f"{p}_match"], to=TensorProto.FLOAT, name=f"{p}_match_float"),
                    helper.make_node("Slice", ["ch4", f"{p}_starts", f"{p}_ends", f"{p}_axes", f"{p}_steps"], [f"{p}_body"], name=f"{p}_slice"),
                    helper.make_node("Pad", [f"{p}_body", f"{p}_pads", "zero_f"], [f"{p}_shifted"], name=f"{p}_pad"),
                    helper.make_node("Mul", [f"{p}_shifted", f"{p}_match"], [f"{p}_selected"], name=f"{p}_select"),
                ]
            )
            selected.append(f"{p}_selected")

    nodes.extend(
        [
            helper.make_node("Sum", selected, ["shifted4"], name="sum_shifted_candidates"),
            helper.make_node("ConvTranspose", ["shifted4", "scale2_weight"], ["out4_18"], name="scale_color4_2x", strides=[2, 2]),
            helper.make_node("ReduceMax", ["shifted4", "axes_width"], ["shifted_row_has"], keepdims=1, name="shifted_row_has"),
            helper.make_node("ReduceMax", ["shifted4", "axes_height"], ["shifted_col_has"], keepdims=1, name="shifted_col_has"),
            helper.make_node("ConvTranspose", ["shifted_row_has", "scale2_row_weight"], ["row_mask18"], name="scale_row_mask", strides=[2, 1]),
            helper.make_node("ConvTranspose", ["shifted_col_has", "scale2_col_weight"], ["col_mask18"], name="scale_col_mask", strides=[1, 2]),
            helper.make_node("Mul", ["row_mask18", "col_mask18"], ["bbox_mask18"], name="bbox_mask18"),
            helper.make_node("Sub", ["bbox_mask18", "out4_18"], ["out0_18"], name="background18"),
            helper.make_node("Mul", ["out4_18", "zero_f"], ["zero18"], name="zero18"),
            helper.make_node(
                "Concat",
                ["out0_18", "zero18", "zero18", "zero18", "out4_18", "zero18", "zero18", "zero18", "zero18", "zero18"],
                ["out18"],
                axis=1,
                name="concat_18",
            ),
            helper.make_node("Pad", ["out18", "pad18_to30", "zero_f"], ["output"], name="pad_to_static30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task384_static_bbox_scale",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=[
            vi("x9", TensorProto.FLOAT, [1, 10, 9, 9]),
            vi("ch4", TensorProto.FLOAT, [1, 1, 9, 9]),
            vi("row_has", TensorProto.FLOAT, [1, 1, 9, 1]),
            vi("col_has", TensorProto.FLOAT, [1, 1, 1, 9]),
            vi("top", TensorProto.INT64, [1, 1, 1, 1]),
            vi("left", TensorProto.INT64, [1, 1, 1, 1]),
            vi("shifted4", TensorProto.FLOAT, [1, 1, 9, 9]),
            vi("out4_18", TensorProto.FLOAT, [1, 1, 18, 18]),
            vi("bbox_mask18", TensorProto.FLOAT, [1, 1, 18, 18]),
            vi("out18", TensorProto.FLOAT, [1, 10, 18, 18]),
        ],
    )
    model = helper.make_model(graph, producer_name="neurogolf_task384_static_bbox_scale", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task384_static_bbox_scale"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task384.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task384.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task384.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
