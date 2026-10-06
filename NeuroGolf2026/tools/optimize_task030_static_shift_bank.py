#!/usr/bin/env python3
"""Build a strict-static shift-bank model for task030.

The task aligns the colored objects to the top row of color 1.  Older compact
graphs used dynamic Slice/Pad lengths, which now fail the metric-v2 static
shape check.  This version keeps every intermediate shape fixed at 10x10 by
materializing all row shifts and selecting the one matching the top-row delta.
"""

from __future__ import annotations

import argparse
import json
import math
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


def make_shift_nodes(channel: str, prefix: str) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str]:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    selected: list[str] = []

    for delta in range(-9, 10):
        dname = f"{prefix}_d{delta:+d}".replace("+", "p").replace("-", "m")
        inits.append(i64(f"{dname}_delta", np.array([[[[delta]]]], dtype=np.int64)))
        nodes.append(helper.make_node("Equal", [f"{prefix}_delta", f"{dname}_delta"], [f"{dname}_match_bool"], name=f"{dname}_match"))
        nodes.append(helper.make_node("Cast", [f"{dname}_match_bool"], [f"{dname}_match"], to=TensorProto.FLOAT, name=f"{dname}_match_float"))

        if delta == 0:
            nodes.append(helper.make_node("Identity", [channel], [f"{dname}_shifted"], name=f"{dname}_identity"))
        elif delta > 0:
            inits.extend(slice_init(dname, [0, 0, 0, 0], [1, 1, 10 - delta, 10], [0, 1, 2, 3], [1, 1, 1, 1]))
            inits.append(i64(f"{dname}_pads", [0, 0, delta, 0, 0, 0, 0, 0]))
            nodes.append(
                helper.make_node(
                    "Slice",
                    [channel, f"{dname}_starts", f"{dname}_ends", f"{dname}_axes", f"{dname}_steps"],
                    [f"{dname}_body"],
                    name=f"{dname}_slice",
                )
            )
            nodes.append(helper.make_node("Pad", [f"{dname}_body", f"{dname}_pads", "zero_f"], [f"{dname}_shifted"], name=f"{dname}_pad"))
        else:
            cut = -delta
            inits.extend(slice_init(dname, [0, 0, cut, 0], [1, 1, 10, 10], [0, 1, 2, 3], [1, 1, 1, 1]))
            inits.append(i64(f"{dname}_pads", [0, 0, 0, 0, 0, 0, cut, 0]))
            nodes.append(
                helper.make_node(
                    "Slice",
                    [channel, f"{dname}_starts", f"{dname}_ends", f"{dname}_axes", f"{dname}_steps"],
                    [f"{dname}_body"],
                    name=f"{dname}_slice",
                )
            )
            nodes.append(helper.make_node("Pad", [f"{dname}_body", f"{dname}_pads", "zero_f"], [f"{dname}_shifted"], name=f"{dname}_pad"))

        nodes.append(helper.make_node("Mul", [f"{dname}_shifted", f"{dname}_match"], [f"{dname}_selected"], name=f"{dname}_select"))
        selected.append(f"{dname}_selected")

    out = f"{prefix}_shifted"
    nodes.append(helper.make_node("Sum", selected, [out], name=f"{prefix}_sum_shift_bank"))
    return nodes, inits, out


def build_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        f32("zero_f", 0.0),
        i64("crop10_starts", [0, 0, 0, 0]),
        i64("crop10_ends", [1, 10, 10, 10]),
        i64("axes4", [0, 1, 2, 3]),
        i64("steps4", [1, 1, 1, 1]),
        i64("axes_width", [3]),
        i64("axes_ch_width", [1, 3]),
        i64("pad10_to30", [0, 0, 0, 0, 0, 0, 20, 20]),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "crop10_starts", "crop10_ends", "axes4", "steps4"], ["x10"], name="crop_10x10"),
        helper.make_node("Slice", ["x10", "ch0_starts", "ch0_ends", "axes4", "steps4"], ["ch0"], name="slice_ch0"),
        helper.make_node("Slice", ["x10", "ch1_starts", "ch1_ends", "axes4", "steps4"], ["ch1"], name="slice_ch1"),
        helper.make_node("Slice", ["x10", "ch2_starts", "ch2_ends", "axes4", "steps4"], ["ch2"], name="slice_ch2"),
        helper.make_node("Slice", ["x10", "ch4_starts", "ch4_ends", "axes4", "steps4"], ["ch4"], name="slice_ch4"),
        helper.make_node("ReduceMax", ["x10", "axes_ch_width"], ["row_valid"], keepdims=1, name="row_valid_mask"),
        helper.make_node("ReduceMax", ["ch1", "axes_width"], ["row_has_1"], keepdims=1, name="row_has_1"),
        helper.make_node("ReduceMax", ["ch2", "axes_width"], ["row_has_2"], keepdims=1, name="row_has_2"),
        helper.make_node("ReduceMax", ["ch4", "axes_width"], ["row_has_4"], keepdims=1, name="row_has_4"),
        helper.make_node("ArgMax", ["row_has_1"], ["top_1"], axis=2, keepdims=1, name="top_1"),
        helper.make_node("ArgMax", ["row_has_2"], ["top_2"], axis=2, keepdims=1, name="top_2"),
        helper.make_node("ArgMax", ["row_has_4"], ["top_4"], axis=2, keepdims=1, name="top_4"),
        helper.make_node("Sub", ["top_1", "top_2"], ["shift2_delta"], name="delta_2_to_1"),
        helper.make_node("Sub", ["top_1", "top_4"], ["shift4_delta"], name="delta_4_to_1"),
    ]
    inits.extend(slice_init("ch0", [0, 0, 0, 0], [1, 1, 10, 10], [0, 1, 2, 3], [1, 1, 1, 1]))
    inits.extend(slice_init("ch1", [0, 1, 0, 0], [1, 2, 10, 10], [0, 1, 2, 3], [1, 1, 1, 1]))
    inits.extend(slice_init("ch2", [0, 2, 0, 0], [1, 3, 10, 10], [0, 1, 2, 3], [1, 1, 1, 1]))
    inits.extend(slice_init("ch4", [0, 4, 0, 0], [1, 5, 10, 10], [0, 1, 2, 3], [1, 1, 1, 1]))

    shift2_nodes, shift2_inits, shifted2 = make_shift_nodes("ch2", "shift2")
    shift4_nodes, shift4_inits, shifted4 = make_shift_nodes("ch4", "shift4")
    nodes.extend(shift2_nodes)
    nodes.extend(shift4_nodes)
    inits.extend(shift2_inits)
    inits.extend(shift4_inits)

    nodes.extend(
        [
            helper.make_node("Mul", ["ch1", "row_valid"], ["out1"], name="mask_ch1_to_height"),
            helper.make_node("Mul", [shifted2, "row_valid"], ["out2"], name="mask_ch2_to_height"),
            helper.make_node("Mul", [shifted4, "row_valid"], ["out4"], name="mask_ch4_to_height"),
            helper.make_node("Sum", ["out1", "out2", "out4"], ["occupied"], name="occupied_non_bg"),
            helper.make_node("Sub", ["row_valid", "occupied"], ["out0"], name="background_from_height"),
            helper.make_node("Mul", ["out1", "zero_f"], ["zero_ch"], name="make_zero_channel"),
            helper.make_node(
                "Concat",
                ["out0", "out1", "out2", "zero_ch", "out4", "zero_ch", "zero_ch", "zero_ch", "zero_ch", "zero_ch"],
                ["out10"],
                axis=1,
                name="concat_output10",
            ),
            helper.make_node("Pad", ["out10", "pad10_to30", "zero_f"], ["output"], name="pad_to_static30"),
        ]
    )

    value_info = [
        vi("x10", TensorProto.FLOAT, [1, 10, 10, 10]),
        vi("ch0", TensorProto.FLOAT, [1, 1, 10, 10]),
        vi("ch1", TensorProto.FLOAT, [1, 1, 10, 10]),
        vi("ch2", TensorProto.FLOAT, [1, 1, 10, 10]),
        vi("ch4", TensorProto.FLOAT, [1, 1, 10, 10]),
        vi("row_valid", TensorProto.FLOAT, [1, 1, 10, 1]),
        vi("row_has_1", TensorProto.FLOAT, [1, 1, 10, 1]),
        vi("row_has_2", TensorProto.FLOAT, [1, 1, 10, 1]),
        vi("row_has_4", TensorProto.FLOAT, [1, 1, 10, 1]),
        vi("top_1", TensorProto.INT64, [1, 1, 1, 1]),
        vi("top_2", TensorProto.INT64, [1, 1, 1, 1]),
        vi("top_4", TensorProto.INT64, [1, 1, 1, 1]),
        vi("shift2_delta", TensorProto.INT64, [1, 1, 1, 1]),
        vi("shift4_delta", TensorProto.INT64, [1, 1, 1, 1]),
        vi("out10", TensorProto.FLOAT, [1, 10, 10, 10]),
    ]

    graph = helper.make_graph(
        nodes,
        "task030_static_shift_bank",
        [vi("input")],
        [vi("output")],
        initializer=inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, producer_name="neurogolf_task030_static_shift_bank", opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task030_static_shift_bank"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task030.onnx"
    zip_path = args.out_dir / "submission.zip"
    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task030.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = score_network(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task030.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
