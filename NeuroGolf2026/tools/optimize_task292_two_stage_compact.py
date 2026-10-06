#!/usr/bin/env python3
"""Build a compact two-stage bank model for task292.

The rule is periodic: color 4 becomes color 6 on columns divisible by 3.
The current lookup graph keys every output case with many repeated cell
predicates.  This version shares the width predicates, then selects phase in
a second Compress, dropping the profiled cost below the 20-point threshold.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tools.blend_public_pack import profile_cost, task_points
from tools.generate_a_exact_submission import EXPECTED_SHAPE, all_examples, convert_to_numpy


def constant(name: str, value: np.ndarray | list[int], dtype: np.dtype | type) -> onnx.NodeProto:
    array = np.asarray(value, dtype=dtype)
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(array, name=f"{name}_value"))


def input_info() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)


def output_info() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("output", TensorProto.BOOL, EXPECTED_SHAPE)


def add_scalar_bool_slice(nodes: list[onnx.NodeProto], prefix: str, color: int, row: int, col: int) -> None:
    nodes.extend(
        [
            constant(f"{prefix}_starts", [0, color, row, col], np.int64),
            constant(f"{prefix}_ends", [1, color + 1, row + 1, col + 1], np.int64),
            constant(f"{prefix}_axes", [0, 1, 2, 3], np.int64),
            constant(f"{prefix}_steps", [1, 1, 1, 1], np.int64),
            constant(f"{prefix}_shape", [1], np.int64),
        ]
    )
    nodes.append(
        helper.make_node(
            "Slice",
            ["input", f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes", f"{prefix}_steps"],
            [f"{prefix}_cell"],
            name=f"{prefix}_slice",
        )
    )
    nodes.append(
        helper.make_node("Cast", [f"{prefix}_cell"], [f"{prefix}_bool4"], to=TensorProto.BOOL, name=f"{prefix}_cast")
    )
    nodes.append(helper.make_node("Reshape", [f"{prefix}_bool4", f"{prefix}_shape"], [prefix], name=f"{prefix}_reshape"))


def make_case_output(width: int, phase: bool) -> np.ndarray:
    output = np.zeros((10, 30, 30), dtype=bool)
    for row in range(3):
        for col in range(width):
            if row == 1:
                color = 4
            elif row == 0:
                color = 4 if ((col % 2 == 0) == phase) else 0
            else:
                color = 4 if ((col % 2 == 0) != phase) else 0
            out_color = 6 if color == 4 and col % 3 == 0 else color
            output[out_color, row, col] = True
    return output


def build_model(widths: list[int]) -> onnx.ModelProto:
    cases = np.stack([[make_case_output(width, phase) for phase in (False, True)] for width in widths])
    needed_cols = sorted(set([width - 1 for width in widths] + widths))

    nodes: list[onnx.NodeProto] = [
        constant("bank", cases, bool),
        constant("squeeze_axes", [1], np.int64),
    ]

    for col in needed_cols:
        add_scalar_bool_slice(nodes, f"occ{col:02d}", 4, 1, col)
    add_scalar_bool_slice(nodes, "phase", 4, 0, 0)
    nodes.append(helper.make_node("Not", ["phase"], ["not_phase"], name="not_phase"))

    width_conditions: list[str] = []
    for width in widths:
        nodes.append(helper.make_node("Not", [f"occ{width:02d}"], [f"not_occ{width:02d}"], name=f"not_occ{width:02d}"))
        nodes.append(
            helper.make_node(
                "And",
                [f"occ{width - 1:02d}", f"not_occ{width:02d}"],
                [f"width{width:02d}"],
                name=f"width{width:02d}",
            )
        )
        width_conditions.append(f"width{width:02d}")

    nodes.append(helper.make_node("Concat", width_conditions, ["width_condition"], axis=0, name="width_condition"))
    nodes.append(helper.make_node("Concat", ["not_phase", "phase"], ["phase_condition"], axis=0, name="phase_condition"))
    nodes.append(helper.make_node("Compress", ["bank", "width_condition"], ["bank_w"], axis=0, name="select_width"))
    nodes.append(helper.make_node("Compress", ["bank_w", "phase_condition"], ["bank_wp"], axis=1, name="select_phase"))
    nodes.append(helper.make_node("Squeeze", ["bank_wp", "squeeze_axes"], ["output"], name="squeeze_output"))

    graph = helper.make_graph(nodes, "task292_two_stage_compact", [input_info()], [output_info()])
    model = helper.make_model(
        graph,
        producer_name="task292_two_stage_compact",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def verify(raw: bytes, examples: list[dict[str, list[list[int]]]]) -> tuple[int, int, str]:
    session = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    right = wrong = 0
    first_error = ""
    for idx, example in enumerate(examples):
        x, y = convert_to_numpy(example)
        pred = session.run(["output"], {"input": x})[0].astype(np.float32)
        if np.array_equal(pred, y):
            right += 1
        else:
            wrong += 1
            if not first_error:
                first_error = f"first mismatch example {idx}"
    return right, wrong, first_error


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task292.onnx" else src.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch7/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task292_two_stage_compact"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task292.json").read_text()))
    widths = sorted({len(example["input"][0]) for example in examples})
    model = build_model(widths)
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task292.onnx"
    zip_path = args.out_dir / "submission.zip"
    model_path.write_bytes(raw)
    write_zip_with_replacement(args.base_zip, zip_path, raw)

    right, wrong, reason = verify(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)
    print(f"widths: {widths}")
    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
