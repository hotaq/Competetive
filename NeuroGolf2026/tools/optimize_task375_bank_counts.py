#!/usr/bin/env python3
"""Build task375 as a count-keyed output bank."""

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
from tools.audit_clean_best_by_triage import audit_member  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def const_node(name: str, output: str, value: np.ndarray | list[int] | list[float], dtype: np.dtype) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=dtype), name=f"{output}__value")
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def make_output(size: int, color: int) -> np.ndarray:
    out = np.zeros((10, 30, 30), dtype=np.bool_)
    for r in range(size):
        for c in range(size):
            out[color, r, c] = True
    for i in range(size):
        out[color, i, i] = False
        out[0, i, i] = True
        j = size - 1 - i
        out[color, i, j] = False
        out[0, i, j] = True
    return out


def build_bank(max_size: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = list(range(3, max_size + 1, 2))
    rows: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    for size in sizes:
        for color in range(1, 10):
            rows.append(make_output(size, color))
            threshold = np.zeros(10, dtype=np.float32)
            threshold[0] = 1.0
            threshold[color] = float(size * size - 1)
            thresholds.append(threshold)
    return np.stack(rows, axis=0), np.stack(thresholds, axis=0)


def build_model(dynamic_io: bool, max_size: int, add_zero_tail: bool, tail_width: int) -> onnx.ModelProto:
    bank, thresholds = build_bank(max_size)
    io_shape = [None, None, None, None] if dynamic_io else [1, 10, 30, 30]
    nodes: list[onnx.NodeProto] = [
        const_node("spatial_count_axes_value", "spatial_count_axes", [0, 2, 3], np.int64),
        const_node("channel_axis_value", "channel_axis", [1], np.int64),
        const_node("ten_value", "ten", [10.0], np.float32),
        const_node("thresholds_value", "thresholds", thresholds, np.float32),
        const_node("bank_value", "bank", bank, np.bool_),
        helper.make_node("ReduceSum", ["input", "spatial_count_axes"], ["counts"], name="channel_counts", keepdims=0),
        helper.make_node("Equal", ["counts", "thresholds"], ["eq_counts"], name="match_counts"),
        helper.make_node("Cast", ["eq_counts"], ["eq_counts_f"], name="match_counts_f", to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["eq_counts_f", "channel_axis"], ["eq_total"], name="match_total", keepdims=0),
        helper.make_node("Equal", ["eq_total", "ten"], ["condition"], name="select_condition"),
        helper.make_node("Compress", ["bank", "condition"], ["selected_bool"], name="select_output", axis=0),
        helper.make_node("Cast", ["selected_bool"], ["selected" if add_zero_tail else "output"], name="selected_float", to=TensorProto.FLOAT),
    ]
    if add_zero_tail:
        nodes.extend(
            [
                const_node("zero_tail_value", "zero_tail", np.zeros((1, 1, 1, tail_width), dtype=np.float32), np.float32),
                helper.make_node("Add", ["selected", "zero_tail"], ["output"], name="cost_floor_add_zero"),
            ]
        )

    m = bank.shape[0]
    value_info = [
        helper.make_tensor_value_info("counts", TensorProto.FLOAT, [10]),
        helper.make_tensor_value_info("eq_counts", TensorProto.BOOL, [m, 10]),
        helper.make_tensor_value_info("eq_counts_f", TensorProto.FLOAT, [m, 10]),
        helper.make_tensor_value_info("eq_total", TensorProto.FLOAT, [m]),
        helper.make_tensor_value_info("condition", TensorProto.BOOL, [m]),
        helper.make_tensor_value_info("selected_bool", TensorProto.BOOL, [1, 10, 30, 30]),
        helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1, 10, 30, 30]),
    ]
    graph = helper.make_graph(
        nodes,
        "task375_bank_counts",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, io_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, io_shape)],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_zip(base_zip: Path, raw: bytes, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task375.onnx" else src.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_gate_const_batch3_task312/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task375_bank_counts"))
    parser.add_argument("--static-io", action="store_true")
    parser.add_argument("--max-size", type=int, default=29)
    parser.add_argument("--add-zero-tail", action="store_true")
    parser.add_argument("--tail-width", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model(
        dynamic_io=not args.static_io,
        max_size=args.max_size,
        add_zero_tail=args.add_zero_tail,
        tail_width=args.tail_width,
    ).SerializeToString()
    model_path = args.out_dir / "task375.onnx"
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task375.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"verify task375: {right}/{len(examples)} wrong={wrong} reason={reason}")
    if wrong:
        raise SystemExit(1)

    zip_path = args.out_dir / "submission.zip"
    write_zip(args.base_zip, raw, zip_path)
    audit = audit_member("task375", raw, args.out_dir / "audit_work" / "task375.onnx", args.data_root)
    print(
        "audit task375: "
        f"reason={audit.get('reason')} cost={audit.get('cost')} points={audit.get('points')} "
        f"macs={audit.get('macs')} memory={audit.get('memory')} params={audit.get('params')} "
        f"file_bytes={len(raw)} ops={audit.get('ops')}"
    )
    print(f"model: {model_path}")
    print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
