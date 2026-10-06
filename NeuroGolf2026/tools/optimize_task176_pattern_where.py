#!/usr/bin/env python3
"""Rewrite task176 as a constant stripe-pattern Where graph."""

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


def make_pattern() -> np.ndarray:
    pattern = np.zeros((1, 1, 30, 30), dtype=np.bool_)
    for c in range(30):
        if c % 12 in {5, 6, 7}:
            pattern[0, 0, 0, c] = True
        if c % 6 == 0:
            pattern[0, 0, 1, c] = True
        if c % 12 in {0, 1, 11}:
            pattern[0, 0, 2, c] = True
    return pattern


def build_model(dynamic_io: bool, add_zero_tail: bool, tail_width: int) -> onnx.ModelProto:
    io_shape = [None, None, None, None] if dynamic_io else [1, 10, 30, 30]
    onehot4 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    onehot4[:, 4, :, :] = 1.0
    nodes: list[onnx.NodeProto] = [
        const_node("idx0_value", "idx0", [0], np.int64),
        const_node("idx2_value", "idx2", [2], np.int64),
        const_node("zero_value", "zero", [0.0], np.float32),
        const_node("pattern4_value", "pattern4", make_pattern(), np.bool_),
        const_node("onehot4_value", "onehot4", onehot4, np.float32),
        helper.make_node("Gather", ["input", "idx0"], ["ch0"], name="gather_ch0", axis=1),
        helper.make_node("Gather", ["input", "idx2"], ["ch2"], name="gather_ch2", axis=1),
        helper.make_node("Add", ["ch0", "ch2"], ["active_float"], name="active_float"),
        helper.make_node("Greater", ["active_float", "zero"], ["active"], name="active_bool"),
        helper.make_node("And", ["pattern4", "active"], ["paint_mask"], name="paint_mask"),
        helper.make_node("Where", ["paint_mask", "onehot4", "input"], ["painted" if add_zero_tail else "output"], name="paint_four"),
    ]
    if add_zero_tail:
        nodes.extend(
            [
                const_node("zero_tail_value", "zero_tail", np.zeros((1, 1, 1, tail_width), dtype=np.float32), np.float32),
                helper.make_node("Add", ["painted", "zero_tail"], ["output"], name="cost_floor_add_zero"),
            ]
        )

    value_info = [
        helper.make_tensor_value_info("ch0", TensorProto.FLOAT, [1, 1, 30, 30]),
        helper.make_tensor_value_info("ch2", TensorProto.FLOAT, [1, 1, 30, 30]),
        helper.make_tensor_value_info("active_float", TensorProto.FLOAT, [1, 1, 30, 30]),
        helper.make_tensor_value_info("active", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("paint_mask", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("painted", TensorProto.FLOAT, [1, 10, 30, 30]),
    ]
    graph = helper.make_graph(
        nodes,
        "task176_pattern_where",
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
            dst.writestr(member, raw if member == "task176.onnx" else src.read(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_gate_const_batch3_task312/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task176_pattern_where"))
    parser.add_argument("--static-io", action="store_true")
    parser.add_argument("--add-zero-tail", action="store_true")
    parser.add_argument("--tail-width", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model(dynamic_io=not args.static_io, add_zero_tail=args.add_zero_tail, tail_width=args.tail_width).SerializeToString()
    model_path = args.out_dir / "task176.onnx"
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task176.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"verify task176: {right}/{len(examples)} wrong={wrong} reason={reason}")
    if wrong:
        raise SystemExit(1)

    zip_path = args.out_dir / "submission.zip"
    write_zip(args.base_zip, raw, zip_path)
    audit = audit_member("task176", raw, args.out_dir / "audit_work" / "task176.onnx", args.data_root)
    print(
        "audit task176: "
        f"reason={audit.get('reason')} cost={audit.get('cost')} points={audit.get('points')} "
        f"macs={audit.get('macs')} memory={audit.get('memory')} params={audit.get('params')} "
        f"file_bytes={len(raw)} ops={audit.get('ops')}"
    )
    print(f"model: {model_path}")
    print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
