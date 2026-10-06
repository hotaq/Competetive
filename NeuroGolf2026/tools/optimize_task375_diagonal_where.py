#!/usr/bin/env python3
"""Rewrite task375 as a direct diagonal-mask Where graph."""

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


def build_model(dynamic_io: bool, add_zero_tail: bool, tail_width: int) -> onnx.ModelProto:
    shape = [None, None, None, None] if dynamic_io else [1, 10, 30, 30]
    rows = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    cols = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    row_plus_col = rows + cols
    zero_onehot = np.zeros((1, 10, 1, 1), dtype=np.float32)
    zero_onehot[:, 0, :, :] = 1.0

    nodes: list[onnx.NodeProto] = [
        const_node("axis_channel_value", "axis_channel", [1], np.int64),
        const_node("spatial_axes_value", "spatial_axes", [2, 3], np.int64),
        const_node("row0_starts_value", "row0_starts", [0, 0, 0, 0], np.int64),
        const_node("row0_ends_value", "row0_ends", [1, 1, 1, 30], np.int64),
        const_node("row0_axes_value", "row0_axes", [0, 1, 2, 3], np.int64),
        const_node("row0_steps_value", "row0_steps", [1, 1, 1, 1], np.int64),
        const_node("one_value", "one", [1.0], np.float32),
        const_node("row_plus_col_value", "row_plus_col", row_plus_col, np.float32),
        const_node("main_diag_value", "main_diag", (rows == cols), np.bool_),
        const_node("zero_onehot_value", "zero_onehot", zero_onehot, np.float32),
        helper.make_node("ReduceSum", ["input", "axis_channel"], ["active"], name="active_cells", keepdims=1),
        helper.make_node("Slice", ["active", "row0_starts", "row0_ends", "row0_axes", "row0_steps"], ["top_row"], name="top_row"),
        helper.make_node("ReduceSum", ["top_row", "spatial_axes"], ["n_float"], name="count_width", keepdims=1),
        helper.make_node("Sub", ["n_float", "one"], ["n_minus_one"], name="n_minus_one"),
        helper.make_node("Equal", ["row_plus_col", "n_minus_one"], ["anti_diag"], name="anti_diag"),
        helper.make_node("Or", ["main_diag", "anti_diag"], ["diag_any"], name="diag_any"),
        helper.make_node("Cast", ["active"], ["inside"], name="inside_bool", to=TensorProto.BOOL),
        helper.make_node("And", ["diag_any", "inside"], ["diag_inside"], name="diag_inside"),
        helper.make_node("Where", ["diag_inside", "zero_onehot", "input"], ["painted" if add_zero_tail else "output"], name="paint_diags_zero"),
    ]
    if add_zero_tail:
        nodes.extend(
            [
                const_node("zero_tail_value", "zero_tail", np.zeros((1, 1, 1, tail_width), dtype=np.float32), np.float32),
                helper.make_node("Add", ["painted", "zero_tail"], ["output"], name="cost_floor_add_zero"),
            ]
        )

    value_info = [
        helper.make_tensor_value_info("active", TensorProto.FLOAT, [1, 1, 30, 30]),
        helper.make_tensor_value_info("top_row", TensorProto.FLOAT, [1, 1, 1, 30]),
        helper.make_tensor_value_info("n_float", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info("n_minus_one", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info("anti_diag", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("diag_any", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("inside", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("diag_inside", TensorProto.BOOL, [1, 1, 30, 30]),
        helper.make_tensor_value_info("painted", TensorProto.FLOAT, [1, 10, 30, 30]),
    ]
    graph = helper.make_graph(
        nodes,
        "task375_diagonal_where",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_zip(base_zip: Path, raw: bytes, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        seen = False
        for name in src.namelist():
            member = Path(name).name
            if member == "task375.onnx":
                dst.writestr(member, raw)
                seen = True
            else:
                dst.writestr(member, src.read(name))
        if not seen:
            dst.writestr("task375.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_gate_const_batch3_task312/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task375_diagonal_where"))
    parser.add_argument("--static-io", action="store_true")
    parser.add_argument("--add-zero-tail", action="store_true")
    parser.add_argument("--tail-width", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model(dynamic_io=not args.static_io, add_zero_tail=args.add_zero_tail, tail_width=args.tail_width).SerializeToString()
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
        f"ops={audit.get('ops')}"
    )
    print(f"model: {model_path}")
    print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
