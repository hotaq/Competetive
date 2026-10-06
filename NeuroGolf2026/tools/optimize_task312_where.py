#!/usr/bin/env python3
"""Rewrite task312 as Gather + Expand + Where."""

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


def build_model(static_input: bool, static_output: bool, add_zero_tail: bool, tail_width: int) -> onnx.ModelProto:
    static_shape = [1, 10, 30, 30]
    input_shape = static_shape if static_input else [None, None, None, None]
    output_shape = static_shape if static_output else [None, None, None, None]
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["idx_col0"],
            value=numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx_col0_value"),
            name="const_idx_col0",
        ),
        helper.make_node(
            "Constant",
            [],
            ["idx_ch5"],
            value=numpy_helper.from_array(np.array([5], dtype=np.int64), name="idx_ch5_value"),
            name="const_idx_ch5",
        ),
        helper.make_node("Gather", ["input", "idx_col0"], ["row_anchor"], axis=3, name="row_anchor"),
        helper.make_node("Shape", ["input"], ["input_shape"], name="input_shape"),
        helper.make_node("Expand", ["row_anchor", "input_shape"], ["row_fill"], name="row_fill"),
        helper.make_node("Gather", ["input", "idx_ch5"], ["mask5"], axis=1, name="mask5"),
        helper.make_node("Cast", ["mask5"], ["mask5_bool"], to=TensorProto.BOOL, name="mask5_bool"),
        helper.make_node(
            "Where",
            ["mask5_bool", "row_fill", "input"],
            ["painted" if add_zero_tail else "output"],
            name="paint_5_from_row_anchor",
        ),
    ]
    if add_zero_tail:
        nodes.extend(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["zero_w30"],
                    value=numpy_helper.from_array(
                        np.zeros((1, 1, 1, tail_width), dtype=np.float32),
                        name="zero_w30_value",
                    ),
                    name="const_zero_w30",
                ),
                helper.make_node("Add", ["painted", "zero_w30"], ["output"], name="cost_floor_add_zero"),
            ]
        )
    graph = helper.make_graph(nodes, "task312_where", [input_vi], [output_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_zip(base_zip: Path, raw: bytes, zip_out: Path) -> None:
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        seen = False
        for name in src.namelist():
            member = Path(name).name
            if member == "task312.onnx":
                dst.writestr(member, raw)
                seen = True
            else:
                dst.writestr(member, src.read(name))
        if not seen:
            dst.writestr("task312.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/custom_gate_const_batch2/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task312_where"))
    parser.add_argument("--static-shape", action="store_true")
    parser.add_argument("--static-input", action="store_true")
    parser.add_argument("--static-output", action="store_true")
    parser.add_argument("--add-zero-tail", action="store_true")
    parser.add_argument("--tail-width", type=int, default=30)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = build_model(
        static_input=args.static_shape or args.static_input,
        static_output=args.static_shape or args.static_output,
        add_zero_tail=args.add_zero_tail,
        tail_width=args.tail_width,
    ).SerializeToString()
    model_path = args.out_dir / "task312.onnx"
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task312.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"verify task312: {right}/{len(examples)} wrong={wrong} reason={reason}")
    if wrong:
        raise SystemExit(1)

    zip_path = args.out_dir / "submission.zip"
    write_zip(args.base_zip, raw, zip_path)
    audit = audit_member("task312", raw, args.out_dir / "audit_work" / "task312.onnx", args.data_root)
    print(
        "audit task312: "
        f"reason={audit.get('reason')} cost={audit.get('cost')} points={audit.get('points')} "
        f"macs={audit.get('macs')} memory={audit.get('memory')} params={audit.get('params')} "
        f"ops={audit.get('ops')}"
    )
    print(f"model: {model_path}")
    print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
