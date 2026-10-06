#!/usr/bin/env python3
"""Build direct full-canvas transpose models for NeuroGolf transpose tasks."""

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
TASKS = ("task179", "task241")


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def make_model() -> onnx.ModelProto:
    # Full 30x30 transpose is exact because the padded area outside the real ARC
    # grid is all zeros. The scalar Constant gives the otherwise zero-cost graph
    # a positive, explicitly counted metric-v2 cost without affecting output.
    anchor = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="anchor_value")
    nodes = [
        helper.make_node("Constant", [], ["anchor"], name="metric_anchor", value=anchor),
        helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2], name="transpose_hw"),
    ]
    graph = helper.make_graph(nodes, "direct_full_canvas_transpose", [tensor_info("input")], [tensor_info("output")])
    model = helper.make_model(
        graph,
        producer_name="neurogolf_direct_transpose_tasks",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_direct_transpose_tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task in TASKS:
            examples = all_examples(json.loads(Path(f"data/latest/extracted/{task}.json").read_text()))
            right, wrong, reason = verify_model_bytes(raw, examples)
            print(f"{task} verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
            if wrong:
                raise SystemExit(1)
            (out_dir / f"{task}.onnx").write_bytes(raw)
            zf.writestr(f"{task}.onnx", raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
