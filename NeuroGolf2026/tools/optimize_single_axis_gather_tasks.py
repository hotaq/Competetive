#!/usr/bin/env python3
"""Build clean one-Gather ONNX models for single-axis ARC transforms."""

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
TASKS: dict[str, tuple[int, list[int]]] = {
    "task053": (2, [2, 0, 1]),
    "task116": (2, [2, 1, 0, 0, 1, 2]),
    "task164": (3, [0, 1, 2, 2, 1, 0]),
    "task172": (2, [0, 1, 2, 2, 1, 0]),
    "task210": (2, [0, 1, 2, 2, 1, 0]),
    "task311": (3, [0, 1, 2, 2, 1, 0]),
    "task385": (2, [9, 8, 7, 6, 5, 5, 6, 7, 8, 9]),
}


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def make_model(task: str, axis: int, prefix_indices: list[int]) -> onnx.ModelProto:
    indices = prefix_indices + list(range(len(prefix_indices), 30))
    init = numpy_helper.from_array(np.asarray(indices, dtype=np.int64), name="idx")
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis, name=f"{task}_single_axis_gather")
    graph = helper.make_graph([node], f"{task}_single_axis_gather", [tensor_info("input")], [tensor_info("output")], initializer=[init])
    model = helper.make_model(
        graph,
        producer_name="neurogolf_single_axis_gather",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out_dir = Path("generated/custom_single_axis_gather_tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, bytes] = {}
    for task, (axis, prefix_indices) in TASKS.items():
        model = make_model(task, axis, prefix_indices)
        raw = model.SerializeToString()
        examples = all_examples(json.loads(Path(f"data/latest/extracted/{task}.json").read_text()))
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(f"{task} verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
        if wrong:
            raise SystemExit(1)
        (out_dir / f"{task}.onnx").write_bytes(raw)
        raws[f"{task}.onnx"] = raw

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(raws.items()):
            zf.writestr(name, raw)
    print(f"wrote {out_dir / 'submission.zip'}")


if __name__ == "__main__":
    main()
