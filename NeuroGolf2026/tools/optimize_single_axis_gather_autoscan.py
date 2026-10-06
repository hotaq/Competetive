#!/usr/bin/env python3
"""Auto-build clean one-Gather models for exact single-axis remaps.

This scans every task for transformations representable as exactly one ONNX
Gather over channel, row, or column on the padded 1x10x30x30 tensor.
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
from tools.generate_a_exact_submission import all_examples, convert_to_numpy, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import examples_fit_30  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
AXES = ((1, 10), (2, 30), (3, 30))


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def infer_single_axis_gather(examples: list[dict[str, list[list[int]]]]) -> tuple[int, list[int]] | None:
    arrays = [convert_to_numpy(example) for example in examples]
    for axis, length in AXES:
        indices: list[int] = []
        for out_index in range(length):
            candidates: list[int] = []
            for in_index in range(length):
                if all(
                    np.array_equal(
                        np.take(inp, in_index, axis=axis),
                        np.take(out, out_index, axis=axis),
                    )
                    for inp, out in arrays
                ):
                    candidates.append(in_index)
            if not candidates:
                break
            indices.append(out_index if out_index in candidates else candidates[0])
        else:
            if indices != list(range(length)):
                return axis, indices
    return None


def make_model(task: str, axis: int, indices: list[int]) -> onnx.ModelProto:
    idx = numpy_helper.from_array(np.asarray(indices, dtype=np.int64), name="idx")
    node = helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis, name=f"{task}_single_axis_gather")
    graph = helper.make_graph(
        [node],
        f"{task}_single_axis_gather",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=[idx],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_single_axis_gather_autoscan",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    data_root = Path("data/latest/extracted")
    out_dir = Path("generated/custom_single_axis_gather_autoscan")
    out_dir.mkdir(parents=True, exist_ok=True)

    raws: dict[str, bytes] = {}
    for task_path in sorted(data_root.glob("task*.json")):
        task = task_path.stem
        examples = all_examples(json.loads(task_path.read_text()))
        if not examples_fit_30(examples):
            continue
        inferred = infer_single_axis_gather(examples)
        if inferred is None:
            continue
        axis, indices = inferred
        model = make_model(task, axis, indices)
        raw = model.SerializeToString()
        right, wrong, reason = verify_model_bytes(raw, examples)
        print(
            f"{task} axis={axis} cost_hint={len(indices)} "
            f"verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}"
        )
        if wrong:
            raise SystemExit(1)
        (out_dir / f"{task}.onnx").write_bytes(raw)
        raws[f"{task}.onnx"] = raw

    with zipfile.ZipFile(out_dir / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, raw in sorted(raws.items()):
            zf.writestr(name, raw)
    print(f"wrote {out_dir / 'submission.zip'} with {len(raws)} tasks")


if __name__ == "__main__":
    main()
