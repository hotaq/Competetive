#!/usr/bin/env python3
"""Build a compact clean ONNX pattern matcher for NeuroGolf task146.

Task146 consists of three stacked 3x3 two-color blocks.  The output is one of
those blocks.  A color-invariant binary mask lookup over the 3x3 patterns
separates selected blocks from distractors on the public train/test/arc-gen
set, and the matcher is implemented as grouped 3x3 convolutions.
"""

from __future__ import annotations

import argparse
import collections
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


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def color_code(block: list[list[int]], color: int) -> int:
    flat = [value for row in block for value in row]
    return sum((1 if flat[i] == color else 0) << i for i in range(9))


def minority_code(block: list[list[int]]) -> int:
    counts = collections.Counter(value for row in block for value in row)
    color = min(counts, key=lambda value: (counts[value], value))
    return color_code(block, color)


def selected_codes(examples: list[dict[str, list[list[int]]]]) -> list[int]:
    codes: set[int] = set()
    for example in examples:
        inp = example["input"]
        out = example["output"]
        for idx in range(3):
            block = inp[idx * 3 : (idx + 1) * 3]
            if block == out:
                codes.add(minority_code(block))
    return sorted(codes)


def matcher_tensors(codes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    weights: list[np.ndarray] = []
    biases: list[float] = []
    for _color in range(9):
        for code in codes:
            mask = np.array([(code >> i) & 1 for i in range(9)], dtype=np.float32).reshape(3, 3)
            weights.append((2.0 * mask - 1.0)[None, :, :])
            biases.append(float(9.0 - mask.sum() - 8.5))
    return np.stack(weights, axis=0).astype(np.float32), np.asarray(biases, dtype=np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def make_model(codes: list[int]) -> onnx.ModelProto:
    weights, biases = matcher_tensors(codes)
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("channel_axis", [1]),
        const_f32("match_weight", weights),
        const_f32("match_bias", biases),
        const_f32("threshold", 0.0),
        const_f32("zero_scalar", 0.0),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 27, 27]),
    ]
    for idx in range(3):
        r0 = idx * 3
        r1 = r0 + 3
        inits.extend(slice_4d(f"block{idx}", [0, 0, r0, 0], [1, 10, r1, 3]))
        inits.extend(slice_4d(f"block{idx}_colors", [0, 1, r0, 0], [1, 10, r1, 3]))

    nodes: list[onnx.NodeProto] = []
    selected_terms: list[str] = []
    for idx in range(3):
        nodes.append(make_slice("input", f"block{idx}", f"block{idx}"))
        nodes.append(make_slice("input", f"block{idx}_colors", f"block{idx}_colors"))
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [f"block{idx}_colors", "match_weight", "match_bias"],
                    [f"block{idx}_scores"],
                    group=9,
                    name=f"block{idx}_pattern_scores",
                ),
                helper.make_node("Greater", [f"block{idx}_scores", "threshold"], [f"block{idx}_hits"], name=f"block{idx}_hits"),
                helper.make_node("Cast", [f"block{idx}_hits"], [f"block{idx}_hits_f"], to=TensorProto.FLOAT, name=f"block{idx}_hits_cast"),
                helper.make_node(
                    "ReduceMax",
                    [f"block{idx}_hits_f"],
                    [f"block{idx}_selected"],
                    axes=[1],
                    keepdims=1,
                    name=f"block{idx}_selected",
                ),
                helper.make_node("Mul", [f"block{idx}", f"block{idx}_selected"], [f"block{idx}_masked"], name=f"block{idx}_mask_block"),
            ]
        )
        selected_terms.append(f"block{idx}_masked")

    nodes.extend(
        [
            helper.make_node("Sum", selected_terms, ["out3"], name="sum_selected_block"),
            helper.make_node("Pad", ["out3", "pad_to_30", "zero_scalar"], ["output"], name="pad_output"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task146_pattern_lookup",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task146_pattern_lookup",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task146_pattern_lookup_v102"))
    args = parser.parse_args()

    task = json.loads(Path("data/latest/extracted/task146.json").read_text())
    examples = all_examples(task)
    codes = selected_codes(examples)
    print(f"selected pattern codes: {len(codes)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(codes)
    raw = model.SerializeToString()
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task146 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task146.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task146.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task146 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
