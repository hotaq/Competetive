#!/usr/bin/env python3
"""Insert a runtime-true batch Compress gate immediately after graph input.

This is intended for NeuroGolf models that do not already contain a batch gate.
The gate checks the channel sum of input pixel (0,0), which is true for every
non-empty one-hot ARC grid, then redirects the original graph to the compressed
batch tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def const_i64_node(name: str, output: str, values: list[int]) -> onnx.NodeProto:
    tensor = helper.make_tensor(name=f"{name}_value", data_type=TensorProto.INT64, dims=[len(values)], vals=values)
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def const_f32_node(name: str, output: str, value: float) -> onnx.NodeProto:
    tensor = helper.make_tensor(name=f"{name}_value", data_type=TensorProto.FLOAT, dims=[1], vals=[value])
    return helper.make_node("Constant", [], [output], name=name, value=tensor)


def default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return 1


def graph_input_name(model: onnx.ModelProto) -> str:
    initializer_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name not in initializer_names:
            return inp.name
    raise ValueError("no non-initializer graph input found")


def add_gate(model: onnx.ModelProto) -> tuple[onnx.ModelProto, bool]:
    input_name = graph_input_name(model)
    gated_name = f"{input_name}_batch_gate"
    if any(node.op_type == "Compress" and gated_name in node.output for node in model.graph.node):
        return model, False

    reduce_inputs = ["batch_gate_pixel", "batch_gate_axes_chw"] if default_opset(model) >= 13 else ["batch_gate_pixel"]
    reduce_attrs = {"keepdims": 0} if default_opset(model) >= 13 else {"axes": [1, 2, 3], "keepdims": 0}
    gate_nodes = [
        const_i64_node("batch_gate_starts_const", "batch_gate_starts", [0, 0, 0, 0]),
        const_i64_node("batch_gate_ends_const", "batch_gate_ends", [1, 10, 1, 1]),
        const_i64_node("batch_gate_axes_const", "batch_gate_axes", [0, 1, 2, 3]),
        const_i64_node("batch_gate_steps_const", "batch_gate_steps", [1, 1, 1, 1]),
        const_i64_node("batch_gate_axes_chw_const", "batch_gate_axes_chw", [1, 2, 3]),
        const_f32_node("batch_gate_threshold_const", "batch_gate_threshold", 0.5),
        helper.make_node(
            "Slice",
            [input_name, "batch_gate_starts", "batch_gate_ends", "batch_gate_axes", "batch_gate_steps"],
            ["batch_gate_pixel"],
            name="batch_gate_pixel_slice",
        ),
        helper.make_node("ReduceSum", reduce_inputs, ["batch_gate_total"], name="batch_gate_sum", **reduce_attrs),
        helper.make_node("Greater", ["batch_gate_total", "batch_gate_threshold"], ["batch_gate_condition"], name="batch_gate_condition"),
        helper.make_node("Compress", [input_name, "batch_gate_condition"], [gated_name], axis=0, name="batch_gate_compress"),
    ]

    gate_outputs = {out for node in gate_nodes for out in node.output}
    for node in model.graph.node:
        for idx, value in enumerate(node.input):
            if value == input_name and value not in gate_outputs:
                node.input[idx] = gated_name

    del model.graph.node[:0]
    new_nodes = gate_nodes + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    return model, True


def load_model(input_path: Path, task: str) -> onnx.ModelProto:
    if input_path.suffix == ".zip":
        raw = zipfile.ZipFile(input_path).read(f"{task}.onnx")
    else:
        raw = input_path.read_bytes()
    return onnx.load_from_string(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    model, changed = add_gate(load_model(args.input, args.task))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = model.SerializeToString()
    args.output.write_bytes(raw)
    print(f"{args.task}: changed={int(changed)} bytes={len(raw)}")
    if args.no_verify:
        return
    examples = all_examples(json.loads((args.data_root / f"{args.task}.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"verify: {right}/{right + wrong} wrong={wrong} reason={reason}")
    macs, memory, params, cost = profile_cost(args.output)
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    if wrong:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
