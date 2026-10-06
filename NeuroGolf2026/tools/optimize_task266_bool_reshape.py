#!/usr/bin/env python3
"""Make task266's scalar feature reshapes operate on bool tensors."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


FEATURE_RE = re.compile(r"f(\d{3})_")


def build_optimized_model(base_raw: bytes) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load_from_string(base_raw))
    reshape_by_feature: dict[str, onnx.NodeProto] = {}
    skip_names: set[str] = set()

    for node in model.graph.node:
        if node.op_type != "Reshape" or not node.name.endswith("_flat"):
            continue
        match = FEATURE_RE.match(node.name)
        if not match:
            continue
        feature = f"f{match.group(1)}"
        reshape_by_feature[feature] = node
        skip_names.add(node.name)

    new_nodes: list[onnx.NodeProto] = []
    rewired = 0
    for node in model.graph.node:
        if node.name in skip_names:
            continue
        if node.op_type == "Cast" and node.name.endswith("_bool"):
            match = FEATURE_RE.match(node.name)
            if match:
                feature = f"f{match.group(1)}"
                reshape = reshape_by_feature.get(feature)
                if reshape is not None:
                    bool_4d = f"{feature}_bool_4d"
                    cast_node = helper.make_node(
                        "Cast",
                        [f"{feature}_cell"],
                        [bool_4d],
                        name=f"{feature}_bool_cast_before_reshape",
                        to=onnx.TensorProto.BOOL,
                    )
                    reshape_node = helper.make_node(
                        "Reshape",
                        [bool_4d, f"{feature}_shape"],
                        [f"{feature}_bool"],
                        name=f"{feature}_bool_flatten",
                    )
                    new_nodes.extend([cast_node, reshape_node])
                    rewired += 1
                    continue
        new_nodes.append(copy.deepcopy(node))

    if rewired != 15:
        raise ValueError(f"expected to rewire 15 feature paths, rewired {rewired}")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    model.producer_name = "neurogolf_task266_bool_reshape"
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submit_compact_canvas_batch6/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task266_bool_reshape"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task266.onnx"
    zip_path = args.out_dir / "submission.zip"

    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task266.onnx")

    model = build_optimized_model(base_raw)
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task266.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task266.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
