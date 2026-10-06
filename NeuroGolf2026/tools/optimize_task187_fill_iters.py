#!/usr/bin/env python3
"""Shorten task187's outside flood-fill loop while preserving exact outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


def shorten_iterations(model: onnx.ModelProto, keep_iteration: int) -> onnx.ModelProto:
    if not -1 <= keep_iteration <= 29:
        raise ValueError("keep_iteration must be in [-1, 29]")
    optimized = copy.deepcopy(model)
    replacement = "seed" if keep_iteration < 0 else f"cur_{keep_iteration}"

    for node in optimized.graph.node:
        for i, name in enumerate(node.input):
            if name == "cur_29":
                node.input[i] = replacement

    kept_nodes = []
    for node in optimized.graph.node:
        outputs = set(node.output)
        remove = False
        for i in range(keep_iteration + 1, 30):
            if outputs & {f"mp_{i}", f"cur_{i}"}:
                remove = True
                break
        if not remove:
            kept_nodes.append(node)

    del optimized.graph.node[:]
    optimized.graph.node.extend(kept_nodes)
    optimized.producer_name = f"neurogolf_task187_fill_{keep_iteration}"
    onnx.checker.check_model(optimized)
    return optimized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v97_plus_task077_v98/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task187_fill12_v99"))
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()

    with zipfile.ZipFile(args.base_zip) as zf:
        raw_base = zf.read("task187.onnx")
    model = shorten_iterations(onnx.load_from_string(raw_base), args.iterations)
    raw = model.SerializeToString()

    examples = all_examples(json.loads((args.data_root / "task187.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task187 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "task187.onnx"
    onnx_path.write_bytes(raw)
    with zipfile.ZipFile(args.out_dir / "single_task_submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task187.onnx", raw)

    macs, memory, params, cost = score_network(onnx_path)
    print(f"task187 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"wrote {onnx_path}")


if __name__ == "__main__":
    main()
