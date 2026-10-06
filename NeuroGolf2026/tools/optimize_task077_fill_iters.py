#!/usr/bin/env python3
"""Shorten task077's flood-fill loop while preserving exact outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def shorten_iterations(model: onnx.ModelProto, keep_iterations: int) -> onnx.ModelProto:
    """Use yellow_{keep_iterations} as the fixed point and delete later nodes."""

    if not 0 <= keep_iterations <= 30:
        raise ValueError("keep_iterations must be between 0 and 30")
    optimized = copy.deepcopy(model)
    replacement = "init_yellow" if keep_iterations == 0 else f"yellow_{keep_iterations}"

    for node in optimized.graph.node:
        for i, name in enumerate(node.input):
            if name == "yellow_30":
                node.input[i] = replacement

    kept_nodes = []
    for node in optimized.graph.node:
        outputs = set(node.output)
        remove = False
        for i in range(keep_iterations, 30):
            if outputs & {
                f"combined_{i}",
                f"counts_{i}",
                f"ge_2_{i}",
                f"not_red_{i}",
                f"to_fill_{i}",
                f"yellow_{i + 1}",
            }:
                remove = True
                break
        if not remove:
            kept_nodes.append(node)

    del optimized.graph.node[:]
    optimized.graph.node.extend(kept_nodes)
    optimized.producer_name = f"neurogolf_task077_fill_{keep_iterations}"
    onnx.checker.check_model(optimized)
    return optimized


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task077.onnx" else src.read(name))


def write_single_probe(output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task077.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task077_fill6"))
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task077.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task077.onnx")

    model = shorten_iterations(onnx.load_from_string(base_raw), args.iterations)
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task077.onnx"
    full_zip = args.out_dir / "submission.zip"
    probe_zip = args.out_dir / "probe" / "submission.zip"
    model_path.write_bytes(raw)
    write_zip_with_replacement(args.base_zip, full_zip, raw)
    write_single_probe(probe_zip, raw)

    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)
    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"full zip: {full_zip} ({full_zip.stat().st_size} bytes)")
    print(f"probe zip: {probe_zip} ({probe_zip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
