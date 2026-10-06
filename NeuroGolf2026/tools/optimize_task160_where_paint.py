#!/usr/bin/env python3
"""Replace task160's final one-hot paint arithmetic with Where selects."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Use boolean paint masks directly instead of channel-vector multiplications."""

    remove_outputs = {
        "erase",
        "plus",
        "inside",
        "keep_mask",
        "kept",
        "notplus",
        "erase_only",
        "ch0_mask",
        "ch0_paint",
        "plus_in",
        "plus_paint",
        "output",
    }
    kept_nodes = [node for node in model.graph.node if not (set(node.output) & remove_outputs)]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.node.append(helper.make_node("Where", ["gt_erase", "CH0", "input"], ["bg_painted"], name="where_bg"))
    model.graph.node.append(helper.make_node("Where", ["gt_plus", "CH2", "bg_painted"], ["output"], name="where_plus"))
    model.producer_name = "neurogolf_task160_where_paint"
    onnx.checker.check_model(model)
    return model


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task160.onnx" else src.read(name))


def write_single_probe(output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task160.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task160_where_paint"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task160.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task160.onnx")

    model = optimize_model(onnx.load_from_string(base_raw))
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task160.onnx"
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
