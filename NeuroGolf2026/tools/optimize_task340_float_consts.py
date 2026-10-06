#!/usr/bin/env python3
"""Remove task340's constant uint8-to-float casts."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


CAST_OUTPUTS = {
    "one_u8": "one",
    "ch_nz_mask_u8": "ch_nz_mask",
    "oh0p_u8": "oh0p",
}


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    for node in model.graph.node:
        if node.op_type != "Constant" or node.output[0] not in CAST_OUTPUTS:
            continue
        new_output = CAST_OUTPUTS[node.output[0]]
        value_attr = next(attr for attr in node.attribute if attr.name == "value")
        arr = numpy_helper.to_array(value_attr.t).astype(np.float32)
        value_attr.t.CopyFrom(numpy_helper.from_array(arr, name=f"{new_output}__constant_value"))
        node.output[0] = new_output

    remove_outputs = set(CAST_OUTPUTS.values())
    kept_nodes = [
        node
        for node in model.graph.node
        if not (node.op_type == "Cast" and set(node.output) & remove_outputs)
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.producer_name = "neurogolf_task340_float_consts"
    onnx.checker.check_model(model)
    return model


def write_zip_with_replacement(base_zip: Path, output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as src, zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            member = Path(name).name
            dst.writestr(member, raw if member == "task340.onnx" else src.read(name))


def write_single_probe(output_zip: Path, raw: bytes) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task340.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-zip", type=Path, default=Path("generated/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task340_float_consts"))
    args = parser.parse_args()

    examples = all_examples(json.loads((args.data_root / "task340.json").read_text()))
    with zipfile.ZipFile(args.base_zip) as zf:
        base_raw = zf.read("task340.onnx")

    model = optimize_model(onnx.load_from_string(base_raw))
    raw = model.SerializeToString()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task340.onnx"
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
