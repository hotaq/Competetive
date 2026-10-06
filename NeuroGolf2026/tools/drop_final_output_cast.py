#!/usr/bin/env python3
"""Remove final Cast(... -> FLOAT) nodes that only feed graph output.

NeuroGolf scorer thresholds output values with (output > 0), so a bool/int
pre-cast tensor can be a valid graph output. This saves the final Cast and the
pre-cast intermediate memory when the pre-cast tensor is renamed to `output`.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import onnx
from onnx import helper


def value_type(model: onnx.ModelProto, name: str):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if vi.name == name and vi.type.HasField("tensor_type"):
            return vi.type
    return None


def drop_final_cast(raw: bytes) -> tuple[bytes, bool, str]:
    model = onnx.load_from_string(raw)
    if len(model.graph.output) != 1:
        return raw, False, "not_single_output"
    out_name = model.graph.output[0].name
    final_nodes = [node for node in model.graph.node if out_name in node.output]
    if len(final_nodes) != 1:
        return raw, False, "no_unique_final_producer"
    cast = final_nodes[0]
    if cast.op_type != "Cast" or len(cast.input) != 1 or len(cast.output) != 1:
        return raw, False, "final_not_cast"
    to_attrs = [attr for attr in cast.attribute if attr.name == "to"]
    if not to_attrs or to_attrs[0].i != onnx.TensorProto.FLOAT:
        return raw, False, "cast_not_to_float"
    src = cast.input[0]
    consumers = [node for node in model.graph.node for inp in node.input if inp == src]
    if consumers != [cast]:
        return raw, False, "precast_has_other_consumers"
    producers = [node for node in model.graph.node if src in node.output]
    if len(producers) != 1:
        return raw, False, "precast_no_unique_producer"

    src_type = value_type(model, src)
    if src_type is None:
        return raw, False, "missing_precast_type"

    producer = producers[0]
    for i, name in enumerate(producer.output):
        if name == src:
            producer.output[i] = out_name

    kept = [node for node in model.graph.node if node is not cast]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    # Remove stale value_info for the old pre-cast tensor name.
    kept_vi = [vi for vi in model.graph.value_info if vi.name != src]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)

    model.graph.output[0].type.CopyFrom(src_type)
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString(), True, "dropped"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-in", type=Path, required=True)
    ap.add_argument("--zip-out", type=Path, required=True)
    args = ap.parse_args()
    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    changed = []
    reasons = {}
    with zipfile.ZipFile(args.zip_in) as zin, zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name in sorted(zin.namelist()):
            raw = zin.read(name)
            if name.endswith(".onnx"):
                raw2, ok, reason = drop_final_cast(raw)
                reasons[reason] = reasons.get(reason, 0) + 1
                if ok:
                    changed.append(Path(name).stem)
                raw = raw2
            zout.writestr(name, raw)
    print("changed", len(changed), ",".join(changed))
    print("reasons", reasons)
    print("zip", args.zip_out)


if __name__ == "__main__":
    main()
