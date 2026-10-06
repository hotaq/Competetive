#!/usr/bin/env python3
"""Build a cheaper clean task143 ONNX model.

Task143 always uses a 10x10 grid.  The public baseline scans the full padded
30x30 canvas to find a shape matching the top-left 3x3 reference.  This model
crops to 10x10 first, runs the same shape-match rule, then pads the result back
to the static 1x10x30x30 NeuroGolf tensor shape.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.blend_public_pack import profile_cost, task_points  # noqa: E402
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]


def const_i64(name: str, value: np.ndarray | list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def const_f32(name: str, value: np.ndarray | list[float] | float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def build_model() -> onnx.ModelProto:
    ch5 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    ch5[:, 5, :, :] = 1.0

    initializers = [
        const_i64("content_starts", [0, 0, 0, 0]),
        const_i64("content_ends", [1, 10, 10, 10]),
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("cand_idx", [1, 2, 3, 4, 6, 7, 8, 9]),
        const_i64("starts_00", [0, 0]),
        const_i64("ends_33", [3, 3]),
        const_i64("ends_11", [1, 1]),
        const_i64("axes_hw", [2, 3]),
        const_i64("axis_c", [1]),
        const_i64("reps_8", [8, 1, 1, 1]),
        const_i64("pad_10_to_30", [0, 0, 0, 0, 0, 0, 20, 20]),
        const_f32("zero", [0.0]),
        const_f32("one", [1.0]),
        const_f32("ch5", ch5),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "content_starts", "content_ends", "axes4", "steps4"], ["input10"], name="crop_input10"),
        helper.make_node("Gather", ["input10", "cand_idx"], ["cand"], axis=1, name="candidate_channels"),
        helper.make_node("Slice", ["cand", "starts_00", "ends_33", "axes_hw"], ["ref_part"], name="reference_part"),
        helper.make_node("ReduceSum", ["ref_part", "axis_c"], ["ref"], keepdims=1, name="reference_mask"),
        helper.make_node("ReduceSum", ["ref", "axes_hw"], ["ref_count"], keepdims=1, name="reference_count"),
        helper.make_node("Tile", ["ref", "reps_8"], ["ref8"], name="depthwise_kernel"),
        helper.make_node("Conv", ["cand", "ref8"], ["corr"], group=8, kernel_shape=[3, 3], pads=[0, 0, 2, 2], name="find_matching_shape"),
        helper.make_node("Equal", ["corr", "ref_count"], ["match_b"], name="match_bool"),
        helper.make_node("Cast", ["match_b"], ["match"], to=TensorProto.FLOAT, name="match_float"),
        helper.make_node("ReduceSum", ["match", "axes_hw"], ["match_sum"], keepdims=1, name="match_sum"),
        helper.make_node("Slice", ["match", "starts_00", "ends_11", "axes_hw"], ["match_00"], name="exclude_reference_match"),
        helper.make_node("Sub", ["match_sum", "match_00"], ["match_else"], name="other_matches"),
        helper.make_node("Greater", ["match_else", "zero"], ["has_match_b"], name="has_other_match_bool"),
        helper.make_node("Cast", ["has_match_b"], ["has_match"], to=TensorProto.FLOAT, name="has_other_match"),
        helper.make_node("ReduceSum", ["cand", "axes_hw"], ["ch_count"], keepdims=1, name="channel_counts"),
        helper.make_node("Equal", ["ch_count", "ref_count"], ["count_ok_b"], name="same_count_bool"),
        helper.make_node("Cast", ["count_ok_b"], ["count_ok"], to=TensorProto.FLOAT, name="same_count"),
        helper.make_node("Mul", ["has_match", "count_ok"], ["target_ch"], name="target_channel"),
        helper.make_node("Mul", ["cand", "target_ch"], ["target8"], name="target8"),
        helper.make_node("ReduceSum", ["target8", "axis_c"], ["target_sp"], keepdims=1, name="target_spatial"),
        helper.make_node("Greater", ["target_sp", "zero"], ["target_b"], name="target_bool"),
        helper.make_node("Cast", ["target_b"], ["target"], to=TensorProto.FLOAT, name="target_float"),
        helper.make_node("Sub", ["one", "target"], ["keep"], name="keep_mask"),
        helper.make_node("Mul", ["input10", "keep"], ["kept"], name="kept_input10"),
        helper.make_node("Mul", ["target", "ch5"], ["paint"], name="paint_five"),
        helper.make_node("Add", ["kept", "paint"], ["out10"], name="out10"),
        helper.make_node("Pad", ["out10", "pad_10_to_30", "zero"], ["output"], name="pad_to_30"),
    ]

    graph = helper.make_graph(
        nodes,
        "task143_crop10_shape_match",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, EXPECTED_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task143_crop10",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/custom_task143_crop10"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "task143.onnx"
    zip_path = args.out_dir / "submission.zip"

    model = build_model()
    raw = model.SerializeToString()
    model_path.write_bytes(raw)

    examples = all_examples(json.loads((args.data_root / "task143.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    macs, memory, params, cost = profile_cost(model_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, "task143.onnx")

    print(f"wrote: {model_path} ({len(raw)} bytes)")
    print(f"verify: right={right} wrong={wrong} reason={reason!r}")
    print(f"profile: macs={macs} memory={memory} params={params} cost={cost} points={task_points(cost):.6f}")
    print(f"zip: {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
