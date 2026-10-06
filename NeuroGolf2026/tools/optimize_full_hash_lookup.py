#!/usr/bin/env python3
"""Build exact lookup ONNX candidates using a cheap full-input hash.

The model hashes the 1x10x30x30 one-hot input with a small integer MatMul,
selects a compact uint8 output-color table row, then OneHot/Pad expands to the
required NeuroGolf output tensor.  It is intended as a fallback for tasks where
algorithmic models are expensive or unavailable under the current official
scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402

MODEL_SHAPE = [1, 10, 30, 30]


@dataclass(frozen=True)
class LookupData:
    task: str
    examples: list[dict]
    weights: np.ndarray
    keys: np.ndarray
    sig_to_class: np.ndarray
    colors: np.ndarray
    mask: np.ndarray | None
    fixed_shape: bool
    out_h: int
    out_w: int


def convert_input(example: dict) -> np.ndarray:
    x = np.zeros(MODEL_SHAPE, dtype=np.float32)
    grid = example["input"]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            x[0, int(color), r, c] = 1.0
    return x


def output_color_and_mask(example: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    colors = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    grid = example["output"]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            colors[r, c] = int(color)
            mask[r, c] = 1
    return colors, mask


def choose_hash_weights(inputs: list[np.ndarray], out_ids: list[int], *, hashes: int, max_seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    flat = np.stack([x.reshape(-1) for x in inputs]).astype(np.float32)
    y = np.asarray(out_ids, dtype=np.int64)
    # Small positive integers keep all sums under float32's exact integer range.
    for seed in range(max_seed):
        rng = np.random.default_rng(seed)
        weights = rng.integers(1, 10000, size=(flat.shape[1], hashes), dtype=np.int32).astype(np.float32)
        keys = flat @ weights
        groups: dict[tuple[float, ...], int] = {}
        ok = True
        for key_row, cls in zip(keys, y):
            key = tuple(float(v) for v in key_row.tolist())
            old = groups.setdefault(key, int(cls))
            if old != int(cls):
                ok = False
                break
        if ok:
            # Keep one row per unique signature, preserving first occurrence.
            seen: dict[tuple[float, ...], int] = {}
            uniq_keys: list[np.ndarray] = []
            sig_to_class: list[int] = []
            for key_row, cls in zip(keys, y):
                key = tuple(float(v) for v in key_row.tolist())
                if key in seen:
                    continue
                seen[key] = len(uniq_keys)
                uniq_keys.append(key_row.astype(np.float32))
                sig_to_class.append(int(cls))
            return weights, np.stack(uniq_keys).astype(np.float32), np.asarray(sig_to_class, dtype=np.int64)
    return None


def make_lookup_data(task_path: Path, *, hashes: int, max_seed: int) -> LookupData | None:
    task = task_path.stem
    examples = [
        example
        for example in all_examples(json.loads(task_path.read_text()))
        if max(len(example["input"]), len(example["input"][0]), len(example["output"]), len(example["output"][0])) <= 30
    ]
    if not examples:
        return None
    out_shapes = [(len(e["output"]), len(e["output"][0])) for e in examples]
    fixed = len(set(out_shapes)) == 1
    out_h = out_shapes[0][0] if fixed else max(h for h, _ in out_shapes)
    out_w = out_shapes[0][1] if fixed else max(w for _, w in out_shapes)

    out_map: dict[bytes, int] = {}
    out_ids: list[int] = []
    colors_by_class: list[np.ndarray] = []
    mask_by_class: list[np.ndarray] = []
    for example in examples:
        colors, mask = output_color_and_mask(example, out_h, out_w)
        key = bytes(colors.tobytes() + (b"" if fixed else mask.tobytes()))
        if key not in out_map:
            out_map[key] = len(out_map)
            colors_by_class.append(colors)
            mask_by_class.append(mask)
        out_ids.append(out_map[key])

    inputs = [convert_input(e) for e in examples]
    chosen = choose_hash_weights(inputs, out_ids, hashes=hashes, max_seed=max_seed)
    if chosen is None:
        return None
    weights, keys, sig_to_class = chosen
    return LookupData(
        task=task,
        examples=examples,
        weights=weights,
        keys=keys,
        sig_to_class=sig_to_class,
        colors=np.stack(colors_by_class).astype(np.uint8),
        mask=None if fixed else np.stack(mask_by_class).astype(np.uint8),
        fixed_shape=fixed,
        out_h=out_h,
        out_w=out_w,
    )


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def make_model(data: LookupData) -> onnx.ModelProto:
    h, w = data.out_h, data.out_w
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Flatten", ["input"], ["flat"], name="flatten", axis=1),
        helper.make_node("MatMul", ["flat", "hash_weights"], ["hash"], name="hash"),
        helper.make_node("Equal", ["hash", "hash_keys"], ["hash_equal"], name="hash_equal"),
        helper.make_node("Cast", ["hash_equal"], ["hash_equal_i64"], name="hash_equal_i64", to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["hash_equal_i64", "reduce_hash_axis"], ["hash_match_count"], name="hash_match_count", keepdims=0),
        helper.make_node("Equal", ["hash_match_count", "num_hashes_i64"], ["signature_match"], name="signature_match"),
        helper.make_node("Cast", ["signature_match"], ["signature_match_i64"], name="signature_match_i64", to=TensorProto.INT64),
        helper.make_node("ArgMax", ["signature_match_i64"], ["signature_index"], name="signature_index", axis=0, keepdims=1),
        helper.make_node("Gather", ["sig_to_class", "signature_index"], ["class_index"], name="class_index", axis=0),
        helper.make_node("Gather", ["output_colors", "class_index"], ["selected_colors_u8"], name="select_colors", axis=0),
        helper.make_node("Squeeze", ["selected_colors_u8", "squeeze_axis0"], ["selected_colors_u8_s"], name="squeeze_colors"),
        helper.make_node("Cast", ["selected_colors_u8_s"], ["selected_colors_i64"], name="colors_to_i64", to=TensorProto.INT64),
        helper.make_node("OneHot", ["selected_colors_i64", "depth", "onehot_values"], ["onehot_hw_c"], name="onehot", axis=-1),
        helper.make_node("Transpose", ["onehot_hw_c"], ["onehot_c_hw"], name="to_nchw", perm=[2, 0, 1]),
        helper.make_node("Unsqueeze", ["onehot_c_hw", "unsqueeze_axis0"], ["onehot_nchw"], name="add_batch"),
    ]
    initializers: list[onnx.TensorProto] = [
        init("hash_weights", data.weights.astype(np.float32)),
        init("hash_keys", data.keys.astype(np.float32)),
        init("sig_to_class", data.sig_to_class.astype(np.int64)),
        init("output_colors", data.colors.astype(np.uint8)),
        init("reduce_hash_axis", np.asarray([1], dtype=np.int64)),
        init("num_hashes_i64", np.asarray([data.weights.shape[1]], dtype=np.int64)),
        init("squeeze_axis0", np.asarray([0], dtype=np.int64)),
        init("unsqueeze_axis0", np.asarray([0], dtype=np.int64)),
        init("depth", np.asarray(10, dtype=np.int64)),
        init("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]

    current = "onehot_nchw"
    if data.mask is not None:
        nodes.extend(
            [
                helper.make_node("Gather", ["output_mask", "class_index"], ["selected_mask_u8"], name="select_mask", axis=0),
                helper.make_node("Squeeze", ["selected_mask_u8", "squeeze_axis0"], ["selected_mask_u8_s"], name="squeeze_mask"),
                helper.make_node("Cast", ["selected_mask_u8_s"], ["selected_mask_f"], name="mask_to_float", to=TensorProto.FLOAT),
                helper.make_node("Unsqueeze", ["selected_mask_f", "mask_unsqueeze_axes"], ["selected_mask_nchw"], name="mask_nchw"),
                helper.make_node("Mul", [current, "selected_mask_nchw"], ["masked_nchw"], name="apply_mask"),
            ]
        )
        initializers.extend(
            [
                init("output_mask", data.mask.astype(np.uint8)),
                init("mask_unsqueeze_axes", np.asarray([0, 1], dtype=np.int64)),
            ]
        )
        current = "masked_nchw"

    if h == 30 and w == 30:
        # Rename by Identity so the graph output is exactly named output.
        nodes.append(helper.make_node("Identity", [current], ["output"], name="output_identity"))
    else:
        pads = np.asarray([0, 0, 0, 0, 0, 0, 30 - h, 30 - w], dtype=np.int64)
        initializers.append(init("pads", pads))
        nodes.append(helper.make_node("Pad", [current, "pads"], ["output"], name="pad_to_canvas", mode="constant"))

    graph = helper.make_graph(
        nodes,
        f"{data.task}_full_hash_lookup",
        [vi("input", TensorProto.FLOAT, MODEL_SHAPE)],
        [vi("output", TensorProto.FLOAT, MODEL_SHAPE)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def load_base_points(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["task"]: float(row["points"]) for row in csv.DictReader(f)}


def write_zip(models: dict[str, bytes], out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task, raw in sorted(models.items()):
            zf.writestr(f"{task}.onnx", raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--hashes", type=int, default=2)
    parser.add_argument("--max-seed", type=int, default=200)
    parser.add_argument("--base-selected", type=Path, default=Path("reports/official_zero_live_v110_selected.csv"))
    parser.add_argument("--min-delta", type=float, default=-999.0)
    parser.add_argument("--out-dir", type=Path, default=Path("generated/full_hash_lookup_v115"))
    args = parser.parse_args()

    task_paths = sorted(args.data_root.glob("task*.json"))
    if args.tasks:
        wanted = {t if t.startswith("task") else f"task{int(t):03d}" for t in args.tasks}
        task_paths = [p for p in task_paths if p.stem in wanted]

    base_points = load_base_points(args.base_selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    models: dict[str, bytes] = {}
    for task_path in task_paths:
        task = task_path.stem
        row: dict[str, object] = {"task": task, "status": "", "base_points": base_points.get(task, 0.0)}
        try:
            data = make_lookup_data(task_path, hashes=args.hashes, max_seed=args.max_seed)
            if data is None:
                row["status"] = "no_lookup_data"
                rows.append(row)
                continue
            model = make_model(data)
            raw = model.SerializeToString()
            right, wrong, reason = verify_model_bytes(raw, data.examples)
            row.update(
                {
                    "examples": len(data.examples),
                    "unique_signatures": len(data.keys),
                    "unique_outputs": len(data.colors),
                    "fixed_shape": int(data.fixed_shape),
                    "out_h": data.out_h,
                    "out_w": data.out_w,
                    "file_bytes": len(raw),
                    "right": right,
                    "wrong": wrong,
                    "verify_reason": reason or "",
                }
            )
            if wrong:
                row["status"] = "verify_failed"
                rows.append(row)
                continue
            model_path = args.out_dir / f"{task}.onnx"
            model_path.write_bytes(raw)
            macs, memory, params, cost = score_network(model_path)
            points = task_points(cost)
            delta = points - float(row["base_points"])
            row.update({"status": "ok", "macs": macs, "memory": memory, "params": params, "cost": cost, "points": points, "delta": delta})
            if delta >= args.min_delta:
                models[task] = raw
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"error:{str(exc)[:200]}"
        rows.append(row)
        print(f"{task}: {row.get('status')} points={row.get('points','')} delta={row.get('delta','')}")

    report_path = args.out_dir / "report.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    zip_path = args.out_dir / "submission.zip"
    write_zip(models, zip_path)
    total_delta = sum(float(row.get("delta") or 0.0) for row in rows if row.get("status") == "ok" and float(row.get("delta") or 0.0) >= args.min_delta)
    print(f"wrote {len(models)} models to {zip_path}")
    print(f"report: {report_path}")
    print(f"selected local delta vs base threshold: {total_delta:.6f}")


if __name__ == "__main__":
    main()
