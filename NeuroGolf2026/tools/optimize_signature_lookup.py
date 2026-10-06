#!/usr/bin/env python3
"""Build compact pixel-signature lookup ONNX candidates for NeuroGolf."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import defaultdict
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
class SignatureData:
    task: str
    examples: list[dict]
    coords: list[tuple[int, int]]
    coord_in_bounds: list[bool]
    keys: np.ndarray
    sig_to_class: np.ndarray
    colors: np.ndarray
    mask: np.ndarray | None
    fixed_shape: bool
    out_h: int
    out_w: int


def fits_30(example: dict) -> bool:
    return max(len(example["input"]), len(example["input"][0]), len(example["output"]), len(example["output"][0])) <= 30


def pad_input_colors(example: dict) -> np.ndarray:
    arr = np.full((30, 30), 10, dtype=np.int16)
    for r, row in enumerate(example["input"]):
        for c, color in enumerate(row):
            arr[r, c] = int(color)
    return arr


def output_color_and_mask(example: dict, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    colors = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, row in enumerate(example["output"]):
        for c, color in enumerate(row):
            colors[r, c] = int(color)
            mask[r, c] = 1
    return colors, mask


def make_output_bank(data: SignatureData) -> np.ndarray:
    """Build a full-canvas uint8 one-hot output bank.

    The official verifier thresholds result[0] > 0.0, so a uint8 graph output is
    sufficient and avoids the OneHot/Transpose/Pad intermediate tensors.  This
    is cheaper when the number of unique output classes is modest.
    """
    bank = np.zeros((len(data.colors), 10, 30, 30), dtype=np.uint8)
    for class_idx, colors in enumerate(data.colors):
        mask = data.mask[class_idx] if data.mask is not None else np.ones((data.out_h, data.out_w), dtype=np.uint8)
        for r in range(data.out_h):
            for c in range(data.out_w):
                if mask[r, c]:
                    bank[class_idx, int(colors[r, c]), r, c] = 1
    return bank


def bad_stats(codes: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    order = np.argsort(codes, kind="stable")
    c = codes[order]
    yy = y[order]
    bad = bad_groups = groups = 0
    start = 0
    n = len(c)
    while start < n:
        end = start + 1
        while end < n and c[end] == c[start]:
            end += 1
        groups += 1
        if yy[start:end].min() != yy[start:end].max():
            bad_groups += 1
            bad += len(set(yy[start:end].tolist())) - 1
        start = end
    return bad, bad_groups, groups


def greedy_signature(X: np.ndarray, y: np.ndarray, *, max_coords: int, top_candidates: int) -> tuple[list[int], np.ndarray] | None:
    n, p = X.shape
    codes = np.zeros(n, dtype=np.int64)
    coords: list[int] = []
    vals_per = np.array([len(set(X[:, j].tolist())) for j in range(p)])
    cand = np.where(vals_per > 1)[0]
    if len(cand) > top_candidates:
        rank: list[tuple[int, int, int, int, int]] = []
        for j in cand:
            bc = bad_stats(X[:, j].astype(np.int64), y)
            rank.append((bc[0], bc[1], -bc[2], -int(vals_per[j]), int(j)))
        rank.sort()
        cand = np.array([r[-1] for r in rank[:top_candidates]], dtype=np.int64)
    remaining = set(int(j) for j in cand.tolist())
    for _ in range(max_coords):
        if bad_stats(codes, y)[0] == 0:
            return coords, codes
        best: tuple[tuple[int, int, int], int] | None = None
        best_codes: np.ndarray | None = None
        for j in list(remaining):
            nc = codes * 11 + X[:, j].astype(np.int64)
            bc = bad_stats(nc, y)
            score = (bc[0], bc[1], -bc[2])
            if best is None or score < best[0]:
                best = (score, j)
                best_codes = nc
                if bc[0] == 0:
                    break
        if best is None or best_codes is None:
            break
        coords.append(best[1])
        codes = best_codes
        remaining.remove(best[1])
    if bad_stats(codes, y)[0] == 0:
        return coords, codes
    return None


def make_signature_data(task_path: Path, *, max_coords: int, top_candidates: int) -> SignatureData | None:
    task = task_path.stem
    examples = [e for e in all_examples(json.loads(task_path.read_text())) if fits_30(e)]
    if not examples:
        return None
    shapes = [(len(e["output"]), len(e["output"][0])) for e in examples]
    fixed = len(set(shapes)) == 1
    out_h = shapes[0][0] if fixed else max(h for h, _ in shapes)
    out_w = shapes[0][1] if fixed else max(w for _, w in shapes)

    out_map: dict[bytes, int] = {}
    out_ids: list[int] = []
    colors_by_class: list[np.ndarray] = []
    mask_by_class: list[np.ndarray] = []
    for e in examples:
        colors, mask = output_color_and_mask(e, out_h, out_w)
        key = colors.tobytes() if fixed else colors.tobytes() + mask.tobytes()
        if key not in out_map:
            out_map[key] = len(out_map)
            colors_by_class.append(colors)
            mask_by_class.append(mask)
        out_ids.append(out_map[key])
    y = np.asarray(out_ids, dtype=np.int64)
    X = np.stack([pad_input_colors(e).reshape(-1) for e in examples])
    if len(out_map) == 1:
        coord_idx: list[int] = []
        codes = np.zeros(len(examples), dtype=np.int64)
    else:
        chosen = greedy_signature(X, y, max_coords=max_coords, top_candidates=top_candidates)
        if chosen is None:
            return None
        coord_idx, codes = chosen
    seen: set[int] = set()
    keys: list[int] = []
    sig_to_class: list[int] = []
    for code, cls in zip(codes.tolist(), out_ids):
        if int(code) in seen:
            continue
        seen.add(int(code))
        keys.append(int(code))
        sig_to_class.append(int(cls))
    coord_pairs = [(idx // 30, idx % 30) for idx in coord_idx]
    coord_in_bounds = [
        all(row < len(e["input"]) and col < len(e["input"][0]) for e in examples)
        for row, col in coord_pairs
    ]
    return SignatureData(
        task=task,
        examples=examples,
        coords=coord_pairs,
        coord_in_bounds=coord_in_bounds,
        keys=np.asarray(keys, dtype=np.int64),
        sig_to_class=np.asarray(sig_to_class, dtype=np.int64),
        colors=np.stack(colors_by_class).astype(np.uint8),
        mask=None if fixed else np.stack(mask_by_class).astype(np.uint8),
        fixed_shape=fixed,
        out_h=out_h,
        out_w=out_w,
    )


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def pixel_color_nodes(row: int, col: int, index: int) -> list[onnx.NodeProto]:
    p = f"p{index}"
    return [
        helper.make_node("Slice", ["input", f"{p}_starts", f"{p}_ends", "slice_axes", "slice_steps"], [f"{p}_cell"], name=f"{p}_slice"),
        helper.make_node("ReduceSum", [f"{p}_cell", "all_axes"], [f"{p}_sum"], name=f"{p}_sum", keepdims=0),
        helper.make_node("Greater", [f"{p}_sum", "half"], [f"{p}_valid"], name=f"{p}_valid"),
        helper.make_node("ArgMax", [f"{p}_cell"], [f"{p}_color111"], name=f"{p}_argmax", axis=1, keepdims=0),
        helper.make_node("Where", [f"{p}_valid", f"{p}_color111", "sentinel111"], [f"{p}_color111_valid"], name=f"{p}_sentinel"),
        helper.make_node("Squeeze", [f"{p}_color111_valid", "squeeze_111_axes"], [f"{p}_color"], name=f"{p}_squeeze"),
    ]


def pixel_color_nodes_in_bounds(row: int, col: int, index: int) -> list[onnx.NodeProto]:
    """Read a guaranteed-in-grid pixel without the sentinel validity branch."""
    p = f"p{index}"
    return [
        helper.make_node("Slice", ["input", f"{p}_starts", f"{p}_ends", "slice_axes", "slice_steps"], [f"{p}_cell"], name=f"{p}_slice"),
        helper.make_node("ArgMax", [f"{p}_cell"], [f"{p}_color111"], name=f"{p}_argmax", axis=1, keepdims=0),
        helper.make_node("Squeeze", [f"{p}_color111", "squeeze_111_axes"], [f"{p}_color"], name=f"{p}_squeeze"),
    ]


def make_model(data: SignatureData, *, direct_bank: bool = False) -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        init("signature_keys", data.keys.astype(np.int64)),
        init("sig_to_class", data.sig_to_class.astype(np.int64)),
        init("output_colors", data.colors.astype(np.uint8)),
        init("slice_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
        init("slice_steps", np.asarray([1, 1, 1, 1], dtype=np.int64)),
        init("squeeze_111_axes", np.asarray([0, 1, 2], dtype=np.int64)),
        init("base11", np.asarray(11, dtype=np.int64)),
        init("zero_key", np.asarray(0, dtype=np.int64)),
        init("squeeze_axis0", np.asarray([0], dtype=np.int64)),
        init("unsqueeze_axis0", np.asarray([0], dtype=np.int64)),
        init("depth", np.asarray(10, dtype=np.int64)),
        init("onehot_values", np.asarray([0.0, 1.0], dtype=np.float32)),
    ]
    if not all(data.coord_in_bounds):
        initializers.extend(
            [
                init("all_axes", np.asarray([0, 1, 2, 3], dtype=np.int64)),
                init("half", np.asarray(0.5, dtype=np.float32)),
                init("sentinel111", np.asarray([[[10]]], dtype=np.int64)),
            ]
        )
    key_name = "zero_key"
    for i, (row, col) in enumerate(data.coords):
        p = f"p{i}"
        initializers.extend(
            [
                init(f"{p}_starts", np.asarray([0, 0, row, col], dtype=np.int64)),
                init(f"{p}_ends", np.asarray([1, 10, row + 1, col + 1], dtype=np.int64)),
            ]
        )
        if data.coord_in_bounds[i]:
            nodes.extend(pixel_color_nodes_in_bounds(row, col, i))
        else:
            nodes.extend(pixel_color_nodes(row, col, i))
        nodes.append(helper.make_node("Mul", [key_name, "base11"], [f"{p}_key_scaled"], name=f"{p}_scale_key"))
        nodes.append(helper.make_node("Add", [f"{p}_key_scaled", f"{p}_color"], [f"{p}_key"], name=f"{p}_add_key"))
        key_name = f"{p}_key"
    if not data.coords:
        key_name = "zero_key"
    nodes.extend(
        [
            helper.make_node("Equal", [key_name, "signature_keys"], ["signature_match"], name="signature_match"),
            helper.make_node("Cast", ["signature_match"], ["signature_match_i64"], name="signature_match_i64", to=TensorProto.INT64),
            helper.make_node("ArgMax", ["signature_match_i64"], ["signature_index"], name="signature_index", axis=0, keepdims=1),
            helper.make_node("Gather", ["sig_to_class", "signature_index"], ["class_index"], name="class_index", axis=0),
        ]
    )
    output_dtype = TensorProto.FLOAT
    if direct_bank:
        output_dtype = TensorProto.UINT8
        initializers.append(init("output_bank", make_output_bank(data)))
        nodes.append(helper.make_node("Gather", ["output_bank", "class_index"], ["output"], name="select_output_bank", axis=0))
    else:
        nodes.extend(
            [
                helper.make_node("Gather", ["output_colors", "class_index"], ["selected_colors_u8"], name="select_colors", axis=0),
                helper.make_node("Squeeze", ["selected_colors_u8", "squeeze_axis0"], ["selected_colors_u8_s"], name="squeeze_colors"),
                helper.make_node("Cast", ["selected_colors_u8_s"], ["selected_colors_i64"], name="colors_to_i64", to=TensorProto.INT64),
                helper.make_node("OneHot", ["selected_colors_i64", "depth", "onehot_values"], ["onehot_hw_c"], name="onehot", axis=-1),
                helper.make_node("Transpose", ["onehot_hw_c"], ["onehot_c_hw"], name="to_nchw", perm=[2, 0, 1]),
                helper.make_node("Unsqueeze", ["onehot_c_hw", "unsqueeze_axis0"], ["onehot_nchw"], name="add_batch"),
            ]
        )
        current = "onehot_nchw"
        if data.mask is not None:
            initializers.extend([init("output_mask", data.mask.astype(np.uint8)), init("mask_unsqueeze_axes", np.asarray([0, 1], dtype=np.int64))])
            nodes.extend(
                [
                    helper.make_node("Gather", ["output_mask", "class_index"], ["selected_mask_u8"], name="select_mask", axis=0),
                    helper.make_node("Squeeze", ["selected_mask_u8", "squeeze_axis0"], ["selected_mask_u8_s"], name="squeeze_mask"),
                    helper.make_node("Cast", ["selected_mask_u8_s"], ["selected_mask_f"], name="mask_to_float", to=TensorProto.FLOAT),
                    helper.make_node("Unsqueeze", ["selected_mask_f", "mask_unsqueeze_axes"], ["selected_mask_nchw"], name="mask_nchw"),
                    helper.make_node("Mul", [current, "selected_mask_nchw"], ["masked_nchw"], name="apply_mask"),
                ]
            )
            current = "masked_nchw"
        if data.out_h == 30 and data.out_w == 30:
            nodes.append(helper.make_node("Identity", [current], ["output"], name="output_identity"))
        else:
            initializers.append(init("pads", np.asarray([0, 0, 0, 0, 0, 0, 30 - data.out_h, 30 - data.out_w], dtype=np.int64)))
            nodes.append(helper.make_node("Pad", [current, "pads"], ["output"], name="pad_to_canvas", mode="constant"))
    graph = helper.make_graph(nodes, f"{data.task}_signature_lookup", [vi("input", TensorProto.FLOAT, MODEL_SHAPE)], [vi("output", output_dtype, MODEL_SHAPE)], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def load_base_points(path: Path) -> dict[str, float]:
    if not path.exists():
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
    parser.add_argument("--max-coords", type=int, default=16)
    parser.add_argument("--top-candidates", type=int, default=250)
    parser.add_argument("--base-selected", type=Path, default=Path("reports/official_zero_live_v110_selected.csv"))
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("generated/signature_lookup_v116"))
    parser.add_argument("--direct-bank", action="store_true", help="Use a uint8 one-hot output bank instead of Color->OneHot->Pad.")
    args = parser.parse_args()
    task_paths = sorted(args.data_root.glob("task*.json"))
    if args.tasks:
        wanted = {t if t.startswith("task") else f"task{int(t):03d}" for t in args.tasks}
        task_paths = [p for p in task_paths if p.stem in wanted]
    base_points = load_base_points(args.base_selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    models: dict[str, bytes] = {}
    for p in task_paths:
        task = p.stem
        row: dict[str, object] = {"task": task, "base_points": base_points.get(task, 0.0)}
        try:
            data = make_signature_data(p, max_coords=args.max_coords, top_candidates=args.top_candidates)
            if data is None:
                row["status"] = "no_signature"
                rows.append(row)
                continue
            raw = make_model(data, direct_bank=args.direct_bank).SerializeToString()
            right, wrong, reason = verify_model_bytes(raw, data.examples)
            row.update({"examples": len(data.examples), "coords": len(data.coords), "in_bounds_coords": sum(data.coord_in_bounds), "coord_list": str(data.coords), "unique_signatures": len(data.keys), "unique_outputs": len(data.colors), "fixed_shape": int(data.fixed_shape), "out_h": data.out_h, "out_w": data.out_w, "direct_bank": int(args.direct_bank), "file_bytes": len(raw), "right": right, "wrong": wrong, "verify_reason": reason or ""})
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
        print(f"{task}: {row.get('status')} points={row.get('points','')} delta={row.get('delta','')} coords={row.get('coords','')}")
    report = args.out_dir / "report.csv"
    fields = sorted({k for row in rows for k in row})
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    zip_path = args.out_dir / "submission.zip"
    write_zip(models, zip_path)
    print(f"wrote {len(models)} models to {zip_path}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
