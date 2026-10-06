#!/usr/bin/env python3
"""Build a clean static block-lattice line interpolation model for task314."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_a_exact_submission import all_examples, verify_model_bytes  # noqa: E402
from tools.metric_v2_select import score_network, task_points  # noqa: E402


EXPECTED_SHAPE = [1, 10, 30, 30]
H = 8
W = 8
BLOCKS = [0, 3, 6]


def tensor_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, EXPECTED_SHAPE)


def arr(name: str, value: np.ndarray | list[int] | float | int, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def const_i64(name: str, value: list[int]) -> onnx.TensorProto:
    return arr(name, value, np.int64)


def const_f32(name: str, value: np.ndarray | float) -> onnx.TensorProto:
    return arr(name, value, np.float32)


def slice_4d(name: str, starts: list[int], ends: list[int]) -> list[onnx.TensorProto]:
    return [const_i64(f"{name}_starts", starts), const_i64(f"{name}_ends", ends)]


def make_slice(src: str, dst: str, prefix: str) -> onnx.NodeProto:
    return helper.make_node("Slice", [src, f"{prefix}_starts", f"{prefix}_ends", "axes4", "steps4"], [dst], name=f"slice_{dst}")


def max_chain(inputs: list[str], output: str, nodes: list[onnx.NodeProto]) -> None:
    cur = inputs[0]
    for i, name in enumerate(inputs[1:], start=1):
        dst = output if i == len(inputs) - 1 else f"{output}_{i}"
        nodes.append(helper.make_node("Max", [cur, name], [dst], name=f"max_{dst}"))
        cur = dst


def scalar_slice(
    color: int,
    row: int,
    col: int,
    nodes: list[onnx.NodeProto],
    inits: list[onnx.TensorProto],
    cache: dict[tuple[int, int, int], str],
) -> str:
    key = (color, row, col)
    if key in cache:
        return cache[key]
    name = f"c{color}_r{row}_c{col}"
    inits.extend(slice_4d(name, [0, color, row, col], [1, color + 1, row + 1, col + 1]))
    nodes.append(make_slice("x8", name, name))
    cache[key] = name
    return name


def one_pixel_mask(name: str, row: int, col: int) -> onnx.TensorProto:
    mask = np.zeros((1, 1, H, W), dtype=np.float32)
    mask[:, :, row, col] = 1.0
    return const_f32(name, mask)


def make_model() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        const_i64("axes4", [0, 1, 2, 3]),
        const_i64("steps4", [1, 1, 1, 1]),
        const_i64("axis_channel", [1]),
        const_i64("pad_to_30", [0, 0, 0, 0, 0, 0, 30 - H, 30 - W]),
        const_f32("zero_scalar", 0.0),
        const_f32("half", 0.5),
        const_f32("one_scalar", 1.0),
        *slice_4d("crop8", [0, 0, 0, 0], [1, 10, H, W]),
    ]
    for color in range(10):
        inits.extend(slice_4d(f"ch{color}", [0, color, 0, 0], [1, color + 1, H, W]))

    nodes: list[onnx.NodeProto] = [make_slice("input", "x8", "crop8")]
    scalar_cache: dict[tuple[int, int, int], str] = {}
    for color in range(10):
        nodes.append(make_slice("x8", f"ch{color}", f"ch{color}"))

    channel_outs: list[str] = []
    filled_terms: list[str] = []
    for color in range(2, 10):
        color_terms = [f"ch{color}"]
        for lr in range(2):
            for lc in range(2):
                for br, r0 in enumerate(BLOCKS):
                    left = scalar_slice(color, r0 + lr, BLOCKS[0] + lc, nodes, inits, scalar_cache)
                    right = scalar_slice(color, r0 + lr, BLOCKS[2] + lc, nodes, inits, scalar_cache)
                    gate = f"c{color}_lr{lr}_lc{lc}_row{br}_gate"
                    mask = f"c{color}_lr{lr}_lc{lc}_row{br}_mask"
                    fill = f"c{color}_lr{lr}_lc{lc}_row{br}_fill"
                    inits.append(one_pixel_mask(mask, r0 + lr, BLOCKS[1] + lc))
                    nodes.extend(
                        [
                            helper.make_node("Mul", [left, right], [gate], name=gate),
                            helper.make_node("Mul", [gate, mask], [fill], name=fill),
                        ]
                    )
                    color_terms.append(fill)
                    filled_terms.append(fill)
                for bc, c0 in enumerate(BLOCKS):
                    top = scalar_slice(color, BLOCKS[0] + lr, c0 + lc, nodes, inits, scalar_cache)
                    bot = scalar_slice(color, BLOCKS[2] + lr, c0 + lc, nodes, inits, scalar_cache)
                    gate = f"c{color}_lr{lr}_lc{lc}_col{bc}_gate"
                    mask = f"c{color}_lr{lr}_lc{lc}_col{bc}_mask"
                    fill = f"c{color}_lr{lr}_lc{lc}_col{bc}_fill"
                    inits.append(one_pixel_mask(mask, BLOCKS[1] + lr, c0 + lc))
                    nodes.extend(
                        [
                            helper.make_node("Mul", [top, bot], [gate], name=gate),
                            helper.make_node("Mul", [gate, mask], [fill], name=fill),
                        ]
                    )
                    color_terms.append(fill)
                    filled_terms.append(fill)
        out_name = f"out{color}"
        max_chain(color_terms, out_name, nodes)
        channel_outs.append(out_name)

    max_chain(filled_terms, "filled_raw", nodes)
    nodes.extend(
        [
            helper.make_node("Greater", ["filled_raw", "half"], ["filled_b"], name="filled_b"),
            helper.make_node("Cast", ["filled_b"], ["filled"], to=TensorProto.FLOAT, name="filled"),
            helper.make_node("Sub", ["one_scalar", "filled"], ["not_filled"], name="not_filled"),
            helper.make_node("Mul", ["ch1", "not_filled"], ["out1"], name="out1"),
            helper.make_node("Concat", ["ch0", "out1", *channel_outs], ["out10"], axis=1, name="assemble8"),
            helper.make_node("Pad", ["out10", "pad_to_30", "zero_scalar"], ["output"], name="pad_output30"),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task314_block_lattice_lines",
        [tensor_info("input")],
        [tensor_info("output")],
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf_task314_block_lattice_lines",
        ir_version=10,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


def write_zip_with_task(base_zip: Path, out_zip: Path, task_name: str, raw: bytes) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for name in sorted(names):
            zout.writestr(name, raw if name == task_name else zin.read(name))
        if task_name not in names:
            zout.writestr(task_name, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize task314 with clean block-lattice line interpolation.")
    parser.add_argument("--base-zip", type=Path, default=Path("generated/metric_v2_v91_plus_boolreshape_150155_017_002_v92/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("data/latest/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("generated/task314_block_lattice_lines_v93"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_model()
    raw = model.SerializeToString()
    examples = all_examples(json.loads((args.data_root / "task314.json").read_text()))
    right, wrong, reason = verify_model_bytes(raw, examples)
    print(f"task314 verify: right={right} wrong={wrong} examples={len(examples)} reason={reason}")
    if wrong:
        raise SystemExit(1)

    onnx_path = args.out_dir / "task314.onnx"
    onnx_path.write_bytes(raw)
    macs, memory, params, cost = score_network(onnx_path)
    points = task_points(cost)
    print(f"task314 metric-v2: macs={macs} memory={memory} params={params} cost={cost} points={points:.6f}")

    one_task_zip = args.out_dir / "single_task_submission.zip"
    with zipfile.ZipFile(one_task_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("task314.onnx", raw)
    full_zip = args.out_dir / "submission.zip"
    write_zip_with_task(args.base_zip, full_zip, "task314.onnx", raw)

    report_path = args.out_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "right", "wrong", "macs", "memory", "params", "cost", "points", "onnx_bytes"])
        writer.writeheader()
        writer.writerow(
            {
                "task": "task314",
                "right": right,
                "wrong": wrong,
                "macs": macs,
                "memory": memory,
                "params": params,
                "cost": cost,
                "points": f"{points:.6f}",
                "onnx_bytes": len(raw),
            }
        )
    print(f"wrote {onnx_path}")
    print(f"one-task zip: {one_task_zip}")
    print(f"full zip: {full_zip}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
