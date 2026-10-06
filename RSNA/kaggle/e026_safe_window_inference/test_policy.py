#!/usr/bin/env python3
"""Property check that E026 center selection matches the E025 winning policy."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SLOTS = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
]
DEPTH = sum(slot[2] for slot in SLOTS)
K = 42


def load_generated_policy():
    source = (HERE / "e026-single-coatnet-safe-windows.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_eval_centers"
    )
    namespace = {"np": np, "SLOTS": SLOTS}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "policy", "exec"), namespace)
    return namespace["_eval_centers"]


def e025_reference(mask: np.ndarray, series: np.ndarray) -> list[int]:
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    legacy_candidates = list(range(lo + 1, hi)) or [
        max(1, min((lo + hi) // 2, len(mask) - 2))
    ]
    safe = [
        center
        for center in range(1, len(mask) - 1)
        if bool(mask[center - 1:center + 2].all())
        and len(set(series[center - 1:center + 2])) == 1
        and series[center] is not None
    ]
    candidates = safe or legacy_candidates
    indexes = np.linspace(0, len(candidates) - 1, K).round().astype(int)
    return [int(candidates[index]) for index in indexes]


def main() -> None:
    policy = load_generated_policy()
    rng = np.random.default_rng(26)
    for _ in range(1_000):
        mask = np.zeros(DEPTH, dtype=np.uint8)
        series = np.full(DEPTH, None, dtype=object)
        cursor = 0
        for slot_index, (_, _, depth) in enumerate(SLOTS):
            if rng.random() > 0.22:
                mask[cursor:cursor + depth] = 1
                series[cursor:cursor + depth] = f"series-{slot_index}"
                failures = rng.random(depth) < 0.04
                mask[cursor:cursor + depth][failures] = 0
            cursor += depth
        actual = policy(mask, DEPTH, K)
        expected = e025_reference(mask, series)
        assert actual == expected, (mask, actual, expected)
    print("E026 matches E025 safe_resample on 1,000 randomized studies")


if __name__ == "__main__":
    main()
