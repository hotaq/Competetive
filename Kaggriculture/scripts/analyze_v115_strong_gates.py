# ruff: noqa: E501
"""V115 gates analysis: pooled points per arm + paired money diff vs a reference arm.

Seed-cluster bootstrap (resample seeds, keep both seats + all opponents per seed),
2000 replicates, 95% percentile CI. Reads per-game JSONL rows, writes gates JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pooled_points(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        pts = sum(r["point"] for r in sub)
        out[arm] = {"point": pts / len(sub), "games": len(sub),
                    "mean_margin": float(np.mean([r["margin"] for r in sub])),
                    "mean_money": float(np.mean([r["money"] for r in sub]))}
    return out


def paired_diff_ci(rows: list[dict], arm: str, ref: str, n_boot: int = 2000,
                   seed: int = 7) -> dict:
    a = {(r["opponent"], r["seed"], r["seat"]): r["money"]
         for r in rows if r["arm"] == arm}
    b = {(r["opponent"], r["seed"], r["seat"]): r["money"]
         for r in rows if r["arm"] == ref}
    keys = sorted(set(a) & set(b))
    diffs = np.array([a[k] - b[k] for k in keys], dtype=float)
    by_seed: dict[int, list[float]] = {}
    for k, d in zip(keys, diffs, strict=True):
        by_seed.setdefault(k[1], []).append(d)
    seed_ids = sorted(by_seed)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(seed_ids), len(seed_ids))
        vals = [d for s in pick for d in by_seed[seed_ids[s]]]
        boot[i] = float(np.mean(vals))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"pairs": len(keys), "mean": float(np.mean(diffs)),
            "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "beats_ref": bool(float(np.mean(diffs)) > 0 and lo > 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--ref", default="YANNIK")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    points = pooled_points(rows)
    cands = sorted([a for a in points if a != args.ref and a != "OMAR"],
                   key=lambda a: -points[a]["point"])
    result = {"arms": points, "ranked_candidates": cands,
              "vs_ref": {a: paired_diff_ci(rows, a, args.ref, args.boot) for a in cands}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for a in cands:
        p = points[a]["point"]
        v = result["vs_ref"][a]
        print(f"{a}: point={p:.4f} n={points[a]['games']} "
              f"diff_vs_{args.ref}={v['mean']:.0f} CI=[{v['ci_lo']:.0f},{v['ci_hi']:.0f}] "
              f"beats={v['beats_ref']}", flush=True)
    for r in (args.ref, "OMAR"):
        if r in points:
            print(f"{r}: point={points[r]['point']:.4f} n={points[r]['games']}", flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
