"""V104 gate computation from screen.json (paired bootstrap, seed-clustered)."""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    base = {(r["opponent"], r["seed"], r["seat"]): r
            for r in rows if r["arm"] == "base"}
    cand = [r for r in rows if r["arm"] == "cand"]
    pairs = [(r, base[(r["opponent"], r["seed"], r["seat"])]) for r in cand]
    diffs = [c["margin"] - b["margin"] for c, b in pairs]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (c, _), d in zip(pairs, diffs, strict=True):
        by_seed[c["seed"]].append(d)
    seeds = sorted(by_seed)
    rng = random.Random(args.seed)
    means = []
    for _ in range(args.resamples):
        sample = [d for s in (rng.choice(seeds) for _ in seeds) for d in by_seed[s]]
        means.append(st.fmean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]

    def point(arm: str) -> float:
        group = [r["point"] for r in rows if r["arm"] == arm]
        return st.fmean(group)

    per_opp = {}
    for opp in sorted({r["opponent"] for r in cand}):
        group = [c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
                 for c in cand if c["opponent"] == opp]
        per_opp[opp] = {"n": len(group), "mean_diff": st.fmean(group)}

    seat_diff = {}
    for seat in (0, 1):
        group = [c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
                 for c in cand if c["seat"] == seat]
        seat_diff[str(seat)] = st.fmean(group)

    mean_diff = st.fmean(diffs)
    gates = {
        "n_pairs": len(pairs),
        "mean_paired_diff": mean_diff,
        "ci95": [lo, hi],
        "point_base": point("base"),
        "point_cand": point("cand"),
        "point_diff": point("cand") - point("base"),
        "per_opponent": per_opp,
        "seat_mean_diff": seat_diff,
        "G1": bool(mean_diff > 0 and lo > 0),
        "G2": bool(point("cand") - point("base") >= -0.01),
        "G3": bool(all(v["mean_diff"] >= -200 for v in per_opp.values())),
    }
    gates["OVERALL"] = bool(gates["G1"] and gates["G2"] and gates["G3"])
    args.output.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
