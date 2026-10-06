"""Gates analysis for analysis/refresh_20260923 (stdlib only, run as file).

Reads holdout.json rows, computes pooled points, paired-money seed-cluster
bootstrap CIs for W-live and W2-live, per-opponent tables. Writes gates.json.
Usage: uv run python analysis/refresh_20260923/analyze_gates.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

OUT = Path("analysis/refresh_20260923")
N_BOOT = 10000


def load_rows():
    with open(OUT / "holdout.json") as f:
        return json.load(f)


def pooled(rows, arm):
    sel = [r for r in rows if r["arm"] == arm]
    g = len(sel)
    p = sum(r["point"] for r in sel)
    return p / g, g


def pair_key(r):
    return (r["opponent"], r["seed"], r["seat"])


def paired_money_ci(rows, arm_a, arm_b):
    # pair on (opponent, seed, seat); cluster bootstrap over seeds
    ma = {pair_key(r): r["money"] for r in rows if r["arm"] == arm_a}
    mb = {pair_key(r): r["money"] for r in rows if r["arm"] == arm_b}
    common = sorted(set(ma) & set(mb))
    assert common, "no paired games"
    by_seed = defaultdict(list)
    for k in common:
        by_seed[k[1]].append(ma[k] - mb[k])
    seeds = sorted(by_seed)
    obs = sum(d for k in common for d in [ma[k] - mb[k]]) / len(common)
    rng = random.Random(20260923)
    boot = []
    for _ in range(N_BOOT):
        s = 0.0
        n = 0
        for _ in range(len(seeds)):
            diffs = by_seed[rng.choice(seeds)]
            s += sum(diffs)
            n += len(diffs)
        boot.append(s / n)
    boot.sort()
    lo = boot[int(0.025 * N_BOOT)]
    hi = boot[int(0.975 * N_BOOT) - 1]
    return {"mean": obs, "ci95": [lo, hi], "n_paired": len(common), "n_seeds": len(seeds)}


def per_opponent(rows, arm):
    out = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        o = out.setdefault(r["opponent"], [0, 0.0, 0.0])
        o[0] += 1
        o[1] += r["point"]
        o[2] += r["margin"]
    return {o: {"games": v[0], "point": v[1] / v[0], "mean_margin": v[2] / v[0]}
            for o, v in out.items()}


def main():
    rows = load_rows()
    arms = sorted(set(r["arm"] for r in rows))
    print("arms:", arms, "rows:", len(rows))
    pw, nw = pooled(rows, "W")
    pl, nl = pooled(rows, "live")
    p2, n2 = pooled(rows, "W2")
    g1 = paired_money_ci(rows, "W", "live")
    g1w2 = paired_money_ci(rows, "W2", "live")
    gates = {
        "W": {
            "point": pw,
            "games": nw,
            "minus_live_point": pw - pl,
            "paired_money": g1,
            "G1": g1["ci95"][0] > 0,
            "G2": (pw - pl) >= -0.01,
        },
        "W2": {
            "point": p2,
            "games": n2,
            "minus_live_point": p2 - pl,
            "paired_money": g1w2,
            "G1": g1w2["ci95"][0] > 0,
            "G2": (p2 - pl) >= -0.01,
        },
        "live": {"point": pl, "games": nl},
        "per_opponent": {a: per_opponent(rows, a) for a in ("W", "W2", "live")},
    }
    with open(OUT / "gates.json", "w") as f:
        json.dump(gates, f, indent=2, sort_keys=True)
    for a in ("W", "W2"):
        g = gates[a]
        print(f"{a}: point={g['point']:.4f} dPoint={g['minus_live_point']:+.4f} "
              f"dMoney={g['paired_money']['mean']:+.1f} "
              f"CI=[{g['paired_money']['ci95'][0]:+.1f},{g['paired_money']['ci95'][1]:+.1f}] "
              f"G1={'PASS' if g['G1'] else 'FAIL'} G2={'PASS' if g['G2'] else 'FAIL'}")


if __name__ == "__main__":
    main()
