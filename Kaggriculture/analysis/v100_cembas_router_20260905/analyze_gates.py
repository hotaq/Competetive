# ruff: noqa: E501
"""Read the V100 holdout once and evaluate the frozen gates (arm 'two' is v97's table, i.e. the live agent)."""

from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from kaggriculture_rl.policy.shop_router_v1 import load_routes, route_index_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOOT = 10_000


def units(rows, arm, key="point"):
    acc = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            acc[(r["opponent"], r["seed"])].append(float(r[key]))
    return {k: st.mean(v) for k, v in acc.items()}


def diff_ci(a, b, rng):
    by_seed = defaultdict(list)
    for k, v in a.items():
        if k in b:
            by_seed[k[1]].append(v - b[k])
    seeds = sorted(by_seed)
    point = st.mean(x for xs in by_seed.values() for x in xs)
    draws = []
    for _ in range(BOOT):
        tot = cnt = 0.0
        for s in rng.choices(seeds, k=len(seeds)):
            tot += sum(by_seed[s]); cnt += len(by_seed[s])
        draws.append(tot / cnt)
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def main():
    rows = json.loads((HERE / "holdout.json").read_text(encoding="utf-8"))
    rows = [dict(r, arm="v97" if r["arm"] == "two" else r["arm"]) for r in rows]
    rng = random.Random(20260905)
    arms = sorted({r["arm"] for r in rows}); opps = sorted({r["opponent"] for r in rows})
    p = {a: units(rows, a) for a in arms}
    per = {a: {o: st.mean(r["point"] for r in rows if r["arm"] == a and r["opponent"] == o) for o in opps} for a in arms}
    g1 = diff_ci(p["v100"], p["base"], rng)
    g2 = {o: per["v100"][o] - per["base"][o] for o in opps}
    data = load_routes(ROOT / "artifacts/shop_router_v3/routes.json")
    kw = dict(first_routes=data["first_routes"], second_routes=data["second_routes"], first_branch_step=data["first_branch_step"], second_branch_step=data["second_branch_step"])
    bad = sum(1 for r in rows if r["arm"] == "v100" and (r["route_final"] != route_index_for(718, [s for s in (r["shop1"], r["shop2"]) if s], **kw) or r["route_at_second"] != r["route_final"]))
    g4_gap = diff_ci(p["base"], p["v97"], rng)
    yuk_name = next(o for o in opps if o.startswith("yuk__")); lag_name = next(o for o in opps if o.startswith("lagrangian"))
    gates = {
        "G1": {"diff": g1[0], "ci": [g1[1], g1[2]], "pass": g1[0] >= 0.02 and g1[1] > 0},
        "G2": {"per_opponent": g2, "pass": all(v >= -0.02 for v in g2.values())},
        "G3": {"violations": bad, "pass": bad == 0},
        "G4": {"base_minus_v97": g4_gap[0], "ci": [g4_gap[1], g4_gap[2]], "base_vs_yuk": per["base"][yuk_name], "base_vs_lagrangian": per["base"][lag_name],
               "pass": g4_gap[0] >= 0.20 and per["base"][yuk_name] >= 0.5 and per["base"][lag_name] >= 0.5},
    }
    decision = "BUILD v100" if all(gates[g]["pass"] for g in gates) else ("BUILD base tape" if gates["G4"]["pass"] else "NO BUILD")
    summary = {"games": len(rows), "gates": gates, "decision": decision,
               "arms": {a: {"point": st.mean(r["point"] for r in rows if r["arm"] == a), "mean_margin": st.mean(r["margin"] for r in rows if r["arm"] == a), "money": st.mean(r["money"] for r in rows if r["arm"] == a)} for a in arms},
               "per_opponent": per}
    (HERE / "gates.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"games {len(rows)}   decision: {decision}")
    print(f"  G1 v100-base: {'PASS' if gates['G1']['pass'] else 'FAIL'} {g1[0]:+.4f} CI [{g1[1]:+.4f}, {g1[2]:+.4f}]")
    print(f"  G2 per-opp  : {'PASS' if gates['G2']['pass'] else 'FAIL'} worst {min(g2.values()):+.4f}")
    print(f"  G3 routing  : {'PASS' if gates['G3']['pass'] else 'FAIL'} violations {bad}")
    print(f"  G4 lineage  : {'PASS' if gates['G4']['pass'] else 'FAIL'} base-v97 {g4_gap[0]:+.4f} CI [{g4_gap[1]:+.4f}, {g4_gap[2]:+.4f}], base vs yuk {per['base'][yuk_name]:.3f}, vs lagrangian {per['base'][lag_name]:.3f}")
    print(f"\n{'arm':>10} {'point':>7} {'money':>9} {'margin':>9}")
    for a in arms:
        s = summary["arms"][a]; print(f"{a:>10} {s['point']:>7.4f} {s['money']:>9,.0f} {s['mean_margin']:>+9,.0f}")
    print(f"\n{'opponent':<42}" + "".join(f"{a:>11}" for a in arms))
    for o in opps:
        print(f"{o:<42}" + "".join(f"{per[a][o]:>11.3f}" for a in arms))


if __name__ == "__main__":
    main()
