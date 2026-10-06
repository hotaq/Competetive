# ruff: noqa: E501
"""Read the V97 holdout once and evaluate the frozen gates in PREREGISTRATION.md.

Arms: v97 (fitted table), two (v1 table = V96 port), live (56016840 tapes), base.
G1-G4 come from holdout.json (tape opponents, Rust loop); G5 from holdout_v21.json (slow
harness, --no-sanitize). Unit = (opponent, seed) with seats averaged; seed-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.shop_router_v1 import load_routes, route_index_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOOT = 10_000
Unit = tuple[str, int]


def units(rows: list[dict[str, Any]], arm: str, key: str = "point") -> dict[Unit, float]:
    acc: dict[Unit, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            acc[(row["opponent"], row["seed"])].append(float(row[key]))
    return {k: st.mean(v) for k, v in acc.items()}


def diff_ci(a: dict[Unit, float], b: dict[Unit, float], rng: random.Random) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (opp, seed), value in a.items():
        if (opp, seed) in b:
            by_seed[seed].append(value - b[(opp, seed)])
    if not by_seed:
        return float("nan"), float("nan"), float("nan")
    seeds = sorted(by_seed)
    point = st.mean(v for vs in by_seed.values() for v in vs)
    draws = []
    for _ in range(BOOT):
        total = count = 0.0
        for seed in rng.choices(seeds, k=len(seeds)):
            total += sum(by_seed[seed])
            count += len(by_seed[seed])
        draws.append(total / count)
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def per_opponent(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        table[r["opponent"]][r["arm"]].append(r["point"])
    return {opp: {arm: st.mean(v) for arm, v in arms.items()} for opp, arms in table.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HERE / "holdout.json")
    parser.add_argument("--holdout-v21", type=Path, default=HERE / "holdout_v21.json")
    parser.add_argument("--routes", type=Path, default=ROOT / "artifacts/shop_router_v2/routes.json")
    parser.add_argument("--output", type=Path, default=HERE / "gates.json")
    args = parser.parse_args()
    rows = json.loads(args.holdout.read_text(encoding="utf-8"))
    rng = random.Random(20260905)
    arms = sorted({r["arm"] for r in rows})
    p = {arm: units(rows, arm) for arm in arms}
    m = {arm: units(rows, arm, "margin") for arm in arms}

    g1 = diff_ci(p["v97"], p["live"], rng)
    g2 = diff_ci(p["v97"], p["two"], rng)
    table = per_opponent(rows)
    g3 = {opp: {"v97": t["v97"], "live": t["live"], "two": t["two"],
                "pass": t["v97"] >= t["live"] - 0.02 and t["v97"] >= t["two"] - 0.02}
          for opp, t in sorted(table.items())}
    data = load_routes(args.routes)
    kw = dict(first_routes=data["first_routes"], second_routes=data["second_routes"],
              first_branch_step=data["first_branch_step"], second_branch_step=data["second_branch_step"])
    bad = 0
    checked = 0
    for r in rows:
        if r["arm"] != "v97":
            continue
        shops = [s for s in (r["shop1"], r["shop2"]) if s]
        checked += 1
        if r["route_final"] != route_index_for(718, shops, **kw) or r["route_at_second"] != r["route_final"]:
            bad += 1
    g5: dict[str, Any] = {"available": args.holdout_v21.exists()}
    if g5["available"]:
        v21 = json.loads(args.holdout_v21.read_text(encoding="utf-8"))
        t = per_opponent(v21)
        cell = next(iter(t.values()))
        g5.update({"n": len(v21), "v97": cell.get("v97"), "live": cell.get("live"), "two": cell.get("two"),
                   "pass": cell.get("v97", 0) >= cell.get("live", 1) - 0.02
                   and cell.get("v97", 0) >= cell.get("two", 1) - 0.02})
    else:
        g5["pass"] = False

    gates = {
        "G1": {"diff": g1[0], "ci": [g1[1], g1[2]], "pass": g1[0] >= 0.02 and g1[1] > 0},
        "G2": {"diff": g2[0], "ci": [g2[1], g2[2]], "pass": g2[1] > -0.01},
        "G3": {"per_opponent": g3, "pass": all(v["pass"] for v in g3.values())},
        "G4": {"checked": checked, "violations": bad, "pass": bad == 0},
        "G5": g5,
    }
    upload = all(gates[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5"))
    margins = {f"{a}_minus_{b}": diff_ci(m[a], m[b], rng)[0] for a, b in (("v97", "live"), ("v97", "two"), ("two", "live"))}
    summary = {
        "games": len(rows), "gates": gates, "upload_decision": upload,
        "arms": {arm: {"point": st.mean(r["point"] for r in rows if r["arm"] == arm),
                       "mean_margin": st.mean(r["margin"] for r in rows if r["arm"] == arm)} for arm in arms},
        "margin_diffs": margins, "two_minus_live": diff_ci(p["two"], p["live"], rng),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"games {len(rows)}   upload decision: {'BUILD' if upload else 'NO SUBMISSION'}")
    print(f"  G1 v97-live: {'PASS' if gates['G1']['pass'] else 'FAIL'} {g1[0]:+.4f} CI [{g1[1]:+.4f}, {g1[2]:+.4f}]")
    print(f"  G2 v97-two : {'PASS' if gates['G2']['pass'] else 'FAIL'} {g2[0]:+.4f} CI [{g2[1]:+.4f}, {g2[2]:+.4f}]")
    print(f"  G3 per-opp : {'PASS' if gates['G3']['pass'] else 'FAIL'}")
    print(f"  G4 routing : {'PASS' if gates['G4']['pass'] else 'FAIL'} checked {checked} violations {bad}")
    print(f"  G5 vs V21  : {'PASS' if g5['pass'] else 'FAIL/N-A'} {g5}")
    print(f"  two-live (this holdout): {summary['two_minus_live'][0]:+.4f}")
    print(f"\n{'arm':>5} {'point':>7} {'mean m':>9}")
    for arm in arms:
        print(f"{arm:>5} {summary['arms'][arm]['point']:>7.4f} {summary['arms'][arm]['mean_margin']:>+9,.0f}")
    print(f"\n{'opponent':<36}{'base':>8}{'live':>8}{'two':>8}{'v97':>8}{'v97-live':>10}{'v97-two':>9}{'G3':>5}")
    for opp, t in sorted(table.items()):
        line = f"{opp:<36}" + "".join(f"{t.get(a, float('nan')):>8.4f}" for a in ("base", "live", "two", "v97"))
        print(line + f"{t['v97'] - t['live']:>+10.4f}{t['v97'] - t['two']:>+9.4f}{'ok' if g3[opp]['pass'] else 'FAIL':>5}")


if __name__ == "__main__":
    main()
