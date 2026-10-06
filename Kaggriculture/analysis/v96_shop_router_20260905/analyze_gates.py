"""Read V96 games once and evaluate the frozen gates in PREREGISTRATION.md.

Unit of analysis is (opponent, seed) with the two seats averaged; CIs are percentile
bootstraps over seeds (cluster resampling), 10,000 draws, fixed RNG seed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.shop_router_v1 import DEFAULT_ROUTES_PATH, load_routes, route_index_for

ROOT = Path(__file__).resolve().parents[2]
BOOT = 10_000
Unit = tuple[str, int]


def units(rows: list[dict[str, Any]], arm: str, key: str) -> dict[Unit, float]:
    acc: dict[Unit, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            acc[(row["opponent"], row["seed"])].append(float(row[key]))
    return {k: st.mean(v) for k, v in acc.items()}


def paired_diff(a: dict[Unit, float], b: dict[Unit, float]) -> dict[Unit, float]:
    return {k: a[k] - b[k] for k in a if k in b}


def bootstrap_ci(diff: dict[Unit, float], rng: random.Random) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (_, seed), value in diff.items():
        by_seed[seed].append(value)
    if not by_seed:
        return float("nan"), float("nan"), float("nan")
    seeds = sorted(by_seed)
    point = st.mean(v for vs in by_seed.values() for v in vs)
    draws = []
    for _ in range(BOOT):
        sample = rng.choices(seeds, k=len(seeds))
        total = 0.0
        count = 0
        for seed in sample:
            total += sum(by_seed[seed])
            count += len(by_seed[seed])
        draws.append(total / count)
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]


def check_routing(rows: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    kw = dict(
        first_routes=data["first_routes"], second_routes=data["second_routes"],
        first_branch_step=data["first_branch_step"],
        second_branch_step=data["second_branch_step"],
    )
    bad: list[dict[str, Any]] = []
    checked = 0
    for row in rows:
        arm = row["arm"]
        if arm not in ("base", "one", "two"):
            continue
        shops = [s for s in (row["shop1"], row["shop2"]) if s]
        depth = {"base": 0, "one": 1, "two": 2}[arm]
        expected = route_index_for(718, shops, max_shops=depth, **kw)
        ok = row["route_final"] == expected and row["route_at_second"] == row["route_final"]
        checked += 1
        if not ok:
            keys = ("arm", "opponent", "seed", "seat", "shop1", "shop2",
                    "route_at_second", "route_final")
            bad.append({k: row[k] for k in keys} | {"expected": expected})
    return {"checked": checked, "violations": len(bad), "examples": bad[:10]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = ROOT / "analysis/v96_shop_router_20260905"
    parser.add_argument("--games", type=Path, default=here / "games.json")
    parser.add_argument("--output", type=Path, default=here / "gates.json")
    args = parser.parse_args()
    rows = json.loads(args.games.read_text(encoding="utf-8"))
    data = load_routes(DEFAULT_ROUTES_PATH)
    rng = random.Random(20260905)
    arms = sorted({r["arm"] for r in rows})
    opponents = sorted({r["opponent"] for r in rows})

    # headline per arm / per (arm, opponent)
    table: dict[str, Any] = {}
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        table[arm] = {
            "n": len(group), "point": st.mean(r["point"] for r in group),
            "mean_margin": st.mean(r["margin"] for r in group),
            "median_margin": st.median(r["margin"] for r in group),
            "by_opponent": {
                opp: {
                    "n": len(g), "point": st.mean(r["point"] for r in g),
                    "mean_margin": st.mean(r["margin"] for r in g),
                    "seat0_point": st.mean(r["point"] for r in g if r["seat"] == 0),
                    "seat1_point": st.mean(r["point"] for r in g if r["seat"] == 1),
                }
                for opp in opponents
                for g in [[r for r in group if r["opponent"] == opp]]
            },
        }

    p_two, p_live, p_base, p_one = (units(rows, a, "point") for a in ("two", "live", "base", "one"))
    g1_point, g1_lo, g1_hi = bootstrap_ci(paired_diff(p_two, p_live), rng)
    g2_point, g2_lo, g2_hi = bootstrap_ci(paired_diff(p_two, p_base), rng)
    one_point, one_lo, one_hi = bootstrap_ci(paired_diff(p_one, p_base), rng)
    m_two, m_live, m_base = (units(rows, a, "margin") for a in ("two", "live", "base"))
    mg_live = bootstrap_ci(paired_diff(m_two, m_live), rng)
    mg_base = bootstrap_ci(paired_diff(m_two, m_base), rng)

    g3 = {}
    for opp in opponents:
        two = table["two"]["by_opponent"][opp]["point"]
        live = table["live"]["by_opponent"][opp]["point"]
        g3[opp] = {"two": two, "live": live, "diff": two - live, "pass": two >= live - 0.02}
    g4 = check_routing(rows, data)

    gates = {
        "G1": {"diff": g1_point, "ci": [g1_lo, g1_hi], "pass": g1_point >= 0.02 and g1_lo > 0},
        "G2": {"diff": g2_point, "ci": [g2_lo, g2_hi], "pass": g2_point > 0 and g2_lo > 0},
        "G3": {"per_opponent": g3, "pass": all(v["pass"] for v in g3.values())},
        "G4": {**g4, "pass": g4["violations"] == 0},
    }
    upload = gates["G1"]["pass"] and gates["G3"]["pass"] and gates["G4"]["pass"]

    # shop-pair cells and specialist coverage (from the `two` arm; pairs are identical across arms
    # only if the prefix is shared, so record per arm)
    cells: dict[str, Counter] = {}
    for arm in arms:
        cells[arm] = Counter(f"{r['shop1']}__{r['shop2']}" for r in rows if r["arm"] == arm)
    specialist = {pair for pair, idx in data["second_routes"].items()
                  if idx != data["first_routes"][pair.split("__", 1)[0]]}
    in_spec = {(r["opponent"], r["seed"]) for r in rows if r["arm"] == "two"
               and f"{r['shop1']}__{r['shop2']}" in specialist}
    spec_rows = {}
    for arm in ("two", "base", "one", "live"):
        pts = units(rows, arm, "point")
        sub = [pts[k] for k in pts if k in in_spec]
        spec_rows[arm] = {"units": len(sub), "point": st.mean(sub) if sub else None}
    spec_pairs = {k: v for k, v in paired_diff(p_two, p_base).items() if k in in_spec}
    spec_diff = bootstrap_ci(spec_pairs, rng) if spec_pairs else None

    tape_pairs = defaultdict(dict)
    for r in rows:
        if not r["opponent"].startswith("agent:"):
            tape_pairs[(r["arm"], r["opponent"], r["seed"])][r["seat"]] = r["margin"]
    symmetric = sum(1 for v in tape_pairs.values() if v.get(0) == v.get(1))
    seconds = {}
    for kind in ("agent", "tape"):
        is_agent = kind == "agent"
        picked = [r["seconds"] for r in rows
                  if r["opponent"].startswith("agent:") == is_agent and r.get("seconds") is not None]
        seconds[kind] = st.mean(picked) if picked else None

    out = {
        "games": len(rows), "arms": table, "gates": gates, "upload_decision": upload,
        "one_minus_base": {"diff": one_point, "ci": [one_lo, one_hi]},
        "margin_two_minus_live": {"diff": mg_live[0], "ci": mg_live[1:]},
        "margin_two_minus_base": {"diff": mg_base[0], "ci": mg_base[1:]},
        "cells_two": dict(cells["two"].most_common()),
        "cells_identical_across_router_arms": cells["two"] == cells["base"] == cells["one"],
        "specialist_cells_hit": sorted(c for c in cells["two"] if c in specialist),
        "specialist_units": spec_rows,
        "specialist_two_minus_base": (
            None if spec_diff is None else {"diff": spec_diff[0], "ci": spec_diff[1:]}
        ),
        "tape_seat_symmetric": [symmetric, len(tape_pairs)],
        "seconds_per_game": seconds,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"games {len(rows)}   upload decision: {'BUILD' if upload else 'NO SUBMISSION'}")
    for g, v in gates.items():
        extra = ""
        if g in ("G1", "G2"):
            extra = f" diff {v['diff']:+.4f} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"
        if g == "G4":
            extra = f" checked {v['checked']} violations {v['violations']}"
        print(f"  {g}: {'PASS' if v['pass'] else 'FAIL'}{extra}")
    print(f"  one-base diff {one_point:+.4f} CI [{one_lo:+.4f}, {one_hi:+.4f}]")
    print(f"  margin two-live {mg_live[0]:+,.0f} CI [{mg_live[1]:+,.0f}, {mg_live[2]:+,.0f}]"
          f"   two-base {mg_base[0]:+,.0f} CI [{mg_base[1]:+,.0f}, {mg_base[2]:+,.0f}]")
    print(f"\n{'arm':>5} {'point':>7} {'mean m':>9} {'median':>9}")
    for arm in arms:
        t = table[arm]
        print(f"{arm:>5} {t['point']:>7.4f} {t['mean_margin']:>+9,.0f}"
              f" {t['median_margin']:>+9,.0f}")
    header = f"\n{'opponent':<36}" + "".join(f"{a:>8}" for a in arms)
    print(header + f"{'two-live':>10}{'G3':>5}")
    for opp in opponents:
        cells_ = "".join(f"{table[a]['by_opponent'][opp]['point']:>8.4f}" for a in arms)
        verdict = "ok" if g3[opp]["pass"] else "FAIL"
        print(f"{opp:<36}{cells_}{g3[opp]['diff']:>+10.4f}{verdict:>5}")
    hit = len(out["specialist_cells_hit"])
    print(f"\nspecialist cells hit: {hit}/23, units {spec_rows['two']['units']}")
    for arm, v in spec_rows.items():
        print(f"  {arm:>5} point in specialist cells {v['point']}")
    if spec_diff:
        print(f"  two-base in specialist cells {spec_diff[0]:+.4f}"
              f" CI [{spec_diff[1]:+.4f}, {spec_diff[2]:+.4f}]")
    print(f"cells identical across router arms: {out['cells_identical_across_router_arms']}")
    print(f"tape seat-symmetric {symmetric}/{len(tape_pairs)}   seconds/game {seconds}")
    agent_name = "agent:recovered_boatlee_v21_source"
    per_seat = {}
    for arm in arms:
        cell = table[arm]["by_opponent"].get(agent_name)
        if cell:
            per_seat[arm] = (round(cell["seat0_point"], 4), round(cell["seat1_point"], 4))
    print("V21 per seat (seat0, seat1):", per_seat)


if __name__ == "__main__":
    main()
