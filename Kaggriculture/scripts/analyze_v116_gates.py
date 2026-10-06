"""V116 gate analysis: pooled points, seed-cluster bootstrap CIs, G4 audit, E0 rerun.

Reads screen_rows.json (evaluate_shop_router_fast.py row schema), applies the
frozen FIT/GATE shop-draw split ((i1*8+i2)%2==0 -> FIT), and writes gates.json
with the five frozen gates plus an overall verdict. E0 replays the new arm on
the first 10 gate seeds x all opponents x both seats through play_batch and
requires 0 money mismatches vs the screen rows.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rs import ShopRouter, Tape, play_batch

from kaggriculture_rl.policy.shop_router_v1 import route_index_for

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
SHOP_INDEX = {shop: i for i, shop in enumerate(SHOPS)}
N_BOOT = 2000


def is_fit_pair(shop1: str, shop2: str) -> bool:
    return (SHOP_INDEX[shop1] * 8 + SHOP_INDEX[shop2]) % 2 == 0


def pooled(rows: list[dict[str, Any]]) -> float:
    return sum(row["point"] for row in rows) / len(rows)


def boot_point_diff(rows: list[dict[str, Any]], arm_a: str, arm_b: str,
                    seed: int = 116) -> tuple[float, float, float]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    seeds = sorted(by_seed)
    rng = random.Random(seed)
    diffs = []
    for _ in range(N_BOOT):
        sample = [by_seed[rng.choice(seeds)] for _ in seeds]
        flat = [row for group in sample for row in group]
        pa = pooled([row for row in flat if row["arm"] == arm_a])
        pb = pooled([row for row in flat if row["arm"] == arm_b])
        diffs.append(pa - pb)
    diffs.sort()
    obs = pooled([row for row in rows if row["arm"] == arm_a]) - pooled(
        [row for row in rows if row["arm"] == arm_b])
    return obs, diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = json.loads(args.rows.read_text(encoding="utf-8"))
    data = json.loads(args.routes.read_text(encoding="utf-8"))
    first_routes = {str(k): int(v) for k, v in data["first_routes"].items()}
    second_routes = {str(k): int(v) for k, v in data["second_routes"].items()}
    first_step = int(data["first_branch_step"])
    second_step = int(data["second_branch_step"])

    opponents = sorted({row["opponent"] for row in rows})
    new_rows = [row for row in rows if row["arm"] == "new"]

    # G4: every new-arm game played the table lookup route.
    audit_bad = 0
    for row in new_rows:
        shops = [s for s in (row["shop1"], row["shop2"]) if s]
        expect_second = route_index_for(second_step, shops[:1], first_routes=first_routes,
                                        second_routes=second_routes,
                                        first_branch_step=first_step,
                                        second_branch_step=second_step, max_shops=2)
        expect_final = route_index_for(719, shops, first_routes=first_routes,
                                       second_routes=second_routes,
                                       first_branch_step=first_step,
                                       second_branch_step=second_step, max_shops=2)
        if row["route_at_second"] != expect_second or row["route_final"] != expect_final:
            audit_bad += 1
    g4 = {"pass": audit_bad == 0, "mismatches": audit_bad, "games": len(new_rows)}

    obs_y, lo_y, hi_y = boot_point_diff(rows, "new", "yannik")
    obs_o, lo_o, hi_o = boot_point_diff(rows, "new", "omar")
    g1 = {"pass": bool(obs_y >= 0.02 and lo_y > 0),
          "point_diff": obs_y, "ci": [lo_y, hi_y]}
    g2 = {"pass": bool(lo_o > -0.01), "point_diff": obs_o, "ci": [lo_o, hi_o]}

    per_opp = {}
    g3_pass = True
    for opp in opponents:
        group = [row for row in rows if row["opponent"] == opp]
        pn = pooled([row for row in group if row["arm"] == "new"])
        py = pooled([row for row in group if row["arm"] == "yannik"])
        po = pooled([row for row in group if row["arm"] == "omar"])
        ok = pn >= py - 0.02 and pn >= po - 0.02
        g3_pass = g3_pass and ok
        per_opp[opp] = {"new": pn, "yannik": py, "omar": po, "pass": ok}
    g3 = {"pass": g3_pass, "per_opponent": per_opp}

    gate_rows = [row for row in rows
                 if not is_fit_pair(row["shop1"], row["shop2"])]
    fit_rows = [row for row in rows if is_fit_pair(row["shop1"], row["shop2"])]
    pn_gate = pooled([row for row in gate_rows if row["arm"] == "new"])
    py_gate = pooled([row for row in gate_rows if row["arm"] == "yannik"])
    g5 = {"pass": bool(pn_gate >= py_gate), "new": pn_gate, "yannik": py_gate,
          "gate_games": len(gate_rows), "fit_games": len(fit_rows)}

    # E0: determinism re-run of the new arm on the first 10 gate seeds.
    router = ShopRouter(args.routes.read_text(encoding="utf-8"), 2)
    opp_files: dict[str, Path] = {}
    v104 = Path("analysis/v104_carrot_orderpri2_20260922/opponents")
    pool = Path("artifacts/tape_pool_20260922_top10/tapes")
    for opp in opponents:
        candidate = v104 / f"{opp}.json"
        opp_files[opp] = candidate if candidate.exists() else pool / f"{opp}.json"
    taped = {opp: Tape(json.dumps(json.loads(path.read_text(encoding="utf-8"))["actions"][1:720]))
             for opp, path in opp_files.items()}
    probe_seeds = sorted({row["seed"] for row in rows})[:10]
    jobs = [(seed, seat, 0, 1 + i) for seed in probe_seeds
            for i, _opp in enumerate(opponents) for seat in (0, 1)]
    players: list[Any] = [router, *[taped[opp] for opp in opponents]]
    results = play_batch(json.dumps(ENGINE_CONFIG), players, jobs, 0)
    screen_money = {(row["seed"], row["seat"], row["opponent"]): row["money"]
                    for row in new_rows}
    mismatches = 0
    for (seed, seat, _a, b), result in zip(jobs, results, strict=True):
        if float(result["money"][seat]) != screen_money[(seed, seat, opponents[b - 1])]:
            mismatches += 1
    e0 = {"pass": mismatches == 0, "mismatches": mismatches, "games": len(jobs)}

    gates = {
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "E0": e0,
        "points": {arm: pooled([row for row in rows if row["arm"] == arm])
                   for arm in ("new", "yannik", "omar")},
        "overall": "PASS" if all(g["pass"] for g in (g1, g2, g3, g4, g5, e0)) else "FAIL",
    }
    args.output.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k != "G3" else v["pass"]) for k, v in gates.items()}, indent=2))


if __name__ == "__main__":
    main()
