"""V117: build a same-lineage shop-router (Yusuke method) on yannik's own tapes.

Base route = yannik. Each of the 7 yannik donor tapes is verified (720 actions,
exact action keys) and checked for canonical prefix identity through the branch
steps (v97 `canonical` rule). Splicing uses `splice()` semantics and ONLY where
the prefix is identical through the branch step -- never across divergent farms
(v106 chimeras die). Cells with no eligible donor, or no donor clearing the v97
fit thresholds, stay at incumbent 0.

Cell fit (from a donor fit screen played with UNMODIFIED
evaluate_shop_router_fast.py, donor tapes as `--tape` arms vs the frozen panel
on fit seeds, FIT-pair games only): per shop1 cell / shop-pair cell, among
eligible donors pick the best pooled point; a cell changes from incumbent 0
only with min_cell_games 30, min_opponent_games 8 (distinct opponents),
min_gain 0.01, tolerance 0.05 (no opponent-point regression vs incumbent on
fit). Second-level eligibility is checked against the chosen first-level route.

Outputs: artifacts/shop_router_v117/routes.json + analysis dir fit.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action

from kaggriculture_rl.policy.shop_router_v1 import splice

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "analysis" / "refresh_20260923" / (
    "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)
POOL = ROOT / "artifacts" / "tape_pool_20260922_top10" / "tapes"
OUT_ROUTES = ROOT / "artifacts" / "shop_router_v117" / "routes.json"
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
SHOP_INDEX = {shop: i for i, shop in enumerate(SHOPS)}
BASE_STEM = "yannik_schiffner__ep111952809__seat0"
DONORS = [
    "yannik_schiffner__ep111936440__seat0",
    "yannik_schiffner__ep111952809__seat0",
    "yannik_schiffner__ep111953944__seat1",
    "yannik_schiffner__ep111955057__seat1",
    "yannik_schiffner__ep111956211__seat1",
    "yannik_schiffner__ep111957345__seat1",
    "yannik_schiffner__ep111958489__seat1",
]
MIN_CELL_GAMES = 30
MIN_OPPONENT_GAMES = 8
MIN_GAIN = 0.01
TOLERANCE = 0.05


def is_fit_pair(shop1: str, shop2: str) -> bool:
    return (SHOP_INDEX[shop1] * 8 + SHOP_INDEX[shop2]) % 2 == 0


def canonical(actions: list[Any]) -> list[dict[str, Any]]:
    return [normalize_legacy_action(a) for a in actions]


def load_donor_route(stem: str) -> list[dict[str, Any]]:
    payload = json.loads((POOL / f"{stem}.json").read_text(encoding="utf-8"))
    actions = payload["actions"]
    assert len(actions) == 720, f"{stem}: length {len(actions)} != 720"
    assert all(set(a) == {"farmer", "hands", "market"} for a in actions), f"{stem}: bad keys"
    return actions[1:720]


def pooled(rows: list[dict[str, Any]]) -> float:
    return sum(row["point"] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT_ROUTES)
    parser.add_argument("--fit-output", type=Path, required=True)
    args = parser.parse_args()

    base_data = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    first_step = int(base_data["first_branch_step"])
    second_step = int(base_data["second_branch_step"])
    base = base_data["routes"][0]
    assert len(base) == 719, f"base length {len(base)}"
    base_canon = canonical(base)
    assert splice(base_canon, base_canon, first_step) == base_canon
    assert splice(base_canon, base_canon, second_step) == base_canon

    donor_routes = {stem: load_donor_route(stem) for stem in DONORS}
    donor_canon = {stem: canonical(route) for stem, route in donor_routes.items()}
    eligible_first = {
        stem: donor_canon[stem][:first_step] == base_canon[:first_step]
        for stem in DONORS
    }
    eligible_second_vs_base = {
        stem: donor_canon[stem][:second_step] == base_canon[:second_step]
        for stem in DONORS
    }
    print(json.dumps({"first": eligible_first, "second_vs_base": eligible_second_vs_base},
                     indent=1))
    usable = [stem for stem in DONORS if stem != BASE_STEM and eligible_first[stem]]
    if len(usable) < 2:
        print(f"fewer than 2 eligible non-base donors {usable}: negative construction, stop")
        raise SystemExit(2)
    identity72 = {}
    identity144 = {}
    for stem in DONORS:
        identity72[stem] = sum(1 for i in range(first_step)
                               if donor_canon[stem][i] == base_canon[i]) / first_step
        identity144[stem] = sum(1 for i in range(second_step)
                                if donor_canon[stem][i] == base_canon[i]) / second_step

    rows: list[dict[str, Any]] = json.loads(args.fit_rows.read_text(encoding="utf-8"))
    fit = [row for row in rows if is_fit_pair(row["shop1"], row["shop2"])]
    print(f"fit games: {len(fit)} / {len(rows)} rows on FIT pairs")

    def group(level: str, cell: str, arm: str) -> list[dict[str, Any]]:
        out = []
        for row in fit:
            if row["arm"] != arm:
                continue
            key = row["shop1"] if level == "first" else f"{row['shop1']}__{row['shop2']}"
            if key == cell:
                out.append(row)
        return out

    inc = BASE_STEM
    routes: list[list[Any]] = [base]
    route_names = list(base_data.get("route_names", ["yannik_base"]))
    route_index = {BASE_STEM: 0}

    def add_route(stem: str, boundary: int) -> int:
        if stem in route_index:
            return route_index[stem]
        route = splice(base, donor_routes[stem], boundary)
        assert canonical(route)[:boundary] == base_canon[:boundary]
        routes.append(route)
        route_names.append(stem)
        route_index[stem] = len(routes) - 1
        return route_index[stem]

    first_routes = {shop: 0 for shop in SHOPS}
    first_route_canon = {shop: base_canon for shop in SHOPS}
    cells_changed = []
    for shop in SHOPS:
        inc_rows = group("first", shop, inc)
        if len(inc_rows) < MIN_CELL_GAMES:
            continue
        inc_point = pooled(inc_rows)
        inc_by_opp = {opp: pooled([r for r in inc_rows if r["opponent"] == opp])
                      for opp in {r["opponent"] for r in inc_rows}}
        best = None
        for stem in usable:
            cand = group("first", shop, stem)
            if len(cand) < MIN_CELL_GAMES:
                continue
            if len({r["opponent"] for r in cand}) < MIN_OPPONENT_GAMES:
                continue
            gain = pooled(cand) - inc_point
            if gain < MIN_GAIN:
                continue
            worst = min(pooled([r for r in cand if r["opponent"] == opp])
                        - inc_by_opp.get(opp, 1.0)
                        for opp in {r["opponent"] for r in cand})
            if worst < -TOLERANCE:
                continue
            if best is None or gain > best[1]:
                best = (stem, gain, len(cand))
        if best is not None:
            stem, gain, n = best
            first_routes[shop] = add_route(stem, first_step)
            first_route_canon[shop] = canonical(routes[first_routes[shop]])
            cells_changed.append({"cell": shop, "donor": stem, "gain": gain, "n": n})

    second_routes = {f"{a}__{b}": 0 for a in SHOPS for b in SHOPS}
    for a in SHOPS:
        for b in SHOPS:
            cell = f"{a}__{b}"
            inc_rows = group("second", cell, inc)
            if len(inc_rows) < MIN_CELL_GAMES:
                continue
            inc_point = pooled(inc_rows)
            inc_by_opp = {opp: pooled([r for r in inc_rows if r["opponent"] == opp])
                          for opp in {r["opponent"] for r in inc_rows}}
            ref = first_route_canon[a]
            best = None
            for stem in usable:
                if donor_canon[stem][:second_step] != ref[:second_step]:
                    continue
                cand = group("second", cell, stem)
                if len(cand) < MIN_CELL_GAMES:
                    continue
                if len({r["opponent"] for r in cand}) < MIN_OPPONENT_GAMES:
                    continue
                gain = pooled(cand) - inc_point
                if gain < MIN_GAIN:
                    continue
                worst = min(pooled([r for r in cand if r["opponent"] == opp])
                            - inc_by_opp.get(opp, 1.0)
                            for opp in {r["opponent"] for r in cand})
                if worst < -TOLERANCE:
                    continue
                if best is None or gain > best[1]:
                    best = (stem, gain, len(cand))
            if best is not None:
                stem, gain, n = best
                idx = add_route(stem, second_step)
                # second-level route must extend the chosen first-level route.
                assert canonical(routes[idx])[:second_step] == ref[:second_step]
                second_routes[cell] = idx
                cells_changed.append({"cell": cell, "donor": stem, "gain": gain, "n": n})

    route_blob = {
        "first_branch_step": first_step,
        "second_branch_step": second_step,
        "first_routes": first_routes,
        "second_routes": second_routes,
        "routes": routes,
        "route_names": route_names,
        "route_sha256": [hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                         for r in routes],
        "schema": "shop_router_v4/base_tape",
        "source": {"v117": "same-lineage Yusuke-method refit on yannik's own tapes; base = yannik",
                   "base": str(BASE_PATH.relative_to(ROOT)),
                   "fit_rows": str(args.fit_rows)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(route_blob) + "\n", encoding="utf-8")
    fit_note = {
        "eligible_donors": usable,
        "donor_identity72": identity72,
        "donor_identity144": identity144,
        "cells_changed": cells_changed,
        "thresholds": {"min_cell_games": MIN_CELL_GAMES,
                       "min_opponent_games": MIN_OPPONENT_GAMES,
                       "min_gain": MIN_GAIN, "tolerance": TOLERANCE},
    }
    args.fit_output.parent.mkdir(parents=True, exist_ok=True)
    args.fit_output.write_text(json.dumps(fit_note, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(routes)} routes, {len(cells_changed)} cells changed)"
          f" + {args.fit_output}")


if __name__ == "__main__":
    main()
