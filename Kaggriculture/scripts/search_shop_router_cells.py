# ruff: noqa: E501
"""Fit a two-shop route table on train seeds with Yusuke's no-regression rule.

Every donor is played verbatim (it already carries the shared prefix) against every opponent
over the train seeds, both seats, through `kaggriculture_rs.play_batch`. Because all donors of a
family share the prefix to the branch step, the shop pair a game draws is the same for every
donor, so "the router picks donor d in cell (A, B)" has exactly the outcome of donor d in the
games that drew (A, B). The table is then pure post-processing:

- per first shop A choose the family (route-0 prefix or route-1 prefix) whose best cells give the
  higher pooled point over games with shop1 == A, without regressing against any opponent;
- per cell (A, B) replace the incumbent (the v1 table) only by the donor with the best cell point
  rate that gains >= --min-gain, has >= --min-cell-games games, and is not worse than the
  incumbent by more than --tolerance against any opponent with >= --min-opponent-games in the cell.

Writes a routes file with the same schema as artifacts/shop_router_v1/yusuke_routes.json, so
`ShopRouterV1.from_file(path)` and the standalone builder consume it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rs import Tape, play_batch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
Rows = list[dict[str, Any]]


def load_pool_tape(path: Path) -> Tape:
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    return Tape(json.dumps(actions[1:720]))


def play_donors(donors: dict[str, Any], opponents: dict[str, Tape], seeds: range, threads: int,
                chunk: int) -> Rows:
    players: list[Any] = [Tape(json.dumps(route)) for route in donors["routes"]]
    names = list(opponents)
    offset = len(players)
    players.extend(opponents[n] for n in names)
    jobs = [(seed, seat, d, offset + o)
            for d in range(len(donors["routes"])) for o in range(len(names))
            for seed in seeds for seat in (0, 1)]
    rows: Rows = []
    started = time.perf_counter()
    for start in range(0, len(jobs), chunk):
        batch = jobs[start:start + chunk]
        for (seed, seat, d, o), result in zip(batch, play_batch(json.dumps(ENGINE_CONFIG), players, batch, threads),
                                              strict=True):
            rows.append({"donor": d, "opponent": names[o - offset], "seed": seed, "seat": seat,
                         "point": result["point"], "margin": result["margin"],
                         "shop1": result["shop1"], "shop2": result["shop2"]})
        rate = len(rows) / (time.perf_counter() - started)
        print(f"  {len(rows)}/{len(jobs)} games  {rate:,.0f}/s", flush=True)
    return rows


def point(rows: Rows) -> float:
    return st.mean(r["point"] for r in rows) if rows else float("nan")


def fit(rows: Rows, donors: dict[str, Any], incumbent: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    families = donors["family"]
    by_donor: dict[int, Rows] = defaultdict(list)
    for r in rows:
        by_donor[r["donor"]].append(r)
    # incumbent v1 route index i == donor index i (the donor builder puts v1 routes first, in order)
    inc_first = {shop: int(incumbent["first_routes"].get(shop, 0)) for shop in SHOPS}
    inc_second = {pair: int(idx) for pair, idx in incumbent["second_routes"].items()}
    family_base = {0: 0, 1: 1}  # donor index that is the plain prefix continuation per family
    opponents = sorted({r["opponent"] for r in rows})

    def cell_rows(d: int, a: str, b: str) -> Rows:
        return [r for r in by_donor[d] if r["shop1"] == a and r["shop2"] == b]

    def not_worse(cand: Rows, inc: Rows) -> tuple[bool, dict[str, Any]]:
        checks = {}
        ok = True
        for opp in opponents:
            c = [r for r in cand if r["opponent"] == opp]
            i = [r for r in inc if r["opponent"] == opp]
            if len(c) < args.min_opponent_games or len(i) < args.min_opponent_games:
                checks[opp] = None
                continue
            diff = point(c) - point(i)
            checks[opp] = round(diff, 4)
            if diff < -args.tolerance:
                ok = False
        return ok, checks

    first_routes: dict[str, int] = {}
    second_routes: dict[str, int] = {}
    report: dict[str, Any] = {"first_shop": {}, "cells": {}}
    for a in SHOPS:
        options = {}
        for fam in (0, 1):
            eligible = [d for d, f in enumerate(families) if f == fam]
            if not eligible:
                continue
            base = family_base[fam]
            chosen: dict[str, int] = {}
            cell_info: dict[str, Any] = {}
            for b in SHOPS:
                pair = f"{a}__{b}"
                inc_d = inc_second.get(pair, inc_first[a]) if inc_first[a] == base else base
                if families[inc_d] != fam:
                    inc_d = base
                inc = cell_rows(inc_d, a, b)
                best_d, best_rows, best_checks = inc_d, inc, {}
                for d in eligible:
                    if d == inc_d:
                        continue
                    cand = cell_rows(d, a, b)
                    if len(cand) < args.min_cell_games or len(inc) < args.min_cell_games:
                        continue
                    if point(cand) < point(best_rows) + args.min_gain:
                        continue
                    ok, checks = not_worse(cand, inc)
                    if ok:
                        best_d, best_rows, best_checks = d, cand, checks
                chosen[pair] = best_d
                cell_info[pair] = {"incumbent": inc_d, "chosen": best_d, "n": len(best_rows),
                                   "incumbent_point": round(point(inc), 4) if inc else None,
                                   "chosen_point": round(point(best_rows), 4) if best_rows else None,
                                   "checks": best_checks}
            # pooled outcome for games with shop1 == a under this family and these cells
            pooled = []
            per_opp: dict[str, list[float]] = defaultdict(list)
            for b in SHOPS:
                for r in cell_rows(chosen[f"{a}__{b}"], a, b):
                    pooled.append(r["point"])
                    per_opp[r["opponent"]].append(r["point"])
            options[fam] = {"cells": chosen, "info": cell_info, "n": len(pooled),
                            "point": st.mean(pooled) if pooled else float("nan"),
                            "per_opponent": {o: st.mean(v) for o, v in per_opp.items()}}
        inc_fam = families[inc_first[a]]
        pick = inc_fam
        for fam, opt in options.items():
            if fam == inc_fam or opt["n"] < args.min_cell_games:
                continue
            ref = options[inc_fam]
            gain = opt["point"] - ref["point"]
            regress = any(opt["per_opponent"].get(o, 0) < ref["per_opponent"].get(o, 0) - args.tolerance
                          for o in opponents)
            if gain >= args.min_gain and not regress:
                pick = fam
        first_routes[a] = family_base[pick]
        for pair, d in options[pick]["cells"].items():
            second_routes[pair] = d
        report["first_shop"][a] = {"family": pick, "incumbent_family": inc_fam,
                                   "options": {str(f): {"n": o["n"], "point": round(o["point"], 4)}
                                               for f, o in options.items()}}
        report["cells"].update({pair: options[pick]["info"][pair] for pair in options[pick]["cells"]})
    return {"first_routes": first_routes, "second_routes": second_routes, "report": report}


def screen(donors: dict[str, Any], opponents: dict[str, Tape], args: argparse.Namespace) -> dict[str, Any]:
    """Rank every donor by pooled point on the screen seeds; keep the top K plus the always-keep set.

    Family-1 donors are ranked among themselves and all kept (there are few), so the first-shop
    family decision still has candidates.
    """
    seeds = range(args.screen_seed_start, args.screen_seed_start + args.screen_seed_count)
    print(f"screen: {len(donors['routes'])} donors x {len(opponents)} opponents x {len(seeds)} seeds x 2 seats", flush=True)
    rows = play_donors(donors, opponents, seeds, args.threads, args.chunk)
    pooled: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        pooled[r["donor"]].append(r["point"])
    ranked = sorted(pooled, key=lambda d: -st.mean(pooled[d]))
    prefixes = tuple(p for p in args.always_keep.split(",") if p)
    keep = {d for d in ranked if donors["family"][d] == 1}
    keep |= {d for d in ranked if donors["names"][d].startswith(prefixes)}
    keep |= set([d for d in ranked if donors["family"][d] == 0][:args.screen_keep])
    order = [d for d in range(len(donors["routes"])) if d in keep]  # keep original order (v1 routes first)
    table = [{"donor": d, "name": donors["names"][d], "family": donors["family"][d],
              "point": round(st.mean(pooled[d]), 4), "kept": d in keep} for d in ranked]
    if args.output_screen:
        args.output_screen.parent.mkdir(parents=True, exist_ok=True)
        args.output_screen.write_text(json.dumps({"seeds": [seeds.start, seeds.stop - 1], "ranking": table},
                                                 indent=1) + "\n", encoding="utf-8")
    print(f"screen kept {len(order)} donors; top 10: " + ", ".join(f"{donors['names'][d]}={st.mean(pooled[d]):.3f}" for d in ranked[:10]))
    return {**donors, "routes": [donors["routes"][d] for d in order], "names": [donors["names"][d] for d in order],
            "family": [donors["family"][d] for d in order]}


def compact(donors: dict[str, Any], first_routes: dict[str, int], second_routes: dict[str, int]) -> dict[str, Any]:
    used = sorted(set(first_routes.values()) | set(second_routes.values()) | {0, 1})
    remap = {old: new for new, old in enumerate(used)}
    routes = [donors["routes"][old] for old in used]
    return {
        "schema": "shop_router_v2/routes",
        "first_branch_step": donors["first_branch_step"], "second_branch_step": donors["second_branch_step"],
        "first_routes": {k: remap[v] for k, v in first_routes.items()},
        "second_routes": {k: remap[v] for k, v in second_routes.items()},
        "routes": routes,
        "route_names": [donors["names"][old] for old in used],
        "route_sha256": [hashlib.sha256(json.dumps(r, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
                         for r in routes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donors", type=Path, default=ROOT / "artifacts/shop_router_v2/donors.json")
    parser.add_argument("--incumbent", type=Path, default=None,
                        help="routes.json whose table is the incumbent; default = donor 0 in every cell")
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--screen-seed-start", type=int, default=None,
                        help="if given, rank all donors on these seeds first and keep --screen-keep of them")
    parser.add_argument("--screen-seed-count", type=int, default=0)
    parser.add_argument("--screen-keep", type=int, default=24)
    parser.add_argument("--always-keep", default="yusuke_route_,yusuke_hayashi__",
                        help="comma-separated name prefixes that survive the screen regardless of rank")
    parser.add_argument("--output-screen", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--min-cell-games", type=int, default=30)
    parser.add_argument("--min-opponent-games", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--output-routes", type=Path, required=True)
    parser.add_argument("--output-fit", type=Path, required=True)
    parser.add_argument("--output-games", type=Path, required=True)
    args = parser.parse_args()

    donors = json.loads(args.donors.read_text(encoding="utf-8"))
    if args.incumbent is not None:
        incumbent = json.loads(args.incumbent.read_text(encoding="utf-8"))
        for i in range(len(incumbent["routes"])):
            if donors["names"][i] != f"yusuke_route_{i}":
                raise SystemExit("donor file must start with the incumbent routes in order")
    else:
        incumbent = {"routes": [donors["routes"][0]], "first_routes": {s: 0 for s in SHOPS},
                     "second_routes": {f"{a}__{b}": 0 for a in SHOPS for b in SHOPS}}
    opponents = {p.stem: load_pool_tape(p) for p in args.opponents}
    if args.screen_seed_start is not None and args.screen_seed_count > 0:
        donors = screen(donors, opponents, args)
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    print(f"{len(donors['routes'])} donors x {len(opponents)} opponents x {len(seeds)} seeds x 2 seats", flush=True)
    rows = play_donors(donors, opponents, seeds, args.threads, args.chunk)
    args.output_games.parent.mkdir(parents=True, exist_ok=True)
    args.output_games.write_text(json.dumps(rows) + "\n", encoding="utf-8")

    fitted = fit(rows, donors, incumbent, args)
    routes = compact(donors, fitted["first_routes"], fitted["second_routes"])
    routes["fit"] = {"donors": str(args.donors), "incumbent": str(args.incumbent) if args.incumbent else "donor 0",
                     "opponents": sorted(opponents), "seed_start": args.seed_start, "seed_count": args.seed_count,
                     "min_cell_games": args.min_cell_games, "min_opponent_games": args.min_opponent_games,
                     "tolerance": args.tolerance, "min_gain": args.min_gain}
    args.output_routes.parent.mkdir(parents=True, exist_ok=True)
    args.output_routes.write_text(json.dumps(routes, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    args.output_fit.write_text(json.dumps(fitted["report"], indent=1) + "\n", encoding="utf-8")

    changed = {p: c for p, c in fitted["report"]["cells"].items() if c["chosen"] != c["incumbent"]}
    print(f"\nfirst_routes: {fitted['first_routes']}")
    print(f"cells changed: {len(changed)}/64; donors used: {routes['route_names']}")
    for pair, c in sorted(changed.items()):
        print(f"  {pair:<34} {donors['names'][c['incumbent']]:<28} -> {donors['names'][c['chosen']]:<28}"
              f" {c['incumbent_point']} -> {c['chosen_point']} (n={c['n']})")
    # in-sample pooled point of incumbent vs fitted (labelled in-sample; the holdout run is the evidence)
    by = defaultdict(list)
    for r in rows:
        by[(r["donor"], r["shop1"], r["shop2"])].append(r["point"])
    def pooled(table_first: dict[str, int], table_second: dict[str, int]) -> float:
        pts = []
        for (d, a, b), v in by.items():
            if b is None or a is None:
                continue
            if table_second.get(f"{a}__{b}", table_first.get(a, 0)) == d:
                pts.extend(v)
        return st.mean(pts) if pts else float("nan")
    inc_first = {s: int(incumbent["first_routes"].get(s, 0)) for s in SHOPS}
    inc_second = {k: int(v) for k, v in incumbent["second_routes"].items()}
    print(f"in-sample pooled point: incumbent {pooled(inc_first, inc_second):.4f}"
          f" -> fitted {pooled(fitted['first_routes'], fitted['second_routes']):.4f}")
    print(f"wrote {args.output_routes}")


if __name__ == "__main__":
    main()
