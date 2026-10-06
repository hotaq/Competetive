"""V111 screen driver: bake tomato-top-up tapes, play via Rust play_batch.

Arms: base (Rust ShopRouter on the omar base routes) vs cand (per-game baked
base+tomato_topup tapes). Chunked by seed-block so baked tapes never pile up
in RAM. Aborts unless replay money equals bake money on every candidate game
(E0) and unless the Python-router base tapes replay equal to the Rust-router
base arm on the verify subset. Writes rows + gates.json (G1 money diff with
seed-clustered bootstrap CI, G2 point diff, G3 per-opponent floor).
See analysis/v111_topup_20260922/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bake_tomato_topup_tapes import ENGINE_CONFIG, bake_one, load_opponent_actions
from kaggriculture_rs import ShopRouter, Tape, play_batch

BLOCK = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-seeds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes_json = args.routes.read_text(encoding="utf-8")
    rust_router = ShopRouter(routes_json, max_shops=2)
    opponent_names = [path.stem for path in args.opponents]
    opponent_actions = {path.stem: load_opponent_actions(path) for path in args.opponents}
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    verify_seeds = set(seeds[: args.verify_seeds])

    opponent_tapes = {name: Tape(json.dumps(acts)) for name, acts in opponent_actions.items()}

    rows: list[dict] = []
    mismatches = 0
    verify_bad = 0
    verify_total = 0
    total_topups = 0
    total_baked = 0
    started = time.perf_counter()
    blocks = [seeds[i : i + BLOCK] for i in range(0, len(seeds), BLOCK)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bi, block in enumerate(blocks):
            payloads = [
                (name, routes_json, opponent_actions[name], seed, seat, True)
                for name in opponent_names
                for seed in block
                for seat in (0, 1)
            ]
            verify_payloads = [
                (name, routes_json, opponent_actions[name], seed, seat, False)
                for name in opponent_names
                for seed in block
                if seed in verify_seeds
                for seat in (0, 1)
            ]
            cand_tapes: dict[tuple[str, int, int], str] = {}
            bake_money: dict[tuple[str, int, int], float] = {}
            for name, seed, seat, tape_json, money, topups in pool.map(
                bake_one, payloads, chunksize=4
            ):
                cand_tapes[(name, seed, seat)] = tape_json
                bake_money[(name, seed, seat)] = money
                total_topups += topups
                total_baked += 1
            base_tapes: dict[tuple[str, int, int], str] = {}
            for name, seed, seat, tape_json, _money, _trig in pool.map(
                bake_one, verify_payloads, chunksize=1
            ):
                base_tapes[(name, seed, seat)] = tape_json

            cand_objs = {k: Tape(v) for k, v in cand_tapes.items()}
            base_objs = {k: Tape(v) for k, v in base_tapes.items()}

            def slot(obj: object, index: dict[int, int], store: list[object]) -> int:
                key = id(obj)
                if key not in index:
                    index[key] = len(store)
                    store.append(obj)
                return index[key]

            packed_players: list[object] = []
            index: dict[int, int] = {}
            jobs: list[tuple[str, str, int, int]] = []
            packed: list[tuple[int, int, int, int]] = []
            for name in opponent_names:
                for seed in block:
                    for seat in (0, 1):
                        jobs.append(("base", name, seed, seat))
                        packed.append((
                            seed, seat,
                            slot(rust_router, index, packed_players),
                            slot(opponent_tapes[name], index, packed_players),
                        ))
                        jobs.append(("cand", name, seed, seat))
                        packed.append((
                            seed, seat,
                            slot(cand_objs[(name, seed, seat)], index, packed_players),
                            slot(opponent_tapes[name], index, packed_players),
                        ))
                        if (name, seed, seat) in base_tapes:
                            jobs.append(("basebake", name, seed, seat))
                            packed.append((
                                seed, seat,
                                slot(base_objs[(name, seed, seat)], index, packed_players),
                                slot(opponent_tapes[name], index, packed_players),
                            ))
            results = play_batch(
                json.dumps(ENGINE_CONFIG), packed_players, packed, args.threads)
            base_money: dict[tuple[str, int, int], float] = {}
            block_rows: list[dict] = []
            for (arm, opponent, seed, seat), result in zip(jobs, results, strict=True):
                money = result["money"]
                row = {
                    "arm": arm, "opponent": opponent, "seed": seed, "seat": seat,
                    "money": money[seat], "opponent_money": money[1 - seat],
                    "margin": result["margin"], "point": result["point"],
                }
                if arm == "cand":
                    key = (opponent, seed, seat)
                    row["bake_money"] = bake_money[key]
                    if abs(float(money[seat]) - float(bake_money[key])) > 1e-9:
                        mismatches += 1
                if arm == "base":
                    base_money[(opponent, seed, seat)] = float(money[seat])
                rows.append(row)
                block_rows.append(row)
            for r in block_rows:
                if r["arm"] == "basebake":
                    verify_total += 1
                    if abs(float(r["money"]) - float(
                        base_money[(r["opponent"], r["seed"], r["seat"])]
                    )) > 1e-9:
                        verify_bad += 1
            del cand_objs, base_objs, packed_players
            rate = total_baked / (time.perf_counter() - started)
            print(f"  block {bi + 1}/{len(blocks)} baked+played  {rate:.1f} bakes/s",
                  flush=True)

    print(f"E0 replay-vs-bake mismatches: {mismatches}/{total_baked}")
    if mismatches:
        raise SystemExit(f"E0 FAIL: {mismatches} replay mismatches, aborting")
    print(f"router-parity mismatches (py vs rust): {verify_bad}/{verify_total}")
    if verify_bad:
        raise SystemExit(f"router parity FAIL: {verify_bad} mismatches, aborting")

    # Frozen gates: paired money-margin diff, seed-clustered bootstrap CI.
    base = {(r["opponent"], r["seed"], r["seat"]): r for r in rows if r["arm"] == "base"}
    cand = [r for r in rows if r["arm"] == "cand"]
    pairs = [(r, base[(r["opponent"], r["seed"], r["seat"])]) for r in cand]
    diffs = [c["margin"] - b["margin"] for c, b in pairs]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (c, _), d in zip(pairs, diffs, strict=True):
        by_seed[c["seed"]].append(d)
    seed_ids = sorted(by_seed)
    rng = random.Random(99)
    means = []
    for _ in range(args.bootstrap):
        sample = [d for s in (rng.choice(seed_ids) for _ in seed_ids) for d in by_seed[s]]
        means.append(st.fmean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]

    def point(arm: str) -> float:
        return st.fmean(r["point"] for r in rows if r["arm"] == arm)

    per_opp = {}
    for opp in sorted({r["opponent"] for r in cand}):
        group = [
            c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
            for c in cand
            if c["opponent"] == opp
        ]
        per_opp[opp] = {"n": len(group), "mean_diff": st.fmean(group)}
    seat_diff = {
        str(seat): st.fmean(
            c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
            for c in cand
            if c["seat"] == seat
        )
        for seat in (0, 1)
    }
    mean_diff = st.fmean(diffs)
    pt_base, pt_cand = point("base"), point("cand")
    gates = {
        "n_pairs": len(pairs),
        "mean_paired_diff": mean_diff,
        "ci95": [lo, hi],
        "point_base": pt_base,
        "point_cand": pt_cand,
        "point_diff": pt_cand - pt_base,
        "per_opponent": per_opp,
        "seat_mean_diff": seat_diff,
        "topups_fired": total_topups,
        "games_baked": total_baked,
        "G1": bool(mean_diff > 0 and lo > 0),
        "G2": bool(pt_cand - pt_base >= -0.01),
        "G3": bool(all(v["mean_diff"] >= -200 for v in per_opp.values())),
    }
    gates["OVERALL"] = bool(gates["G1"] and gates["G2"] and gates["G3"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    (args.output.parent / "gates.json").write_text(
        json.dumps(gates, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output} in "
          f"{time.perf_counter() - started:.1f}s")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
