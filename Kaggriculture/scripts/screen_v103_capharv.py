"""V103 screen driver: bake CapHarvRule tapes, play via Rust play_batch.

Arms: base (Rust ShopRouter on CemBas live v3 routes) vs cand (per-game baked
base+CapHarvRule tapes). Chunked by seed-block so baked tapes never pile up in
RAM. Aborts unless replay money equals bake money on every candidate game (E0)
and unless the Python-router base tapes replay equal to the Rust-router base
arm on the verify subset. Writes rows + gates.json (G1 with seed-clustered
bootstrap CI). See analysis/v103_capharv_20260922/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bake_capharv_tapes import ENGINE_CONFIG, bake_one, load_opponent_actions
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
    parser.add_argument("--bootstrap", type=int, default=2000)
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
    total_triggers = 0
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
            for name, seed, seat, tape_json, money, triggers in pool.map(
                bake_one, payloads, chunksize=4
            ):
                cand_tapes[(name, seed, seat)] = tape_json
                bake_money[(name, seed, seat)] = money
                total_triggers += triggers
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

    # G1: paired point diff cand-base with seed-clustered bootstrap CI.
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        if r["arm"] in ("base", "cand"):
            by_seed.setdefault(r["seed"], []).append(r)
    seed_ids = sorted(by_seed)
    pair_diff: dict[int, float] = {}
    for s in seed_ids:
        base_pts = {(r["opponent"], r["seat"]): r["point"]
                    for r in by_seed[s] if r["arm"] == "base"}
        diffs = [float(r["point"]) - float(base_pts[(r["opponent"], r["seat"])])
                 for r in by_seed[s] if r["arm"] == "cand"]
        pair_diff[s] = sum(diffs) / len(diffs)
    mean_diff = sum(pair_diff.values()) / len(pair_diff)
    rng = random.Random(0)
    boot = []
    for _ in range(args.bootstrap):
        sample = [pair_diff[rng.choice(seed_ids)] for _ in seed_ids]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
    point_cand = sum(r["point"] for r in rows if r["arm"] == "cand")
    n_cand = sum(1 for r in rows if r["arm"] == "cand")
    point_base = sum(r["point"] for r in rows if r["arm"] == "base")
    g1 = {
        "mean_paired_point_diff": mean_diff,
        "ci95": [lo, hi],
        "point_cand": point_cand / n_cand,
        "point_base": point_base / sum(1 for r in rows if r["arm"] == "base"),
        "n_paired": n_cand,
        "pass": bool(mean_diff >= 0.20 and lo > 0),
    }
    cand_rows = sorted(
        (r for r in rows if r["arm"] == "cand"), key=lambda r: -r["money"]
    )[:25]
    top25 = {
        "point_cand": sum(r["point"] for r in cand_rows) / len(cand_rows),
        "mean_money_cand": sum(r["money"] for r in cand_rows) / len(cand_rows),
    }
    gates = {"G1": g1, "top25_supplementary": top25,
             "triggers": total_triggers, "games": total_baked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    (args.output.parent / "gates.json").write_text(
        json.dumps(gates, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output} in "
          f"{time.perf_counter() - started:.1f}s")
    print(f"G1 diff={mean_diff:.4f} CI=[{lo:.4f},{hi:.4f}] pass={g1['pass']} "
          f"triggers={total_triggers}")


if __name__ == "__main__":
    main()
