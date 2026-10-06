"""V104 screen driver: bake CARROT2+ORDERPRI2 tapes, play via Rust play_batch.

Arms: base (Rust ShopRouter on the omar base routes) vs cand (per-game baked
candidate tapes). Rows match the evaluate_shop_router_fast.py schema plus
bake diagnostics. Aborts unless replay money equals bake money on every
candidate game (prereg gate E0) and unless the Python-router base tapes
replay equal to the Rust-router base arm on the verify subset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bake_carrot_orderpri2_tapes import ENGINE_CONFIG, bake_game, load_opponent_actions
from kaggriculture_rs import ShopRouter, Tape, play_batch

from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1, load_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-seeds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes_data = load_routes(args.routes)
    python_router = ShopRouterV1(routes_data, max_shops=2)
    rust_router = ShopRouter(args.routes.read_text(encoding="utf-8"), max_shops=2)
    opponent_names = [path.stem for path in args.opponents]
    opponent_actions = {path.stem: load_opponent_actions(path) for path in args.opponents}
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    players: list[object] = [rust_router]
    tapes: dict[tuple[str, int, int], Tape] = {}
    base_verify: dict[tuple[str, int, int], Tape] = {}
    bake_money: dict[tuple[str, int, int], float] = {}
    bake_stats: dict[tuple[str, int, int], dict] = {}
    verify_keys: set[tuple[str, int, int]] = set()
    for name in opponent_names:
        for seed in seeds[: args.verify_seeds]:
            for seat in (0, 1):
                verify_keys.add((name, seed, seat))

    total = len(opponent_names) * len(seeds) * 2
    done = 0
    started = time.perf_counter()
    for name in opponent_names:
        for seed in seeds:
            for seat in (0, 1):
                baked = bake_game(python_router, opponent_actions[name], seed, seat)
                key = (name, seed, seat)
                tapes[key] = Tape(json.dumps(baked["tape"]))
                players.append(tapes[key])
                bake_money[key] = baked["money"]
                bake_stats[key] = {
                    "swaps": baked["swaps"],
                    "rescues": baked["rescues"],
                    "reordered_turns": baked["reordered_turns"],
                }
                if key in verify_keys:
                    pure = bake_game(
                        python_router, opponent_actions[name], seed, seat,
                        overlays=False,
                    )
                    base_verify[key] = Tape(json.dumps(pure["tape"]))
                    players.append(base_verify[key])
                done += 1
                if done % 60 == 0:
                    rate = done / (time.perf_counter() - started)
                    print(f"  baked {done}/{total}  {rate:.1f} games/s", flush=True)
    print(f"baked {done} games in {time.perf_counter() - started:.1f}s", flush=True)

    opponent_tapes = {name: Tape(json.dumps(opponent_actions[name])) for name in opponent_names}
    players.extend(opponent_tapes.values())

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
        for seed in seeds:
            for seat in (0, 1):
                jobs.append(("base", name, seed, seat))
                packed.append((seed, seat, slot(rust_router, index, packed_players),
                               slot(opponent_tapes[name], index, packed_players)))
                jobs.append(("cand", name, seed, seat))
                packed.append((seed, seat, slot(tapes[(name, seed, seat)], index, packed_players),
                               slot(opponent_tapes[name], index, packed_players)))
                if (name, seed, seat) in verify_keys:
                    jobs.append(("basebake", name, seed, seat))
                    slot_base = slot(
                        base_verify[(name, seed, seat)], index, packed_players
                    )
                    packed.append(
                        (seed, seat, slot_base,
                         slot(opponent_tapes[name], index, packed_players))
                    )

    results = play_batch(json.dumps(ENGINE_CONFIG), packed_players, packed, args.threads)
    rows: list[dict] = []
    mismatches = 0
    for (arm, opponent, seed, seat), result in zip(jobs, results, strict=True):
        money = result["money"]
        row = {
            "arm": arm,
            "opponent": opponent,
            "seed": seed,
            "seat": seat,
            "money": money[seat],
            "opponent_money": money[1 - seat],
            "margin": result["margin"],
            "point": result["point"],
        }
        if arm == "cand":
            row["bake_money"] = bake_money[(opponent, seed, seat)]
            row["bake_stats"] = bake_stats[(opponent, seed, seat)]
            if abs(float(money[seat]) - float(bake_money[(opponent, seed, seat)])) > 1e-9:
                mismatches += 1
        rows.append(row)
    print(f"E0 replay-vs-bake mismatches: {mismatches}/{len(bake_money)}")
    if mismatches:
        raise SystemExit(f"E0 FAIL: {mismatches} replay mismatches, aborting")

    base_money = {(r["opponent"], r["seed"], r["seat"]): r["money"]
                  for r in rows if r["arm"] == "base"}
    verify_bad = sum(
        1 for r in rows if r["arm"] == "basebake"
        and abs(float(r["money"]) - float(base_money[(r["opponent"], r["seed"], r["seat"])])) > 1e-9
    )
    print(f"router-parity mismatches (py vs rust): {verify_bad}/{len(verify_keys)}")
    if verify_bad:
        raise SystemExit(f"router parity FAIL: {verify_bad} mismatches, aborting")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
