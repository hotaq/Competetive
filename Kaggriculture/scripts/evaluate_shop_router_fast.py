"""Rust-loop twin of evaluate_shop_router_policy.py for tape opponents.

Same arms (base / one / two / live), same row schema, same seed and seat grid, but every game
runs inside `kaggriculture_rs.play_fast`, so tape-vs-tape sweeps run ~100x faster. Closed-loop
Python opponents (file agents) are not supported here; use the slow harness for those.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggriculture_rs import ShopRouter, Tape, play_batch

from kaggriculture_rl.policy.phased_router_v2 import YUSUKE_SCHEDULES
from kaggriculture_rl.policy.shop_router_v1 import DEFAULT_ROUTES_PATH

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
DEPTH = {"base": 0, "one": 1, "two": 2}


def load_pool_tape(path: Path) -> Tape:
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    return Tape(json.dumps(actions[1:720]))


def load_schedule_tape(path: Path) -> Tape:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Tape(json.dumps([json.loads(line) for line in lines][:719]))


def build_arms(names: list[str], routes_path: Path,
               extra_routers: dict[str, tuple[Path, int]] | None = None,
               extra_tapes: dict[str, Path] | None = None) -> dict[str, dict[int, Any]]:
    routes_json = routes_path.read_text(encoding="utf-8")
    arms: dict[str, dict[int, Any]] = {}
    for name, (path, depth) in (extra_routers or {}).items():
        router = ShopRouter(path.read_text(encoding="utf-8"), max_shops=depth)
        arms[name] = {0: router, 1: router}
    for name, path in (extra_tapes or {}).items():
        tape = load_pool_tape(path)
        arms[name] = {0: tape, 1: tape}
    for name in names:
        if name in arms:
            continue
        if name in DEPTH:
            router = ShopRouter(routes_json, max_shops=DEPTH[name])
            arms[name] = {0: router, 1: router}
        elif name == "live":
            arms[name] = {seat: load_schedule_tape(path) for seat, path in YUSUKE_SCHEDULES.items()}
        else:
            raise ValueError("unknown arm " + name)
    return arms


def play_jobs(arms: dict[str, dict[int, Any]], opponents: dict[str, Tape],
              jobs: list[tuple[str, str, int, int]], threads: int = 0) -> list[dict[str, Any]]:
    """Run (arm, opponent, seed, seat) jobs through one Rust batch; rows come back in order."""
    players: list[Any] = []
    index: dict[int, int] = {}

    def slot(obj: Any) -> int:
        key = id(obj)
        if key not in index:
            index[key] = len(players)
            players.append(obj)
        return index[key]

    packed = [
        (seed, seat, slot(arms[arm][seat]), slot(opponents[opp])) for arm, opp, seed, seat in jobs
    ]
    results = play_batch(json.dumps(ENGINE_CONFIG), players, packed, threads)
    rows = []
    for (arm, opponent, seed, seat), result in zip(jobs, results, strict=True):
        money = result["money"]
        rows.append({
            "arm": arm, "opponent": opponent, "seed": seed, "seat": seat,
            "money": money[seat], "opponent_money": money[1 - seat], "margin": result["margin"],
            "point": result["point"], "shop1": result["shop1"], "shop2": result["shop2"],
            "route_at_second": result["route_at_second"], "route_final": result["route_final"],
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> str:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
        by_pair[(row["arm"], row["opponent"])].append(row)

    def line(label: str, name: str, group: list[dict[str, Any]]) -> str:
        return (f"{label:>6} {name:<36} {len(group):>6} {st.mean(r['point'] for r in group):>7.4f}"
                f" {st.mean(r['margin'] for r in group):>+12,.0f}"
                f" {st.median(r['margin'] for r in group):>+9,.0f}")

    lines = [f"{'arm':>6} {'opponent':<36} {'n':>6} {'point':>7} {'mean margin':>12} {'median':>9}"]
    for arm in sorted(by_arm):
        lines.append(line(arm, "ALL", by_arm[arm]))
        for (a, opp), group in sorted(by_pair.items()):
            if a == arm:
                lines.append(line("", opp, group))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arms", nargs="+", default=["base", "one", "two", "live"])
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES_PATH)
    parser.add_argument("--router", action="append", default=[], metavar="NAME=PATH[:DEPTH]",
                        help="extra router arm from another routes file (depth defaults to 2)")
    parser.add_argument("--tape", action="append", default=[], metavar="NAME=PATH",
                        help="extra open-loop arm from a pool-format tape")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunk", type=int, default=5000, help="games per Rust batch call")
    args = parser.parse_args()

    extra: dict[str, tuple[Path, int]] = {}
    for spec in args.router:
        name, _, rest = spec.partition("=")
        path, _, depth = rest.partition(":")
        extra[name] = (Path(path), int(depth) if depth else 2)
    tapes_extra: dict[str, Path] = {}
    for spec in args.tape:
        name, _, path = spec.partition("=")
        tapes_extra[name] = Path(path)
    wanted = args.arms + [n for n in list(extra) + list(tapes_extra) if n not in args.arms]
    arms = build_arms(wanted, args.routes, extra, tapes_extra)
    args.arms = list(arms)
    opponents = {path.stem: load_pool_tape(path) for path in args.opponents}
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    jobs = [(arm, opp, seed, seat)
            for arm in args.arms for opp in opponents for seed in seeds for seat in (0, 1)]
    print(f"{len(args.arms)} arms x {len(opponents)} opponents x {len(seeds)} seeds x 2 seats"
          f" = {len(jobs)} games on {args.threads} Rust threads", flush=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(jobs), args.chunk):
        rows.extend(play_jobs(arms, opponents, jobs[start:start + args.chunk], args.threads))
        rate = len(rows) / (time.perf_counter() - started)
        print(f"  {len(rows)}/{len(jobs)}  {rate:.0f} games/s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output} in {time.perf_counter() - started:.1f}s")
    print(summarize(rows))


if __name__ == "__main__":
    main()
