"""Play closed-loop router arms against fixed opponents; one row per (arm, opponent, seed, seat).

Unlike `evaluate_shop_router.py` (tape vs tape, routing read off afterwards), this drives the
actual policy object every step, so the routed arm is the thing that would be uploaded.
See `analysis/v96_shop_router_20260905/PREREGISTRATION.md`.

Arms (observation -> action):
  base / one / two   ShopRouterV1 with max_shops 0 / 1 / 2; routes from
                     artifacts/shop_router_v1/yusuke_routes.json unless --routes is given
  live               the per-seat Yusuke tapes inside live submission 56016840 (phased_router_v2)
Opponents:
  --opponents        pool-format tapes: actions[k+1] is the action taken at step k
  --agent-opponents  self-contained file agents exposing agent(observation, configuration)
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import statistics as st
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action
from kaggriculture_rs import Engine
from play_against_tape import clean

from kaggriculture_rl.policy.idle_hand_overlay import IdleHandOverlay
from kaggriculture_rl.policy.late_game_manager import LateGameHandover
from kaggriculture_rl.policy.phased_router_v2 import YUSUKE_SCHEDULES, DustaSchedule, load_schedule
from kaggriculture_rl.policy.shop_router_v1 import DEFAULT_ROUTES_PATH, ShopRouterV1, load_routes

FIRST_SHOP_STEP = 72
SECOND_SHOP_STEP = 144
STEPS = 719
ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}

_ARMS: dict[str, Any] = {}
_OPPONENTS: dict[str, Any] = {}
_ROUTES_PATH: Path = DEFAULT_ROUTES_PATH
_EXTRA_ROUTERS: dict[str, tuple[str, int]] = {}


def build_arm(name: str) -> Any:
    if name in _EXTRA_ROUTERS:
        path, depth = _EXTRA_ROUTERS[name]
        router = ShopRouterV1(load_routes(Path(path)), max_shops=depth)
        if name.endswith("+idle"):
            return IdleHandOverlay(router.agent)
        if "+late" in name:
            start = int(name.rsplit("+late", 1)[1] or 432)
            return LateGameHandover(router.agent, start_step=start)
        return router
    if name in ("base", "one", "two"):
        depth = {"base": 0, "one": 1, "two": 2}[name]
        return ShopRouterV1(load_routes(_ROUTES_PATH), max_shops=depth)
    if name == "live":
        return DustaSchedule({seat: load_schedule(path) for seat, path in YUSUKE_SCHEDULES.items()})
    raise ValueError("unknown arm " + name)


class TapeOpponent:
    def __init__(self, path: Path) -> None:
        self.actions = json.loads(Path(path).read_text(encoding="utf-8"))["actions"]
        if len(self.actions) < STEPS + 1:
            raise ValueError("tape too short: " + str(path))

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Any:
        del configuration
        return self.actions[int(observation["step"]) + 1]


class FileAgentOpponent:
    def __init__(self, path: Path) -> None:
        namespace = runpy.run_path(str(path))
        self._agent = namespace["agent"]

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Any:
        return self._agent(observation, configuration)


def _init(arms: list[str], tapes: dict[str, str], agents: dict[str, str], routes_path: str,
          sanitize: bool = True, extra_routers: dict[str, tuple[str, int]] | None = None) -> None:
    global _ROUTES_PATH, _SANITIZE
    _ROUTES_PATH = Path(routes_path)
    _SANITIZE = sanitize
    _EXTRA_ROUTERS.update(extra_routers or {})
    for name in arms:
        _ARMS[name] = build_arm(name)
    for name, path in tapes.items():
        _OPPONENTS[name] = TapeOpponent(Path(path))
    for name, path in agents.items():
        _OPPONENTS[name] = FileAgentOpponent(Path(path))


_SANITIZE = True


def legalize(engine: Engine, seat: int, action: Any) -> Any:
    """Canonicalise the schema (the live env defaults PICKUP/PLACE count to 1).

    With `_SANITIZE` (the V94-V96 default) the executor's fail-closed `sanitize_action` also
    drops orders the executor would reject. That is stricter than the official env, which fills
    market orders per unit and no-ops only what is impossible; `--no-sanitize` reproduces the
    official env (verified against kaggle_environments in bindings/python/tests/test_fastplay.py).
    """
    canonical = normalize_legacy_action(clean(action))
    return engine.sanitize_action(seat, canonical) if _SANITIZE else canonical


def play(arm: str, opponent: str, seed: int, seat: int) -> dict[str, Any]:
    ours, theirs = _ARMS[arm], _OPPONENTS[opponent]
    engine = Engine({**ENGINE_CONFIG, "seed": seed})
    shops: list[list[str]] = [[], []]
    route_at_second: int | None = None
    route_final: int | None = None
    started = time.perf_counter()
    for step in range(STEPS):
        mine = engine.observation(seat)
        if step == FIRST_SHOP_STEP:
            shops[0] = list(mine["town"].get("unlocked_shops", []))
        elif step == SECOND_SHOP_STEP:
            shops[1] = list(mine["town"].get("unlocked_shops", []))
            if hasattr(ours, "route_index"):
                route_at_second = ours.route_index(mine)
        if step == STEPS - 1 and hasattr(ours, "route_index"):
            route_final = ours.route_index(mine)
        actions: list[Any] = [None, None]
        actions[seat] = legalize(engine, seat, ours.agent(mine, None))
        their_observation = engine.observation(1 - seat)
        actions[1 - seat] = legalize(engine, 1 - seat, theirs.agent(their_observation, None))
        engine.step_typed({"schema_version": 1, "actions": actions})
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    margin = money[seat] - money[1 - seat]
    return {
        "arm": arm, "opponent": opponent, "seed": seed, "seat": seat,
        "money": money[seat], "opponent_money": money[1 - seat], "margin": margin,
        "point": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "shop1": shops[0][0] if shops[0] else None,
        "shop2": shops[1][1] if len(shops[1]) > 1 else None,
        "route_at_second": route_at_second, "route_final": route_final,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _work(job: tuple[str, str, int, int]) -> dict[str, Any]:
    return play(*job)


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
    parser.add_argument("--opponents", type=Path, nargs="*", default=[])
    parser.add_argument("--agent-opponents", type=Path, nargs="*", default=[])
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES_PATH)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no-sanitize", action="store_true",
                        help="skip executor sanitize_action; matches the official env")
    parser.add_argument("--router", action="append", default=[], metavar="NAME=PATH[:DEPTH]",
                        help="extra router arm from another routes file (depth defaults to 2);"
                             " a NAME ending in '+idle' wraps it in IdleHandOverlay")
    args = parser.parse_args()
    extra: dict[str, tuple[str, int]] = {}
    for spec in args.router:
        name, _, rest = spec.partition("=")
        path, _, depth = rest.partition(":")
        extra[name] = (path, int(depth) if depth else 2)
    args.arms = args.arms + [n for n in extra if n not in args.arms]

    tapes = {path.stem: str(path) for path in args.opponents}
    agents = {"agent:" + path.parent.name: str(path) for path in args.agent_opponents}
    if not tapes and not agents:
        raise SystemExit("give at least one opponent")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    opponents = list(tapes) + list(agents)
    jobs = [
        (arm, opp, seed, seat)
        for arm in args.arms for opp in opponents for seed in seeds for seat in (0, 1)
    ]
    print(f"{len(args.arms)} arms x {len(opponents)} opponents x {len(seeds)} seeds x 2 seats"
          f" = {len(jobs)} games on {args.workers} workers", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.jsonl")
    rows: list[dict[str, Any]] = []
    finished: set[tuple[str, str, int, int]] = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            finished.add((row["arm"], row["opponent"], row["seed"], row["seat"]))
    todo = [job for job in jobs if job not in finished]
    print(f"resuming {len(rows)} rows, {len(todo)} to play", flush=True)

    started = time.perf_counter()
    with partial.open("a", encoding="utf-8") as sink, ProcessPoolExecutor(
        args.workers, initializer=_init,
        initargs=(args.arms, tapes, agents, str(args.routes), not args.no_sanitize, extra),
    ) as pool:
        for done, row in enumerate(pool.map(_work, todo, chunksize=4), 1):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            if done % 200 == 0 or done == len(todo):
                sink.flush()
                rate = done / (time.perf_counter() - started)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"  {done}/{len(todo)}  {rate:.1f} games/s  eta {eta:.1f} min", flush=True)

    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
