"""Play every candidate tape against fixed opponent tapes and record the revealed shops.

Emits one row per (candidate, opponent, seed, seat) with the margin and the first two
shops revealed in that game, which is everything the V94 gates need. See
`analysis/v94_shop_router_20260830/PREREGISTRATION.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action
from kaggriculture_rs import Engine
from play_against_tape import clean

FIRST_SHOP_STEP = 72
SECOND_SHOP_STEP = 144

_PATHS: dict[str, str] = {}
_TAPES: dict[str, list[Any]] = {}
_PINNED: set[str] = set()
_CACHE_LIMIT = 128


def load_tape(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))["actions"]
    return [normalize_legacy_action(clean(action)) for action in raw]


def _init(paths: dict[str, str], pinned: tuple[str, ...] = ()) -> None:
    _PATHS.update(paths)
    _PINNED.update(pinned)


def tape(name: str) -> list[Any]:
    """Load on first use. Jobs are candidate-major, so a worker holds only a few at once."""
    cached = _TAPES.get(name)
    if cached is None:
        if len(_TAPES) >= _CACHE_LIMIT:
            for key in [k for k in _TAPES if k not in _PINNED]:
                del _TAPES[key]
        cached = _TAPES[name] = load_tape(Path(_PATHS[name]))
    return cached


def play(candidate: str, opponent: str, seed: int, seat: int) -> dict[str, Any]:
    ours, theirs = tape(candidate), tape(opponent)
    engine = Engine({"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005})
    shops: list[list[str]] = [[], []]
    for step in range(719):
        index = step + 1
        if index >= len(ours) or index >= len(theirs):
            break
        if step == FIRST_SHOP_STEP:
            shops[0] = list(engine.observation(seat)["town"].get("unlocked_shops", []))
        elif step == SECOND_SHOP_STEP:
            shops[1] = list(engine.observation(seat)["town"].get("unlocked_shops", []))
        actions: list[Any] = [None, None]
        actions[seat] = engine.sanitize_action(seat, ours[index])
        actions[1 - seat] = engine.sanitize_action(1 - seat, theirs[index])
        engine.step_typed({"schema_version": 1, "actions": actions})
    money = [farm["money"] for farm in engine.state["farms"]]
    margin = money[seat] - money[1 - seat]
    return {
        "candidate": candidate, "opponent": opponent, "seed": seed, "seat": seat,
        "money": money[seat], "opponent_money": money[1 - seat], "margin": margin,
        "point": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "shop1": shops[0][0] if shops[0] else None,
        "shop2": shops[1][1] if len(shops[1]) > 1 else None,
    }


def _work(job: tuple[str, str, int, int]) -> dict[str, Any]:
    return play(*job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    def key(path: Path) -> str:
        return f"{path.parent.parent.name}/{path.stem}"

    paths = {key(p): str(p) for p in list(args.candidates) + list(args.opponents)}
    candidates = [key(p) for p in args.candidates]
    opponents = [key(p) for p in args.opponents]
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    jobs = [(c, o, s, seat)
            for c in candidates for o in opponents for s in seeds for seat in (0, 1)]
    print(f"{len(candidates)} candidates x {len(opponents)} opponents x {len(seeds)} seeds "
          f"x 2 seats = {len(jobs)} games on {args.workers} workers")

    # Results are appended per game so a kill costs only the games still in flight, and a
    # rerun with the same arguments resumes instead of replaying what is already recorded.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.jsonl")
    rows: list[dict[str, Any]] = []
    finished: set[tuple[str, str, int, int]] = set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a torn final line from an interrupted run
            rows.append(row)
            finished.add((row["candidate"], row["opponent"], row["seed"], row["seat"]))
        print(f"resuming: {len(rows)} games already recorded")
    jobs = [job for job in jobs if job not in finished]
    print(f"  {len(jobs)} games left to play")

    with partial.open("a", encoding="utf-8") as sink, \
            ProcessPoolExecutor(args.workers, initializer=_init, initargs=(paths, tuple(opponents))) as pool:
        for done, row in enumerate(pool.map(_work, jobs, chunksize=4), 1):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            if done % 200 == 0 or done == len(jobs):
                sink.flush()
                print(f"  {done}/{len(jobs)}", flush=True)

    args.output.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
