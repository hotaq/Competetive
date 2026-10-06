# ruff: noqa: E501
"""V115 strong-panel screen: play tape arms + router reference arms vs a fixed opponent set, verbatim.

Per-game JSONL rows (arm, opponent, seed, seat, money, opponent_money, margin, point)
so paired money-diff CIs can be computed downstream. Tape-vs-tape self-pairings
(arm stem == opponent stem) are skipped. All play is open-loop verbatim: Tape for
candidates/opponents, ShopRouter for reference route arms. Both seats always.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from kaggriculture_rs import ShopRouter, Tape, play_batch

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}


def load_tape(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Tape(json.dumps(payload["actions"][1:720]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=Path, nargs="*", default=[],
                        help="candidate/opponent tape files (played as Tape)")
    parser.add_argument("--router", action="append", default=[],
                        help="reference arm as NAME=routes.json (played as ShopRouter); repeatable")
    parser.add_argument("--opponents", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    arm_names: list[str] = []
    arm_players: list[Any] = []
    arm_is_tape: list[bool] = []
    for path in args.tapes:
        arm_names.append(path.stem)
        arm_players.append(load_tape(path))
        arm_is_tape.append(True)
    for spec in args.router:
        name, routes_path = spec.split("=", 1)
        arm_names.append(name)
        arm_players.append(ShopRouter(Path(routes_path).read_text(encoding="utf-8"), 2))
        arm_is_tape.append(False)

    opp_names = [p.stem for p in args.opponents]
    opp_players = [load_tape(p) for p in args.opponents]
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    jobs: list[tuple[int, int, int, int]] = []
    for i, name in enumerate(arm_names):
        for o, opp in enumerate(opp_names):
            if arm_is_tape[i] and name == opp:
                continue  # no mirror self-games; excluded from pooled points
            for seed in seeds:
                for seat in (0, 1):
                    jobs.append((seed, seat, i, o))

    offset = len(arm_players)
    allp = arm_players + opp_players
    batch = [(seed, seat, i, offset + o) for (seed, seat, i, o) in jobs]
    results = play_batch(json.dumps(ENGINE_CONFIG), allp, batch, args.threads)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for (seed, seat, i, o), r in zip(jobs, results, strict=True):
            fh.write(json.dumps({
                "arm": arm_names[i],
                "opponent": opp_names[o],
                "seed": seed,
                "seat": seat,
                "money": r["money"][seat],
                "opponent_money": r["money"][1 - seat],
                "margin": r["money"][seat] - r["money"][1 - seat],
                "point": r["point"],
            }) + "\n")
    elapsed = time.perf_counter() - started
    print(f"wrote {args.output}: {len(jobs)} games in {elapsed:.0f}s "
          f"({len(jobs) / max(elapsed, 1e-9):.1f} games/s)", flush=True)


if __name__ == "__main__":
    main()
