"""Reduce named live episodes to compact tapes carrying both seats' money trajectory.

Live replays are ~32 MB each, so every replay is downloaded, reduced, and discarded
before the next. What is kept per episode: the seed, both action tapes, and the
per-step money series for both seats — enough to locate where a lost game diverged
without re-simulating it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from download_top1_replays import replay_bytes
from kaggle.api.kaggle_api_extended import KaggleApi

TEAM_ID = 16621481
TEAM_NAME = "zero waste"


def our_seat(replay: dict[str, Any]) -> int | None:
    names = replay.get("info", {}).get("TeamNames", [])
    wanted = " ".join(TEAM_NAME.casefold().split())
    for seat, name in enumerate(names):
        if " ".join(str(name).casefold().split()) == wanted:
            return seat
    return None


def reduce_replay(replay: dict[str, Any], episode_id: int) -> dict[str, Any] | None:
    seat = our_seat(replay)
    if seat is None:
        return None
    steps = replay["steps"]
    money: list[list[float]] = [[], []]
    actions: list[list[Any]] = [[], []]
    for step in steps:
        observation = step[0].get("observation") or {}
        farms = observation.get("farms") or []
        for player in (0, 1):
            actions[player].append(step[player].get("action"))
            farm = farms[player] if player < len(farms) else {}
            money[player].append(float(farm.get("money", "nan")))
    return {
        "episode": episode_id,
        "seed": replay["info"].get("seed"),
        "teams": replay["info"].get("TeamNames"),
        "our_seat": seat,
        "rewards": replay.get("rewards"),
        "our_money_final": replay["rewards"][seat],
        "opponent_money_final": replay["rewards"][1 - seat],
        "length": len(steps),
        "money": {"ours": money[seat], "theirs": money[1 - seat]},
        "actions": {"ours": actions[seat], "theirs": actions[1 - seat]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode_id in args.episode:
        destination = args.output_dir / f"ep{episode_id}.json"
        if destination.exists():
            print(f"{episode_id}: already harvested")
            continue
        replay = json.loads(replay_bytes(api, episode_id))
        payload = reduce_replay(replay, episode_id)
        del replay
        if payload is None:
            print(f"{episode_id}: our team not in this replay")
            continue
        destination.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"{episode_id}: seed {payload['seed']} seat {payload['our_seat']} "
              f"{payload['our_money_final']:,.0f} vs {payload['opponent_money_final']:,.0f} "
              f"-> {destination.name}")


if __name__ == "__main__":
    main()
