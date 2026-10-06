"""Inspect compact state snapshots from a file agent versus one replay tape."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--steps", default="600,624,648,672,696,708,716,718")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tile_counts(farm: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict):
                kind = str(tile.get("kind", "UNKNOWN"))
                detail = tile.get("crop") or tile.get("animal") or kind
                counts[str(detail)] += 1
    return dict(sorted(counts.items()))


def snapshot(
    observation: dict[str, Any],
    proposed: dict[str, Any],
    sanitized: dict[str, Any],
) -> dict[str, Any]:
    player = int(observation["player"])
    farms = observation["farms"]
    return {
        "step": int(observation["step"]),
        "money": [float(farm["money"]) for farm in farms],
        "candidate_farm": {
            "farmer": farms[player]["farmer"],
            "hands": farms[player]["hands"],
            "tiles": tile_counts(farms[player]),
        },
        "opponent_farm": {
            "farmer": farms[1 - player]["farmer"],
            "hands": farms[1 - player]["hands"],
            "tiles": tile_counts(farms[1 - player]),
        },
        "private": observation.get("private", {}),
        "prices": observation.get("market", {}).get("prices", {}),
        "proposed": proposed,
        "sanitized": sanitized,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = next(
        value
        for value in manifest["episodes"]
        if int(value["episode_id"]) == args.episode_id
    )
    replay = json.loads(
        (args.manifest.parent / f"{args.episode_id}.json").read_text(
            encoding="utf-8"
        )
    )
    module, agent, accepts_config = load_agent(args.agent)
    del module
    config = engine_config(replay)
    engine = Engine(config)
    top1_seat = int(row["top1_seat"] if "top1_seat" in row else row["agent_seat"])
    candidate_seat = 1 - top1_seat
    selected_steps = {int(value) for value in args.steps.split(",")}
    snapshots = []
    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        proposed = normalize_legacy_action(
            run_agent(agent, accepts_config, observation, config)
        )
        sanitized = engine.sanitize_action(candidate_seat, proposed)
        if int(observation["step"]) in selected_steps:
            snapshots.append(snapshot(observation, proposed, sanitized))
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = sanitized
        engine.step(actions)

    final_observation = engine.observation(candidate_seat)
    report = {
        "agent": str(args.agent),
        "episode_id": args.episode_id,
        "candidate_seat": candidate_seat,
        "snapshots": snapshots,
        "final": {
            "step": int(final_observation["step"]),
            "money": [float(farm["money"]) for farm in final_observation["farms"]],
            "private": final_observation.get("private", {}),
            "candidate_tiles": tile_counts(final_observation["farms"][candidate_seat]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
