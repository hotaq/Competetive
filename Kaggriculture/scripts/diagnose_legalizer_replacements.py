"""Explain which replay-tape commands an observation-only legalizer rejects."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.primitive_legalizer import legalize_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def command_key(command: Any) -> str:
    return json.dumps(command, separators=(",", ":"))


def actor_context(observation: dict[str, Any], actor: int) -> dict[str, Any]:
    player = int(observation["player"])
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    inventories = observation.get("private", {}).get("inventories", [])
    if actor >= len(positions):
        return {"missing_actor": True}
    x, y = positions[actor]
    return {
        "position": [x, y],
        "tile": farm["tiles"][y][x],
        "inventory": inventories[actor] if actor < len(inventories) else {},
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["episodes"][: args.limit or None]
    _, agent, accepts_config = load_agent(args.agent_file)
    counts: Counter[tuple[str, int, str, str]] = Counter()
    examples: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        episode_id = int(row["episode_id"])
        top1_seat = int(row["top1_seat"])
        candidate_seat = 1 - top1_seat
        replay = json.loads((args.manifest.parent / f"{episode_id}.json").read_text())
        config = engine_config(replay)
        engine = Engine(config)
        for frame in replay["steps"][1:]:
            observation = engine.observation(candidate_seat)
            step = int(observation.get("step", 0))
            proposed = normalize_legacy_action(
                run_agent(agent, accepts_config, observation, config)
            )
            legalized = legalize_action(observation, proposed, config)

            proposed_units = [proposed["farmer"], *proposed["hands"]]
            legal_units = [legalized["farmer"], *legalized["hands"]]
            for actor, (before, after) in enumerate(
                zip_longest(proposed_units, legal_units, fillvalue=None)
            ):
                if before == after:
                    continue
                key = ("unit", step, command_key(before), command_key(after))
                counts[key] += 1
                if len(examples[key]) < 3:
                    examples[key].append(
                        {"episode_id": episode_id, **actor_context(observation, actor)}
                    )

            before_market = [command_key(order) for order in proposed["market"]]
            after_market = [command_key(order) for order in legalized["market"]]
            remaining = Counter(after_market)
            for before in before_market:
                if remaining[before]:
                    remaining[before] -= 1
                    continue
                key = ("market", step, before, "<dropped>")
                counts[key] += 1
                if len(examples[key]) < 3:
                    player_farm = observation["farms"][candidate_seat]
                    examples[key].append(
                        {
                            "episode_id": episode_id,
                            "money": player_farm.get("money"),
                            "hires_today": player_farm.get("hires_today"),
                            "shed": observation.get("private", {}).get("shed", {}),
                        }
                    )

            actions = [frame_action(frame, 0), frame_action(frame, 1)]
            actions[candidate_seat] = legalized
            engine.step(actions)

    ranked = [
        {
            "component": key[0],
            "step": key[1],
            "proposed": json.loads(key[2]) if key[2] != "<dropped>" else key[2],
            "legalized": json.loads(key[3]) if key[3] != "<dropped>" else key[3],
            "count": count,
            "examples": examples[key],
        }
        for key, count in counts.most_common()
    ]
    payload = {
        "agent_file": str(args.agent_file),
        "manifest": str(args.manifest),
        "episodes": len(rows),
        "replacement_events": sum(counts.values()),
        "by_component": dict(Counter(key[0] for key in counts.elements())),
        "replacements": ranked,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "replacements"}
    print(json.dumps(summary, indent=2))
    print(json.dumps(ranked[:30], indent=2))


if __name__ == "__main__":
    main()
