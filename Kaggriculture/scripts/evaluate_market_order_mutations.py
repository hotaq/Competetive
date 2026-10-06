"""Evaluate one targeted market-order quantity mutation on a file agent."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    play_against_tape,
    run_agent,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--quantities", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module, base_agent, accepts_config = load_agent(args.agent)
    del module
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
    configurations = []
    for quantity in (int(value) for value in args.quantities.split(",")):

        def agent(
            observation: dict[str, Any],
            configuration: dict[str, Any],
            *,
            _quantity: int = quantity,
        ) -> dict[str, Any]:
            action = copy.deepcopy(
                run_agent(base_agent, accepts_config, observation, configuration)
            )
            if int(observation.get("step", 0) or 0) != args.step:
                return action
            output = []
            for order in action.get("market", []):
                if (
                    order
                    and order[0] == args.operation
                    and len(order) >= 2
                    and order[1] == args.item
                ):
                    if _quantity > 0:
                        output.append([*order[:2], _quantity])
                else:
                    output.append(order)
            action["market"] = output
            return action

        result = play_against_tape(
            replay,
            episode_id=args.episode_id,
            top1_seat=int(row["top1_seat"]),
            agent=agent,
            accepts_config=True,
        )
        configurations.append(
            {
                "quantity": quantity,
                "summary": summarize([result]),
                "episode": asdict(result),
            }
        )

    report = {
        "agent": str(args.agent),
        "episode_id": args.episode_id,
        "step": args.step,
        "operation": args.operation,
        "item": args.item,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(configurations, key=lambda value: value["summary"]["mean_margin"])
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
