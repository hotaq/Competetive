"""Merge compatible population evaluation reports in manifest action order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--allow-extra-actions",
        action="store_true",
        help="ignore input rankings that are absent from the target population",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    action_order = [int(agent["episode_id"]) for agent in population["agents"]]

    rankings: dict[int, dict[str, Any]] = {}
    manifests: list[str] | None = None
    episode_ids: list[int] | None = None
    inputs: list[str] = []
    for input_path in args.input:
        report = json.loads(input_path.read_text(encoding="utf-8"))
        inputs.append(str(input_path))
        current_manifests = [str(path) for path in report["evaluation_manifests"]]
        current_episode_ids = [int(value) for value in report.get("episode_ids", [])]
        if not current_episode_ids and report["rankings"]:
            current_episode_ids = [
                int(episode["episode_id"])
                for episode in report["rankings"][0]["episodes"]
            ]
        if manifests is None:
            manifests = current_manifests
            episode_ids = current_episode_ids
        elif current_manifests != manifests or current_episode_ids != episode_ids:
            raise ValueError(f"evaluation contexts differ in {input_path}")

        for ranking in report["rankings"]:
            action_id = int(ranking["agent"]["episode_id"])
            if args.allow_extra_actions and action_id not in action_order:
                continue
            if action_id in rankings:
                raise ValueError(f"duplicate action {action_id} in {input_path}")
            rankings[action_id] = ranking

    missing = [action_id for action_id in action_order if action_id not in rankings]
    extra = sorted(set(rankings).difference(action_order))
    if missing or extra:
        raise ValueError(f"population mismatch: missing={missing}, extra={extra}")

    output = {
        "population": str(args.population),
        "evaluation_manifests": manifests,
        "episode_ids": episode_ids,
        "inputs": inputs,
        "rankings": [rankings[action_id] for action_id in action_order],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "actions": len(output["rankings"]),
                "contexts": len(episode_ids or []),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
