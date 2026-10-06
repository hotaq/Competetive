"""Merge agent population manifests while rejecting duplicate action IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents: list[dict[str, Any]] = []
    seen: set[int] = set()
    for population_path in args.population:
        population = json.loads(population_path.read_text(encoding="utf-8"))
        for agent in population["agents"]:
            action_id = int(agent["episode_id"])
            if action_id in seen:
                raise ValueError(f"duplicate action ID {action_id}")
            seen.add(action_id)
            agents.append(agent)

    result = {
        "agents": agents,
        "holdout": False,
        "source_populations": [str(path) for path in args.population],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"agents": len(agents), "output": str(args.output)}))


if __name__ == "__main__":
    main()
