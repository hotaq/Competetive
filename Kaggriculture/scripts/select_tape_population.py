"""Select an explicit replay-tape subpopulation by source episode id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    requested = list(dict.fromkeys(args.episode_id))
    by_id = {int(row["episode_id"]): row for row in population["agents"]}
    missing = [episode_id for episode_id in requested if episode_id not in by_id]
    if missing:
        raise ValueError(f"episodes are not in population: {missing}")

    selected = {
        **population,
        "agents": [by_id[episode_id] for episode_id in requested],
        "selection": {
            "source_population": str(args.population),
            "episode_ids": requested,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected["selection"], indent=2))


if __name__ == "__main__":
    main()
