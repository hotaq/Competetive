"""Filter a replay-tape population to an exact compatible action prefix."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--base-episode-id", type=int, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"trace_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load replay tape {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def resolve_path(population_path: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    sibling = population_path.parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    rows = population["agents"]
    by_id = {int(row["episode_id"]): row for row in rows}
    if args.base_episode_id not in by_id:
        raise ValueError(f"base episode {args.base_episode_id} is not in population")

    base_path = resolve_path(
        args.population, by_id[args.base_episode_id]["agent_file"]
    )
    base_prefix = load_trace(base_path)[: args.branch_step]
    compatible = []
    for row in rows:
        path = resolve_path(args.population, row["agent_file"])
        if load_trace(path)[: args.branch_step] == base_prefix:
            compatible.append(row)

    filtered = {
        **population,
        "agents": compatible,
        "compatibility_filter": {
            "source_population": str(args.population),
            "base_episode_id": args.base_episode_id,
            "branch_step": args.branch_step,
            "source_agents": len(rows),
            "compatible_agents": len(compatible),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "compatible_agents": len(compatible),
                "episode_ids": [int(row["episode_id"]) for row in compatible],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
