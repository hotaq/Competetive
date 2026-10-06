"""Aggregate disjoint raw-agent replay evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_file_agent_vs_replay import FileAgentTapeResult, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes: list[dict[str, object]] = []
    seen: set[int] = set()
    for path in args.input:
        report = json.loads(path.read_text(encoding="utf-8"))
        for episode in report["episodes"]:
            episode_id = int(episode["episode_id"])
            if episode_id in seen:
                raise ValueError(f"duplicate episode {episode_id}")
            seen.add(episode_id)
            episodes.append(episode)
    results = [FileAgentTapeResult(**episode) for episode in episodes]
    output = {
        "inputs": [str(path) for path in args.input],
        "episodes": episodes,
        "summary": summarize(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
