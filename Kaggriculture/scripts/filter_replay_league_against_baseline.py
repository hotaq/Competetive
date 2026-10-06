#!/usr/bin/env python3
"""Write per-team replay manifests excluding episodes in a baseline league."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-name", default="manifest_fresh.json")
    parser.add_argument("--index-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    teams: list[dict[str, Any]] = []
    for candidate_manifest in sorted(args.candidate_root.glob("r*/manifest.json")):
        team = candidate_manifest.parent.name
        baseline_manifest = args.baseline_root / team / "manifest.json"
        baseline = json.loads(baseline_manifest.read_text())
        candidate = json.loads(candidate_manifest.read_text())
        baseline_ids = {int(row["episode_id"]) for row in baseline["episodes"]}
        fresh = [row for row in candidate["episodes"] if int(row["episode_id"]) not in baseline_ids]
        output = {
            **candidate,
            "episodes": fresh,
            "filter": {
                "baseline_manifest": str(baseline_manifest),
                "baseline_episode_ids": sorted(baseline_ids),
                "candidate_episodes": len(candidate["episodes"]),
                "excluded_overlap": len(candidate["episodes"]) - len(fresh),
                "fresh_episodes": len(fresh),
            },
        }
        output_path = candidate_manifest.parent / args.output_name
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        teams.append(
            {
                "team": team,
                "manifest": str(output_path),
                "fresh_episodes": len(fresh),
                "episode_ids": [int(row["episode_id"]) for row in fresh],
            }
        )
    payload = {
        "baseline_root": str(args.baseline_root),
        "candidate_root": str(args.candidate_root),
        "contexts": sum(team["fresh_episodes"] for team in teams),
        "minimum_per_team": min(team["fresh_episodes"] for team in teams),
        "maximum_per_team": max(team["fresh_episodes"] for team in teams),
        "teams": teams,
    }
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
