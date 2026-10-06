"""Merge disjoint replay-population evaluation matrices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(episode["candidate_margin"]) for episode in episodes]
    money = [float(episode["candidate_money"]) for episode in episodes]
    ordered = sorted(margins)
    tail_count = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(episodes),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "win_rate": sum(margin > 0 for margin in margins) / len(margins),
        "mean_margin": fmean(margins),
        "median_margin": median(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail_count]),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "fallback_components": sum(
            int(episode.get("fallback_components", 0)) for episode in episodes
        ),
        "normalized_steps": sum(
            int(episode.get("normalized_steps", 0)) for episode in episodes
        ),
        "sanitized_steps": sum(
            int(episode.get("sanitized_steps", 0)) for episode in episodes
        ),
    }


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    population = reports[0]["population"]
    agents: dict[int, dict[str, Any]] = {}
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_contexts: dict[int, set[int]] = defaultdict(set)

    for report in reports:
        if report["population"] != population:
            raise ValueError("population manifests differ")
        for ranking in report["rankings"]:
            action_id = int(ranking["agent"]["episode_id"])
            if action_id in agents and ranking["agent"] != agents[action_id]:
                raise ValueError(f"agent metadata differs for {action_id}")
            agents[action_id] = ranking["agent"]
            for episode in ranking["episodes"]:
                context_id = int(episode["episode_id"])
                if context_id in seen_contexts[action_id]:
                    raise ValueError(
                        f"duplicate context {context_id} for action {action_id}"
                    )
                seen_contexts[action_id].add(context_id)
                episodes[action_id].append(episode)

    rankings = [
        {
            "agent": agents[action_id],
            "episodes": sorted(
                episodes[action_id], key=lambda episode: int(episode["episode_id"])
            ),
            "summary": summarize(episodes[action_id]),
        }
        for action_id in agents
    ]
    rankings.sort(
        key=lambda row: (
            -int(row["summary"]["wins"]),
            -float(row["summary"]["mean_margin"]),
            int(row["agent"]["episode_id"]),
        )
    )
    output = {
        "population": population,
        "evaluation_manifests": [
            value
            for report in reports
            for value in report.get("evaluation_manifests", [])
        ],
        "exclude_source_episode": any(
            bool(report.get("exclude_source_episode")) for report in reports
        ),
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "actions": len(rankings),
                "contexts": len(rankings[0]["episodes"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
