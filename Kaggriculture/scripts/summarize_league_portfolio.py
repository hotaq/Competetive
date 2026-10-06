"""Measure fixed-policy and per-game oracle coverage for league reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def margin_summary(margins: list[float]) -> dict[str, float | int]:
    ordered = sorted(margins)
    tail_count = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(margins),
        "wins": sum(value > 0 for value in margins),
        "draws": sum(value == 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "win_rate": sum(value > 0 for value in margins) / len(margins),
        "mean_margin": fmean(margins),
        "median_margin": median(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail_count]),
    }


def report_games(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    games = {}
    for team in report["teams"]:
        for episode in team["episodes"]:
            key = (int(team["team_id"]), int(episode["episode_id"]))
            games[key] = {
                "snapshot_rank": int(team["snapshot_rank"]),
                "team_id": int(team["team_id"]),
                "team_name": team["team_name"],
                "episode_id": int(episode["episode_id"]),
                "margin": float(episode["candidate_margin"]),
            }
    return games


def main() -> None:
    args = parse_args()
    if len(args.reports) < 2:
        raise SystemExit("provide at least two league reports")

    loaded = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in args.reports]
    labels = [label for label, _ in loaded]
    if len(set(labels)) != len(labels):
        raise SystemExit("report filenames must have unique stems")
    source_hashes = {report["source"]["league_manifest_sha256"] for _, report in loaded}
    if len(source_hashes) != 1:
        raise RuntimeError("reports use different league snapshots")

    games_by_label = {label: report_games(report) for label, report in loaded}
    reference_keys = set(next(iter(games_by_label.values())))
    for label, games in games_by_label.items():
        if set(games) != reference_keys:
            raise RuntimeError(f"game set mismatch in {label}")

    choices: Counter[str] = Counter()
    oracle_margins: list[float] = []
    uncovered = []
    team_oracle_margins: dict[int, list[float]] = {}
    team_metadata: dict[int, dict[str, Any]] = {}
    for key in sorted(reference_keys):
        candidates = {
            label: games_by_label[label][key]["margin"] for label in labels
        }
        winner = max(labels, key=lambda label: (candidates[label], -labels.index(label)))
        margin = candidates[winner]
        choices[winner] += 1
        oracle_margins.append(margin)
        exemplar = games_by_label[labels[0]][key]
        team_id = exemplar["team_id"]
        team_metadata[team_id] = exemplar
        team_oracle_margins.setdefault(team_id, []).append(margin)
        if margin <= 0:
            uncovered.append({**exemplar, "candidate_margins": candidates})

    per_team = []
    for team_id, margins in sorted(
        team_oracle_margins.items(),
        key=lambda item: team_metadata[item[0]]["snapshot_rank"],
    ):
        metadata = team_metadata[team_id]
        fixed_candidates = {}
        for label in labels:
            candidate_margins = [
                game["margin"]
                for (candidate_team_id, _), game in games_by_label[label].items()
                if candidate_team_id == team_id
            ]
            fixed_candidates[label] = margin_summary(candidate_margins)
        best_fixed = max(
            labels,
            key=lambda label: (
                -fixed_candidates[label]["losses"],
                fixed_candidates[label]["worst_margin"],
                fixed_candidates[label]["mean_margin"],
            ),
        )
        per_team.append(
            {
                "snapshot_rank": metadata["snapshot_rank"],
                "team_id": team_id,
                "team_name": metadata["team_name"],
                "oracle": margin_summary(margins),
                "best_fixed_candidate": best_fixed,
                "fixed_candidates": fixed_candidates,
            }
        )

    payload = {
        "schema_version": 1,
        "league_manifest_sha256": source_hashes.pop(),
        "reports": {
            label: {
                "path": str(path),
                "candidate": report["candidate"],
                "summary": margin_summary(
                    [game["margin"] for game in games_by_label[label].values()]
                ),
            }
            for (label, report), path in zip(loaded, args.reports, strict=True)
        },
        "oracle": margin_summary(oracle_margins),
        "oracle_choice_counts": dict(sorted(choices.items())),
        "uncovered_games": uncovered,
        "per_team": per_team,
        "selection_objective": [
            "minimize_losses",
            "maximize_worst_margin",
            "maximize_win_rate",
            "maximize_mean_margin",
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "oracle": payload["oracle"],
                "oracle_choice_counts": payload["oracle_choice_counts"],
                "uncovered_games": len(uncovered),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
