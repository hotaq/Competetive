"""Compare two candidate reports on the exact same frozen league games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def games(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {}
    for team in report["teams"]:
        for episode in team["episodes"]:
            key = (int(team["team_id"]), int(episode["episode_id"]))
            indexed[key] = {
                "snapshot_rank": int(team["snapshot_rank"]),
                "team_id": int(team["team_id"]),
                "team_name": team["team_name"],
                "episode_id": int(episode["episode_id"]),
                "margin": float(episode["candidate_margin"]),
                "money": float(episode["candidate_money"]),
            }
    return indexed


def main() -> None:
    args = parse_args()
    base_report = json.loads(args.base.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    if (
        base_report["source"]["league_manifest_sha256"]
        != candidate_report["source"]["league_manifest_sha256"]
    ):
        raise RuntimeError("reports do not use the same league snapshot")
    base_games = games(base_report)
    candidate_games = games(candidate_report)
    if set(base_games) != set(candidate_games):
        raise RuntimeError("reports do not contain the same game set")

    rows = []
    for key in sorted(base_games):
        base = base_games[key]
        candidate = candidate_games[key]
        base_win = base["margin"] > 0
        candidate_win = candidate["margin"] > 0
        rows.append(
            {
                "snapshot_rank": base["snapshot_rank"],
                "team_id": base["team_id"],
                "team_name": base["team_name"],
                "episode_id": base["episode_id"],
                "base_margin": base["margin"],
                "candidate_margin": candidate["margin"],
                "margin_delta": candidate["margin"] - base["margin"],
                "money_delta": candidate["money"] - base["money"],
                "outcome": (
                    "loss_to_win"
                    if not base_win and candidate_win
                    else "win_to_loss"
                    if base_win and not candidate_win
                    else "win_to_win"
                    if base_win
                    else "loss_to_loss"
                ),
            }
        )

    deltas = [row["margin_delta"] for row in rows]
    changed = [row for row in rows if row["margin_delta"] != 0]
    payload = {
        "schema_version": 1,
        "base": str(args.base),
        "candidate": str(args.candidate),
        "summary": {
            "games": len(rows),
            "changed_games": len(changed),
            "loss_to_win": sum(row["outcome"] == "loss_to_win" for row in rows),
            "win_to_loss": sum(row["outcome"] == "win_to_loss" for row in rows),
            "mean_margin_delta": fmean(deltas),
            "median_margin_delta": median(deltas),
            "minimum_margin_delta": min(deltas),
            "maximum_margin_delta": max(deltas),
        },
        "changed_games": sorted(
            changed,
            key=lambda row: (row["snapshot_rank"], row["episode_id"]),
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
