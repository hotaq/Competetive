"""Build a paired, post-unblinding diagnostic from two league reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_games(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for team in report["teams"]:
        for episode in team["episodes"]:
            key = (int(team["team_id"]), int(episode["episode_id"]))
            indexed[key] = {
                "snapshot_rank": int(team["snapshot_rank"]),
                "team_id": int(team["team_id"]),
                "team_name": str(team["team_name"]),
                "episode_id": int(episode["episode_id"]),
                "top1_seat": int(episode["top1_seat"]),
                "margin": float(episode["candidate_margin"]),
                "money": float(episode["candidate_money"]),
            }
    return indexed


def outcome(margin: float) -> str:
    if margin > 0:
        return "win"
    if margin < 0:
        return "loss"
    return "draw"


def metrics(rows: Iterable[dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected = list(rows)
    margins = [float(row[f"{prefix}_margin"]) for row in selected]
    money = [float(row[f"{prefix}_money"]) for row in selected]
    ordered = sorted(margins)
    tail_count = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(selected),
        "wins": sum(value > 0 for value in margins),
        "draws": sum(value == 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "win_rate": sum(value > 0 for value in margins) / len(margins),
        "mean_margin": fmean(margins),
        "cvar10_margin": fmean(ordered[:tail_count]),
        "worst_margin": min(margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: dict[str, int] = defaultdict(int)
    for row in rows:
        transitions[str(row["outcome_transition"])] += 1
    margin_deltas = [float(row["margin_delta"]) for row in rows]
    money_deltas = [float(row["money_delta"]) for row in rows]
    base = metrics(rows, "base")
    candidate = metrics(rows, "candidate")
    return {
        "base": base,
        "candidate": candidate,
        "delta": {
            "wins": candidate["wins"] - base["wins"],
            "losses": candidate["losses"] - base["losses"],
            "win_rate": candidate["win_rate"] - base["win_rate"],
            "mean_margin": candidate["mean_margin"] - base["mean_margin"],
            "cvar10_margin": candidate["cvar10_margin"] - base["cvar10_margin"],
            "worst_margin": candidate["worst_margin"] - base["worst_margin"],
            "mean_money": candidate["mean_money"] - base["mean_money"],
            "worst_money": candidate["worst_money"] - base["worst_money"],
        },
        "outcome_transitions": dict(sorted(transitions.items())),
        "loss_to_win": transitions["loss_to_win"],
        "win_to_loss": transitions["win_to_loss"],
        "net_win_flips": transitions["loss_to_win"] - transitions["win_to_loss"],
        "mean_paired_margin_delta": fmean(margin_deltas),
        "mean_paired_money_delta": fmean(money_deltas),
        "minimum_paired_margin_delta": min(margin_deltas),
        "maximum_paired_margin_delta": max(margin_deltas),
    }


def main() -> None:
    args = parse_args()
    base_report = json.loads(args.base.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    base_hash = base_report["source"]["league_manifest_sha256"]
    candidate_hash = candidate_report["source"]["league_manifest_sha256"]
    if base_hash != candidate_hash:
        raise RuntimeError("reports do not use the same league manifest")

    base_games = index_games(base_report)
    candidate_games = index_games(candidate_report)
    if set(base_games) != set(candidate_games):
        raise RuntimeError("reports do not contain the same game set")

    rows: list[dict[str, Any]] = []
    for key in sorted(base_games):
        base = base_games[key]
        candidate = candidate_games[key]
        if base["top1_seat"] != candidate["top1_seat"]:
            raise RuntimeError(f"seat mismatch for {key}")
        base_outcome = outcome(base["margin"])
        candidate_outcome = outcome(candidate["margin"])
        rows.append(
            {
                "snapshot_rank": base["snapshot_rank"],
                "team_id": base["team_id"],
                "team_name": base["team_name"],
                "episode_id": base["episode_id"],
                "opponent_top1_seat": base["top1_seat"],
                "candidate_seat": 1 - base["top1_seat"],
                "base_margin": base["margin"],
                "candidate_margin": candidate["margin"],
                "margin_delta": candidate["margin"] - base["margin"],
                "base_money": base["money"],
                "candidate_money": candidate["money"],
                "money_delta": candidate["money"] - base["money"],
                "base_outcome": base_outcome,
                "candidate_outcome": candidate_outcome,
                "outcome_transition": f"{base_outcome}_to_{candidate_outcome}",
            }
        )

    by_seat = []
    for top1_seat in sorted({int(row["opponent_top1_seat"]) for row in rows}):
        selected = [row for row in rows if row["opponent_top1_seat"] == top1_seat]
        by_seat.append(
            {
                "opponent_top1_seat": top1_seat,
                "candidate_seat": 1 - top1_seat,
                **paired_summary(selected),
            }
        )

    by_team = []
    team_keys = sorted(
        {(row["snapshot_rank"], row["team_id"], row["team_name"]) for row in rows}
    )
    for snapshot_rank, team_id, team_name in team_keys:
        selected = [row for row in rows if row["team_id"] == team_id]
        by_team.append(
            {
                "snapshot_rank": snapshot_rank,
                "team_id": team_id,
                "team_name": team_name,
                **paired_summary(selected),
            }
        )

    flips = [
        row for row in rows if row["base_outcome"] != row["candidate_outcome"]
    ]
    payload = {
        "schema_version": 1,
        "analysis_status": "post_unblinding_diagnostic_only",
        "promotion_eligible": False,
        "limitation": args.note,
        "source": {
            "base_report": str(args.base),
            "base_report_sha256": sha256(args.base),
            "candidate_report": str(args.candidate),
            "candidate_report_sha256": sha256(args.candidate),
            "league_manifest_sha256": base_hash,
        },
        "overall": paired_summary(rows),
        "by_seat": by_seat,
        "by_team": by_team,
        "outcome_flips": sorted(
            flips, key=lambda row: (row["snapshot_rank"], row["episode_id"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
