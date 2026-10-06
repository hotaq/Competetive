"""Evaluate one Kaggle agent file against a frozen replay league."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, play_against_tape, summarize
from evaluate_replay_tape import validate_original_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument("--league-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-teams", type=int, default=0)
    parser.add_argument("--limit-episodes-per-team", type=int, default=0)
    parser.add_argument("--team-manifest-name", default="manifest.json")
    parser.add_argument("--phase-boundary", type=int, default=120)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    league_path = args.league_manifest.resolve()
    league = json.loads(league_path.read_text(encoding="utf-8"))
    team_rows = list(league["teams"])
    if args.limit_teams > 0:
        team_rows = team_rows[: args.limit_teams]
    if not team_rows:
        raise SystemExit("league manifest contains no selected teams")

    _, agent, accepts_config = load_agent(args.agent_file)
    all_results = []
    team_reports: list[dict[str, Any]] = []
    verified_hashes = 0

    for team in team_rows:
        team_dir = league_path.parent / team["directory"]
        team_manifest_path = team_dir / args.team_manifest_name
        team_manifest = json.loads(team_manifest_path.read_text(encoding="utf-8"))
        episodes = list(team_manifest["episodes"])
        if args.limit_episodes_per_team > 0:
            episodes = episodes[: args.limit_episodes_per_team]
        if not episodes:
            raise RuntimeError(f"no episodes selected for rank {team['snapshot_rank']}")

        results = []
        for row in episodes:
            replay_path = team_dir / f"{row['episode_id']}.json"
            digest = file_sha256(replay_path)
            if digest != row["sha256"]:
                raise RuntimeError(f"replay hash mismatch: {replay_path}")
            verified_hashes += 1
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            transitions, rewards = validate_original_replay(replay)
            if transitions != len(replay["steps"]) - 1:
                raise RuntimeError(
                    f"replay transition validation failed: {row['episode_id']}"
                )
            if rewards != tuple(float(value) for value in replay["rewards"]):
                raise RuntimeError(f"replay reward validation failed: {row['episode_id']}")

            opponent_seat = int(
                row["agent_seat"] if "agent_seat" in row else row["top1_seat"]
            )
            result = play_against_tape(
                replay,
                episode_id=int(row["episode_id"]),
                top1_seat=opponent_seat,
                agent=agent,
                accepts_config=accepts_config,
                phase_boundary=args.phase_boundary,
            )
            results.append(result)
            all_results.append(result)

        team_summary = summarize(results)
        team_reports.append(
            {
                "snapshot_rank": team["snapshot_rank"],
                "team_id": team["team_id"],
                "team_name": team["team_name"],
                "score": team["score"],
                "submission_id": team["submission"]["submission_id"],
                "summary": team_summary,
                "episodes": [asdict(result) for result in results],
            }
        )
        print(
            json.dumps(
                {
                    "rank": team["snapshot_rank"],
                    "team": team["team_name"],
                    "wins": team_summary["wins"],
                    "games": team_summary["games"],
                    "mean_margin": team_summary["mean_margin"],
                    "worst_margin": team_summary["worst_margin"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    overall = summarize(all_results)
    per_team_win_rates = [row["summary"]["win_rate"] for row in team_reports]
    per_team_mean_margins = [row["summary"]["mean_margin"] for row in team_reports]
    robust = {
        "teams": len(team_reports),
        "teams_swept": sum(
            row["summary"]["wins"] == row["summary"]["games"]
            for row in team_reports
        ),
        "teams_with_any_loss": sum(row["summary"]["losses"] > 0 for row in team_reports),
        "teams_with_majority_wins": sum(
            row["summary"]["wins"] > row["summary"]["games"] / 2
            for row in team_reports
        ),
        "worst_team_win_rate": min(per_team_win_rates),
        "mean_team_win_rate": fmean(per_team_win_rates),
        "worst_team_mean_margin": min(per_team_mean_margins),
        "mean_team_mean_margin": fmean(per_team_mean_margins),
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": {
            "agent_file": str(args.agent_file),
            "sha256": file_sha256(args.agent_file),
        },
        "source": {
            "league_manifest": str(args.league_manifest),
            "league_manifest_sha256": file_sha256(league_path),
            "leaderboard_checked_at": league["leaderboard_checked_at"],
            "requested_teams": league["requested_teams"],
            "requested_episodes_per_team": league["requested_episodes_per_team"],
        },
        "validation": {
            "verified_replay_hashes": verified_hashes,
            "all_original_replays_exact": True,
            "candidate_actions_sanitized": True,
            "phase_boundary": args.phase_boundary,
        },
        "overall": overall,
        "robust": robust,
        "teams": team_reports,
        "limitation": (
            "Counterfactual candidate actions change shared state, so later recorded "
            "opponent actions may become stale or illegal and are sanitized by the engine."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"overall": overall, "robust": robust}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
