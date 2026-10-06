"""Freeze a Kaggriculture leaderboard league and download its public replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from download_top1_replays import (
    COMPETITION,
    completed_public_episodes,
    get_any,
    plain,
    replay_bytes,
    scoring_submission,
    top1_seat,
)
from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest, KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=20)
    parser.add_argument("--episodes-per-team", type=int, default=4)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.75)
    parser.add_argument(
        "--frozen-league-manifest",
        type=Path,
        help="reuse exact team and submission identities from an earlier snapshot",
    )
    parser.add_argument(
        "--min-episodes-per-team",
        type=int,
        default=1,
        help="fail instead of publishing an under-covered league snapshot",
    )
    return parser.parse_args()


def leaderboard_teams(api: KaggleApi, limit: int) -> list[dict[str, Any]]:
    request = ApiGetLeaderboardRequest()
    request.competition_name = COMPETITION
    request.page_size = max(20, limit)
    with api.build_kaggle_client() as client:
        response = client.competitions.competition_api_client.get_leaderboard(request)

    by_team: dict[int, dict[str, Any]] = {}
    for source_position, item in enumerate(response.submissions or [], start=1):
        row = plain(item)
        team_id = get_any(row, "teamId", "team_id")
        score = get_any(row, "score", "publicScore", "public_score")
        if team_id is None or score is None:
            continue
        candidate = {
            "source_position": source_position,
            "team_id": int(team_id),
            "team_name": str(get_any(row, "teamName", "team_name", default="")),
            "score": float(score),
            "submission_date": str(
                get_any(row, "submissionDate", "submission_date", default="")
            ),
        }
        previous = by_team.get(candidate["team_id"])
        if previous is None or candidate["score"] > previous["score"]:
            by_team[candidate["team_id"]] = candidate

    ordered = sorted(
        by_team.values(),
        key=lambda row: (-row["score"], row["source_position"], row["team_id"]),
    )[:limit]
    if len(ordered) != limit:
        raise RuntimeError(
            f"Kaggle leaderboard returned {len(ordered)} unique teams; expected {limit}"
        )
    for rank, row in enumerate(ordered, start=1):
        row["snapshot_rank"] = rank
    return ordered


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:48] or "team"


def frozen_teams(
    manifest_path: Path, limit: int
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    resolved = manifest_path.resolve()
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    source_rows = list(manifest["teams"])
    if len(source_rows) < limit:
        raise RuntimeError(
            f"frozen manifest has {len(source_rows)} teams; expected at least {limit}"
        )
    keys = (
        "source_position",
        "snapshot_rank",
        "team_id",
        "team_name",
        "score",
        "submission_date",
    )
    teams = []
    for source in source_rows[:limit]:
        team = {key: source[key] for key in keys}
        team["_frozen_submission"] = dict(source["submission"])
        teams.append(team)
    identity_source = {
        "mode": "frozen_league_manifest",
        "path": str(manifest_path),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }
    return manifest["leaderboard_checked_at"], teams, identity_source


def replay_record(
    raw: bytes,
    *,
    episode_id: int,
    team_name: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    replay = json.loads(raw)
    seat = top1_seat(replay, team_name)
    steps = replay.get("steps")
    if seat is None or not isinstance(steps, list) or len(steps) < 2:
        return None
    return replay, {
        "episode_id": episode_id,
        "agent_seat": seat,
        "steps": len(steps),
        "rewards": replay.get("rewards"),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.teams <= 0:
        raise SystemExit("--teams must be positive")
    if args.episodes_per_team <= 0:
        raise SystemExit("--episodes-per-team must be positive")
    if args.episode_offset < 0:
        raise SystemExit("--episode-offset must be non-negative")
    if not 0 <= args.min_episodes_per_team <= args.episodes_per_team:
        raise SystemExit(
            "--min-episodes-per-team must be between zero and --episodes-per-team"
        )

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()

    api = KaggleApi()
    api.authenticate()
    if args.frozen_league_manifest is None:
        checked_at = datetime.now(UTC).isoformat()
        teams = leaderboard_teams(api, args.teams)
        identity_source = {"mode": "live_leaderboard"}
    else:
        checked_at, teams, identity_source = frozen_teams(
            args.frozen_league_manifest, args.teams
        )

    try:
        league_entries: list[dict[str, Any]] = []
        for team in teams:
            public_team = {
                key: value for key, value in team.items() if not key.startswith("_")
            }
            submission = (
                dict(team["_frozen_submission"])
                if "_frozen_submission" in team
                else scoring_submission(api, team)
            )
            public = completed_public_episodes(api, submission["submission_id"])
            selected = public[args.episode_offset :]
            directory_name = (
                f"r{team['snapshot_rank']:02d}_{team['team_id']}_"
                f"{safe_slug(team['team_name'])}"
            )
            team_dir = staging / directory_name
            team_dir.mkdir()
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []

            for episode in selected:
                if len(accepted) >= args.episodes_per_team:
                    break
                episode_id = int(episode["id"])
                try:
                    raw = replay_bytes(api, episode_id)
                    parsed = replay_record(
                        raw,
                        episode_id=episode_id,
                        team_name=team["team_name"],
                    )
                except Exception as exc:
                    rejected.append(
                        {
                            "episode_id": episode_id,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if parsed is None:
                    rejected.append(
                        {"episode_id": episode_id, "reason": "invalid replay or team absent"}
                    )
                    continue
                _, record = parsed
                (team_dir / f"{episode_id}.json").write_bytes(raw)
                accepted.append(record)
                if args.request_delay > 0:
                    time.sleep(args.request_delay)

            team_manifest = {
                "schema_version": 1,
                "competition": COMPETITION,
                "leaderboard_checked_at": checked_at,
                "created_at": datetime.now(UTC).isoformat(),
                "episode_offset": args.episode_offset,
                "team": public_team,
                "submission": submission,
                "available_public_episodes": len(public),
                "requested_episodes": args.episodes_per_team,
                "episodes": accepted,
                "rejected_episodes": rejected,
            }
            (team_dir / "manifest.json").write_text(
                json.dumps(team_manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            league_entries.append(
                {
                    **public_team,
                    "submission": submission,
                    "directory": directory_name,
                    "available_public_episodes": len(public),
                    "downloaded_episodes": len(accepted),
                    "rejected_episodes": len(rejected),
                }
            )

        under_covered = [
            row
            for row in league_entries
            if row["downloaded_episodes"] < args.min_episodes_per_team
        ]
        if under_covered:
            summary = ", ".join(
                f"r{row['snapshot_rank']}={row['downloaded_episodes']}"
                for row in under_covered
            )
            raise RuntimeError(f"league snapshot is under-covered: {summary}")

        manifest = {
            "schema_version": 1,
            "competition": COMPETITION,
            "api_source": "official Kaggle API",
            "identity_source": identity_source,
            "leaderboard_checked_at": checked_at,
            "created_at": datetime.now(UTC).isoformat(),
            "requested_teams": args.teams,
            "requested_episodes_per_team": args.episodes_per_team,
            "episode_offset": args.episode_offset,
            "teams": league_entries,
            "coverage": {
                "teams": len(league_entries),
                "episodes": sum(row["downloaded_episodes"] for row in league_entries),
                "minimum_per_team": min(
                    row["downloaded_episodes"] for row in league_entries
                ),
                "maximum_per_team": max(
                    row["downloaded_episodes"] for row in league_entries
                ),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging)
        raise

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
