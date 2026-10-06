"""Download current Kaggriculture Top-1 public replays from the official Kaggle API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import (
    ApiGetEpisodeReplayRequest,
    ApiGetLeaderboardRequest,
    KaggleApi,
)

COMPETITION = "kaggriculture"


def plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ("to_dict", "model_dump"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, dict):
                return converted
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def get_any(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def leaderboard_top1(api: KaggleApi) -> dict[str, Any]:
    request = ApiGetLeaderboardRequest()
    request.competition_name = COMPETITION
    request.page_size = 20
    with api.build_kaggle_client() as client:
        response = client.competitions.competition_api_client.get_leaderboard(request)
    rows = []
    for item in response.submissions or []:
        row = plain(item)
        team_id = get_any(row, "teamId", "team_id")
        score = get_any(row, "score", "publicScore", "public_score")
        if team_id is None or score is None:
            continue
        rows.append(
            {
                "team_id": int(team_id),
                "team_name": str(get_any(row, "teamName", "team_name", default="")),
                "score": float(score),
                "submission_date": str(
                    get_any(row, "submissionDate", "submission_date", default="")
                ),
            }
        )
    if not rows:
        raise RuntimeError("Kaggle leaderboard returned no teams")
    return min(rows, key=lambda row: (-row["score"], row["team_id"]))


def scoring_submission(api: KaggleApi, leader: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for item in api.competition_team_submissions(leader["team_id"]):
        row = plain(item)
        submission_id = get_any(row, "id", "ref", "submissionId", "submission_id")
        if submission_id is None:
            continue
        score = get_any(row, "publicScore", "public_score", "score")
        try:
            public_score = float(score)
        except (TypeError, ValueError):
            public_score = math.nan
        candidates.append(
            {
                "submission_id": int(submission_id),
                "public_score": public_score,
                "date": str(
                    get_any(row, "dateSubmitted", "date_submitted", "date", default="")
                ),
            }
        )
    if not candidates:
        raise RuntimeError(f"no submissions found for Top-1 team {leader['team_name']!r}")
    return min(
        candidates,
        key=lambda row: (
            abs(row["public_score"] - leader["score"])
            if not math.isnan(row["public_score"])
            else math.inf,
            -row["submission_id"],
        ),
    )


def completed_public_episodes(api: KaggleApi, submission_id: int) -> list[dict[str, Any]]:
    rows = []
    for item in api.competition_list_episodes(submission_id) or []:
        row = plain(item)
        episode_id = get_any(row, "id", "episodeId", "episode_id")
        if episode_id is None:
            continue
        episode_type = str(get_any(row, "type", "episodeType", default="PUBLIC")).upper()
        state = str(get_any(row, "state", "status", default="COMPLETED")).upper()
        if "PUBLIC" in episode_type and "COMPLETE" in state:
            rows.append({**row, "id": int(episode_id)})
    return sorted(rows, key=lambda row: row["id"], reverse=True)


def replay_bytes(api: KaggleApi, episode_id: int) -> bytes:
    request = ApiGetEpisodeReplayRequest()
    request.episode_id = episode_id
    with api.build_kaggle_client() as client:
        response = client.competitions.competition_api_client.get_episode_replay(request)
    response.raise_for_status()
    return bytes(response.content)


def top1_seat(replay: dict[str, Any], team_name: str) -> int | None:
    names = replay.get("info", {}).get("TeamNames", [])
    wanted = " ".join(team_name.casefold().split())
    for seat, name in enumerate(names):
        if " ".join(str(name).casefold().split()) == wanted:
            return seat
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("replays/top1_current"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="skip this many newest completed public episodes before downloading",
    )
    parser.add_argument("--request-delay", type=float, default=0.75)
    parser.add_argument("--expected-team-id", type=int)
    parser.add_argument("--expected-submission-id", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.episode_offset < 0:
        raise SystemExit("--episode-offset must be non-negative")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing replay directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()

    api = KaggleApi()
    api.authenticate()
    leader = leaderboard_top1(api)
    submission = scoring_submission(api, leader)
    if (
        args.expected_team_id is not None
        and leader["team_id"] != args.expected_team_id
    ):
        raise RuntimeError(
            f"Top-1 changed: expected team {args.expected_team_id}, "
            f"found {leader['team_id']} ({leader['team_name']})"
        )
    if (
        args.expected_submission_id is not None
        and submission["submission_id"] != args.expected_submission_id
    ):
        raise RuntimeError(
            f"Top-1 submission changed: expected {args.expected_submission_id}, "
            f"found {submission['submission_id']}"
        )
    leaderboard_checked_at = datetime.now(UTC).isoformat()
    all_episodes = completed_public_episodes(api, submission["submission_id"])
    episodes = all_episodes[args.episode_offset :]
    accepted = []
    try:
        for episode in episodes:
            if len(accepted) >= args.episodes:
                break
            raw = replay_bytes(api, episode["id"])
            replay = json.loads(raw)
            seat = top1_seat(replay, leader["team_name"])
            steps = replay.get("steps")
            if seat is None or not isinstance(steps, list) or len(steps) < 2:
                continue
            path = staging / f"{episode['id']}.json"
            path.write_bytes(raw)
            accepted.append(
                {
                    "episode_id": episode["id"],
                    "top1_seat": seat,
                    "steps": len(steps),
                    "rewards": replay.get("rewards"),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            if args.request_delay > 0:
                time.sleep(args.request_delay)
        if not accepted:
            raise RuntimeError("no valid Top-1 public replay was downloaded")
        manifest = {
            "schema_version": 2,
            "competition": COMPETITION,
            "api_source": "official Kaggle API",
            "leaderboard_checked_at": leaderboard_checked_at,
            "created_at": datetime.now(UTC).isoformat(),
            "episode_offset": args.episode_offset,
            "leader": leader,
            "submission": submission,
            "available_public_episodes_total": len(all_episodes),
            "available_public_episodes": len(episodes),
            "episodes": accepted,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        raise

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
