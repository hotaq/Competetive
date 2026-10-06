"""Download an unseen Top-1 replay holdout disjoint from existing manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from download_top1_replays import (
    COMPETITION,
    completed_public_episodes,
    leaderboard_top1,
    replay_bytes,
    scoring_submission,
    top1_seat,
)
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument(
        "--exclude-root",
        type=Path,
        default=Path("replays"),
        help="root whose top1*/manifest*.json episode IDs are excluded",
    )
    return parser.parse_args()


def excluded_episode_ids(root: Path) -> tuple[set[int], list[str]]:
    episode_ids: set[int] = set()
    manifests: list[str] = []
    for path in sorted(root.glob("top1*/manifest*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(str(path))
        episode_ids.update(
            int(row["episode_id"]) for row in payload.get("episodes", [])
        )
    return episode_ids, manifests


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing replay directory: {output}")

    excluded, exclude_manifests = excluded_episode_ids(args.exclude_root)
    api = KaggleApi()
    api.authenticate()
    leader = leaderboard_top1(api)
    submission = scoring_submission(api, leader)
    public_episodes = completed_public_episodes(api, submission["submission_id"])
    unseen = [row for row in public_episodes if int(row["id"]) not in excluded]
    if len(unseen) < args.episodes:
        raise RuntimeError(
            f"only {len(unseen)} unseen Top-1 episodes; need {args.episodes}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    accepted: list[dict[str, Any]] = []
    for episode in unseen:
        if len(accepted) >= args.episodes:
            break
        episode_id = int(episode["id"])
        raw = replay_bytes(api, episode_id)
        replay = json.loads(raw)
        seat = top1_seat(replay, leader["team_name"])
        steps = replay.get("steps")
        if seat is None or not isinstance(steps, list) or len(steps) < 2:
            continue
        (staging / f"{episode_id}.json").write_bytes(raw)
        accepted.append(
            {
                "bytes": len(raw),
                "episode_id": episode_id,
                "rewards": replay.get("rewards"),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "steps": len(steps),
                "top1_seat": seat,
            }
        )
        if args.request_delay > 0:
            time.sleep(args.request_delay)
    if len(accepted) != args.episodes:
        raise RuntimeError(f"downloaded only {len(accepted)} valid unseen episodes")

    excluded_digest = hashlib.sha256(
        json.dumps(sorted(excluded), separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "competition": COMPETITION,
        "created_at": datetime.now(UTC).isoformat(),
        "episodes": accepted,
        "excluded_episode_count": len(excluded),
        "excluded_episode_ids_sha256": excluded_digest,
        "exclude_manifests": exclude_manifests,
        "leader": leader,
        "submission": submission,
        "immutable": False,
        "exact_validated": False,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    staging.replace(output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
