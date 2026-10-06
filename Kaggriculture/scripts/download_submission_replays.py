"""Download public replays for a specific Kaggriculture submission."""

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
    replay_bytes,
    top1_seat,
)
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.5)
    return parser.parse_args()


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
        "bytes": len(raw),
        "episode_id": episode_id,
        "rewards": replay.get("rewards"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "steps": len(steps),
        # Keep top1_seat for compatibility with the existing tape builders.
        "top1_seat": seat,
        "agent_seat": seat,
    }


def main() -> None:
    args = parse_args()
    if args.submission_id <= 0:
        raise SystemExit("--submission-id must be positive")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.episode_offset < 0:
        raise SystemExit("--episode-offset must be non-negative")

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing replay directory: {output}")

    api = KaggleApi()
    api.authenticate()
    public = completed_public_episodes(api, args.submission_id)
    selected = public[args.episode_offset :]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    accepted: list[dict[str, Any]] = []
    try:
        for episode in selected:
            if len(accepted) >= args.episodes:
                break
            episode_id = int(episode["id"])
            raw = replay_bytes(api, episode_id)
            parsed = replay_record(
                raw,
                episode_id=episode_id,
                team_name=args.team_name,
            )
            if parsed is None:
                continue
            _, record = parsed
            (staging / f"{episode_id}.json").write_bytes(raw)
            accepted.append(record)
            if args.request_delay > 0:
                time.sleep(args.request_delay)
        if not accepted:
            raise RuntimeError("no valid public replay was downloaded")

        manifest = {
            "competition": COMPETITION,
            "created_at": datetime.now(UTC).isoformat(),
            "team_name": args.team_name,
            "episode_offset": args.episode_offset,
            "available_public_episodes": len(public),
            "submission": {"submission_id": args.submission_id},
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
