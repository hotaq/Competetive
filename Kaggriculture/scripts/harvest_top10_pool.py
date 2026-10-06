"""Harvest a dated tape pool for the current top-10 leaderboard teams.

Follows the scripts/harvest_tape_pool.py pattern exactly: one replay download per
episode, both seats kept as compact tapes, dedup by action sha256. Tape JSON schema
is identical to artifacts/tape_pool_20260919 (team_name, episode, seat, seed,
final_money, opponent, opponent_money, length, sha256, actions).

Extras for this dated pool: per-tape provenance in manifest.json (rank, team_id,
score, submission_id, episode date, which top-10 lists referenced the episode),
"zero waste" (own-team) seats skipped, and a sanitize pass (length 720 + action
keys exactly {farmer, hands, market}; rejects are logged, not silently dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from download_top1_replays import (
    completed_public_episodes,
    plain,
    replay_bytes,
    scoring_submission,
)
from harvest_opponent_tapes import leaderboard
from kaggle.api.kaggle_api_extended import KaggleApi

OWN_TEAM = "zero waste"
EXPECTED_KEYS = {"farmer", "hands", "market"}
EXPECTED_LENGTH = 720


def action_hash(actions: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "unknown"


def episode_date(row: dict[str, Any]) -> str:
    row = plain(row)
    for key in ("endTime", "end_time", "createTime", "create_time", "date",
                "dateCompleted", "date_completed", "completedTime",
                "completed_time", "dateStarted", "date_started"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def sanitize(actions: list[Any]) -> str | None:
    if len(actions) != EXPECTED_LENGTH:
        return f"length {len(actions)} != {EXPECTED_LENGTH}"
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or set(action.keys()) != EXPECTED_KEYS:
            keys = sorted(action.keys()) if isinstance(action, dict) else type(action).__name__
            return f"step {index} keys {keys} != {sorted(EXPECTED_KEYS)}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=10)
    parser.add_argument("--episodes-per-team", type=int, default=6)
    parser.add_argument("--request-delay", type=float, default=0.75)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    board = leaderboard(api)[: args.teams]
    harvested_at = datetime.now(UTC).isoformat()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tapes").mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "tapes": {},
        "episodes": [],
        "leaderboard": board,
        "leaderboard_checked_at": harvested_at,
        "own_team_excluded": OWN_TEAM,
    }
    seen_hash: dict[str, str] = {}
    seen_episode: set[int] = set()
    # episode -> {"row": episode row, "date": str, "wanted_by": [team names]}
    wanted: dict[int, dict[str, Any]] = {}
    team_submissions: dict[str, Any] = {}
    no_replays: list[str] = []

    for leader in board:
        try:
            submission = scoring_submission(api, leader)
        except RuntimeError as error:
            print(f"rank {leader['rank']:>2}  {leader['team_name']:<24} skipped: {error}")
            no_replays.append(leader["team_name"])
            continue
        team_submissions[leader["team_name"]] = submission
        episodes = completed_public_episodes(api, submission["submission_id"])
        queued = 0
        for row in episodes[: args.episodes_per_team]:
            entry = wanted.setdefault(row["id"], {"row": row, "wanted_by": []})
            if leader["team_name"] not in entry["wanted_by"]:
                entry["wanted_by"].append(leader["team_name"])
            queued += 1
        print(f"rank {leader['rank']:>2}  {leader['team_name']:<24} "
              f"{len(episodes)} public episodes, queued {queued}")

    queue = sorted(wanted)
    print(f"\n{len(queue)} distinct episodes to download\n")
    dropped: list[dict[str, Any]] = []
    skipped_own = 0

    started = time.monotonic()
    for index, episode in enumerate(queue, 1):
        entry = wanted[episode]
        try:
            replay = json.loads(replay_bytes(api, episode))
        except Exception as error:
            print(f"  [{index}/{len(queue)}] ep{episode} download failed: {error}")
            dropped.append({"episode": episode, "reason": f"download failed: {error}",
                            "wanted_by": entry["wanted_by"]})
            continue
        names = replay.get("info", {}).get("TeamNames", [])
        rewards = replay.get("rewards", [])
        added = 0
        for seat, name in enumerate(names):
            if " ".join(str(name).casefold().split()) == OWN_TEAM:
                skipped_own += 1
                continue
            actions = [step[seat]["action"] for step in replay["steps"]]
            digest = action_hash(actions)
            if digest in seen_hash:
                continue
            problem = sanitize(actions)
            if problem is not None:
                dropped.append({"episode": episode, "seat": seat, "team_name": str(name),
                                "reason": f"sanitize: {problem}", "wanted_by": entry["wanted_by"]})
                continue
            key = f"{slug(str(name))}__ep{episode}__seat{seat}"
            payload = {
                "team_name": name, "episode": episode, "seat": seat,
                "seed": replay["info"].get("seed"),
                "final_money": rewards[seat] if seat < len(rewards) else None,
                "opponent": names[1 - seat] if len(names) > 1 else None,
                "opponent_money": rewards[1 - seat] if len(rewards) > 1 else None,
                "length": len(actions), "sha256": digest, "actions": actions,
            }
            (args.output / "tapes" / f"{key}.json").write_text(
                json.dumps(payload, separators=(",", ":")))
            manifest["tapes"][key] = {
                k: v for k, v in payload.items() if k != "actions"
            } | {
                "date": episode_date(entry["row"]),
                "wanted_by": entry["wanted_by"],
                "own_tape_for": [t for t in entry["wanted_by"] if str(t) == str(name)],
            }
            seen_hash[digest] = key
            added += 1
        seen_episode.add(episode)
        manifest["episodes"] = sorted(seen_episode)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        rate = (time.monotonic() - started) / index
        print(f"  [{index}/{len(queue)}] ep{episode} {' vs '.join(map(str, names))}"
              f"  +{added} new  (pool {len(manifest['tapes'])}, {rate:.1f}s/ep, "
              f"eta {rate * (len(queue) - index) / 60:.0f}m)", flush=True)
        del replay
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    manifest["teams_with_no_replays"] = no_replays
    manifest["dropped"] = dropped
    manifest["own_team_seats_skipped"] = skipped_own
    manifest["pool_size"] = len(manifest["tapes"])
    manifest["episode_count"] = len(seen_episode)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\npool: {len(manifest['tapes'])} unique tapes from {len(seen_episode)} episodes")
    counts: dict[str, int] = {}
    for tape in manifest["tapes"].values():
        counts[str(tape["team_name"])] = counts.get(str(tape["team_name"]), 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {name}")
    if dropped:
        print(f"\ndropped {len(dropped)} (logged in manifest['dropped']):")
        for item in dropped:
            print(f"  ep{item.get('episode')} seat {item.get('seat')}: {item['reason']}")
    if skipped_own:
        print(f"\nskipped {skipped_own} own-team ({OWN_TEAM!r}) seats")


if __name__ == "__main__":
    main()
