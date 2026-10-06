"""Build a large deduplicated tape pool from the leaderboard's public replays.

`harvest_opponent_tapes.py` keeps only the target team's seat and re-downloads an episode
once per team that played it. A replay is ~32 MB and the download is the whole cost, so
this keeps **both** seats and visits each episode once, which roughly doubles the tapes per
byte and removes the duplicate downloads between two ranked teams that played each other.

Tapes are stored raw: `actions[i]` is the action recorded at `steps[i]`. The action taken
from state `i` is `actions[i + 1]` and every consumer applies that shift itself
(`scripts/play_against_tape.py`, `scripts/evaluate_shop_router.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from download_top1_replays import completed_public_episodes, replay_bytes, scoring_submission
from harvest_opponent_tapes import leaderboard
from kaggle.api.kaggle_api_extended import KaggleApi


def action_hash(actions: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teams", type=int, default=20)
    parser.add_argument("--episodes-per-team", type=int, default=12)
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    board = leaderboard(api)[: args.teams]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() \
        else {"tapes": {}, "episodes": []}
    seen_hash: dict[str, str] = {t["sha256"]: key for key, t in manifest["tapes"].items()}
    seen_episode = set(manifest["episodes"])

    wanted: list[tuple[dict[str, Any], int]] = []
    for leader in board:
        try:
            submission = scoring_submission(api, leader)
        except RuntimeError as error:
            print(f"rank {leader['rank']:>2}  {leader['team_name']:<24} skipped: {error}")
            continue
        episodes = completed_public_episodes(api, submission["submission_id"])
        for row in episodes[: args.episodes_per_team]:
            if row["id"] not in seen_episode:
                wanted.append((leader, row["id"]))
        print(f"rank {leader['rank']:>2}  {leader['team_name']:<24} "
              f"{len(episodes)} public episodes, queued {min(len(episodes), args.episodes_per_team)}")

    queue = list(dict.fromkeys(episode for _, episode in wanted))
    print(f"\n{len(queue)} distinct episodes to download\n")

    started = time.monotonic()
    for index, episode in enumerate(queue, 1):
        try:
            replay = json.loads(replay_bytes(api, episode))
        except Exception as error:                      # a single bad replay must not stop the run
            print(f"  [{index}/{len(queue)}] ep{episode} download failed: {error}")
            continue
        names = replay.get("info", {}).get("TeamNames", [])
        rewards = replay.get("rewards", [])
        added = 0
        for seat, name in enumerate(names):
            actions = [step[seat]["action"] for step in replay["steps"]]
            digest = action_hash(actions)
            if digest in seen_hash:
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
            (args.output / "tapes").mkdir(parents=True, exist_ok=True)
            (args.output / "tapes" / f"{key}.json").write_text(
                json.dumps(payload, separators=(",", ":")))
            manifest["tapes"][key] = {k: v for k, v in payload.items() if k != "actions"}
            seen_hash[digest] = key
            added += 1
        seen_episode.add(episode)
        manifest["episodes"] = sorted(seen_episode)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rate = (time.monotonic() - started) / index
        print(f"  [{index}/{len(queue)}] ep{episode} {' vs '.join(map(str, names))}"
              f"  +{added} new  (pool {len(manifest['tapes'])}, {rate:.1f}s/ep, "
              f"eta {rate * (len(queue) - index) / 60:.0f}m)", flush=True)
        del replay

    print(f"\npool: {len(manifest['tapes'])} unique tapes from {len(seen_episode)} episodes")
    counts: dict[str, int] = {}
    for tape in manifest["tapes"].values():
        counts[str(tape["team_name"])] = counts.get(str(tape["team_name"]), 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {name}")


if __name__ == "__main__":
    main()
