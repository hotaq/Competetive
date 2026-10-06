"""Find which safe semantic profile works best on each Top-1 tape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from kaggriculture_rl.learning.profile_ppo import load_profile_checkpoint
from kaggriculture_rl.learning.replay_profile_ppo import (
    collect_tape_rollouts,
    load_top1_tapes,
)
from kaggriculture_rl.policy.production_profile import ADAPTIVE_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_v1.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("replays/top1_current/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/profile_tape_catalog.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters, config, feature_checkpoint, _ = load_profile_checkpoint(args.checkpoint)
    tapes = load_top1_tapes(args.manifest)
    rows: list[dict[str, object]] = []
    by_episode: dict[int, list[dict[str, object]]] = {}
    for profile_id, profile in enumerate(ADAPTIVE_PROFILES):
        rollout = collect_tape_rollouts(
            parameters,
            config,
            feature_checkpoint,
            tapes,
            stochastic=False,
            random_seed=0,
            force_profile_id=profile_id,
        )
        money = [float(game["candidate_money"]) for game in rollout.games]
        margins = [float(game["margin"]) for game in rollout.games]
        row: dict[str, object] = {
            "profile_id": profile_id,
            "profile": profile.name,
            "mean_money": fmean(money),
            "worst_money": min(money),
            "mean_margin": fmean(margins),
            "worst_margin": min(margins),
            "ineffective_actions": sum(
                int(game["ineffective_actions"]) for game in rollout.games
            ),
            "mean_stale_tape_actions": fmean(
                int(game["stale_tape_actions"]) for game in rollout.games
            ),
        }
        rows.append(row)
        for game in rollout.games:
            by_episode.setdefault(int(game["episode_id"]), []).append(
                {
                    "profile_id": profile_id,
                    "profile": profile.name,
                    "candidate_money": float(game["candidate_money"]),
                    "margin": float(game["margin"]),
                    "stale_tape_actions": int(game["stale_tape_actions"]),
                }
            )

    for episode_rows in by_episode.values():
        episode_rows.sort(
            key=lambda row: (float(row["margin"]), float(row["candidate_money"])),
            reverse=True,
        )
    rows.sort(
        key=lambda row: (float(row["worst_margin"]), float(row["mean_margin"])),
        reverse=True,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "profiles": rows,
        "episodes": by_episode,
        "best_profile_ids": {
            str(episode_id): int(episode_rows[0]["profile_id"])
            for episode_id, episode_rows in by_episode.items()
        },
        "distinct_best_profiles": sorted(
            np.unique([int(rows[0]["profile_id"]) for rows in by_episode.values()]).tolist()
        ),
        "limitation": (
            "Recorded Top-1 actions are open-loop tapes; changed states can make later "
            "actions stale, so this is a hard counterfactual suite rather than live play."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profiles": rows, "best_profile_ids": payload["best_profile_ids"]}, indent=2))


if __name__ == "__main__":
    main()
