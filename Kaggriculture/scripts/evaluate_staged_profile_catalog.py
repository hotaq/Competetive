"""Screen staged livestock investments before training a sequential policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.learning.replay_profile_ppo import Top1Tape, load_top1_tapes
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    INCUMBENT_PROFILE,
    StickyProfileController,
    profile_targets,
)

SCHEDULES: dict[str, tuple[int, ...]] = {
    "cow4_now": (3,),
    "cow2_hold": (2,),
    "cow2_upgrade_144": (2, 3),
    "cow2_upgrade_216": (2, 2, 3),
    "cow2_upgrade_288": (2, 2, 2, 3),
    "cow2_mixed_144": (2, 5),
    "cow2_mixed_216": (2, 2, 5),
    "mixed_now": (5,),
    "cow4_mixed_144": (3, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/staged_profile_catalog.json"),
    )
    return parser.parse_args()


def _play(tape: Top1Tape, schedule: tuple[int, ...]) -> dict[str, Any]:
    engine = Engine(tape.config)
    candidate_seat = 1 - tape.top1_seat
    controller = StickyProfileController()
    ineffective_actions = 0
    for turn, tape_action in enumerate(tape.actions):
        if turn < 72:
            selected = INCUMBENT_PROFILE
        else:
            decision = min((turn - 72) // 72, len(schedule) - 1)
            selected = ADAPTIVE_PROFILES[schedule[decision]]
        effective = controller.select(selected)
        observation = engine.observation(candidate_seat)
        planned = engine.plan(candidate_seat, profile_targets(effective, observation))
        candidate_action = planned["joint_action"]["actions"][candidate_seat]
        if engine.sanitize_action(candidate_seat, candidate_action) != candidate_action:
            ineffective_actions += 1
        actions = planned["joint_action"]["actions"]
        actions[tape.top1_seat] = tape_action
        engine.step(actions)
    state = engine.state
    candidate_money = float(state["farms"][candidate_seat]["money"])
    opponent_money = float(state["farms"][tape.top1_seat]["money"])
    return {
        "episode_id": tape.episode_id,
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": candidate_money - opponent_money,
        "ineffective_actions": ineffective_actions,
    }


def main() -> None:
    args = parse_args()
    tapes = tuple(
        tape
        for manifest in args.manifest
        for tape in load_top1_tapes(manifest)
    )
    results = {
        name: [_play(tape, schedule) for tape in tapes]
        for name, schedule in SCHEDULES.items()
    }
    baseline = results["cow4_now"]
    rows = []
    for name, games in results.items():
        money_deltas = [
            float(game["candidate_money"] - base["candidate_money"])
            for game, base in zip(games, baseline, strict=True)
        ]
        margin_deltas = [
            float(game["margin"] - base["margin"])
            for game, base in zip(games, baseline, strict=True)
        ]
        rows.append(
            {
                "schedule": name,
                "profile_ids": list(SCHEDULES[name]),
                "games": len(games),
                "mean_money_delta": fmean(money_deltas),
                "worst_money_delta": min(money_deltas),
                "mean_margin_delta": fmean(margin_deltas),
                "worst_margin_delta": min(margin_deltas),
                "better": sum(delta > 0.0 for delta in margin_deltas),
                "equal": sum(delta == 0.0 for delta in margin_deltas),
                "worse": sum(delta < 0.0 for delta in margin_deltas),
                "ineffective_actions": sum(
                    int(game["ineffective_actions"]) for game in games
                ),
                "safe": min(money_deltas) >= 0.0 and min(margin_deltas) >= 0.0,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["worst_money_delta"]),
            float(row["worst_margin_delta"]),
            float(row["mean_margin_delta"]),
        ),
        reverse=True,
    )
    payload = {
        "manifests": [str(path) for path in args.manifest],
        "profiles": [profile.name for profile in ADAPTIVE_PROFILES],
        "leaderboard": rows,
        "games": results,
        "limitation": "Top-1 actions are open-loop counterfactual tapes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
