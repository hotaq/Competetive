"""Evaluate the observable wait-then-respond policy against live profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

from kaggriculture_rl.evaluation.live_profile_matrix import (
    ProfileGame,
    play_live_policy_game,
    play_live_profile_game,
)
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
)
from kaggriculture_rl.policy.reactive_profile import ReactiveProfilePolicy

BASELINE_NAME = "melon6_cow4_w4"


def _summary(games: list[ProfileGame]) -> dict[str, Any]:
    money = [game.candidate_money for game in games]
    margins = [game.margin for game in games]
    return {
        "games": len(games),
        "wins": sum(margin > 0.0 for margin in margins),
        "draws": sum(margin == 0.0 for margin in margins),
        "losses": sum(margin < 0.0 for margin in margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=230_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/reactive_profile.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    baseline_profile = next(
        profile for profile in ADAPTIVE_PROFILES if profile.name == BASELINE_NAME
    )
    games: list[ProfileGame] = []
    baselines: list[ProfileGame] = []
    for opponent in ADAPTIVE_PROFILES:
        for seed in range(args.base_seed, args.base_seed + args.seeds):
            for seat in (0, 1):
                games.append(
                    play_live_policy_game(
                        "reactive_wait1",
                        ReactiveProfilePolicy(),
                        opponent,
                        seed=seed,
                        candidate_seat=seat,
                    )
                )
                baselines.append(
                    play_live_profile_game(
                        baseline_profile,
                        opponent,
                        seed=seed,
                        candidate_seat=seat,
                    )
                )
    margin_deltas = [
        game.margin - baseline.margin
        for game, baseline in zip(games, baselines, strict=True)
    ]
    money_deltas = [
        game.candidate_money - baseline.candidate_money
        for game, baseline in zip(games, baselines, strict=True)
    ]
    result = {
        "seeds": list(range(args.base_seed, args.base_seed + args.seeds)),
        "policy": "reactive_wait1",
        "summary": _summary(games),
        "baseline_summary": _summary(baselines),
        "paired_to_cow4": {
            "mean_money": fmean(money_deltas),
            "worst_money": min(money_deltas),
            "mean_margin": fmean(margin_deltas),
            "worst_margin": min(margin_deltas),
            "better": sum(delta > 0.0 for delta in margin_deltas),
            "equal": sum(delta == 0.0 for delta in margin_deltas),
            "worse": sum(delta < 0.0 for delta in margin_deltas),
        },
        "by_opponent": {
            opponent.name: _summary(
                [game for game in games if game.opponent == opponent.name]
            )
            for opponent in ADAPTIVE_PROFILES
        },
        "games": [asdict(game) for game in games],
        "baseline_games": [asdict(game) for game in baselines],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"games", "baseline_games"}
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
