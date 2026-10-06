"""Evaluate the legality-masked spatial-GRU policy against an incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/top1_spatial_gru_v1.npz"),
    )
    parser.add_argument("--paired-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument("--opponent-crop", choices=[crop.value for crop in Crop], default="MELON")
    parser.add_argument("--opponent-plots", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def play_game(
    checkpoint: Path,
    opponent: CropHeuristicPolicy,
    *,
    seed: int,
    candidate_seat: int,
) -> dict[str, Any]:
    engine = Engine({"seed": seed})
    policy = SpatialRecurrentPolicy(checkpoint)
    opponent_seat = 1 - candidate_seat
    for _ in range(719):
        candidate_observation = engine.observation(candidate_seat)
        opponent_observation = engine.observation(opponent_seat)
        candidate_action = policy(engine, candidate_observation)
        planned = engine.plan(
            opponent_seat,
            opponent({"shed": opponent_observation["private"]["shed"]}),
        )
        actions = planned["joint_action"]["actions"]
        actions[candidate_seat] = candidate_action
        state = engine.step(actions)

    money = [float(farm["money"]) for farm in state["farms"]]
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_money": money[candidate_seat],
        "opponent_money": money[opponent_seat],
        "margin": money[candidate_seat] - money[opponent_seat],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.paired_seeds <= 0:
        raise ValueError("paired-seeds must be positive")
    opponent = CropHeuristicPolicy(
        crop=args.opponent_crop,
        target_quantity=args.opponent_plots,
        sell_reserve=0,
    )
    games = [
        play_game(
            args.checkpoint,
            opponent,
            seed=args.base_seed + seed_offset,
            candidate_seat=candidate_seat,
        )
        for seed_offset in range(args.paired_seeds)
        for candidate_seat in (0, 1)
    ]
    margins = [game["margin"] for game in games]
    candidate_money = [game["candidate_money"] for game in games]
    result = {
        "checkpoint": str(args.checkpoint),
        "opponent": {"crop": args.opponent_crop, "plots": args.opponent_plots},
        "summary": {
            "games": len(games),
            "wins": sum(margin > 0 for margin in margins),
            "draws": sum(margin == 0 for margin in margins),
            "losses": sum(margin < 0 for margin in margins),
            "mean_candidate_money": fmean(candidate_money),
            "median_candidate_money": median(candidate_money),
            "worst_candidate_money": min(candidate_money),
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "worst_margin": min(margins),
        },
        "games": games,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
