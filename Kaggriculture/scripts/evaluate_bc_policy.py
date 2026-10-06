"""Evaluate a BC checkpoint against the deterministic crop baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.bc import (
    BehaviorCloningPolicy,
    confidence_gate,
)
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/top1_bc_v1.npz"),
    )
    parser.add_argument("--paired-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument(
        "--opponent-crop",
        choices=[crop.value for crop in Crop],
        default="MELON",
    )
    parser.add_argument("--opponent-plots", type=int, default=6)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--bc-prefix-turns", type=int)
    parser.add_argument(
        "--legality-mask",
        action="store_true",
        help="decode with exact Rust legality masks and final sanitizer",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def play_game(
    policy: BehaviorCloningPolicy,
    opponent: CropHeuristicPolicy,
    *,
    seed: int,
    candidate_seat: int,
    confidence_threshold: float | None,
    bc_prefix_turns: int | None,
    legality_mask: bool,
) -> dict[str, Any]:
    engine = Engine({"seed": seed})
    accepted_farmer = 0
    accepted_hands = 0
    total_hands = 0
    accepted_market = 0

    for turn in range(719):
        candidate_observation = engine.observation(candidate_seat)
        opponent_seat = 1 - candidate_seat
        opponent_observation = engine.observation(opponent_seat)

        prediction = (
            policy.predict_masked_with_confidence(engine, candidate_observation)
            if legality_mask
            else policy.predict_with_confidence(candidate_observation)
        )
        candidate_action = prediction.action
        if confidence_threshold is not None or bc_prefix_turns is not None:
            fallback = engine.plan(
                candidate_seat,
                opponent({"shed": candidate_observation["private"]["shed"]}),
            )["joint_action"]["actions"][candidate_seat]
            if bc_prefix_turns is not None:
                use_bc = turn < bc_prefix_turns
                candidate_action = prediction.action if use_bc else fallback
                accepted = {
                    "farmer": int(use_bc),
                    "hands": (len(prediction.hand_confidences) if use_bc else 0),
                    "market": int(use_bc),
                }
            else:
                candidate_action, accepted = confidence_gate(
                    prediction, fallback, confidence_threshold
                )
            accepted_farmer += accepted["farmer"]
            accepted_hands += accepted["hands"]
            total_hands += len(prediction.hand_confidences)
            accepted_market += accepted["market"]

        planned = engine.plan(
            opponent_seat,
            opponent({"shed": opponent_observation["private"]["shed"]}),
        )
        actions = planned["joint_action"]["actions"]
        actions[candidate_seat] = (
            engine.sanitize_action(candidate_seat, candidate_action)
            if legality_mask
            else candidate_action
        )
        state = engine.step(actions)

    money = [float(farm["money"]) for farm in state["farms"]]
    margin = money[candidate_seat] - money[1 - candidate_seat]
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_money": money[candidate_seat],
        "opponent_money": money[1 - candidate_seat],
        "margin": margin,
        "accepted_farmer_rate": accepted_farmer / 719,
        "accepted_hand_rate": accepted_hands / max(total_hands, 1),
        "accepted_market_rate": accepted_market / 719,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.paired_seeds <= 0:
        raise ValueError("paired_seeds must be positive")
    if args.confidence_threshold is not None and args.bc_prefix_turns is not None:
        raise ValueError("confidence_threshold and bc_prefix_turns are mutually exclusive")
    if args.bc_prefix_turns is not None and args.bc_prefix_turns < 0:
        raise ValueError("bc_prefix_turns must be non-negative")

    policy = BehaviorCloningPolicy(args.checkpoint)
    opponent = CropHeuristicPolicy(
        crop=args.opponent_crop,
        target_quantity=args.opponent_plots,
        sell_reserve=0,
    )

    warmup_engine = Engine({"seed": args.base_seed})
    policy(warmup_engine.observation(0))

    games = [
        play_game(
            policy,
            opponent,
            seed=args.base_seed + seed_offset,
            candidate_seat=candidate_seat,
            confidence_threshold=args.confidence_threshold,
                bc_prefix_turns=args.bc_prefix_turns,
                legality_mask=args.legality_mask,
        )
        for seed_offset in range(args.paired_seeds)
        for candidate_seat in (0, 1)
    ]
    margins = [game["margin"] for game in games]
    candidate_money = [game["candidate_money"] for game in games]
    result = {
        "checkpoint": str(args.checkpoint),
        "opponent": {
            "kind": "crop_heuristic",
            "crop": args.opponent_crop,
            "plots": args.opponent_plots,
        },
        "confidence_threshold": args.confidence_threshold,
        "bc_prefix_turns": args.bc_prefix_turns,
        "legality_mask": args.legality_mask,
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
            "mean_accepted_farmer_rate": fmean(game["accepted_farmer_rate"] for game in games),
            "mean_accepted_hand_rate": fmean(game["accepted_hand_rate"] for game in games),
            "mean_accepted_market_rate": fmean(game["accepted_market_rate"] for game in games),
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
