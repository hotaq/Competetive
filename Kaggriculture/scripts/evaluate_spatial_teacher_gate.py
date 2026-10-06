"""Evaluate spatial-GRU components behind an exact standalone-teacher fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.bc import confidence_gate
from kaggriculture_rl.imitation.teacher_dataset import (
    StandaloneFilePolicy,
    sanitize_standalone_action,
)
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "rl/checkpoints/v16_rc5_spatial_gru_bc_v1_e100.npz"
DEFAULT_TEACHER = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--confidence-threshold", type=float, default=1.0)
    parser.add_argument("--paired-seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=620_000)
    parser.add_argument(
        "--opponent-crop",
        choices=[crop.value for crop in Crop],
        default="MELON",
    )
    parser.add_argument("--opponent-plots", type=int, default=6)
    parser.add_argument("--disable-farmer", action="store_true")
    parser.add_argument("--disable-hands", action="store_true")
    parser.add_argument("--disable-market", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def play_game(
    checkpoint: Path,
    teacher_path: Path,
    opponent: CropHeuristicPolicy,
    *,
    confidence_threshold: float,
    enabled_components: frozenset[str],
    seed: int,
    candidate_seat: int,
) -> dict[str, Any]:
    engine = Engine({"seed": seed})
    policy = SpatialRecurrentPolicy(checkpoint)
    teacher = StandaloneFilePolicy(teacher_path, role="fallback")
    opponent_seat = 1 - candidate_seat
    accepted_farmer = 0
    accepted_hands = 0
    total_hands = 0
    accepted_market = 0
    action_differences = 0
    try:
        for _ in range(719):
            candidate_observation = engine.observation(candidate_seat)
            opponent_observation = engine.observation(opponent_seat)
            hands_count = len(
                candidate_observation["farms"][candidate_seat].get("hands", [])
            )
            fallback = sanitize_standalone_action(
                engine,
                candidate_seat,
                teacher(candidate_observation),
                hands_count,
            )
            prediction = policy.predict_with_confidence(engine, candidate_observation)
            candidate_action, accepted = confidence_gate(
                prediction,
                fallback,
                confidence_threshold,
            )
            if "farmer" not in enabled_components:
                candidate_action["farmer"] = fallback["farmer"]
                accepted["farmer"] = 0
            if "hands" not in enabled_components:
                candidate_action["hands"] = fallback["hands"]
                accepted["hands"] = 0
            if "market" not in enabled_components:
                candidate_action["market"] = fallback["market"]
                accepted["market"] = 0
            candidate_action = sanitize_standalone_action(
                engine,
                candidate_seat,
                candidate_action,
                hands_count,
            )
            accepted_farmer += int(accepted["farmer"])
            accepted_hands += int(accepted["hands"])
            total_hands += len(prediction.hand_confidences)
            accepted_market += int(accepted["market"])
            action_differences += int(candidate_action != fallback)

            planned = engine.plan(
                opponent_seat,
                opponent({"shed": opponent_observation["private"]["shed"]}),
            )
            actions = planned["joint_action"]["actions"]
            actions[candidate_seat] = candidate_action
            state = engine.step(actions)
    finally:
        teacher.close()

    money = [float(farm["money"]) for farm in state["farms"]]
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_money": money[candidate_seat],
        "opponent_money": money[opponent_seat],
        "margin": money[candidate_seat] - money[opponent_seat],
        "farmer_acceptance_rate": accepted_farmer / 719,
        "hand_acceptance_rate": accepted_hands / max(total_hands, 1),
        "market_acceptance_rate": accepted_market / 719,
        "action_difference_rate": action_differences / 719,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.paired_seeds <= 0:
        raise ValueError("paired-seeds must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be within 0..1")
    checkpoint = args.checkpoint.resolve()
    teacher = args.teacher.resolve()
    enabled_components = frozenset(
        component
        for component, disabled in (
            ("farmer", args.disable_farmer),
            ("hands", args.disable_hands),
            ("market", args.disable_market),
        )
        if not disabled
    )
    opponent = CropHeuristicPolicy(
        crop=args.opponent_crop,
        target_quantity=args.opponent_plots,
        sell_reserve=0,
    )
    games = [
        play_game(
            checkpoint,
            teacher,
            opponent,
            confidence_threshold=args.confidence_threshold,
            enabled_components=enabled_components,
            seed=args.base_seed + seed_offset,
            candidate_seat=candidate_seat,
        )
        for seed_offset in range(args.paired_seeds)
        for candidate_seat in (0, 1)
    ]
    margins = [float(game["margin"]) for game in games]
    candidate_money = [float(game["candidate_money"]) for game in games]
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "teacher": str(teacher),
        "teacher_sha256": _sha256(teacher),
        "confidence_threshold": args.confidence_threshold,
        "enabled_components": sorted(enabled_components),
        "opponent": {
            "crop": args.opponent_crop,
            "plots": args.opponent_plots,
        },
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
            "mean_farmer_acceptance_rate": fmean(
                game["farmer_acceptance_rate"] for game in games
            ),
            "mean_hand_acceptance_rate": fmean(
                game["hand_acceptance_rate"] for game in games
            ),
            "mean_market_acceptance_rate": fmean(
                game["market_acceptance_rate"] for game in games
            ),
            "mean_action_difference_rate": fmean(
                game["action_difference_rate"] for game in games
            ),
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
