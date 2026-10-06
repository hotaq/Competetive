"""Run fast seat-swapped live matches between two file agents on the Rust engine."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController


def _repair_action(
    mode: str,
    observation: dict[str, Any],
    proposed: dict[str, Any],
    configuration: dict[str, Any],
    weed_recovery: WeedRecoveryController,
    orphan_recovery: OrphanStructureRecoveryController,
) -> dict[str, Any]:
    if mode == "none":
        return proposed
    repaired = weed_recovery.repair(observation, proposed)
    if mode == "orphan":
        repaired = orphan_recovery.apply(observation, repaired)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)


def _play(
    candidate_path: Path,
    reference_path: Path,
    seed: int,
    seat: int,
    *,
    candidate_repair: str = "none",
    reference_repair: str = "none",
) -> dict[str, Any]:
    _, candidate, candidate_accepts_config = load_agent(candidate_path)
    _, reference, reference_accepts_config = load_agent(reference_path)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)
    recovery = {
        "candidate": (WeedRecoveryController(), OrphanStructureRecoveryController()),
        "reference": (WeedRecoveryController(), OrphanStructureRecoveryController()),
    }

    for _ in range(719):
        actions: list[dict[str, Any]] = []
        for player in range(2):
            policy = candidate if player == seat else reference
            accepts_config = (
                candidate_accepts_config if player == seat else reference_accepts_config
            )
            observation = engine.observation(player)
            proposed = normalize_legacy_action(
                run_agent(policy, accepts_config, observation, configuration)
            )
            role = "candidate" if player == seat else "reference"
            mode = candidate_repair if role == "candidate" else reference_repair
            proposed = _repair_action(
                mode,
                observation,
                proposed,
                configuration,
                *recovery[role],
            )
            actions.append(engine.sanitize_action(player, proposed))
        engine.step(actions)

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    return {
        "seed": seed,
        "candidate_seat": seat,
        "candidate_money": money[seat],
        "reference_money": money[1 - seat],
        "margin": money[seat] - money[1 - seat],
        "final_step": int(engine.state["step"]),
    }


def evaluate(
    candidate_path: Path,
    reference_path: Path,
    *,
    starting_seed: int,
    paired_seeds: int,
    candidate_repair: str = "none",
    reference_repair: str = "none",
) -> dict[str, Any]:
    games = [
        _play(
            candidate_path,
            reference_path,
            seed,
            seat,
            candidate_repair=candidate_repair,
            reference_repair=reference_repair,
        )
        for seed in range(starting_seed, starting_seed + paired_seeds)
        for seat in (0, 1)
    ]
    margins = [float(game["margin"]) for game in games]
    tail_count = max(1, math.ceil(len(margins) * 0.1))
    return {
        "engine": "kaggriculture_rs",
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "configuration": {
            "starting_seed": starting_seed,
            "paired_seeds": paired_seeds,
            "games": len(games),
            "candidate_repair": candidate_repair,
            "reference_repair": reference_repair,
        },
        "summary": {
            "wins": sum(margin > 0 for margin in margins),
            "draws": sum(margin == 0 for margin in margins),
            "losses": sum(margin < 0 for margin in margins),
            "win_rate": sum(margin > 0 for margin in margins) / len(margins),
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "worst_margin": min(margins),
            "cvar10_margin": fmean(sorted(margins)[:tail_count]),
            "all_complete": all(game["final_step"] == 719 for game in games),
        },
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired-seeds", type=int, default=12)
    parser.add_argument(
        "--candidate-repair",
        choices=("none", "weed", "orphan"),
        default="none",
    )
    parser.add_argument(
        "--reference-repair",
        choices=("none", "weed", "orphan"),
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.paired_seeds <= 0:
        raise ValueError("--paired-seeds must be positive")

    report = evaluate(
        args.candidate.resolve(),
        args.reference.resolve(),
        starting_seed=args.seed,
        paired_seeds=args.paired_seeds,
        candidate_repair=args.candidate_repair,
        reference_repair=args.reference_repair,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
