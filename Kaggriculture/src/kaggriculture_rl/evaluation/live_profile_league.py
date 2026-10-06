"""Seat-balanced live league for state-responsive semantic policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.learning.profile_ppo import ProfilePPOPolicy
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    INCUMBENT_PROFILE,
    ProductionProfile,
    StickyProfileController,
    profile_targets,
)


@dataclass(frozen=True, slots=True)
class LiveGame:
    checkpoint: str
    opponent: str
    seed: int
    candidate_seat: int
    candidate_money: float
    opponent_money: float
    margin: float
    candidate_blocked_rate: float
    opponent_blocked_rate: float
    candidate_ineffective_actions: int
    opponent_ineffective_actions: int


class WarmupProfilePolicy:
    """A live opponent with the same 72-turn safe opening as profile PPO."""

    def __init__(self, profile: ProductionProfile, warmup_steps: int = 72) -> None:
        self.profile = profile
        self.warmup_steps = warmup_steps
        self._controller = StickyProfileController()

    def reset(self) -> None:
        self._controller.reset()

    def __call__(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        profile = (
            INCUMBENT_PROFILE
            if int(observation.get("step", 0)) < self.warmup_steps
            else self.profile
        )
        return profile_targets(self._controller.select(profile), observation)


def play_live_game(
    checkpoint: Path,
    opponent_profile: ProductionProfile,
    *,
    seed: int,
    candidate_seat: int,
) -> LiveGame:
    engine = Engine(
        {
            "episodeSteps": 720,
            "seed": seed,
            "weedSpawnChance": 0.005,
        }
    )
    candidate = ProfilePPOPolicy(checkpoint)
    opponent = WarmupProfilePolicy(opponent_profile)
    candidate.reset()
    opponent.reset()
    opponent_seat = 1 - candidate_seat
    candidate_blocked = 0
    candidate_targets = 0
    opponent_blocked = 0
    opponent_targets = 0
    candidate_ineffective = 0
    opponent_ineffective = 0

    for _ in range(719):
        candidate_observation = engine.observation(candidate_seat)
        opponent_observation = engine.observation(opponent_seat)
        candidate_plan = engine.plan(candidate_seat, candidate(candidate_observation))
        opponent_plan = engine.plan(opponent_seat, opponent(opponent_observation))
        candidate_action = candidate_plan["joint_action"]["actions"][candidate_seat]
        opponent_action = opponent_plan["joint_action"]["actions"][opponent_seat]
        if engine.sanitize_action(candidate_seat, candidate_action) != candidate_action:
            candidate_ineffective += 1
        if engine.sanitize_action(opponent_seat, opponent_action) != opponent_action:
            opponent_ineffective += 1
        candidate_entries = candidate_plan["report"]["entries"]
        opponent_entries = opponent_plan["report"]["entries"]
        candidate_targets += len(candidate_entries)
        opponent_targets += len(opponent_entries)
        candidate_blocked += sum(
            entry["status"] == "BLOCKED" for entry in candidate_entries
        )
        opponent_blocked += sum(
            entry["status"] == "BLOCKED" for entry in opponent_entries
        )
        actions = candidate_plan["joint_action"]["actions"]
        actions[opponent_seat] = opponent_action
        engine.step(actions)

    state = engine.state
    candidate_money = float(state["farms"][candidate_seat]["money"])
    opponent_money = float(state["farms"][opponent_seat]["money"])
    return LiveGame(
        checkpoint=str(checkpoint),
        opponent=opponent_profile.name,
        seed=seed,
        candidate_seat=candidate_seat,
        candidate_money=candidate_money,
        opponent_money=opponent_money,
        margin=candidate_money - opponent_money,
        candidate_blocked_rate=(
            candidate_blocked / candidate_targets if candidate_targets else 0.0
        ),
        opponent_blocked_rate=(
            opponent_blocked / opponent_targets if opponent_targets else 0.0
        ),
        candidate_ineffective_actions=candidate_ineffective,
        opponent_ineffective_actions=opponent_ineffective,
    )


def _summary(games: list[LiveGame]) -> dict[str, Any]:
    margins = [game.margin for game in games]
    money = [game.candidate_money for game in games]
    return {
        "games": len(games),
        "wins": sum(margin > 0.0 for margin in margins),
        "draws": sum(margin == 0.0 for margin in margins),
        "losses": sum(margin < 0.0 for margin in margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "mean_blocked_rate": fmean(game.candidate_blocked_rate for game in games),
        "ineffective_actions": sum(
            game.candidate_ineffective_actions for game in games
        ),
    }


def evaluate_live_profile_league(
    checkpoints: tuple[Path, ...],
    *,
    seeds: tuple[int, ...],
    opponents: tuple[ProductionProfile, ...] = ADAPTIVE_PROFILES,
) -> dict[str, Any]:
    if not checkpoints or not seeds or not opponents:
        raise ValueError("checkpoints, seeds, and opponents must not be empty")
    games_by_checkpoint: dict[str, list[LiveGame]] = {}
    for checkpoint in checkpoints:
        games = [
            play_live_game(
                checkpoint,
                opponent,
                seed=seed,
                candidate_seat=seat,
            )
            for opponent in opponents
            for seed in seeds
            for seat in (0, 1)
        ]
        games_by_checkpoint[str(checkpoint)] = games

    baseline = games_by_checkpoint[str(checkpoints[0])]
    rows = []
    for checkpoint in checkpoints:
        games = games_by_checkpoint[str(checkpoint)]
        margin_deltas = [
            game.margin - base.margin
            for game, base in zip(games, baseline, strict=True)
        ]
        money_deltas = [
            game.candidate_money - base.candidate_money
            for game, base in zip(games, baseline, strict=True)
        ]
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "summary": _summary(games),
                "paired_to_first": {
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
                    for opponent in opponents
                },
            }
        )
    return {
        "seeds": list(seeds),
        "opponents": [profile.name for profile in opponents],
        "leaderboard": rows,
        "games": {
            checkpoint: [asdict(game) for game in games]
            for checkpoint, games in games_by_checkpoint.items()
        },
    }
