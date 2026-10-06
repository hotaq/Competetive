"""Seat-balanced live payoff matrix for production profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Protocol

from kaggriculture_rs import Engine

from kaggriculture_rl.evaluation.live_profile_league import WarmupProfilePolicy
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    ProductionProfile,
)


@dataclass(frozen=True, slots=True)
class ProfileGame:
    candidate: str
    opponent: str
    seed: int
    candidate_seat: int
    candidate_money: float
    opponent_money: float
    margin: float
    candidate_blocked_rate: float
    candidate_ineffective_actions: int


class LivePolicy(Protocol):
    def reset(self) -> None: ...

    def __call__(self, observation: dict[str, Any]) -> list[dict[str, Any]]: ...


def play_live_policy_game(
    candidate_name: str,
    candidate: LivePolicy,
    opponent_profile: ProductionProfile,
    *,
    seed: int,
    candidate_seat: int,
) -> ProfileGame:
    """Play two live policies from current observations, not replayed actions."""
    engine = Engine(
        {
            "episodeSteps": 720,
            "seed": seed,
            "weedSpawnChance": 0.005,
        }
    )
    opponent = WarmupProfilePolicy(opponent_profile)
    candidate.reset()
    opponent.reset()
    opponent_seat = 1 - candidate_seat
    candidate_blocked = 0
    candidate_targets = 0
    candidate_ineffective = 0

    for _ in range(719):
        candidate_observation = engine.observation(candidate_seat)
        opponent_observation = engine.observation(opponent_seat)
        candidate_plan = engine.plan(
            candidate_seat, candidate(candidate_observation)
        )
        opponent_plan = engine.plan(opponent_seat, opponent(opponent_observation))
        candidate_action = candidate_plan["joint_action"]["actions"][candidate_seat]
        opponent_action = opponent_plan["joint_action"]["actions"][opponent_seat]
        if (
            engine.sanitize_action(candidate_seat, candidate_action)
            != candidate_action
        ):
            candidate_ineffective += 1
        candidate_entries = candidate_plan["report"]["entries"]
        candidate_targets += len(candidate_entries)
        candidate_blocked += sum(
            entry["status"] == "BLOCKED" for entry in candidate_entries
        )
        actions = candidate_plan["joint_action"]["actions"]
        actions[opponent_seat] = opponent_action
        engine.step(actions)

    state = engine.state
    candidate_money = float(state["farms"][candidate_seat]["money"])
    opponent_money = float(state["farms"][opponent_seat]["money"])
    return ProfileGame(
        candidate=candidate_name,
        opponent=opponent_profile.name,
        seed=seed,
        candidate_seat=candidate_seat,
        candidate_money=candidate_money,
        opponent_money=opponent_money,
        margin=candidate_money - opponent_money,
        candidate_blocked_rate=(
            candidate_blocked / candidate_targets if candidate_targets else 0.0
        ),
        candidate_ineffective_actions=candidate_ineffective,
    )


def play_live_profile_game(
    candidate_profile: ProductionProfile,
    opponent_profile: ProductionProfile,
    *,
    seed: int,
    candidate_seat: int,
    warmup_steps: int = 72,
) -> ProfileGame:
    return play_live_policy_game(
        candidate_profile.name,
        WarmupProfilePolicy(candidate_profile, warmup_steps=warmup_steps),
        opponent_profile,
        seed=seed,
        candidate_seat=candidate_seat,
    )


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
        "mean_blocked_rate": fmean(
            game.candidate_blocked_rate for game in games
        ),
        "ineffective_actions": sum(
            game.candidate_ineffective_actions for game in games
        ),
    }


def evaluate_live_profile_matrix(
    *,
    seeds: tuple[int, ...],
    candidates: tuple[ProductionProfile, ...] = ADAPTIVE_PROFILES,
    opponents: tuple[ProductionProfile, ...] = ADAPTIVE_PROFILES,
    baseline_name: str = "melon6_cow4_w4",
    candidate_warmup_steps: int = 72,
) -> dict[str, Any]:
    """Evaluate every candidate/opponent/seed/seat combination."""
    if not seeds or not candidates or not opponents:
        raise ValueError("seeds, candidates, and opponents must not be empty")
    candidate_names = {profile.name for profile in candidates}
    if baseline_name not in candidate_names:
        raise ValueError(f"baseline profile not found: {baseline_name}")

    games = [
        play_live_profile_game(
            candidate,
            opponent,
            seed=seed,
            candidate_seat=seat,
            warmup_steps=candidate_warmup_steps,
        )
        for candidate in candidates
        for opponent in opponents
        for seed in seeds
        for seat in (0, 1)
    ]
    by_candidate = {
        candidate.name: [
            game for game in games if game.candidate == candidate.name
        ]
        for candidate in candidates
    }
    baseline = {
        (game.opponent, game.seed, game.candidate_seat): game
        for game in by_candidate[baseline_name]
    }

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_games = by_candidate[candidate.name]
        opponent_summaries = {
            opponent.name: _summary(
                [
                    game
                    for game in candidate_games
                    if game.opponent == opponent.name
                ]
            )
            for opponent in opponents
        }
        paired_margin = []
        paired_money = []
        for game in candidate_games:
            base = baseline[(game.opponent, game.seed, game.candidate_seat)]
            paired_margin.append(game.margin - base.margin)
            paired_money.append(game.candidate_money - base.candidate_money)
        rows.append(
            {
                "candidate": candidate.name,
                "summary": _summary(candidate_games),
                "worst_opponent_mean_margin": min(
                    summary["mean_margin"]
                    for summary in opponent_summaries.values()
                ),
                "paired_to_baseline": {
                    "baseline": baseline_name,
                    "mean_money": fmean(paired_money),
                    "worst_money": min(paired_money),
                    "mean_margin": fmean(paired_margin),
                    "worst_margin": min(paired_margin),
                    "better": sum(delta > 0.0 for delta in paired_margin),
                    "equal": sum(delta == 0.0 for delta in paired_margin),
                    "worse": sum(delta < 0.0 for delta in paired_margin),
                },
                "by_opponent": opponent_summaries,
            }
        )

    best_responses = {}
    for opponent in opponents:
        ranked = sorted(
            rows,
            key=lambda row: (
                row["by_opponent"][opponent.name]["mean_margin"],
                row["by_opponent"][opponent.name]["mean_money"],
            ),
            reverse=True,
        )
        best_responses[opponent.name] = [
            {
                "candidate": row["candidate"],
                **row["by_opponent"][opponent.name],
            }
            for row in ranked
        ]

    robust_ranking = sorted(
        rows,
        key=lambda row: (
            row["worst_opponent_mean_margin"],
            row["summary"]["worst_money"],
            row["summary"]["mean_margin"],
        ),
        reverse=True,
    )
    return {
        "seeds": list(seeds),
        "candidates": [profile.name for profile in candidates],
        "opponents": [profile.name for profile in opponents],
        "baseline": baseline_name,
        "candidate_warmup_steps": candidate_warmup_steps,
        "robust_ranking": [row["candidate"] for row in robust_ranking],
        "best_responses": best_responses,
        "rows": rows,
        "games": [asdict(game) for game in games],
    }
