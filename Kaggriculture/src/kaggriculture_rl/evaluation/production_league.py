"""Paired, seat-balanced screening for semantic production profiles."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.policy.production_profile import (
    INCUMBENT_PROFILE,
    ProductionProfile,
    profile_targets,
)

OpponentSpec = tuple[str, int]


@dataclass(frozen=True, slots=True)
class ProfileGame:
    profile: str
    opponent: str
    seed: int
    seat: int
    money: float
    opponent_money: float
    margin: float
    blocked_entries: int
    target_entries: int
    ineffective_actions: int


@dataclass(frozen=True, slots=True)
class ProfileResult:
    profile: str
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    mean_money: float
    worst_money: float
    mean_margin: float
    worst_margin: float
    mean_margin_delta: float
    worst_margin_delta: float
    better: int
    equal: int
    worse: int
    blocked_rate: float
    ineffective_actions: int
    viable: bool


def _run_game(
    profile: ProductionProfile,
    opponent_spec: OpponentSpec,
    seed: int,
    seat: int,
    episode_steps: int,
    weed_spawn_chance: float,
) -> ProfileGame:
    engine = Engine(
        {
            "episodeSteps": episode_steps,
            "seed": seed,
            "weedSpawnChance": weed_spawn_chance,
        }
    )
    opponent_seat = 1 - seat
    opponent = CropHeuristicPolicy(
        crop=opponent_spec[0],
        target_quantity=opponent_spec[1],
        sell_reserve=0,
    )
    blocked_entries = 0
    target_entries = 0
    ineffective_actions = 0

    for _ in range(episode_steps):
        candidate_observation = engine.observation(seat)
        opponent_observation = engine.observation(opponent_seat)
        candidate_plan = engine.plan(
            seat,
            profile_targets(profile, candidate_observation),
        )
        opponent_plan = engine.plan(
            opponent_seat,
            opponent({"shed": opponent_observation["private"]["shed"]}),
        )
        candidate_action = candidate_plan["joint_action"]["actions"][seat]
        if engine.sanitize_action(seat, candidate_action) != candidate_action:
            ineffective_actions += 1
        entries = candidate_plan["report"]["entries"]
        target_entries += len(entries)
        blocked_entries += sum(entry["status"] == "BLOCKED" for entry in entries)

        actions = opponent_plan["joint_action"]["actions"]
        actions[seat] = candidate_action
        engine.step(actions)

    state = engine.state
    money = float(state["farms"][seat]["money"])
    opponent_money = float(state["farms"][opponent_seat]["money"])
    return ProfileGame(
        profile=profile.name,
        opponent=f"{opponent_spec[0]}:{opponent_spec[1]}",
        seed=seed,
        seat=seat,
        money=money,
        opponent_money=opponent_money,
        margin=money - opponent_money,
        blocked_entries=blocked_entries,
        target_entries=target_entries,
        ineffective_actions=ineffective_actions,
    )


def _trial_specs(
    opponent_specs: Sequence[OpponentSpec],
    seeds: Sequence[int],
) -> list[tuple[OpponentSpec, int, int]]:
    return [
        (opponent, seed, seat)
        for opponent in opponent_specs
        for seed in seeds
        for seat in (0, 1)
    ]


def evaluate_profiles(
    profiles: Sequence[ProductionProfile],
    *,
    opponent_specs: Sequence[OpponentSpec],
    seeds: Sequence[int],
    episode_steps: int = 720,
    weed_spawn_chance: float = 0.005,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate every profile and exact MELON:6 baseline on matched trials."""

    if not profiles:
        raise ValueError("profiles must not be empty")
    if not opponent_specs or not seeds:
        raise ValueError("opponent_specs and seeds must not be empty")
    if workers <= 0:
        raise ValueError("workers must be positive")

    trials = _trial_specs(opponent_specs, seeds)
    baseline_games = [
        _run_game(
            INCUMBENT_PROFILE,
            opponent,
            seed,
            seat,
            episode_steps,
            weed_spawn_chance,
        )
        for opponent, seed, seat in trials
    ]

    jobs = [
        (profile, opponent, seed, seat, episode_steps, weed_spawn_chance)
        for profile in profiles
        for opponent, seed, seat in trials
    ]
    if workers == 1:
        games = [_run_game(*job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            games = list(executor.map(lambda job: _run_game(*job), jobs))

    baseline_margins = [game.margin for game in baseline_games]
    rows: list[ProfileResult] = []
    raw_games: dict[str, list[dict[str, Any]]] = {}
    width = len(trials)
    for profile_index, profile in enumerate(profiles):
        profile_games = games[profile_index * width : (profile_index + 1) * width]
        margins = [game.margin for game in profile_games]
        money = [game.money for game in profile_games]
        deltas = [
            margin - baseline
            for margin, baseline in zip(margins, baseline_margins, strict=True)
        ]
        wins = sum(margin > 0.0 for margin in margins)
        draws = sum(margin == 0.0 for margin in margins)
        losses = len(margins) - wins - draws
        blocked = sum(game.blocked_entries for game in profile_games)
        targets = sum(game.target_entries for game in profile_games)
        ineffective = sum(game.ineffective_actions for game in profile_games)
        # A profile is safe enough to expose to PPO only if exploration cannot
        # make any matched trial worse than the incumbent and every emitted
        # low-level action survives the engine sanitizer exactly.
        viable = min(deltas) >= 0.0 and ineffective == 0
        rows.append(
            ProfileResult(
                profile=profile.name,
                games=len(margins),
                wins=wins,
                draws=draws,
                losses=losses,
                win_rate=wins / len(margins),
                mean_money=fmean(money),
                worst_money=min(money),
                mean_margin=fmean(margins),
                worst_margin=min(margins),
                mean_margin_delta=fmean(deltas),
                worst_margin_delta=min(deltas),
                better=sum(delta > 0.0 for delta in deltas),
                equal=sum(delta == 0.0 for delta in deltas),
                worse=sum(delta < 0.0 for delta in deltas),
                blocked_rate=blocked / targets if targets else 0.0,
                ineffective_actions=ineffective,
                viable=viable,
            )
        )
        raw_games[profile.name] = [asdict(game) for game in profile_games]

    rows.sort(
        key=lambda row: (
            -row.worst_margin_delta,
            -row.mean_margin_delta,
            -row.mean_margin,
            row.profile,
        )
    )
    return {
        "episode_steps": episode_steps,
        "opponents": [f"{crop}:{plots}" for crop, plots in opponent_specs],
        "seeds": list(seeds),
        "baseline": {
            "profile": INCUMBENT_PROFILE.name,
            "mean_margin": fmean(baseline_margins),
            "worst_margin": min(baseline_margins),
        },
        "leaderboard": [asdict(row) for row in rows],
        "games": raw_games,
    }
