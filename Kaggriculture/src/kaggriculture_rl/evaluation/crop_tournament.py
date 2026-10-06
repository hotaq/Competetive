"""Seat-balanced crop-policy tournaments on the typed Rust engine."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from math import isclose
from statistics import fmean, median
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


@dataclass(frozen=True, slots=True)
class CropConfig:
    crop: Crop | str
    plots: int
    sell_reserve: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop", Crop(self.crop))
        if self.plots <= 0:
            raise ValueError("plots must be positive")
        if self.sell_reserve < 0:
            raise ValueError("sell_reserve must be non-negative")

    @property
    def key(self) -> str:
        return f"{Crop(self.crop).value}:{self.plots}:r{self.sell_reserve}"


@dataclass(frozen=True, slots=True)
class GameResult:
    players: tuple[str, str]
    seed: int
    steps: int
    money: tuple[float, float]
    blocked_reports: tuple[int, int]


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    config: str
    crop: str
    plots: int
    sell_reserve: int
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    match_points: float
    mean_money: float
    median_money: float
    worst_money: float
    mean_margin: float
    p10_margin: float
    worst_margin: float
    blocked_rate: float


@dataclass(frozen=True, slots=True)
class _MatchSpec:
    left: CropConfig
    right: CropConfig
    seed: int
    episode_steps: int
    weed_spawn_chance: float


def _initial_observation() -> dict[str, Any]:
    return {"shed": {crop.value: 0 for crop in Crop}}


def run_match(
    left: CropConfig,
    right: CropConfig,
    *,
    seed: int,
    episode_steps: int = 720,
    weed_spawn_chance: float = 0.005,
) -> GameResult:
    """Play one Kaggle-shaped episode with both policies planning one snapshot."""

    if episode_steps < 4:
        raise ValueError("episode_steps must be at least 4")
    if not 0.0 <= weed_spawn_chance <= 1.0:
        raise ValueError("weed_spawn_chance must be between zero and one")

    engine = Engine(
        {
            "episodeSteps": episode_steps,
            "seed": seed,
            "weedSpawnChance": weed_spawn_chance,
        }
    )
    policies = (
        CropHeuristicPolicy(left.crop, left.plots, left.sell_reserve),
        CropHeuristicPolicy(right.crop, right.plots, right.sell_reserve),
    )
    observations = [_initial_observation(), _initial_observation()]
    blocked = [0, 0]
    steps = 0

    for _ in range(episode_steps):
        plans = [
            engine.plan(player, policies[player](observations[player])) for player in range(2)
        ]
        joint_action = {
            "schema_version": 1,
            "actions": [
                plans[0]["joint_action"]["actions"][0],
                plans[1]["joint_action"]["actions"][1],
            ],
        }
        for player, planned in enumerate(plans):
            blocked[player] += planned["report"]["status"] == "BLOCKED"
        result = engine.step_typed(joint_action)
        observations = result["observations"]
        steps += 1
        if result["done"]:
            break

    state = engine.state
    return GameResult(
        players=(left.key, right.key),
        seed=seed,
        steps=steps,
        money=(float(state["farms"][0]["money"]), float(state["farms"][1]["money"])),
        blocked_reports=(blocked[0], blocked[1]),
    )


def _run_spec(spec: _MatchSpec) -> GameResult:
    return run_match(
        spec.left,
        spec.right,
        seed=spec.seed,
        episode_steps=spec.episode_steps,
        weed_spawn_chance=spec.weed_spawn_chance,
    )


def round_robin(
    configs: Sequence[CropConfig],
    *,
    seeds: Sequence[int],
    episode_steps: int,
    weed_spawn_chance: float = 0.005,
    workers: int = 1,
) -> list[GameResult]:
    """Play every distinct pair in both seats for every seed."""

    if len(configs) < 2:
        raise ValueError("round_robin requires at least two configurations")
    if not seeds:
        raise ValueError("round_robin requires at least one seed")
    if workers <= 0:
        raise ValueError("workers must be positive")
    keys = [config.key for config in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("configuration keys must be unique")

    specs = [
        _MatchSpec(left, right, seed, episode_steps, weed_spawn_chance)
        for first, second in combinations(configs, 2)
        for seed in seeds
        for left, right in ((first, second), (second, first))
    ]
    if workers == 1:
        return [_run_spec(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_spec, specs))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rank_results(
    configs: Sequence[CropConfig], games: Sequence[GameResult]
) -> list[LeaderboardRow]:
    """Aggregate Kaggle money, head-to-head outcomes, and lower-tail margins."""

    by_key = {config.key: config for config in configs}
    money: dict[str, list[float]] = {key: [] for key in by_key}
    margins: dict[str, list[float]] = {key: [] for key in by_key}
    outcomes: dict[str, list[int]] = {key: [] for key in by_key}
    blocked: dict[str, int] = {key: 0 for key in by_key}
    total_steps: dict[str, int] = {key: 0 for key in by_key}

    for game in games:
        for seat, key in enumerate(game.players):
            if key not in by_key:
                raise ValueError(f"unknown configuration in game result: {key}")
            own_money = game.money[seat]
            opponent_money = game.money[1 - seat]
            margin = own_money - opponent_money
            money[key].append(own_money)
            margins[key].append(margin)
            if isclose(margin, 0.0, abs_tol=1e-9):
                outcomes[key].append(0)
            elif margin > 0:
                outcomes[key].append(1)
            else:
                outcomes[key].append(-1)
            blocked[key] += game.blocked_reports[seat]
            total_steps[key] += game.steps

    rows = []
    for key, config in by_key.items():
        if not money[key]:
            raise ValueError(f"configuration has no games: {key}")
        wins = outcomes[key].count(1)
        draws = outcomes[key].count(0)
        losses = outcomes[key].count(-1)
        game_count = len(outcomes[key])
        rows.append(
            LeaderboardRow(
                config=key,
                crop=Crop(config.crop).value,
                plots=config.plots,
                sell_reserve=config.sell_reserve,
                games=game_count,
                wins=wins,
                draws=draws,
                losses=losses,
                win_rate=wins / game_count,
                match_points=(wins + 0.5 * draws) / game_count,
                mean_money=fmean(money[key]),
                median_money=median(money[key]),
                worst_money=min(money[key]),
                mean_margin=fmean(margins[key]),
                p10_margin=_percentile(margins[key], 0.10),
                worst_margin=min(margins[key]),
                blocked_rate=blocked[key] / total_steps[key],
            )
        )

    # Lower-tail margin is the primary guardrail; match points and mean score
    # break ties without hiding the actual metrics behind an arbitrary scalar.
    rows.sort(
        key=lambda row: (
            -row.p10_margin,
            -row.match_points,
            -row.mean_margin,
            -row.mean_money,
            row.config,
        )
    )
    return rows


def format_leaderboard(rows: Sequence[LeaderboardRow], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    lines = [
        "| # | config | games | W/D/L | win | mean $ | worst $ | mean margin | "
        "p10 margin | worst margin | blocked |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(shown, 1):
        lines.append(
            "| "
            f"{rank} | {row.config} | {row.games} | {row.wins}/{row.draws}/{row.losses} | "
            f"{row.win_rate:.1%} | {row.mean_money:.1f} | {row.worst_money:.1f} | "
            f"{row.mean_margin:+.1f} | {row.p10_margin:+.1f} | {row.worst_margin:+.1f} | "
            f"{row.blocked_rate:.1%} |"
        )
    return "\n".join(lines)
