from __future__ import annotations

import pytest

from kaggriculture_rl.evaluation import (
    CropConfig,
    GameResult,
    format_leaderboard,
    rank_results,
    round_robin,
    run_match,
)
from kaggriculture_rl.schemas.models import Crop


def test_rank_results_preserves_money_and_lower_tail_metrics() -> None:
    wheat = CropConfig(Crop.WHEAT, 2)
    carrot = CropConfig(Crop.CARROT, 2)
    games = [
        GameResult(
            players=(wheat.key, carrot.key),
            seed=7,
            steps=100,
            money=(3_100.0, 3_000.0),
            blocked_reports=(0, 10),
        ),
        GameResult(
            players=(carrot.key, wheat.key),
            seed=7,
            steps=100,
            money=(2_990.0, 3_010.0),
            blocked_reports=(20, 0),
        ),
    ]

    rows = rank_results([wheat, carrot], games)

    assert rows[0].config == wheat.key
    assert rows[0].games == 2
    assert rows[0].wins == 2
    assert rows[0].mean_money == 3_055.0
    assert rows[0].worst_money == 3_010.0
    assert rows[0].worst_margin == 20.0
    assert rows[0].blocked_rate == 0.0
    assert rows[1].blocked_rate == pytest.approx(0.15)
    assert wheat.key in format_leaderboard(rows)


def test_match_is_deterministic_for_same_seed_and_seats() -> None:
    wheat = CropConfig(Crop.WHEAT, 1)
    carrot = CropConfig(Crop.CARROT, 1)

    first = run_match(wheat, carrot, seed=29, episode_steps=72, weed_spawn_chance=0.0)
    second = run_match(wheat, carrot, seed=29, episode_steps=72, weed_spawn_chance=0.0)

    assert first == second
    assert first.steps > 0
    assert first.players == (wheat.key, carrot.key)


def test_round_robin_runs_both_seat_orders() -> None:
    wheat = CropConfig(Crop.WHEAT, 1)
    carrot = CropConfig(Crop.CARROT, 1)

    games = round_robin(
        [wheat, carrot],
        seeds=[31],
        episode_steps=12,
        weed_spawn_chance=0.0,
    )

    assert [game.players for game in games] == [
        (wheat.key, carrot.key),
        (carrot.key, wheat.key),
    ]


@pytest.mark.parametrize("plots", [0, -1])
def test_crop_config_rejects_non_positive_plot_count(plots: int) -> None:
    with pytest.raises(ValueError, match="plots must be positive"):
        CropConfig(Crop.WHEAT, plots)
