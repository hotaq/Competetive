from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = str(PROJECT_ROOT)
root_was_importable = project_root in sys.path
if not root_was_importable:
    sys.path.insert(0, project_root)
try:
    overlay = importlib.import_module("artifacts.v68_front_run_robust_20260826.overlay")
finally:
    if not root_was_importable:
        sys.path.remove(project_root)


def _action(market: list[list[Any]]) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": copy.deepcopy(market)}


def _observation(
    step: int,
    *,
    player: int = 0,
    shed: dict[str, int] | None = None,
    shops: tuple[str, ...] = (),
    base_market: list[list[Any]] | None = None,
    future_market: list[list[Any]] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "player": player,
        "private": {"shed": dict(shed or {})},
        "town": {"unlocked_shops": list(shops)},
        "_base_action": _action(base_market or []),
        "_future_action": _action(future_market or []),
    }


def _stub_policy_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        overlay.v59,
        "agent",
        lambda observation, configuration=None: copy.deepcopy(observation["_base_action"]),
    )
    monkeypatch.setattr(
        overlay,
        "_future_action",
        lambda observation, configuration, player, step: copy.deepcopy(
            observation["_future_action"]
        ),
    )


def test_town_demand_uses_day_and_shop_tick_boundaries() -> None:
    observation = {"town": {"unlocked_shops": ["YARN_STORE", "PIZZA_SHOP"]}}

    assert overlay._town_demand(observation, "WOOL", 1) == 0
    assert overlay._town_demand(observation, "WOOL", 4) == 2
    assert overlay._town_demand(observation, "WOOL", 24) == 3
    assert overlay._town_demand(observation, "MILK", 4) == 1
    assert overlay._town_demand(observation, "MILK", 24) == 2
    assert overlay._town_demand(observation, "FERTILIZER", 24) == 0


@pytest.mark.parametrize(
    ("step", "shops", "expected_market"),
    [
        (1, (), [["SELL", "MILK", 3]]),
        (24, (), []),
        (4, ("PIZZA_SHOP",), []),
        (4, ("YARN_STORE",), [["SELL", "MILK", 3]]),
    ],
)
def test_demand_safe_agent_skips_only_current_demand_ticks(
    monkeypatch: pytest.MonkeyPatch,
    step: int,
    shops: tuple[str, ...],
    expected_market: list[list[Any]],
) -> None:
    _stub_policy_sources(monkeypatch)
    policy = overlay.make_agent(("MILK",), skip_current_town_demand=True)
    observation = _observation(
        step,
        shed={"MILK": 5},
        shops=shops,
        future_market=[["SELL", "MILK", 3]],
    )

    assert policy(observation, None)["market"] == expected_market


def test_sum_existing_sells_prevents_duplicate_sell_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_sources(monkeypatch)
    policy = overlay.make_agent(("MELON",), sum_existing_sells=True)
    observation = _observation(
        10,
        shed={"MELON": 10},
        base_market=[["SELL", "MELON", 4], ["SELL", "MELON", 3]],
        future_market=[["SELL", "MELON", 10]],
    )

    result = policy(observation, {"maxMarketOrdersPerTurn": 10})
    melon_sells = [order for order in result["market"] if order[:2] == ["SELL", "MELON"]]

    assert melon_sells == [["SELL", "MELON", 7], ["SELL", "MELON", 3]]
    assert sum(int(order[2]) for order in melon_sells) == 10
    assert observation["_base_action"]["market"] == [
        ["SELL", "MELON", 4],
        ["SELL", "MELON", 3],
    ]


@pytest.mark.parametrize(
    "at_cap_configuration",
    [
        pytest.param({"maxMarketOrdersPerTurn": 1}, id="mapping"),
        pytest.param(SimpleNamespace(maxMarketOrdersPerTurn=1), id="attribute"),
    ],
)
def test_configured_market_order_limit_controls_append_boundary(
    monkeypatch: pytest.MonkeyPatch,
    at_cap_configuration: object,
) -> None:
    _stub_policy_sources(monkeypatch)
    observation = _observation(
        10,
        shed={"MELON": 5},
        base_market=[["BUY_SEED", "WHEAT", 1]],
        future_market=[["SELL", "MELON", 5]],
    )

    at_cap = overlay.make_agent(("MELON",))(observation, at_cap_configuration)
    below_cap = overlay.make_agent(("MELON",))(observation, {"maxMarketOrdersPerTurn": 2})

    assert at_cap["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert below_cap["market"] == [
        ["BUY_SEED", "WHEAT", 1],
        ["SELL", "MELON", 5],
    ]


def test_repayment_debt_is_isolated_by_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_policy_sources(monkeypatch)
    policy = overlay.make_agent(("MELON",))

    front_run = policy(
        _observation(
            10,
            player=0,
            shed={"MELON": 10},
            future_market=[["SELL", "MELON", 3]],
        ),
        None,
    )
    other_player = policy(
        _observation(
            11,
            player=1,
            shed={"MELON": 10},
            base_market=[["SELL", "MELON", 5]],
        ),
        None,
    )
    indebted_player = policy(
        _observation(
            11,
            player=0,
            shed={"MELON": 10},
            base_market=[["SELL", "MELON", 5]],
        ),
        None,
    )

    assert front_run["market"] == [["SELL", "MELON", 3]]
    assert other_player["market"] == [["SELL", "MELON", 5]]
    assert indebted_player["market"] == [["SELL", "MELON", 2]]


@pytest.mark.parametrize("reset_step", [0, 9], ids=["new-episode", "rewind"])
def test_repayment_debt_resets_on_episode_boundary(
    monkeypatch: pytest.MonkeyPatch,
    reset_step: int,
) -> None:
    _stub_policy_sources(monkeypatch)
    policy = overlay.make_agent(("MELON",))
    policy(
        _observation(
            10,
            shed={"MELON": 10},
            future_market=[["SELL", "MELON", 3]],
        ),
        None,
    )

    result = policy(
        _observation(
            reset_step,
            shed={"MELON": 10},
            base_market=[["SELL", "MELON", 5]],
        ),
        None,
    )

    assert result["market"] == [["SELL", "MELON", 5]]


@pytest.mark.parametrize("step", [71, 119, 696, 719])
def test_future_action_excluded_step_boundaries(step: int) -> None:
    assert overlay._future_action({}, None, player=0, step=step) == {}
