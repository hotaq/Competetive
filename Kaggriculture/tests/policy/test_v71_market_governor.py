from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = str(PROJECT_ROOT)
root_was_importable = project_root in sys.path
if not root_was_importable:
    sys.path.insert(0, project_root)
try:
    overlay = importlib.import_module("artifacts.v71_market_governor_20260826.overlay")
finally:
    if not root_was_importable:
        sys.path.remove(project_root)


def _action(market: list[list[Any]]) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": copy.deepcopy(market)}


def _observation(
    step: int,
    *,
    player: int = 0,
    money: float = 3000,
    shed: dict[str, int] | None = None,
    shops: tuple[str, ...] = (),
    prices: dict[str, float] | None = None,
    base_market: list[list[Any]] | None = None,
    future_market: list[list[Any]] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "player": player,
        "farms": [{"money": money}, {"money": money}],
        "private": {"shed": dict(shed or {})},
        "town": {"unlocked_shops": list(shops)},
        "market": {"prices": dict(prices or {})},
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
        overlay.v68,
        "_future_action",
        lambda observation, configuration, player, step, correct_clock=False: copy.deepcopy(
            observation["_future_action"]
        ),
    )


def test_town_demand_matches_engine_clock_and_unlocked_shops() -> None:
    observation = {"town": {"unlocked_shops": ["YARN_STORE", "PIZZA_SHOP"]}}
    assert overlay._town_demand(observation, {}, "WOOL", 24) == 3
    assert overlay._town_demand(observation, {}, "MILK", 24) == 2
    assert overlay._town_demand(observation, {}, "WOOL", 25) == 0


def test_early_exit_moves_quantity_one_step_without_creating_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_sources(monkeypatch)
    agent = overlay.make_agent()

    agent(
        _observation(133, prices={"WOOL": 217}),
        {},
    )
    early = agent(
        _observation(
            157,
            money=3074,
            shed={"WOOL": 4},
            prices={"WOOL": 174},
            future_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )
    repaid = agent(
        _observation(
            158,
            money=3074,
            shed={"WOOL": 2},
            prices={"WOOL": 169},
            base_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )

    assert early["market"] == [["SELL", "WOOL", 2]]
    assert repaid["market"] == []
    assert overlay._sell_quantity(early, "WOOL") + overlay._sell_quantity(
        repaid, "WOOL"
    ) == 2


def test_reserve_moves_quantity_after_current_town_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_sources(monkeypatch)
    agent = overlay.make_agent(early_items=(), reserve_items=("WOOL",))

    agent(_observation(524, prices={"WOOL": 88}), {})
    reserved = agent(
        _observation(
            548,
            shed={"WOOL": 16},
            shops=("YARN_STORE",),
            prices={"WOOL": 93},
            base_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )
    repaid = agent(
        _observation(549, shed={"WOOL": 16}, prices={"WOOL": 95}),
        {},
    )

    assert reserved["market"] == []
    assert repaid["market"] == [["SELL", "WOOL", 2]]
    assert overlay._sell_quantity(reserved, "WOOL") + overlay._sell_quantity(
        repaid, "WOOL"
    ) == 2


def test_market_order_cap_prevents_unpayable_early_debt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_policy_sources(monkeypatch)
    agent = overlay.make_agent()
    full_market = [["BUY_SEED", "WHEAT", 1] for _ in range(10)]

    agent(_observation(133, prices={"WOOL": 217}), {"maxMarketOrdersPerTurn": 10})
    unchanged = agent(
        _observation(
            157,
            money=3074,
            shed={"WOOL": 4},
            prices={"WOOL": 174},
            base_market=full_market,
            future_market=[["SELL", "WOOL", 2]],
        ),
        {"maxMarketOrdersPerTurn": 10},
    )
    next_action = agent(
        _observation(
            158,
            prices={"WOOL": 169},
            base_market=[["SELL", "WOOL", 2]],
        ),
        {"maxMarketOrdersPerTurn": 10},
    )

    assert unchanged["market"] == full_market
    assert next_action["market"] == [["SELL", "WOOL", 2]]


def test_timing_debt_is_isolated_by_player(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_policy_sources(monkeypatch)
    agent = overlay.make_agent()
    agent(_observation(133, player=0, prices={"WOOL": 217}), {})
    agent(_observation(133, player=1, prices={"WOOL": 217}), {})
    agent(
        _observation(
            157,
            player=0,
            shed={"WOOL": 4},
            prices={"WOOL": 174},
            future_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )

    player_one = agent(
        _observation(
            158,
            player=1,
            prices={"WOOL": 169},
            base_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )
    player_zero = agent(
        _observation(
            158,
            player=0,
            prices={"WOOL": 169},
            base_market=[["SELL", "WOOL", 2]],
        ),
        {},
    )

    assert player_one["market"] == [["SELL", "WOOL", 2]]
    assert player_zero["market"] == []
