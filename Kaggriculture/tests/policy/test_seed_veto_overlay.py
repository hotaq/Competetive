from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.seed_veto_overlay import SeedVetoOverlay

ORDERS = [
    ["BUY_SEED", "STRAWBERRY", 5],
    ["BUY_SEED", "WHEAT", 9],
    ["HIRE"],
    ["SELL", "MELON", 6],
]


def buying_policy(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in ORDERS]}


def test_veto_drops_only_matching_day_and_crop() -> None:
    overlay = SeedVetoOverlay(buying_policy)
    vetoed = overlay.agent({"player": 0, "step": 240}, None)
    assert ["BUY_SEED", "STRAWBERRY", 5] not in vetoed["market"]
    assert ["BUY_SEED", "WHEAT", 9] in vetoed["market"]
    assert ["HIRE"] in vetoed["market"]
    assert overlay.dropped_total == 1
    open_day = overlay.agent({"player": 0, "step": 24}, None)
    assert ["BUY_SEED", "STRAWBERRY", 5] in open_day["market"]


def test_custom_veto_set() -> None:
    overlay = SeedVetoOverlay(buying_policy, veto={(1, "WHEAT")})
    action = overlay.agent({"player": 0, "step": 24}, None)
    assert ["BUY_SEED", "WHEAT", 9] not in action["market"]
    assert ["BUY_SEED", "STRAWBERRY", 5] in action["market"]
    overlay.reset()
    assert overlay.dropped_total == 0
