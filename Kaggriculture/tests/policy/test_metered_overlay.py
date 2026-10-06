from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.metered_overlay import MeteredSellingOverlay

BULK = {
    "farmer": ["NORTH"],
    "hands": [],
    "market": [["SELL", "FERTILIZER", 145], ["SELL", "WHEAT", 30]],
}
QUIET = {"farmer": ["NORTH"], "hands": [], "market": []}


def bulk_policy(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    if int(observation.get("step", 0)) == 240:
        return dict(BULK)
    return dict(QUIET)


def test_bulk_dump_is_trimmed_and_backlogged() -> None:
    overlay = MeteredSellingOverlay(bulk_policy)
    first = overlay.agent({"player": 0, "step": 240}, None)
    assert first["farmer"] == ["NORTH"]
    assert ["SELL", "FERTILIZER", 15] in first["market"]
    assert ["SELL", "WHEAT", 12] in first["market"]
    assert overlay.backlog_for[0] == {"FERTILIZER": 130, "WHEAT": 18}


def test_backlog_replays_on_later_days() -> None:
    overlay = MeteredSellingOverlay(bulk_policy)
    overlay.agent({"player": 0, "step": 240}, None)
    replayed: list[list[Any]] = []
    for step in range(264, 290):
        action = overlay.agent({"player": 0, "step": step}, None)
        for order in action["market"]:
            if order[0] == "SELL":
                replayed.append(order)
    fert = sum(o[2] for o in replayed if o[1] == "FERTILIZER")
    wheat = sum(o[2] for o in replayed if o[1] == "WHEAT")
    assert fert + wheat > 0
    assert overlay.backlog_for[0]["FERTILIZER"] < 130


def test_small_sales_pass_through() -> None:
    caps = {"FERTILIZER": 1000, "WHEAT": 1000}
    overlay = MeteredSellingOverlay(bulk_policy, daily_caps=caps)
    action = overlay.agent({"player": 1, "step": 240}, None)
    assert ["SELL", "FERTILIZER", 145] in action["market"]
    assert overlay.backlog_for.get(1, {}) == {}


def test_reset_clears_state() -> None:
    overlay = MeteredSellingOverlay(bulk_policy)
    overlay.agent({"player": 0, "step": 240}, None)
    overlay.reset(0)
    assert overlay.backlog_for.get(0, {}) == {}
    overlay.agent({"player": 0, "step": 240}, None)
    overlay.reset()
    assert overlay.backlog_for == {}
