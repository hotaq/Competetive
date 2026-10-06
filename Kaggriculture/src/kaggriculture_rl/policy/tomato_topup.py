"""Day-18 tomato seed top-up overlay (v111): replay the mined top-up wave.

Source: `analysis/v105_tomato_20260922/TOMATO_PATTERNS.md` §1.2 — a scripted
end-of-day-18 top-up wave recurs across 50+ pool tapes as 1-5-unit trickles
(pool-wide 187 units on day 18). The sharpest instance: all six KawattaTaido
tapes buy 4-5 units of TOMATO seed at turn 435 exactly. The omar base route
buys 0 tomato seeds all game (v105 §2), so this overlay adds that wave as a
seeds-only top-up.

Rule: at observation step == TRIGGER_STEP (435) only, if our cash covers the
buy plus a frozen reserve floor AND the tape leaves a free market slot,
append ["BUY_SEED", "TOMATO", TOPUP_N] after the tape's own orders. Every
other step, and every firing-blocked game, returns the action untouched.

Non-goals (explicit): no planting-labour changes — farmer/hands are never
touched, so the bought seeds are planted only if the tape's own schedule
happens to plant them (see the labour caveat in the v111 REPORT). No herd,
no sell timing, no opening changes.

Wiring: the trigger step is state-independent but the cash/cap gates read
live observations, so like `carrot_swap` the screen bakes per-game tapes
closed-loop (`scripts/bake_tomato_topup_tapes.py`) and plays them with
`play_batch`. This module stays dependency-free and seat-agnostic.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

# Turn 435 exactly: all six KawattaTaido pool tapes buy 4-5 tomato seeds at
# this turn (v105 §1.2). Routes-file index == turn (719 steps, turns 0..718).
TRIGGER_STEP = 435
# 4 units: lower end of the mined 4-5-unit KawattaTaido instance (fail closed).
TOPUP_N = 4
# TOMATO seed price (`rust/kaggriculture-engine/src/rules.rs` crop_rule).
TOMATO_SEED_COST = 50
TOPUP_COST = TOPUP_N * TOMATO_SEED_COST  # 200
# Frozen reserve floor: the tape's remaining known deterministic BUY_SEED
# spend after step 435 is WHEAT 70 x 10 + CARROT 31 x 20 = 1320, plus 80
# BUY_PRODUCT FERTILIZER units at dynamic prices (bounded at 20/unit -> 1600)
# = 2920, rounded to 3000. The buy fires only if cash >= 200 + 3000, so it
# cannot bankrupt the day-18+ land/animal schedule. Hires ride on ongoing
# income exactly as in the base route; the top-up never displaces them
# because it only fills a free market slot (cap check below).
RESERVE_FLOOR = 3000
CASH_GATE = TOPUP_COST + RESERVE_FLOOR  # 3200
# Engine resolves at most 10 market orders per turn, lockstep 1 unit at a
# time; appending past the cap would silently drop an order, so the overlay
# fires only into a free slot and never reorders or drops tape orders.
MAX_MARKET_ORDERS = 10


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def observation_step(observation: Any) -> int:
    """Step index, falling back to day/hour (mirrors `opening_override`)."""
    raw = _get(observation, "step")
    if raw is None:
        step = int(_get(observation, "day", 0) or 0) * 24 + int(
            _get(observation, "hour", 0) or 0
        )
    else:
        step = int(raw or 0)
    return max(step, 0)


def _seat_farm(observation: Any) -> Any:
    player = int(_get(observation, "player", 0) or 0)
    farms = _get(observation, "farms", []) or []
    seat = 1 if player == 1 else 0
    if 0 <= seat < len(farms):
        return farms[seat]
    return {}


def cash(observation: Any) -> float:
    """Our current cash (same read as `market_rules.BuyLandRule`)."""
    farm = _seat_farm(observation)
    return float(farm.get("money", 0.0) or 0.0) if isinstance(farm, dict) else 0.0


def _apply_with_stats(observation: Any, action: Action) -> tuple[Action, bool]:
    """Apply the overlay; also return whether the top-up fired."""
    if observation_step(observation) != TRIGGER_STEP:
        return action, False
    if cash(observation) < CASH_GATE:
        return action, False
    market = list(action.get("market", []) or [])
    if len(market) >= MAX_MARKET_ORDERS:
        return action, False
    updated = copy.deepcopy(action)
    updated["market"] = [*list(updated.get("market", []) or []), ["BUY_SEED", "TOMATO", TOPUP_N]]
    return updated, True


def apply(observation: Any, action: Action) -> Action:
    """Return `action` with the day-18 tomato top-up appended when gated."""
    updated, _ = _apply_with_stats(observation, action)
    return updated


class TomatoTopupOverlay:
    """Wrap a raw-action policy with the day-18 tomato seed top-up."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy
        self._topups = 0

    @property
    def topups(self) -> int:
        return self._topups

    def reset(self) -> None:
        self._topups = 0

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        updated, fired = _apply_with_stats(observation, action)
        self._topups += int(fired)
        return updated

    __call__ = agent
