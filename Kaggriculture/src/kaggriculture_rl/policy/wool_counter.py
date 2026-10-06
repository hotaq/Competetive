"""Wool counter overlay (v114): shop-conditional wool hold + land discipline.

Source: `analysis/v114_wool_counter_20260923/MINING.md` (EP 112504824,
wool winner P0 106,639 vs yannik-route P1 85,130, town YARN_STORE day 3
+ SMOOTHIE/PET_CAFE/BAKERY/BRUNCH/SMOOTHIE/BAKERY/YARN_STORE, wool
200-237 flat all game, no spike).

Two frozen rules, both removal-only (never add/reorder market orders, never
touch farmer/hands — same safety shape as `vt1_tail_trim`):

1. Wool hold (shop-conditional): if the first revealed shop is the wool-only
   buyer (`WOOL_DEMAND_SHOPS` = {YARN_STORE}, from
   `rust/kaggriculture-engine/src/rules.rs::shop_products`) and the day is
   before `HOLD_UNTIL_DAY` (10), drop every `SELL WOOL` order so inventory
   rides into the tape's own day-10+ sells. Mining: the base sells ~26 wool
   units on days 5-9 while its own days-10+ fixed-n wool orders total ~134,
   so held inventory is fully absorbed (no shed-rot; last-frame money ==
   rewards, no terminal liquidation); wool price humps days 10-16 (235-237)
   at or above days 5-9 (229-235). Wheat is never touched (herd FEED eats
   wheat; reserve-floor lesson), hires never touched, the 10-cap can only
   gain free slots.
2. Land discipline (state-gated, v113 lesson 1): drop `BUY_LAND` only when
   our own `unlocked_quadrants` already holds `MAX_QUADRANTS` (4 = NW +
   `LAND_ORDER`'s 3 entries). At max the engine's `do_buy_land` returns
   before charging, so a dropped order is provably dead — never by order
   count, which confused failed orders with dead ones on fresh seeds.

Wiring: the shop gate reads live observations, so like `carrot_swap` /
`tomato_topup` / `petcafe_counter` the screen bakes per-game tapes
closed-loop (`scripts/bake_wool_counter_tapes.py`) and plays them with
`play_batch`. The overlay is stateful across turns (stats only; the land
gate reads live state), so baking holds one instance per game; the pure
helpers stay unit-testable.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

TURNS_PER_DAY = 24
# Wool-only buyer (rules.rs::shop_products): the only shop that buys WOOL.
WOOL_DEMAND_SHOPS = frozenset({"YARN_STORE"})
# Drop SELL WOOL while day < 10; the tape's own day-10+ sells stay intact.
HOLD_UNTIL_DAY = 10
# NW start + LAND_ORDER (NE/SW/SE): at 4 quadrants no unlock remains.
MAX_QUADRANTS = 1 + 3
# Engine resolves at most 10 market orders per turn; removal-only here.
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
        step = int(_get(observation, "day", 0) or 0) * TURNS_PER_DAY + int(
            _get(observation, "hour", 0) or 0
        )
    else:
        step = int(raw or 0)
    return max(step, 0)


def observation_day(observation: Any) -> int:
    raw = _get(observation, "day")
    if raw is None:
        raw = observation_step(observation) // TURNS_PER_DAY
    return max(int(raw or 0), 0)


def revealed_shops(observation: Any) -> list[str]:
    town = _get(observation, "town", {}) or {}
    shops = _get(town, "unlocked_shops", []) or []
    return [str(shop) for shop in shops]


def wool_draw(shops: list[str]) -> bool:
    """True when the first revealed shop is the wool-only buyer."""
    return len(shops) >= 1 and shops[0] in WOOL_DEMAND_SHOPS


def unlocked_quadrant_count(observation: Any) -> int | None:
    """Our unlocked quadrant count, or None when the observation is malformed."""
    player = int(_get(observation, "player", 0) or 0)
    farms = _get(observation, "farms", []) or []
    if not isinstance(farms, list) or not 0 <= player < len(farms):
        return None
    farm = farms[player]
    quadrants = farm.get("unlocked_quadrants", None) if isinstance(farm, dict) else None
    return len(quadrants) if isinstance(quadrants, list) else None


def _is_sell_wool(order: Any) -> bool:
    return isinstance(order, list) and order[:2] == ["SELL", "WOOL"]


def _is_buy_land(order: Any) -> bool:
    return isinstance(order, list) and order[:1] == ["BUY_LAND"]


def apply_rules(observation: Any, action: Action) -> tuple[Action, dict[str, int]]:
    """Pure rule core: returns (updated action, stats)."""
    updated = copy.deepcopy(action)
    stats = {"wool_held_units": 0, "land_dropped": 0}
    market = list(updated.get("market", []) or [])
    assert len(market) <= MAX_MARKET_ORDERS

    if wool_draw(revealed_shops(observation)) and observation_day(
        observation
    ) < HOLD_UNTIL_DAY:
        kept = []
        for order in market:
            if _is_sell_wool(order):
                stats["wool_held_units"] += int(order[2]) if len(order) > 2 else 1
            else:
                kept.append(order)
        market = kept

    quadrants = unlocked_quadrant_count(observation)
    if quadrants is not None and quadrants >= MAX_QUADRANTS:
        kept = []
        for order in market:
            if _is_buy_land(order):
                stats["land_dropped"] += 1
            else:
                kept.append(order)
        market = kept

    updated["market"] = market
    return updated, stats


class WoolCounterOverlay:
    """Wrap a raw-action policy with the v114 wool counter rules."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy
        self._held_units = 0
        self._land_dropped = 0

    @property
    def held_units(self) -> int:
        return self._held_units

    @property
    def land_dropped(self) -> int:
        return self._land_dropped

    def reset(self) -> None:
        self._held_units = 0
        self._land_dropped = 0

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        updated, stats = apply_rules(observation, action)
        self._held_units += stats["wool_held_units"]
        self._land_dropped += stats["land_dropped"]
        return updated

    __call__ = agent
