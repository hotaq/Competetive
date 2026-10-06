"""Pet-cafe counter overlay (v113): shop-conditional carrot hold + land discipline.

Source: `analysis/v113_petcafe_counter_20260923/MINING.md` (EP 112491838,
k hori P0 vs yannik-route P1, town PET_CAFE x4 + SMOOTHIE + PIZZA + BAKERY
+ ICE_CREAM, carrot hinge-scarcity spike 271 -> 326 -> 354 on days 27-29).

Two frozen rules, both removal-only (never add/reorder market orders, never
touch farmer/hands — same safety shape as `vt1_tail_trim`):

1. Carrot hold (shop-conditional): if the first two revealed shops are both
   carrot-demand shops (`CARROT_DEMAND_SHOPS`, from
   `rust/kaggriculture-engine/src/rules.rs::shop_products` — PET_CAFE buys
   CARROT only, FARMERS_MARKET buys WHEAT/CARROT/TOMATO/STRAWBERRY) and the
   day is before `HOLD_UNTIL_DAY` (27), drop every `SELL CARROT` order so
   inventory rides into the tape's own day-27+ sells. Wheat is never
   touched (herd FEED eats wheat; reserve-floor lesson), hires never
   touched, the 10-cap can only gain free slots.
2. Land discipline (unconditional): at most `MAX_LAND_ORDERS` (3)
   BUY_LAND orders per game; the 4th+ is dropped. Mining: the yannik route
   issues 4 (steps 146/198/199/200); the 4th is a dead no-op
   (`do_buy_land` returns before charging once 3 extra quadrants hold, and
   SE never unlocks — LOCKED tiles stuck at 25).

Wiring: the shop gate reads live observations, so like `carrot_swap` /
`tomato_topup` the screen bakes per-game tapes closed-loop
(`scripts/bake_petcafe_counter_tapes.py`) and plays them with
`play_batch`. The overlay is stateful across turns (land count), so baking
holds one instance per game; the pure helpers stay unit-testable.
Seat-agnostic: no farm/private reads at all (town + day + action only).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

TURNS_PER_DAY = 24
# Carrot-demand shops (rules.rs::shop_products): the only shops that buy CARROT.
CARROT_DEMAND_SHOPS = frozenset({"PET_CAFE", "FARMERS_MARKET"})
# Drop SELL CARROT while day < 27; the tape's own day-27+ dumps stay intact.
HOLD_UNTIL_DAY = 27
# LAND_ORDER has 3 entries (NE/SW/SE); a 4th BUY_LAND is always dead.
MAX_LAND_ORDERS = 3
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


def petcafe_draw(shops: list[str]) -> bool:
    """True when the first two revealed shops are both carrot-demand shops."""
    return (
        len(shops) >= 2
        and shops[0] in CARROT_DEMAND_SHOPS
        and shops[1] in CARROT_DEMAND_SHOPS
    )


def _is_sell_carrot(order: Any) -> bool:
    return isinstance(order, list) and order[:2] == ["SELL", "CARROT"]


def _is_buy_land(order: Any) -> bool:
    return isinstance(order, list) and order[:1] == ["BUY_LAND"]


def apply_rules(
    observation: Any, action: Action, land_used: int
) -> tuple[Action, dict[str, int]]:
    """Pure rule core: returns (updated action, stats).

    `land_used` = BUY_LAND orders already issued this game; the caller owns
    the counter (the overlay class threads it per game).
    """
    updated = copy.deepcopy(action)
    stats = {"carrot_held_units": 0, "land_dropped": 0}
    market = list(updated.get("market", []) or [])
    assert len(market) <= MAX_MARKET_ORDERS

    hold = petcafe_draw(revealed_shops(observation)) and observation_day(
        observation
    ) < HOLD_UNTIL_DAY
    if hold:
        kept = []
        for order in market:
            if _is_sell_carrot(order):
                stats["carrot_held_units"] += int(order[2]) if len(order) > 2 else 1
            else:
                kept.append(order)
        market = kept

    if land_used >= MAX_LAND_ORDERS:
        kept = []
        for order in market:
            if _is_buy_land(order):
                stats["land_dropped"] += 1
            else:
                kept.append(order)
        market = kept
        land_allowed = 0
    else:
        land_allowed = MAX_LAND_ORDERS - land_used
    lands = [o for o in market if _is_buy_land(o)]
    if len(lands) > land_allowed:
        drop = len(lands) - land_allowed
        kept = []
        for order in market:
            if _is_buy_land(order) and drop > 0:
                drop -= 1
                stats["land_dropped"] += 1
                continue
            kept.append(order)
        market = kept

    updated["market"] = market
    return updated, stats


class PetcafeCounterOverlay:
    """Wrap a raw-action policy with the v113 pet-cafe counter rules."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy
        self._land_used = 0
        self._held_units = 0
        self._land_dropped = 0

    @property
    def land_used(self) -> int:
        return self._land_used

    @property
    def held_units(self) -> int:
        return self._held_units

    @property
    def land_dropped(self) -> int:
        return self._land_dropped

    def reset(self) -> None:
        self._land_used = 0
        self._held_units = 0
        self._land_dropped = 0

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        lands_now = sum(
            1 for o in (action.get("market", []) or []) if _is_buy_land(o)
        )
        updated, stats = apply_rules(observation, action, self._land_used)
        self._land_used += lands_now - stats["land_dropped"]
        self._held_units += stats["carrot_held_units"]
        self._land_dropped += stats["land_dropped"]
        return updated

    __call__ = agent
