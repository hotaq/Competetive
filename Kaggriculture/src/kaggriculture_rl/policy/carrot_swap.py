"""CARROT2 crop-swap overlay (v9/4 style): swap scheduled PLANT WHEAT to CARROT.

Source claim: CARROT2 measured +54/-10 over 3296 games in the source notebook.
Mechanism: mid-game carrot prices spike (boom regime) while wheat stays flat;
a wheat plant scheduled on days 10..23 earns more as carrot when the ahead
price ratio clears 1.8 (3.5 in boom), provided the herd's wheat reserve holds.

Engine ground truth (`rust/kaggriculture-engine`, verified 2026-09-22):
- crops (`rules.rs::crop_rule`): WHEAT seed 10 first-yield-day 2 max-yield-day 4
  cap 6; CARROT seed 20 first-yield-day 2 max-yield-day 3 cap 4; both
  non-ongoing (one harvest clears the tile).
- new plant (`model.rs::TileObject::plant`): yield starts at 1, lifespan step
  ``(planted_day + max_yield_day + 1) * 24`` (decay bleeds 1 yield every
  2 steps from there, then the tile weeds), ``consecutive_unwatered`` starts
  at 1 -- a plant NOT watered on its plant day weeds at day end.
- watering (`engine.rs`, WATER arm): only inside the age window
  ``(max_yield_day + 1) // 2 ..= max_yield_day`` (wheat ages 2..4, carrot
  ages 2..3); +1 yield, +2 while ``fertilized_until_day >= day``.
- FERTILIZE sets ``max(old, day + 2)`` -- covers day, day+1, day+2 (3 days).
- harvest allowed from age ``first_yield_day``; non-ongoing harvest banks the
  yield and clears the tile. FEED eats 1 WHEAT per animal tile.

Swap rule: a scheduled ``["PLANT", "WHEAT"]`` becomes ``["PLANT", "CARROT"]``
only if ALL hold: ``3 * carrot_price - 20 > 4 * wheat_price - 10`` at ahead
prices (ahead = current quoted price, the only observable proxy), price ratio
``carrot / wheat >= 1.8`` (boom regime at ``>= 3.5``), ``wheat_held`` (shed +
carried) minus a 2-day herd reserve (2 wheat per animal tile) covers 40 wheat
(10 in boom, else a BUY_PRODUCT WHEAT top-up is scheduled in a free slot),
day in 10..23, a carrot seed is held (else BUY_SEED CARROT is staged for a
later turn but today's plant is left alone -- same-turn seeds are unplantable
under the engine atomic rule), and the tile's plant-day watering is assured
(a second unit on the tile waters today, or a PASS unit on the tile is
upgraded to WATER).

Rescue rule: an age-3 carrot (last water-window day) with banked yield that
would miss watering today (no WATER on its tile, not yet watered) becomes
HARVEST on the standing unit -- banking beats risking the weed/decay bleed.

Wiring: both gates are observation-dependent, so unlike the step-0
`opening_override` there is no sound static routes patch -- the Rust-loop
screen bakes per-game tapes closed-loop (`scripts/bake_carrot_orderpri2_tapes.py`)
and plays them with `play_batch`. This module stays dependency-free and
seat-agnostic (pure per-observation reads); it can be inlined into a
standalone bundle.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

TURNS_PER_DAY = 24
GAME_STEPS = 719

# (seed_cost, first_yield_day, max_yield_day, max_yield) from rules.rs.
CROP_FACTS: dict[str, tuple[int, int, int, int]] = {
    "WHEAT": (10, 2, 4, 6),
    "CARROT": (20, 2, 3, 4),
}

SWAP_DAY_LO = 10
SWAP_DAY_HI = 23
RATIO_GATE = 1.8
BOOM_RATIO = 3.5
WHEAT_RESERVE_NORMAL = 40
WHEAT_RESERVE_BOOM = 10
MAX_MARKET_ORDERS = 10


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def observation_day(observation: Any) -> int:
    """Day index, falling back to step when day is missing."""
    raw = _get(observation, "day")
    if raw is None:
        raw = int(_get(observation, "step", 0) or 0) // TURNS_PER_DAY
    return max(int(raw or 0), 0)


def observation_step(observation: Any) -> int:
    raw = _get(observation, "step")
    if raw is None:
        step = int(_get(observation, "day", 0) or 0) * TURNS_PER_DAY + int(
            _get(observation, "hour", 0) or 0
        )
    else:
        step = int(raw or 0)
    return max(step, 0)


def _seat_farm(observation: Any) -> tuple[int, Mapping[str, Any]]:
    player = int(_get(observation, "player", 0) or 0)
    farms = _get(observation, "farms", []) or []
    seat = 1 if player == 1 else 0
    farm: Mapping[str, Any] = farms[seat] if 0 <= seat < len(farms) else {}
    return seat, farm if isinstance(farm, Mapping) else {}


def water_window(crop: str) -> tuple[int, int]:
    """Inclusive age window where watering banks yield (engine.rs WATER arm)."""
    _, _, max_yield_day, _ = CROP_FACTS[crop]
    return (max_yield_day + 1) // 2, max_yield_day


def decay_start_step(planted_day: int, crop: str) -> int:
    """First step of the decay bleed: (planted + myd + 1) * 24 (model.rs)."""
    _, _, max_yield_day, _ = CROP_FACTS[crop]
    return (planted_day + max_yield_day + 1) * TURNS_PER_DAY


def harvest_feasible_today(plant_day: int, crop: str) -> bool:
    """A plant sown today can be harvested (age >= first yield) before step 719."""
    _, first_yield_day, _, _ = CROP_FACTS[crop]
    if decay_start_step(plant_day, crop) <= (plant_day + first_yield_day) * TURNS_PER_DAY:
        return False
    return (plant_day + first_yield_day) * TURNS_PER_DAY <= GAME_STEPS


def _prices(observation: Any) -> Mapping[str, Any]:
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    return prices if isinstance(prices, Mapping) else {}


def price_ratio(observation: Any) -> float:
    """Carrot/wheat ahead-price ratio (ahead = current quote, only proxy)."""
    prices = _prices(observation)
    wheat = float(prices.get("WHEAT", 0.0) or 0.0)
    carrot = float(prices.get("CARROT", 0.0) or 0.0)
    if wheat <= 0:
        return 0.0
    return carrot / wheat


def profit_gate(observation: Any) -> bool:
    """3 * carrot - 20 (seed) > 4 * wheat - 10 (seed) at ahead prices."""
    prices = _prices(observation)
    wheat = float(prices.get("WHEAT", 0.0) or 0.0)
    carrot = float(prices.get("CARROT", 0.0) or 0.0)
    return 3.0 * carrot - 20.0 > 4.0 * wheat - 10.0


def wheat_held(observation: Any) -> int:
    """Shed + carried wheat (herd feed buffer counts both)."""
    private = _get(observation, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    total = int(shed.get("WHEAT", 0) or 0) if isinstance(shed, Mapping) else 0
    inventories = _get(private, "inventories", []) or []
    if isinstance(inventories, list):
        for carried in inventories:
            if isinstance(carried, Mapping):
                total += max(0, int(carried.get("WHEAT", 0) or 0))
    return total


def herd_size(observation: Any) -> int:
    """Animal tiles on our farm (1 wheat/day each via FEED)."""
    _, farm = _seat_farm(observation)
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    count = 0
    if isinstance(tiles, list):
        for row in tiles:
            if not isinstance(row, list):
                continue
            for tile in row:
                if isinstance(tile, Mapping) and tile.get("animal"):
                    count += 1
    return count


def swap_gates(observation: Any) -> tuple[bool, bool]:
    """(gates_pass, boom): price, ratio and day gates for the swap pass."""
    day = observation_day(observation)
    if not SWAP_DAY_LO <= day <= SWAP_DAY_HI:
        return False, False
    ratio = price_ratio(observation)
    boom = ratio >= BOOM_RATIO
    if ratio < RATIO_GATE:
        return False, boom
    if not profit_gate(observation):
        return False, boom
    return True, boom


def wheat_threshold(boom: bool) -> int:
    return WHEAT_RESERVE_BOOM if boom else WHEAT_RESERVE_NORMAL


def _tile_at(farm: Mapping[str, Any], x: int, y: int) -> Any:
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    if not isinstance(tiles, list) or not 0 <= y < len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, list) or not 0 <= x < len(row):
        return None
    return row[x]


def _unit_positions(farm: Mapping[str, Any]) -> list[tuple[int, int] | None]:
    """[farmer, *hands] positions; None where unknown."""
    positions: list[tuple[int, int] | None] = []

    def _pos(value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        return None

    positions.append(_pos(farm.get("farmer") if isinstance(farm, Mapping) else None))
    hands = farm.get("hands", []) if isinstance(farm, Mapping) else []
    if isinstance(hands, list):
        for hand in hands:
            positions.append(_pos(hand))
    return positions


def _unit_actions(action: Action) -> list[Any]:
    return [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]


def growing_plant_count(farm: Mapping[str, Any]) -> int:
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    count = 0
    if isinstance(tiles, list):
        for row in tiles:
            if not isinstance(row, list):
                continue
            for tile in row:
                if isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
                    count += 1
    return count


def apply(observation: Any, action: Action) -> Action:
    """Return `action` with CARROT2 swaps, top-ups and rescues applied."""
    updated, _, _ = _apply_with_stats(observation, action)
    return updated


def _apply_with_stats(
    observation: Any, action: Action
) -> tuple[Action, int, int]:
    """Apply the overlay; also return (swaps, rescues) performed."""
    updated = copy.deepcopy(action)
    swaps = 0
    rescues = 0
    _, farm = _seat_farm(observation)
    day = observation_day(observation)
    gates, boom = swap_gates(observation)
    threshold = wheat_threshold(boom)
    reserve = 2 * herd_size(observation)
    held = wheat_held(observation)
    wheat_ok = held - reserve >= threshold

    private = _get(observation, "private", {}) or {}
    seeds = _get(private, "seeds", {}) or {}
    carrot_seeds = int(seeds.get("CARROT", 0) or 0) if isinstance(seeds, Mapping) else 0

    positions = _unit_positions(farm)
    unit_actions = _unit_actions(updated)
    while len(unit_actions) < len(positions):
        unit_actions.append(["PASS"])

    def _set_unit(index: int, op: list[Any]) -> None:
        if index == 0:
            updated["farmer"] = op
        else:
            hand_index = index - 1
            hands = list(updated.get("hands", []) or [])
            while len(hands) <= hand_index:
                hands.append(["PASS"])
            hands[hand_index] = op
            updated["hands"] = hands

    # Rescue pass first: age-3 carrot missing its watering becomes HARVEST.
    if isinstance(farm, Mapping):
        tiles = farm.get("tiles", [])
        if isinstance(tiles, list):
            for y, row in enumerate(tiles):
                if not isinstance(row, list):
                    continue
                for x, tile in enumerate(row):
                    if not isinstance(tile, Mapping):
                        continue
                    if tile.get("kind") != "PLANT" or tile.get("crop") != "CARROT":
                        continue
                    planted = tile.get("planted_day")
                    if planted is None:
                        continue
                    try:
                        age = day - int(planted)
                    except (TypeError, ValueError):
                        continue
                    if age != CROP_FACTS["CARROT"][2]:
                        continue
                    if int(tile.get("yield_units", 0) or 0) <= 0:
                        continue
                    if bool(tile.get("watered_today", False)):
                        continue
                    on_tile = [
                        i for i, pos in enumerate(positions) if pos == (x, y)
                    ]
                    if not on_tile:
                        continue
                    watered = any(
                        isinstance(unit_actions[i], list)
                        and unit_actions[i][:1] == ["WATER"]
                        for i in on_tile
                        if i < len(unit_actions)
                    )
                    if watered:
                        continue
                    for i in on_tile:
                        if i >= len(unit_actions):
                            continue
                        current = unit_actions[i]
                        if isinstance(current, list) and current[:1] == ["HARVEST"]:
                            break
                        _set_unit(i, ["HARVEST"])
                        unit_actions[i] = ["HARVEST"]
                        rescues += 1
                        break

    # Swap pass: scheduled PLANT WHEAT -> PLANT CARROT under all gates.
    if gates and wheat_ok and harvest_feasible_today(day, "CARROT"):
        waters_today = sum(
            1
            for op in unit_actions
            if isinstance(op, list) and op[:1] == ["WATER"]
        )
        plants = growing_plant_count(farm)
        water_capacity = waters_today >= plants
        if water_capacity:
            carrot_budget = carrot_seeds
            for i, op in enumerate(unit_actions):
                if not (isinstance(op, list) and op[:2] == ["PLANT", "WHEAT"]):
                    continue
                if carrot_budget <= 0:
                    break
                pos = positions[i] if i < len(positions) else None
                if pos is None:
                    continue
                # Plant-day watering is mandatory (missed starts at 1): a
                # second unit must WATER this tile today.
                waterer = None
                for j, other_pos in enumerate(positions):
                    if j == i or other_pos != pos or j >= len(unit_actions):
                        continue
                    other = unit_actions[j]
                    if isinstance(other, list) and other[:1] == ["WATER"]:
                        waterer = j
                        break
                    if isinstance(other, list) and other[:1] == ["PASS"]:
                        waterer = j
                        _set_unit(j, ["WATER"])
                        unit_actions[j] = ["WATER"]
                        break
                if waterer is None:
                    continue
                _set_unit(i, ["PLANT", "CARROT"])
                unit_actions[i] = ["PLANT", "CARROT"]
                carrot_budget -= 1
                swaps += 1

    # Top-up pass: price/day gates hold but wheat reserve is short, or carrot
    # seeds are missing -- stage supplies in free market slots only.
    market = [list(order) for order in (updated.get("market", []) or [])]
    if gates and not wheat_ok and len(market) < MAX_MARKET_ORDERS:
        shortfall = threshold - (held - reserve)
        if shortfall > 0 and not any(
            isinstance(order, list)
            and order[:2] == ["BUY_PRODUCT", "WHEAT"]
            for order in market
        ):
            market.append(["BUY_PRODUCT", "WHEAT", max(1, shortfall)])
    if gates and wheat_ok and carrot_seeds <= 0 and len(market) < MAX_MARKET_ORDERS:
        swaps_today = sum(
            1
            for op in _unit_actions(action)
            if isinstance(op, list) and op[:2] == ["PLANT", "WHEAT"]
        )
        if swaps_today > 0 and not any(
            isinstance(order, list) and order[:2] == ["BUY_SEED", "CARROT"]
            for order in market
        ):
            market.append(["BUY_SEED", "CARROT", swaps_today])
    updated["market"] = market
    return updated, swaps, rescues


class CarrotSwapOverlay:
    """Wrap a raw-action policy with the CARROT2 swap + rescue rules."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy
        self._swaps = 0
        self._rescues = 0

    @property
    def swaps(self) -> int:
        return self._swaps

    @property
    def rescues(self) -> int:
        return self._rescues

    def reset(self) -> None:
        self._swaps = 0
        self._rescues = 0

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        updated, swaps, rescues = _apply_with_stats(observation, action)
        self._swaps += swaps
        self._rescues += rescues
        return updated

    __call__ = agent
