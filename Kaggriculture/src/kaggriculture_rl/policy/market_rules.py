"""Observable market rules that can be layered over an existing policy."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_REFERENCE_PRICES = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "MELON": 250.0,
    "STRAWBERRY": 120.0,
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}

# Exact port of `rust/kaggriculture-engine/src/rules.rs::default_market_params`
# (base, throughput, below_func, below_target, above_func, above_target).
# Equilibrium inventory is 10_000 and the floor is 1 there; the hinge gain is
# 8.0 -- an older script port (`screen_v70_tail_market_oracle.py`) uses 4.0
# and must not be copied.
_MARKET_PARAMS: dict[str, tuple[float, float, str, float, str, float]] = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
_EQUILIBRIUM_INVENTORY = 10_000
_PRICE_FLOOR = 1
_HINGE_GAIN = 8.0

_PRODUCTS = tuple(_MARKET_PARAMS)
_ANIMAL_PRODUCTS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}

# Port of `rules.rs::shop_products` (single-product shops draw double).
_SHOP_PRODUCTS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SHOP_SELL_INTERVAL = 4
_CENTER_SELL_INTERVAL = 24


def _price_shape(function: str, x: float, throughput: float) -> float:
    value = max(0.0, x)
    if function == "linear":
        return value
    if function == "sq":
        return value * value
    if function == "sqrt":
        return math.sqrt(value)
    if function == "log":
        return math.log1p(value)
    if function == "log10":
        return math.log10(1.0 + value)
    if function == "hinge" and throughput > 0.0:
        normalized = value / throughput
        return normalized + _HINGE_GAIN * max(0.0, normalized - 1.0) ** 2
    return value


def market_price(item: str, inventory: int) -> int:
    """Engine price curve port (`rules.rs::market_price_from_param`)."""
    base, throughput, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < _EQUILIBRIUM_INVENTORY:
        amplitude = below_target * base / _price_shape(below_func, throughput, throughput)
        price = base + amplitude * _price_shape(
            below_func, float(_EQUILIBRIUM_INVENTORY - inventory), throughput
        )
    else:
        amplitude = above_target * base / _price_shape(above_func, throughput, throughput)
        price = base - amplitude * _price_shape(
            above_func, float(inventory - _EQUILIBRIUM_INVENTORY), throughput
        )
    # Rust `round_ties_even`; Python `round` on a float is also ties-to-even.
    return max(_PRICE_FLOOR, round(price))


def town_draw(item: str, elapsed_step: int, shops: list[str]) -> int:
    """Town consumption of `item` during the turn played at `elapsed_step`.

    Mirrors `engine.rs::town_consume`: every 4th step each unlocked shop draws
    its products (2x for single-product shops); every 24th step the center
    draws 1 of every product except FERTILIZER. Step 0 draws (0 % n == 0).
    """
    draw = 0
    if elapsed_step % _SHOP_SELL_INTERVAL == 0:
        for shop in shops:
            products = _SHOP_PRODUCTS.get(shop)
            if products is None or item not in products:
                continue
            draw += 2 if len(products) == 1 else 1
    if elapsed_step % _CENTER_SELL_INTERVAL == 0 and item in _PRODUCTS and item != "FERTILIZER":
        draw += 1
    return draw


@dataclass(frozen=True, slots=True)
class LandExpansionRule:
    """Fund a third quadrant from cash and already-scheduled leading sales."""

    start_day: int = 10
    end_day: int = 14
    current_land: int = 2
    estimated_land_cost: float = 1000.0
    cash_reserve: float = 800.0

    def apply(
        self, observation: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(action)
        day = int(observation.get("day", 0) or 0)
        if not self.start_day <= day <= self.end_day:
            return result

        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        quadrants = farm.get("unlocked_quadrants", [])
        if len(quadrants) != self.current_land:
            return result

        market_orders = list(result.get("market", []))
        if any(order and order[0] == "BUY_LAND" for order in market_orders):
            return result

        prices = observation.get("market", {}).get("prices", {})
        available = float(farm.get("money", 0.0) or 0.0)
        insertion_index = 0
        for order in market_orders:
            if not order or order[0] != "SELL":
                break
            if len(order) >= 3:
                item = str(order[1])
                quantity = max(0, int(order[2]))
                available += float(prices.get(item, 0.0) or 0.0) * quantity
            insertion_index += 1

        required = self.estimated_land_cost + self.cash_reserve
        if available < required:
            return result
        market_orders.insert(insertion_index, ["BUY_LAND"])
        result["market"] = market_orders
        return result


@dataclass(frozen=True, slots=True)
class LowPriceSaleDeferralRule:
    """Hold surplus goods instead of repeatedly selling into a depressed market."""

    minimum_price_ratio: float = 0.8
    items: tuple[str, ...] = ("FERTILIZER", "MELON", "MILK", "WOOL")
    terminal_step: int = 718
    pure_sale_actions_only: bool = True

    def apply(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        result = deepcopy(action)
        step = max(0, int(observation.get("step", 0) or 0))
        orders = list(result.get("market", []))
        if step >= self.terminal_step or not orders:
            return result
        if self.pure_sale_actions_only and any(
            not order or order[0] != "SELL" for order in orders
        ):
            return result

        prices = observation.get("market", {}).get("prices", {})
        deferred = set(self.items)
        kept: list[list[Any]] = []
        for order in orders:
            if not isinstance(order, list) or len(order) < 3 or order[0] != "SELL":
                kept.append(order)
                continue
            item = str(order[1])
            reference = _REFERENCE_PRICES.get(item)
            price = float(prices.get(item, 0.0) or 0.0)
            if (
                item in deferred
                and reference is not None
                and price < reference * self.minimum_price_ratio
            ):
                continue
            kept.append(order)

        result["market"] = kept
        return result


@dataclass(frozen=True, slots=True)
class PremiumSaleExpansionRule:
    """Expand an existing premium-priced sale to the observable shed balance."""

    minimum_price_ratio: float = 1.0
    items: tuple[str, ...] = tuple(_REFERENCE_PRICES)

    def apply(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        result = deepcopy(action)
        orders = list(result.get("market", []))
        if not orders:
            return result

        private = observation.get("private", {})
        available = {
            str(item): max(0, int(quantity or 0))
            for item, quantity in private.get("shed", {}).items()
        }
        unit_actions = [result.get("farmer", ["PASS"]), *result.get("hands", [])]
        inventories = private.get("inventories", [])
        for actor, unit_action in enumerate(unit_actions):
            if (
                isinstance(unit_action, list)
                and unit_action
                and unit_action[0] == "DROP"
                and actor < len(inventories)
                and isinstance(inventories[actor], dict)
            ):
                for item, quantity in inventories[actor].items():
                    key = str(item)
                    available[key] = available.get(key, 0) + max(0, int(quantity or 0))

        prices = observation.get("market", {}).get("prices", {})
        expandable = set(self.items)
        expanded: list[list[Any]] = []
        for order in orders:
            if not isinstance(order, list) or len(order) < 3 or order[0] != "SELL":
                expanded.append(order)
                continue

            item = str(order[1])
            requested = max(0, int(order[2] or 0))
            stock = available.get(item, 0)
            reference = _REFERENCE_PRICES.get(item)
            price = float(prices.get(item, 0.0) or 0.0)
            quantity = requested
            if (
                item in expandable
                and reference is not None
                and price >= reference * self.minimum_price_ratio
            ):
                quantity = max(requested, stock)
            quantity = min(quantity, stock)
            if quantity > 0:
                expanded.append(["SELL", item, quantity])
                available[item] = stock - quantity

        result["market"] = expanded
        return result


@dataclass(frozen=True, slots=True)
class SaleQuantityScaleRule:
    """Apply a bounded multiplicative adjustment to selected observable sales."""

    scale: float = 1.0
    items: tuple[str, ...] = tuple(_REFERENCE_PRICES)
    terminal_step: int = 718

    def apply(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        result = deepcopy(action)
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if int(observation.get("step", 0) or 0) >= self.terminal_step:
            return result

        available = {
            str(item): max(0, int(quantity or 0))
            for item, quantity in observation.get("private", {}).get("shed", {}).items()
        }
        selected = set(self.items)
        scaled_orders: list[list[Any]] = []
        for order in result.get("market", []):
            if not isinstance(order, list) or len(order) < 3 or order[0] != "SELL":
                scaled_orders.append(order)
                continue
            item = str(order[1])
            quantity = max(0, int(order[2] or 0))
            stock = available.get(item, 0)
            if item in selected:
                raw = quantity * self.scale
                quantity = math.ceil(raw) if self.scale > 1.0 else math.floor(raw)
                quantity = max(1, quantity)
            quantity = min(quantity, stock)
            if quantity > 0:
                scaled_orders.append(["SELL", item, quantity])
                available[item] = stock - quantity

        result["market"] = scaled_orders
        return result


class MarketOrderPrioritizationRule:
    """ORDERPRI2: front-load the sale most exposed to the rival dump.

    Both players' lists resolve lockstep (10 orders/turn, 1 unit at a time,
    prices refreshed per order index), so a sale placed behind the rival's
    dump realizes the depressed price. Per standing lot (shed item) compute
    ``exposure = now_price - later_price`` with ``later_price`` read off the
    engine curve at ``inventory + batch``, ``batch = min(24, max(8,
    standing_rival_yield))``; the max-exposure SELL moves to the front of the
    sales, and a high-exposure uncovered lot swaps the weakest SELL out when
    the list is full (10).

    Rival tracking (per seat, consecutive steps only): ``standing_rival_yield``
    sums the rival's public field yield (crop tiles by crop, animal tiles by
    product) plus recently sold units inferred from inventory deltas,
    ``rival_sold = inventory_now - inventory_before + town_draw - own_sold``
    (buys ignored, clamped at 0). ``own_sold`` is last turn's scheduled SELL
    volume -- a proxy, since commits are unobservable. Unlike the frozen
    dataclass rules above this one holds cross-turn state and is seat-aware.
    """

    def __init__(self) -> None:
        self._prev_inventory: dict[int, dict[str, int]] = {}
        self._prev_step: dict[int, int] = {}
        self._prev_own_sold: dict[int, dict[str, int]] = {}
        self._rival_sold: dict[int, dict[str, int]] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._prev_inventory.clear()
            self._prev_step.clear()
            self._prev_own_sold.clear()
            self._rival_sold.clear()
            return
        seat = 1 if int(player) == 1 else 0
        self._prev_inventory.pop(seat, None)
        self._prev_step.pop(seat, None)
        self._prev_own_sold.pop(seat, None)
        self._rival_sold.pop(seat, None)

    @property
    def rival_sold(self) -> dict[int, dict[str, int]]:
        return {seat: dict(items) for seat, items in self._rival_sold.items()}

    @staticmethod
    def _seat_of(observation: dict[str, Any]) -> int:
        return 1 if int(observation.get("player", 0) or 0) == 1 else 0

    @staticmethod
    def _market_state(observation: dict[str, Any]) -> tuple[dict[str, int], dict[str, float]]:
        market = observation.get("market", {}) or {}
        raw_inventory = market.get("inventory", {}) or {}
        raw_prices = market.get("prices", {}) or {}
        inventory = {
            str(item): max(0, int(quantity or 0))
            for item, quantity in raw_inventory.items()
        }
        prices = {
            str(item): float(price or 0.0) for item, price in raw_prices.items()
        }
        return inventory, prices

    @staticmethod
    def _shops(observation: dict[str, Any]) -> list[str]:
        town = observation.get("town", {}) or {}
        shops = town.get("unlocked_shops", []) or []
        return [str(shop) for shop in shops]

    @staticmethod
    def _standing_rival_yield(observation: dict[str, Any], seat: int) -> dict[str, int]:
        farms = observation.get("farms", []) or []
        rival = 1 - seat
        standing: dict[str, int] = {}
        if not 0 <= rival < len(farms):
            return standing
        tiles = farms[rival].get("tiles", []) if isinstance(farms[rival], dict) else []
        if not isinstance(tiles, list):
            return standing
        for row in tiles:
            if not isinstance(row, list):
                continue
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                units = max(0, int(tile.get("yield_units", 0) or 0))
                if units <= 0:
                    continue
                item: str | None = None
                if tile.get("kind") == "PLANT" and tile.get("crop"):
                    item = str(tile["crop"])
                elif tile.get("animal"):
                    item = _ANIMAL_PRODUCTS.get(str(tile["animal"]))
                if item is not None:
                    standing[item] = standing.get(item, 0) + units
        return standing

    @staticmethod
    def _sell_totals(market: list[list[Any]]) -> dict[str, int]:
        totals: dict[str, int] = {}
        for order in market:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                totals[str(order[1])] = totals.get(str(order[1]), 0) + max(
                    0, int(order[2] or 0)
                )
        return totals

    def exposure(
        self, observation: dict[str, Any], item: str, batch: int | None = None
    ) -> float:
        """Exposure of one standing lot without touching tracker state."""
        seat = self._seat_of(observation)
        inventory, prices = self._market_state(observation)
        now = prices.get(item, 0.0)
        if item not in _MARKET_PARAMS:
            return 0.0
        if batch is None:
            standing = self._standing_rival_yield(observation, seat)
            sold = self._rival_sold.get(seat, {}).get(item, 0)
            batch = min(24, max(8, standing.get(item, 0) + sold))
        later = float(market_price(item, inventory.get(item, _EQUILIBRIUM_INVENTORY) + batch))
        return now - later

    def apply(
        self, observation: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(action)
        seat = self._seat_of(observation)
        step = max(0, int(observation.get("step", 0) or 0))
        inventory, prices = self._market_state(observation)

        prev_inventory = self._prev_inventory.get(seat)
        prev_step = self._prev_step.get(seat)
        prev_own_sold = self._prev_own_sold.get(seat, {})
        sold: dict[str, int] = {item: 0 for item in _PRODUCTS}
        if prev_inventory is not None and prev_step == step - 1:
            shops = self._shops(observation)
            for item in _PRODUCTS:
                delta = inventory.get(item, 0) - prev_inventory.get(item, 0)
                draw = town_draw(item, prev_step, shops)
                sold[item] = max(0, delta + draw - prev_own_sold.get(item, 0))
        self._rival_sold[seat] = sold

        standing = self._standing_rival_yield(observation, seat)
        exposures: dict[str, float] = {}
        for item in _PRODUCTS:
            batch = min(24, max(8, standing.get(item, 0) + sold.get(item, 0)))
            now = prices.get(item, 0.0)
            later = float(market_price(item, inventory.get(item, _EQUILIBRIUM_INVENTORY) + batch))
            exposures[item] = now - later

        market = [list(order) for order in (result.get("market", []) or [])]
        sell_positions = [
            index
            for index, order in enumerate(market)
            if isinstance(order, list) and order and order[0] == "SELL"
        ]
        ranked = sorted(
            sell_positions, key=lambda i: exposures.get(str(market[i][1]), 0.0), reverse=True
        )
        ordered_sells = [market[i] for i in ranked]
        for position, order in zip(sell_positions, ordered_sells, strict=True):
            market[position] = order

        private = observation.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        scheduled = self._sell_totals(market)
        weakest_exposure = min(
            (exposures.get(str(market[i][1]), 0.0) for i in sell_positions),
            default=None,
        )
        for item in sorted(exposures, key=lambda name: exposures[name], reverse=True):
            if exposures[item] <= 0:
                break
            stock = max(0, int(shed.get(item, 0) or 0))
            uncovered = stock - scheduled.get(item, 0)
            if uncovered <= 0:
                continue
            if weakest_exposure is not None and exposures[item] <= weakest_exposure:
                break
            if len(market) < 10:
                market.append(["SELL", item, uncovered])
                scheduled[item] = scheduled.get(item, 0) + uncovered
            elif sell_positions:
                weakest = min(
                    sell_positions, key=lambda i: exposures.get(str(market[i][1]), 0.0)
                )
                dropped = market[weakest]
                if isinstance(dropped, list) and len(dropped) >= 3 and dropped[0] == "SELL":
                    scheduled[str(dropped[1])] = scheduled.get(str(dropped[1]), 0) - max(
                        0, int(dropped[2] or 0)
                    )
                market[weakest] = ["SELL", item, uncovered]
                scheduled[item] = scheduled.get(item, 0) + uncovered
                weakest_exposure = min(
                    (exposures.get(str(market[i][1]), 0.0) for i in sell_positions),
                    default=None,
                )
            else:
                break

        result["market"] = market
        self._prev_inventory[seat] = dict(inventory)
        self._prev_step[seat] = step
        self._prev_own_sold[seat] = self._sell_totals(market)
        return result


class RaceForwardPullRule:
    """RACE v9/4 forward-pull (never tried before v109; distinct from ORDERPRI2).

    Dissection of the 2945 tape (first seller takes the price, second sells
    into the crash): rival sales are exactly observable above the $1 floor via
    ``rival_sold = inventory_now - inventory_before + town_draw - own_sold``.
    When a rival sale is detected (``sold >= 2``) for a premium item from day
    8 (``step >= 192``), a lead marker is recorded with a 40-turn horizon.
    While the marker is live, an already-planned ``SELL`` of the same item in
    the base action is pulled forward: its quantity is expanded to the full
    observable shed stock, gated by RACEGATE (blocked when
    ``price <= BASE + margin`` with ``BASE`` from ``_REFERENCE_PRICES``).

    Acceleration only, never creation: the rule modifies quantities of
    ``SELL`` orders the base action already schedules and never appends,
    reorders, or removes orders -- so working wheat/feed stock is untouched
    (WHEAT is not a RACE item and no WHEAT order is ever modified). Like
    ``MarketOrderPrioritizationRule`` this rule holds cross-turn tracker state
    and is seat-aware.
    """

    RACE_ITEMS: tuple[str, ...] = (
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
    )
    START_STEP = 192
    HORIZON = 40
    DETECTION_MIN_SOLD = 2

    def __init__(self, margin: float = 0.0) -> None:
        self._margin = float(margin)
        self._prev_inventory: dict[int, dict[str, int]] = {}
        self._prev_step: dict[int, int] = {}
        self._prev_own_sold: dict[int, dict[str, int]] = {}
        self._rival_sold: dict[int, dict[str, int]] = {}
        self._leads: dict[int, dict[str, int]] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._prev_inventory.clear()
            self._prev_step.clear()
            self._prev_own_sold.clear()
            self._rival_sold.clear()
            self._leads.clear()
            return
        seat = 1 if int(player) == 1 else 0
        self._prev_inventory.pop(seat, None)
        self._prev_step.pop(seat, None)
        self._prev_own_sold.pop(seat, None)
        self._rival_sold.pop(seat, None)
        self._leads.pop(seat, None)

    @property
    def rival_sold(self) -> dict[int, dict[str, int]]:
        return {seat: dict(items) for seat, items in self._rival_sold.items()}

    @property
    def lead_expiry(self) -> dict[int, dict[str, int]]:
        """Live lead markers per seat: ``item -> last step the pull applies``."""
        return {seat: dict(items) for seat, items in self._leads.items()}

    def apply(
        self, observation: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(action)
        seat = MarketOrderPrioritizationRule._seat_of(observation)
        step = max(0, int(observation.get("step", 0) or 0))
        inventory, prices = MarketOrderPrioritizationRule._market_state(observation)

        prev_inventory = self._prev_inventory.get(seat)
        prev_step = self._prev_step.get(seat)
        prev_own_sold = self._prev_own_sold.get(seat, {})
        sold: dict[str, int] = {item: 0 for item in _PRODUCTS}
        if prev_inventory is not None and prev_step == step - 1:
            shops = MarketOrderPrioritizationRule._shops(observation)
            for item in _PRODUCTS:
                delta = inventory.get(item, 0) - prev_inventory.get(item, 0)
                draw = town_draw(item, prev_step, shops)
                sold[item] = max(0, delta + draw - prev_own_sold.get(item, 0))
        self._rival_sold[seat] = sold

        leads = self._leads.setdefault(seat, {})
        if step >= self.START_STEP:
            for item in self.RACE_ITEMS:
                if sold.get(item, 0) >= self.DETECTION_MIN_SOLD:
                    leads[item] = step + self.HORIZON

        race_items = set(self.RACE_ITEMS)
        private = observation.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        market = result.get("market", []) or []
        for order in market:
            if (
                not isinstance(order, list)
                or len(order) < 3
                or order[0] != "SELL"
                or str(order[1]) not in race_items
            ):
                continue
            item = str(order[1])
            if leads.get(item, -1) < step:
                continue
            reference = _REFERENCE_PRICES.get(item)
            price = float(prices.get(item, 0.0) or 0.0)
            if reference is None or price <= reference + self._margin:
                continue
            stock = max(0, int(shed.get(item, 0) or 0))
            requested = max(0, int(order[2] or 0))
            if stock > requested:
                order[2] = stock

        result["market"] = market
        self._prev_inventory[seat] = dict(inventory)
        self._prev_step[seat] = step
        self._prev_own_sold[seat] = MarketOrderPrioritizationRule._sell_totals(market)
        return result


class GuardedMarketOrderPrioritizationRule(MarketOrderPrioritizationRule):
    """ORDERPRI2 with a herd/working-stock wheat guard (v107 hypothesis).

    Additive subclass: the unguarded ``MarketOrderPrioritizationRule`` above
    is frozen (existing tests rely on its exact behavior). This subclass runs
    the base rule verbatim, then trims scheduled ``SELL WHEAT`` orders so the
    shed never drops below the wheat reserve:

    ``reserve = 2 * live_animals + pipeline_buffer``

    Formula choice (auditable, from ``rules.rs``/``engine.rs`` constants):

    - One feed costs exactly 1 WHEAT (``engine.rs`` ``UnitAction::Feed`` takes
      1 WHEAT from inventory and sets ``fed_today``); each animal is fed at
      most once per day.
    - Two consecutive unfed days clear the tile (end-of-day
      ``consecutive_unfed >= 2`` resets the tile to its bare structure --
      the animal is lost). So ``2 * N`` wheat lets the whole herd survive
      two full days with zero income: exactly the starvation horizon.
    - ``pipeline_buffer`` = live animals with ``fed_today is not True``:
      today's ration already due before any future income can arrive.

    Live animals are counted from the player's own farm tiles (tile dicts
    with a truthy ``animal`` field). With no herd the reserve is 0 and the
    subclass behaves exactly like the base rule.
    """

    WHEAT_PER_ANIMAL_PER_DAY = 1
    STARVATION_MISSES = 2

    @staticmethod
    def _herd_counts(observation: dict[str, Any]) -> tuple[int, int]:
        """Return ``(live_animals, unfed_animals)`` on the player's own farm."""
        seat = MarketOrderPrioritizationRule._seat_of(observation)
        farms = observation.get("farms", []) or []
        live = 0
        unfed = 0
        if 0 <= seat < len(farms) and isinstance(farms[seat], dict):
            tiles = farms[seat].get("tiles", [])
            if isinstance(tiles, list):
                for row in tiles:
                    if not isinstance(row, list):
                        continue
                    for tile in row:
                        if not isinstance(tile, dict) or not tile.get("animal"):
                            continue
                        live += 1
                        if tile.get("fed_today") is not True:
                            unfed += 1
        return live, unfed

    @classmethod
    def wheat_reserve(cls, observation: dict[str, Any]) -> int:
        """Wheat units the shed must keep: never sell below this."""
        live, unfed = cls._herd_counts(observation)
        return cls.STARVATION_MISSES * live + unfed

    def apply(
        self, observation: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        result = super().apply(observation, action)
        private = observation.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        stock = max(0, int(shed.get("WHEAT", 0) or 0))
        sellable = max(0, stock - self.wheat_reserve(observation))
        market = result.get("market", []) or []
        scheduled = sum(
            max(0, int(order[2] or 0))
            for order in market
            if isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and str(order[1]) == "WHEAT"
        )
        excess = scheduled - sellable
        if excess <= 0:
            return result
        for order in reversed(market):
            if excess <= 0:
                break
            if (
                not isinstance(order, list)
                or len(order) < 3
                or order[0] != "SELL"
                or str(order[1]) != "WHEAT"
            ):
                continue
            quantity = max(0, int(order[2] or 0))
            cut = min(quantity, excess)
            order[2] = quantity - cut
            excess -= cut
        result["market"] = [order for order in market if not (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and str(order[1]) == "WHEAT"
            and max(0, int(order[2] or 0)) <= 0
        )]
        return result
