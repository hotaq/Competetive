"""Observable-only marginal livestock value model for Kaggriculture 1.32.7.

This is an analysis artifact, not an agent policy.  It intentionally uses only
state visible in one observation: shared market/town, both public farms, and
the evaluating player's private state.  Opponent identity, episode id, seeds,
future actions, and fitted decision thresholds are not model inputs.

The model has two deliberately separate layers:

* exact mechanics: price curve, animal production schedule, market lockstep;
* an explicit projection: known shop demand plus the expectation of future
  uniformly drawn shops, and maximum-service output from currently visible
  animals.  This projection is deterministic but is not an oracle.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MAX_SHOP_INSTANCES = 8
BUY_PRODUCTS = frozenset({"WHEAT", "FERTILIZER"})

PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

MARKET_PARAMS: dict[str, dict[str, float | int | str]] = {
    "WHEAT": dict(
        base=25,
        I0=10_000,
        T=400,
        below_func="sqrt",
        below_target=0.80,
        above_func="log",
        above_target=0.20,
    ),
    "CARROT": dict(
        base=35,
        I0=10_000,
        T=450,
        below_func="hinge",
        below_target=1.00,
        above_func="sqrt",
        above_target=0.70,
    ),
    "TOMATO": dict(
        base=60,
        I0=10_000,
        T=200,
        below_func="hinge",
        below_target=0.40,
        above_func="sqrt",
        above_target=0.60,
    ),
    "STRAWBERRY": dict(
        base=120,
        I0=10_000,
        T=100,
        below_func="sqrt",
        below_target=0.70,
        above_func="linear",
        above_target=1.60,
    ),
    "MELON": dict(
        base=250,
        I0=10_000,
        T=300,
        below_func="log",
        below_target=0.20,
        above_func="sq",
        above_target=3.60,
    ),
    "EGG": dict(
        base=50,
        I0=10_000,
        T=332,
        below_func="hinge",
        below_target=0.40,
        above_func="log",
        above_target=0.20,
    ),
    "MILK": dict(
        base=160,
        I0=10_000,
        T=122,
        below_func="sqrt",
        below_target=0.60,
        above_func="linear",
        above_target=1.60,
    ),
    "WOOL": dict(
        base=200,
        I0=10_000,
        T=105,
        below_func="log",
        below_target=0.20,
        above_func="sq",
        above_target=3.20,
    ),
    "FERTILIZER": dict(
        base=100,
        I0=10_000,
        T=200,
        below_func="linear",
        below_target=0.40,
        above_func="linear",
        above_target=0.40,
    ),
}

SHOPS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


@dataclass(frozen=True)
class AnimalRule:
    cost: int
    structure: str
    first_yield_day: int
    interval: int
    max_held: int
    product: str


ANIMALS = {
    "GOOSE": AnimalRule(300, "COOP", 4, 1, 4, "EGG"),
    "COW": AnimalRule(400, "PASTURE", 8, 2, 6, "MILK"),
    "SHEEP": AnimalRule(500, "PASTURE", 6, 3, 6, "WOOL"),
}


@dataclass(frozen=True)
class YieldEvent:
    refresh_day: int
    sale_step: int
    units: int


@dataclass(frozen=True)
class AnimalSchedule:
    animal: str
    placed_day: int
    product: str
    events: tuple[YieldEvent, ...]
    feed_actions: int
    care_actions: int
    harvest_actions: int
    max_fertilizer_units: int

    @property
    def product_units(self) -> int:
        return sum(event.units for event in self.events)


@dataclass(frozen=True)
class ValueResult:
    animal: str
    product: str
    placement_step: int
    placement_day: int
    feasible_same_turn_buy_and_place: bool
    empty_matching_structure_at_observation: bool
    product_units: int
    product_events: tuple[dict[str, int], ...]
    marginal_product_revenue: float
    forced_full_liquidation_portfolio_delta: float
    feed_opportunity_cost: float
    purchase_cost: int
    net_before_labor_and_fertilizer: float
    feed_actions: int
    care_actions: int
    harvest_actions: int
    minimum_logistics_actions: int
    max_fertilizer_units: int
    fertilizer_snapshot_liquidation: float
    net_with_full_fertilizer_snapshot_before_labor: float
    total_required_actions_with_full_fertilizer: int
    break_even_shadow_value_per_required_unit_action: float
    break_even_shadow_value_with_full_fertilizer_per_action: float
    start_prices: dict[str, int]
    start_inventory: dict[str, int]
    visible_live_animals: dict[str, dict[str, int]]
    projection_assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape(func: str, x: float, throughput: float) -> float:
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log1p(x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge" and throughput > 0:
        normalized = x / throughput
        return normalized + HINGE_GAIN * max(0.0, normalized - 1.0) ** 2
    return x


def market_price(
    item: str,
    inventory: float,
    params: Mapping[str, Mapping[str, float | int | str]] = MARKET_PARAMS,
) -> int:
    """Exact Kaggriculture price curve, including Python ties-to-even rounding."""

    p = params[item]
    base = float(p["base"])
    equilibrium = float(p["I0"])
    throughput = float(p["T"])
    if inventory < equilibrium:
        func = str(p["below_func"])
        amplitude = float(p["below_target"]) * base / _shape(func, throughput, throughput)
        price = base + amplitude * _shape(func, equilibrium - inventory, throughput)
    else:
        func = str(p["above_func"])
        amplitude = float(p["above_target"]) * base / _shape(func, throughput, throughput)
        price = base - amplitude * _shape(func, inventory - equilibrium, throughput)
    return max(PRICE_FLOOR, round(price))


def lockstep_sell(
    inventory: float,
    quantities: tuple[int, int],
    item: str,
) -> tuple[float, tuple[float, float]]:
    """Execute one aligned SELL order for both seats exactly per unit lockstep.

    Both seats are quoted from the same pre-commit inventory for each unit pair.
    Commits then occur in seat order.  A floor-price sale pays $1 but does not
    add market supply.
    """

    remaining = [max(0, int(quantities[0])), max(0, int(quantities[1]))]
    revenue = [0.0, 0.0]
    while remaining[0] > 0 or remaining[1] > 0:
        quoted = [
            market_price(item, inventory) if remaining[seat] > 0 else None for seat in range(2)
        ]
        for seat, price in enumerate(quoted):
            if price is None:
                continue
            revenue[seat] += price
            remaining[seat] -= 1
            if price > PRICE_FLOOR:
                inventory += 1
    return inventory, (revenue[0], revenue[1])


def last_processed_step(configuration: Mapping[str, Any]) -> int:
    return max(0, int(configuration.get("episodeSteps", 720)) - 2)


def last_daily_refresh_day(configuration: Mapping[str, Any]) -> int:
    turns = max(1, int(configuration.get("turnsPerDay", 24)))
    processed = last_processed_step(configuration)
    last_eod_step = ((processed + 1) // turns) * turns - 1
    return max(0, (last_eod_step + 1) // turns)


def animal_schedule(
    animal: str,
    placement_step: int,
    configuration: Mapping[str, Any],
    *,
    liquidation_lag_steps: int = 2,
    service_on_placement_day: bool = True,
) -> AnimalSchedule:
    """Maximum-service product schedule with immediate post-production harvest.

    Feed and care happen once per service day.  The runtime consumes an already
    banked care bonus on a production refresh, then banks that day's care for
    the next production.  Harvest between production events keeps the tile's
    max-held cap from destroying later yield.
    """

    rule = ANIMALS[animal]
    turns = max(1, int(configuration.get("turnsPerDay", 24)))
    placed_day = placement_step // turns
    last_refresh = last_daily_refresh_day(configuration)
    production_days = tuple(
        range(placed_day + rule.first_yield_day, last_refresh + 1, rule.interval)
    )
    if not production_days:
        return AnimalSchedule(animal, placed_day, rule.product, (), 0, 0, 0, 0)

    last_product_day = production_days[-1]
    pending_care = 0
    events: list[YieldEvent] = []
    for refresh_day in range(placed_day + 1, last_refresh + 1):
        current_day = refresh_day - 1
        fed = current_day < last_product_day
        cared = fed and (service_on_placement_day or current_day > placed_day)
        if refresh_day in production_days:
            bonus = pending_care if fed else 0
            produced = min(rule.max_held, 1 + bonus)
            if produced > 0:
                eod_step = refresh_day * turns - 1
                events.append(
                    YieldEvent(
                        refresh_day=refresh_day,
                        sale_step=eod_step + liquidation_lag_steps,
                        units=produced,
                    )
                )
            pending_care = 0
        if cared:
            pending_care += 1

    feed_actions = max(0, last_product_day - placed_day)
    care_actions = feed_actions if service_on_placement_day else max(0, feed_actions - 1)
    # One missed refresh after the last useful feed does not cause escape, so
    # fertilizer remains collectable through the final refresh.
    max_fertilizer = max(0, last_refresh - placed_day)
    return AnimalSchedule(
        animal=animal,
        placed_day=placed_day,
        product=rule.product,
        events=tuple(events),
        feed_actions=feed_actions,
        care_actions=care_actions,
        harvest_actions=len(events),
        max_fertilizer_units=max_fertilizer,
    )


def _shop_demand(shop: str, item: str) -> int:
    products = SHOPS[shop]
    if item not in products:
        return 0
    return 2 if len(products) == 1 else 1


def expected_new_shop_demand(item: str) -> float:
    return sum(_shop_demand(shop, item) for shop in SHOPS) / len(SHOPS)


def _animal_tiles(farm: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for row in farm.get("tiles", ()):
        for tile in row:
            if isinstance(tile, Mapping) and tile.get("animal") in ANIMALS:
                yield tile


def visible_live_counts(observation: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for seat, farm in enumerate(observation.get("farms", ())):
        counts = Counter(str(tile["animal"]) for tile in _animal_tiles(farm))
        result[str(seat)] = {animal: counts.get(animal, 0) for animal in ANIMALS}
    return result


def empty_matching_structures(observation: Mapping[str, Any], seat: int, animal: str) -> int:
    structure = ANIMALS[animal].structure
    count = 0
    farm = observation["farms"][seat]
    for row in farm.get("tiles", ()):
        for tile in row:
            if (
                isinstance(tile, Mapping)
                and tile.get("kind") == structure
                and not tile.get("animal")
            ):
                count += 1
    return count


def _existing_visible_supply(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    product: str,
    start_step: int,
    *,
    liquidation_lag_steps: int,
) -> dict[int, list[int]]:
    """Upper-service public supply, grouped as [seat0, seat1] sale quantities."""

    turns = max(1, int(configuration.get("turnsPerDay", 24)))
    current_day = start_step // turns
    last_refresh = last_daily_refresh_day(configuration)
    scheduled: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for seat, farm in enumerate(observation.get("farms", ())):
        for tile in _animal_tiles(farm):
            animal = str(tile["animal"])
            rule = ANIMALS[animal]
            if rule.product != product:
                continue
            held = max(0, int(tile.get("yield_units", 0) or 0))
            if held:
                scheduled[start_step + liquidation_lag_steps][seat] += held
            pending = max(0, int(tile.get("pending_care_bonus", 0) or 0))
            placed_day = int(tile.get("placed_day", current_day) or current_day)
            for refresh_day in range(current_day + 1, last_refresh + 1):
                days_since_first = refresh_day - placed_day - rule.first_yield_day
                if days_since_first >= 0 and days_since_first % rule.interval == 0:
                    produced = min(rule.max_held, 1 + pending)
                    if produced:
                        sale_step = refresh_day * turns - 1 + liquidation_lag_steps
                        scheduled[sale_step][seat] += produced
                    pending = 0
                # Maximum-service forecast: daily feed + care.  Runtime banks
                # current care after consuming a production-day bonus.
                pending += 1
    return scheduled


def _town_tick_demand(
    item: str,
    known_shops: Counter[str],
    expected_future_shops: float,
) -> float:
    exact = sum(count * _shop_demand(shop, item) for shop, count in known_shops.items())
    return exact + expected_future_shops * expected_new_shop_demand(item)


def _project_product_revenue(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    animal: str,
    placement_step: int,
    seat: int,
    *,
    liquidation_lag_steps: int,
) -> tuple[float, float]:
    """Optimal optional-sale and forced-liquidation portfolio deltas.

    The optimal branch may sell any accumulated marginal output after it is
    logistically available, or withhold it.  Dynamic programming includes the
    effect of those sales on later revenue from the farm's already-visible
    animals.  The all-output-immediately branch is returned as a diagnostic;
    it can be negative when dumping a premium product destroys more portfolio
    revenue than the new units earn.
    """

    product = ANIMALS[animal].product
    base_inventory = float(observation["market"]["inventory"][product])
    known_shops = Counter(str(shop) for shop in observation["town"]["unlocked_shops"])
    known_shop_instances = sum(known_shops.values())
    start_step = int(placement_step)
    end_step = last_processed_step(configuration)
    shop_tick = max(1, int(configuration.get("townShopSellInterval", 4)))
    center_tick = max(1, int(configuration.get("townCenterSellInterval", 24)))
    unlock_days = max(1, int(configuration.get("townShopUnlockInterval", 3)))
    turns = max(1, int(configuration.get("turnsPerDay", 24)))

    background = _existing_visible_supply(
        observation,
        configuration,
        product,
        start_step,
        liquidation_lag_steps=liquidation_lag_steps,
    )
    candidate = animal_schedule(
        animal,
        placement_step,
        configuration,
        liquidation_lag_steps=liquidation_lag_steps,
    )
    candidate_by_step = Counter({event.sale_step: event.units for event in candidate.events})

    def run_forced(include_candidate: bool) -> float:
        inventory = base_inventory
        own_revenue = 0.0
        expected_future_shops = 0.0
        for step in range(start_step, end_step + 1):
            quantities = list(background.get(step, [0, 0]))
            if include_candidate:
                quantities[seat] += candidate_by_step.get(step, 0)
            if quantities[0] or quantities[1]:
                inventory, revenue = lockstep_sell(
                    inventory,
                    (quantities[0], quantities[1]),
                    product,
                )
                own_revenue += revenue[seat]

            # Runtime order is market first, then town consumption.
            if step % shop_tick == 0:
                inventory -= _town_tick_demand(
                    product,
                    known_shops,
                    expected_future_shops,
                )
            if product != "FERTILIZER" and step % center_tick == 0:
                inventory -= 1.0

            # Shop unlock is an end-of-day effect after the town tick.
            if (step + 1) % turns == 0:
                next_day = (step + 1) // turns
                if (
                    next_day > 0
                    and next_day % unlock_days == 0
                    and known_shop_instances + expected_future_shops < MAX_SHOP_INSTANCES
                ):
                    expected_future_shops += 1.0
        return own_revenue

    baseline_revenue = run_forced(False)
    forced_delta = run_forced(True) - baseline_revenue

    # State is (market inventory, unsold marginal stock) -> maximum cumulative
    # own revenue, including fixed background sales.  Inventory expectation is
    # fractional because unknown future shop identities are integrated out.
    states: dict[tuple[float, int], float] = {(base_inventory, 0): 0.0}
    expected_future_shops = 0.0
    for step in range(start_step, end_step + 1):
        new_units = candidate_by_step.get(step, 0)
        next_states: dict[tuple[float, int], float] = {}
        for (inventory, stock), cumulative_revenue in states.items():
            available = stock + new_units
            for candidate_sale in range(available + 1):
                quantities = list(background.get(step, [0, 0]))
                quantities[seat] += candidate_sale
                next_inventory, revenue = lockstep_sell(
                    inventory,
                    (quantities[0], quantities[1]),
                    product,
                )
                if step % shop_tick == 0:
                    next_inventory -= _town_tick_demand(
                        product,
                        known_shops,
                        expected_future_shops,
                    )
                if product != "FERTILIZER" and step % center_tick == 0:
                    next_inventory -= 1.0
                key = (round(next_inventory, 9), available - candidate_sale)
                value = cumulative_revenue + revenue[seat]
                if value > next_states.get(key, float("-inf")):
                    next_states[key] = value
        states = next_states
        if (step + 1) % turns == 0:
            next_day = (step + 1) // turns
            if (
                next_day > 0
                and next_day % unlock_days == 0
                and known_shop_instances + expected_future_shops < MAX_SHOP_INSTANCES
            ):
                expected_future_shops += 1.0

    optimal_revenue = max(states.values())
    return optimal_revenue - baseline_revenue, forced_delta


def _project_feed_opportunity_cost(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    schedule: AnimalSchedule,
    placement_step: int,
) -> float:
    """Foregone WHEAT sale value; public-crop supply is intentionally omitted.

    Omitting uncertain future crop sales produces a transparent town-only
    opportunity-cost path instead of learning an opponent-specific behavior.
    """

    turns = max(1, int(configuration.get("turnsPerDay", 24)))
    end_step = last_processed_step(configuration)
    shop_tick = max(1, int(configuration.get("townShopSellInterval", 4)))
    center_tick = max(1, int(configuration.get("townCenterSellInterval", 24)))
    unlock_days = max(1, int(configuration.get("townShopUnlockInterval", 3)))
    known_shops = Counter(str(shop) for shop in observation["town"]["unlocked_shops"])
    known_shop_instances = sum(known_shops.values())
    inventory = float(observation["market"]["inventory"]["WHEAT"])
    expected_future_shops = 0.0
    feed_days = set(range(schedule.placed_day, schedule.placed_day + schedule.feed_actions))
    feed_steps = {max(placement_step, day * turns + turns // 2): day for day in feed_days}
    cost = 0.0
    for step in range(placement_step, end_step + 1):
        if step in feed_steps:
            cost += market_price("WHEAT", inventory)
        if step % shop_tick == 0:
            inventory -= _town_tick_demand("WHEAT", known_shops, expected_future_shops)
        if step % center_tick == 0:
            inventory -= 1.0
        if (step + 1) % turns == 0:
            next_day = (step + 1) // turns
            if (
                next_day > 0
                and next_day % unlock_days == 0
                and known_shop_instances + expected_future_shops < MAX_SHOP_INSTANCES
            ):
                expected_future_shops += 1.0
    return cost


def _snapshot_liquidation(item: str, inventory: int, quantity: int) -> float:
    _, revenue = lockstep_sell(float(inventory), (quantity, 0), item)
    return revenue[0]


def value_one_animal(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    animal: str,
    *,
    placement_step: int = 248,
    seat: int | None = None,
    liquidation_lag_steps: int = 2,
) -> ValueResult:
    """Value one newly placed animal from an observable state.

    The cash result excludes any guessed value per unit action.  Instead the
    function returns the exact required-action burden and its break-even shadow
    value.  Fertilizer is reported separately because collecting it consumes a
    competing action and is identical for COW and SHEEP at the same placement
    day.
    """

    if animal not in ("COW", "SHEEP"):
        raise ValueError("this audit compares COW and SHEEP only")
    if seat is None:
        seat = int(observation["player"])
    schedule = animal_schedule(
        animal,
        placement_step,
        configuration,
        liquidation_lag_steps=liquidation_lag_steps,
    )
    empty = empty_matching_structures(observation, seat, animal)
    product_revenue, forced_liquidation_delta = _project_product_revenue(
        observation,
        configuration,
        animal,
        placement_step,
        seat,
        liquidation_lag_steps=liquidation_lag_steps,
    )
    feed_cost = _project_feed_opportunity_cost(observation, configuration, schedule, placement_step)
    rule = ANIMALS[animal]
    net = product_revenue - feed_cost - rule.cost
    # PICKUP + PLACE; add BUILD when no empty compatible structure exists.
    logistics = 2 + int(empty == 0)
    required_actions = (
        schedule.feed_actions + schedule.care_actions + schedule.harvest_actions + logistics
    )
    fertilizer_value = _snapshot_liquidation(
        "FERTILIZER",
        int(observation["market"]["inventory"]["FERTILIZER"]),
        schedule.max_fertilizer_units,
    )
    net_with_fertilizer = net + fertilizer_value
    actions_with_fertilizer = required_actions + schedule.max_fertilizer_units
    prices = observation["market"]["prices"]
    inventory = observation["market"]["inventory"]
    return ValueResult(
        animal=animal,
        product=rule.product,
        placement_step=placement_step,
        placement_day=schedule.placed_day,
        # Runtime applies units before market, so an animal bought by the
        # current market action is unavailable to PLACE until a later turn.
        feasible_same_turn_buy_and_place=False,
        empty_matching_structure_at_observation=empty > 0,
        product_units=schedule.product_units,
        product_events=tuple(asdict(event) for event in schedule.events),
        marginal_product_revenue=round(product_revenue, 6),
        forced_full_liquidation_portfolio_delta=round(forced_liquidation_delta, 6),
        feed_opportunity_cost=round(feed_cost, 6),
        purchase_cost=rule.cost,
        net_before_labor_and_fertilizer=round(net, 6),
        feed_actions=schedule.feed_actions,
        care_actions=schedule.care_actions,
        harvest_actions=schedule.harvest_actions,
        minimum_logistics_actions=logistics,
        max_fertilizer_units=schedule.max_fertilizer_units,
        fertilizer_snapshot_liquidation=round(fertilizer_value, 6),
        net_with_full_fertilizer_snapshot_before_labor=round(net_with_fertilizer, 6),
        total_required_actions_with_full_fertilizer=actions_with_fertilizer,
        break_even_shadow_value_per_required_unit_action=round(
            net / required_actions if required_actions else net,
            6,
        ),
        break_even_shadow_value_with_full_fertilizer_per_action=round(
            net_with_fertilizer / actions_with_fertilizer
            if actions_with_fertilizer
            else net_with_fertilizer,
            6,
        ),
        start_prices={
            "WHEAT": int(prices["WHEAT"]),
            rule.product: int(prices[rule.product]),
            "FERTILIZER": int(prices["FERTILIZER"]),
        },
        start_inventory={
            "WHEAT": int(inventory["WHEAT"]),
            rule.product: int(inventory[rule.product]),
            "FERTILIZER": int(inventory["FERTILIZER"]),
        },
        visible_live_animals=visible_live_counts(observation),
        projection_assumptions=(
            "daily feed and care through the final useful product refresh",
            "immediate harvest and sale with a two-step post-refresh lag",
            "currently visible animals on both farms receive maximum service and sell output",
            "currently unlocked shops are exact; future uniform-with-replacement "
            "shops use expected demand",
            "no identity, episode, hidden opponent inventory, future action, or "
            "learned threshold input",
            "feed is charged at projected WHEAT opportunity value; public future "
            "crop sales are omitted",
            "fertilizer option value and labor burden are reported separately",
        ),
    )


def compare_cow_sheep(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    placement_step: int = 248,
    seat: int | None = None,
) -> dict[str, Any]:
    cow = value_one_animal(
        observation,
        configuration,
        "COW",
        placement_step=placement_step,
        seat=seat,
    )
    sheep = value_one_animal(
        observation,
        configuration,
        "SHEEP",
        placement_step=placement_step,
        seat=seat,
    )
    delta = sheep.net_before_labor_and_fertilizer - cow.net_before_labor_and_fertilizer
    return {
        "cow": cow.to_dict(),
        "sheep": sheep.to_dict(),
        "sheep_minus_cow_net_before_labor_and_fertilizer": round(delta, 6),
        "preferred_by_mechanical_cash_value": "SHEEP"
        if delta > 0
        else "COW"
        if delta < 0
        else "TIE",
        "common_fertilizer_units_cancel_in_comparison": cow.max_fertilizer_units
        == sheep.max_fertilizer_units,
    }
