"""Observation-only primitive action legalizer for standalone Kaggle agents."""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
UNIT_ITEMS = (*PRODUCTS, *ANIMALS)
SIMPLE_UNITS = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "DROP",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)
FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}
ANIMAL_RULES = {
    "GOOSE": ("COOP", "EGG", 300),
    "COW": ("PASTURE", "MILK", 400),
    "SHEEP": ("PASTURE", "WOOL", 500),
}
SEED_COSTS = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
LAND_PRICES = (1_000, 2_000, 4_000)
MARKET_PARAMS = {
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
SEMANTIC_RECOVERABLE_UNIT_OPS = {
    "PICKUP",
    "DROP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}


class WeedRecoveryController:
    """Repair random-weed tape divergence and rejoin the worker schedule."""

    def __init__(self) -> None:
        self._pending: dict[int, dict[int, list[Any]]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._pending.clear()
            self._last_step.clear()
        else:
            self._pending.pop(player, None)
            self._last_step.pop(player, None)

    def repair(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        repaired = copy.deepcopy(action)
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step <= previous_step or step % 24 == 0:
            self._pending.pop(player, None)
        self._last_step[player] = step

        farm = observation["farms"][player]
        positions = [farm["farmer"], *farm.get("hands", [])]
        commands = [repaired.get("farmer", ["PASS"]), *repaired.get("hands", [])]
        pending = self._pending.setdefault(player, {})

        for actor, delayed in list(pending.items()):
            if actor >= len(commands) or actor >= len(positions):
                pending.pop(actor, None)
                continue
            current = commands[actor]
            self._set_actor(repaired, actor, delayed)
            if current == ["PASS"]:
                pending.pop(actor, None)
            else:
                pending[actor] = copy.deepcopy(current)

        commands = [repaired.get("farmer", ["PASS"]), *repaired.get("hands", [])]
        for actor, (position, command) in enumerate(zip(positions, commands, strict=False)):
            if actor in pending or not isinstance(command, list) or not command:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            if not isinstance(tile, dict) or tile.get("kind") != "WEED":
                continue
            operation = command[0]
            if operation in {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}:
                self._set_actor(repaired, actor, ["DIG"])
                pending[actor] = copy.deepcopy(command)
            elif operation in {"WATER", "HARVEST", "FERTILIZE"}:
                self._set_actor(repaired, actor, ["DIG"])
        if not pending:
            self._pending.pop(player, None)
        return repaired

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[Any]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command


def append_terminal_liquidation(
    observation: dict[str, Any], action: dict[str, Any], *, terminal_step: int = 718
) -> dict[str, Any]:
    """Sell otherwise-worthless terminal shed inventory without duplicating orders."""

    if int(observation.get("step", 0) or 0) < terminal_step:
        return action
    result = copy.deepcopy(action)
    market = result.setdefault("market", [])
    scheduled: Counter[str]
    scheduled = Counter(
        str(order[1])
        for order in market
        if isinstance(order, list) and len(order) == 3 and order[0] == "SELL"
        for _ in range(int(order[2]))
    )
    shed = observation.get("private", {}).get("shed", {})
    for product in PRODUCTS:
        available = int(shed.get(product, 0) or 0) - scheduled[product]
        if available > 0 and len(market) < 10:
            market.append(["SELL", product, available])
    return result


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _canonical_unit(command: Any) -> list[Any]:
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        return ["PASS"]
    operation = command[0]
    if operation in SIMPLE_UNITS:
        return [operation] if len(command) == 1 else ["PASS"]
    if operation == "PLANT":
        return command.copy() if len(command) == 2 and command[1] in CROPS else ["PASS"]
    if operation in {"PICKUP", "PLACE"}:
        if len(command) == 2 and command[1] in UNIT_ITEMS:
            return [operation, command[1], 1]
        if len(command) == 3 and command[1] in UNIT_ITEMS and _positive_int(command[2]) is not None:
            return command.copy()
    return ["PASS"]


def _inventory_add(inventory: dict[str, int], item: str, quantity: int) -> None:
    inventory[item] = int(inventory.get(item, 0)) + quantity


def _inventory_take(inventory: dict[str, int], item: str, quantity: int) -> bool:
    current = int(inventory.get(item, 0))
    if current < quantity:
        return False
    remaining = current - quantity
    if remaining:
        inventory[item] = remaining
    else:
        inventory.pop(item, None)
    return True


def _shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return (
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    )


def _tile_object(tile: Any) -> dict[str, Any] | None:
    return tile if isinstance(tile, dict) else None


def _apply_unit(
    command: list[Any],
    unit_index: int,
    positions: list[list[int]],
    tiles: list[list[Any]],
    inventories: list[dict[str, int]],
    shed: dict[str, int],
    seeds: dict[str, int],
    *,
    day: int,
    board_size: int,
    shed_capacity: int,
) -> bool:
    if unit_index >= len(positions):
        return False
    while len(inventories) <= unit_index:
        inventories.append({})
    inventory = inventories[unit_index]
    x, y = positions[unit_index]
    operation = command[0]

    movement = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }.get(operation)
    if movement is not None:
        next_x, next_y = x + movement[0], y + movement[1]
        if not (0 <= next_x < board_size and 0 <= next_y < board_size):
            return False
        positions[unit_index] = [next_x, next_y]
        return True
    if operation == "PASS":
        return True

    adjacent = (x, y) in _shed_tiles(board_size)
    if operation == "DROP":
        if not adjacent or not any(quantity > 0 for quantity in inventory.values()):
            return False
        room = max(0, shed_capacity - sum(shed.values()))
        for item, quantity in list(inventory.items()):
            accepted = min(max(0, int(quantity)), room)
            if accepted:
                _inventory_add(shed, item, accepted)
                room -= accepted
        inventory.clear()
        return True
    if operation == "PICKUP":
        item, requested = str(command[1]), int(command[2])
        amount = min(requested, int(shed.get(item, 0)))
        if not adjacent or amount <= 0:
            return False
        shed[item] -= amount
        _inventory_add(inventory, item, amount)
        return True
    if operation == "PLACE":
        item, requested = str(command[1]), int(command[2])
        tile = _tile_object(tiles[y][x])
        matching_structure = bool(
            item in ANIMAL_RULES
            and tile is not None
            and tile.get("kind") == ANIMAL_RULES[item][0]
            and tile.get("animal") is None
        )
        if matching_structure:
            if not _inventory_take(inventory, item, 1):
                return False
            tiles[y][x] = {
                "kind": ANIMAL_RULES[item][0],
                "animal": item,
                "placed_day": day,
                "yield_units": 0,
                "fed_today": False,
                "cared_today": False,
                "fertilizer_available": False,
            }
            return True
        room = max(0, shed_capacity - sum(shed.values()))
        amount = min(requested, int(inventory.get(item, 0)), room)
        if not adjacent or amount <= 0:
            return False
        _inventory_take(inventory, item, amount)
        _inventory_add(shed, item, amount)
        return True

    if tiles[y][x] == "LOCKED":
        return False
    tile = _tile_object(tiles[y][x])
    if operation == "PLANT":
        crop = str(command[1])
        if tiles[y][x] is not None or int(seeds.get(crop, 0)) <= 0:
            return False
        seeds[crop] -= 1
        tiles[y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": day,
            "watered_today": False,
            "yield_units": 0 if crop in {"TOMATO", "STRAWBERRY"} else 1,
            "fertilized_until_day": -1,
        }
        return True
    if operation == "WATER":
        if tile is None or tile.get("kind") != "PLANT" or tile.get("watered_today") is True:
            return False
        tile["watered_today"] = True
        return True
    if operation == "HARVEST":
        if tile is None or int(tile.get("yield_units", 0) or 0) <= 0:
            return False
        quantity = int(tile["yield_units"])
        if tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", ""))
            planted = int(tile.get("planted_day", day) or 0)
            if crop not in FIRST_YIELD_DAY or day - planted < FIRST_YIELD_DAY[crop]:
                return False
            _inventory_add(inventory, crop, quantity)
            tile["yield_units"] = 0
            if crop in {"WHEAT", "CARROT", "MELON"}:
                tiles[y][x] = None
            return True
        animal = tile.get("animal")
        if animal not in ANIMAL_RULES:
            return False
        tile["yield_units"] = 0
        _inventory_add(inventory, ANIMAL_RULES[str(animal)][1], quantity)
        return True
    if operation == "FERTILIZE":
        if tile is None or tile.get("kind") != "PLANT":
            return False
        if not _inventory_take(inventory, "FERTILIZER", 1):
            return False
        tile["fertilized_until_day"] = max(int(tile.get("fertilized_until_day", -1) or -1), day + 2)
        return True
    if operation == "DIG":
        if tiles[y][x] is None or (tile is not None and tile.get("animal") is not None):
            return False
        tiles[y][x] = None
        return True
    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        if tiles[y][x] is not None:
            return False
        tiles[y][x] = {"kind": operation.removeprefix("BUILD_")}
        return True
    if operation == "FEED":
        if tile is None or tile.get("animal") is None or tile.get("fed_today") is True:
            return False
        if not _inventory_take(inventory, "WHEAT", 1):
            return False
        tile["fed_today"] = True
        return True
    if operation == "COLLECT_FERTILIZER":
        if (
            tile is None
            or tile.get("animal") is None
            or tile.get("fertilizer_available") is not True
        ):
            return False
        tile["fertilizer_available"] = False
        _inventory_add(inventory, "FERTILIZER", 1)
        return True
    if operation == "CARE":
        if tile is None or tile.get("animal") is None or tile.get("cared_today") is True:
            return False
        tile["cared_today"] = True
        return True
    return False


def _shape(function: str, value: float, throughput: float) -> float:
    value = max(value, 0.0)
    if function == "linear":
        return value
    if function == "sq":
        return value * value
    if function == "sqrt":
        return math.sqrt(value)
    if function == "log":
        return math.log1p(value)
    if function == "hinge" and throughput > 0:
        normalized = value / throughput
        return normalized + 8.0 * max(normalized - 1.0, 0.0) ** 2
    return value


def _market_price(item: str, inventory: int) -> int:
    base, throughput, below_function, below_target, above_function, above_target = MARKET_PARAMS[
        item
    ]
    if inventory < 10_000:
        amplitude = below_target * base / _shape(below_function, throughput, throughput)
        price = base + amplitude * _shape(below_function, 10_000 - inventory, throughput)
    else:
        amplitude = above_target * base / _shape(above_function, throughput, throughput)
        price = base - amplitude * _shape(above_function, inventory - 10_000, throughput)
    return max(1, round(price))


def _fibonacci(index: int) -> int:
    first = second = 1
    for _ in range(index):
        first, second = second, first + second
    return first


def _canonical_market(order: Any) -> list[Any] | None:
    if not isinstance(order, list) or not order or not isinstance(order[0], str):
        return None
    operation = order[0]
    if operation in {"HIRE", "BUY_LAND"}:
        return [operation] if len(order) == 1 else None
    if len(order) != 3 or not isinstance(order[1], str):
        return None
    quantity = _positive_int(order[2])
    if quantity is None:
        return None
    item = order[1]
    valid = (
        (operation == "SELL" and item in PRODUCTS)
        or (operation == "BUY_PRODUCT" and item in {"WHEAT", "FERTILIZER"})
        or (operation == "BUY_SEED" and item in CROPS)
        or (operation == "BUY_ANIMAL" and item in ANIMALS)
    )
    return [operation, item, quantity] if valid else None


def legalize_action(
    observation: dict[str, Any],
    action: dict[str, Any],
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical action whose every component is locally effective."""

    config = configuration or {}
    board_size = int(config.get("boardSize", 10))
    shed_capacity = int(config.get("shedCapacity", 100))
    market_limit = int(config.get("maxMarketOrdersPerTurn", 10))
    hand_cost_multiplier = int(config.get("farmHandCostMult", 1))
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    private = observation.get("private", {})
    positions = copy.deepcopy([farm["farmer"], *farm.get("hands", [])])
    tiles = copy.deepcopy(farm["tiles"])
    inventories = copy.deepcopy(private.get("inventories", []))
    shed = copy.deepcopy(private.get("shed", {}))
    seeds = copy.deepcopy(private.get("seeds", {}))
    day = int(observation.get("day", int(observation.get("step", 0)) // 24))

    proposed_units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    legalized_units: list[list[Any]] = []
    for unit_index, proposed in enumerate(proposed_units[: len(positions)]):
        command = _canonical_unit(proposed)
        if not _apply_unit(
            command,
            unit_index,
            positions,
            tiles,
            inventories,
            shed,
            seeds,
            day=day,
            board_size=board_size,
            shed_capacity=shed_capacity,
        ):
            command = ["PASS"]
        legalized_units.append(command)
    if not legalized_units:
        legalized_units.append(["PASS"])

    money = float(farm.get("money", 0) or 0)
    hires_today = int(farm.get("hires_today", 0) or 0)
    unlocked = len(farm.get("unlocked_quadrants", []))
    market_inventory = copy.deepcopy(observation.get("market", {}).get("inventory", {}))
    accepted_market: list[list[Any]] = []
    for proposed in action.get("market", [])[:market_limit]:
        order = _canonical_market(proposed)
        if order is None:
            continue
        operation = order[0]
        if operation == "HIRE":
            cost = (
                0 if hand_cost_multiplier == 0 else _fibonacci(hires_today) * hand_cost_multiplier
            )
            if money < cost:
                continue
            money -= cost
            hires_today += 1
        elif operation == "BUY_LAND":
            land_index = max(0, unlocked - 1)
            if land_index >= len(LAND_PRICES) or money < LAND_PRICES[land_index]:
                continue
            money -= LAND_PRICES[land_index]
            unlocked += 1
        else:
            item, quantity = str(order[1]), int(order[2])
            committed = 0
            for _ in range(quantity):
                if operation == "SELL":
                    if int(shed.get(item, 0)) <= 0:
                        break
                    price = _market_price(item, int(market_inventory[item]))
                    shed[item] -= 1
                    money += price
                    if price > 1:
                        market_inventory[item] += 1
                elif operation == "BUY_PRODUCT":
                    price = _market_price(item, int(market_inventory[item]) - 1)
                    if money < price or sum(shed.values()) >= shed_capacity:
                        break
                    money -= price
                    _inventory_add(shed, item, 1)
                    market_inventory[item] -= 1
                elif operation == "BUY_SEED":
                    price = SEED_COSTS[item]
                    if money < price:
                        break
                    money -= price
                    _inventory_add(seeds, item, 1)
                else:
                    price = ANIMAL_RULES[item][2]
                    if money < price or sum(shed.values()) >= shed_capacity:
                        break
                    money -= price
                    _inventory_add(shed, item, 1)
                committed += 1
            if committed == 0:
                continue
        accepted_market.append(order)

    return {
        "farmer": legalized_units[0],
        "hands": legalized_units[1:],
        "market": accepted_market,
    }


def _semantic_unit_fallback(observation: dict[str, Any], actor: int) -> list[Any]:
    """Choose one useful local task after a replay command becomes stale.

    The fallback deliberately never moves a unit.  Preserving its position keeps
    the remaining replay route aligned while still recovering work that is
    already available on the current tile.
    """

    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    inventories = observation.get("private", {}).get("inventories", [])
    if actor >= len(positions):
        return ["PASS"]

    x, y = positions[actor]
    tile = farm["tiles"][y][x]
    inventory = inventories[actor] if actor < len(inventories) else {}
    day = int(observation.get("day", int(observation.get("step", 0)) // 24))

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = str(tile.get("crop", ""))
        planted_day = int(tile.get("planted_day", day) or 0)
        harvestable = (
            crop in FIRST_YIELD_DAY
            and int(tile.get("yield_units", 0) or 0) > 0
            and day - planted_day >= FIRST_YIELD_DAY[crop]
        )
        if harvestable:
            return ["HARVEST"]
        if tile.get("watered_today") is not True:
            return ["WATER"]
        if (
            int(inventory.get("FERTILIZER", 0) or 0) > 0
            and int(tile.get("fertilized_until_day", -1) or -1) < day
        ):
            return ["FERTILIZE"]
        return ["PASS"]

    if isinstance(tile, dict) and tile.get("animal") in ANIMAL_RULES:
        if int(tile.get("yield_units", 0) or 0) > 0:
            return ["HARVEST"]
        if tile.get("fed_today") is not True and int(inventory.get("WHEAT", 0) or 0) > 0:
            return ["FEED"]
        if tile.get("cared_today") is not True:
            return ["CARE"]
        if tile.get("fertilizer_available") is True:
            return ["COLLECT_FERTILIZER"]
        return ["PASS"]

    if isinstance(tile, dict) and tile.get("animal") is None:
        structure = tile.get("kind")
        for animal, (required_structure, _, _) in ANIMAL_RULES.items():
            if structure == required_structure and int(inventory.get(animal, 0) or 0) > 0:
                return ["PLACE", animal, 1]

    board_size = len(farm["tiles"])
    if (x, y) in _shed_tiles(board_size) and any(
        int(quantity or 0) > 0 for quantity in inventory.values()
    ):
        return ["DROP"]
    return ["PASS"]


def recover_rejected_unit_intents(
    observation: dict[str, Any],
    proposed: dict[str, Any],
    legalized: dict[str, Any],
    configuration: dict[str, Any] | None = None,
    *,
    allowed_operations: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Replace rejected non-PASS unit commands with state-aware local work.

    A final legality pass projects all replacements together, so two units can
    never consume the same inventory or tile resource in one turn.
    """

    recovered = copy.deepcopy(legalized)
    proposed_units = [proposed.get("farmer", ["PASS"]), *proposed.get("hands", [])]
    legal_units = [legalized.get("farmer", ["PASS"]), *legalized.get("hands", [])]
    for actor, (before, after) in enumerate(zip(proposed_units, legal_units, strict=False)):
        if (
            not isinstance(before, list)
            or not before
            or before[0] == "PASS"
            or before[0] not in SEMANTIC_RECOVERABLE_UNIT_OPS
            or after != ["PASS"]
        ):
            continue
        fallback = _semantic_unit_fallback(observation, actor)
        if fallback != ["PASS"] and (
            allowed_operations is None or fallback[0] in allowed_operations
        ):
            WeedRecoveryController._set_actor(recovered, actor, fallback)
    return legalize_action(observation, recovered, configuration)
