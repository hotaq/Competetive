"""Observable micro rules layered over an existing unit schedule."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
    "EAST": (1, 0),
}


@dataclass(frozen=True, slots=True)
class AdjacentHighValueShedReturnRule:
    """Redirect a loaded unit one tile into an open shed-access tile."""

    minimum_inventory_value: float = 1_000.0

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result

        farm = farms[player]
        tiles = farm.get("tiles", [])
        board_size = len(tiles) if isinstance(tiles, list) else 0
        if board_size < 2:
            return result

        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        inventories = observation.get("private", {}).get("inventories", [])
        if not isinstance(inventories, list):
            return result

        commands = [result.get("farmer", ["PASS"]), *result.get("hands", [])]
        prices = observation.get("market", {}).get("prices", {})
        half = board_size // 2
        shed_tiles = {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }
        occupied = {
            (int(position[0]), int(position[1]))
            for position in positions
            if isinstance(position, list) and len(position) == 2
        }

        for actor, (position, inventory, command) in enumerate(
            zip(positions, inventories, commands, strict=False)
        ):
            if (
                not isinstance(position, list)
                or len(position) != 2
                or not isinstance(inventory, dict)
                or not isinstance(command, list)
                or not command
                or command[0] not in _MOVES
            ):
                continue

            inventory_value = sum(
                max(0, int(quantity or 0)) * float(prices.get(str(item), 0.0) or 0.0)
                for item, quantity in inventory.items()
            )
            if inventory_value < self.minimum_inventory_value:
                continue

            x, y = int(position[0]), int(position[1])
            if (x, y) in shed_tiles:
                continue

            dx, dy = _MOVES[command[0]]
            if (x + dx, y + dy) in shed_tiles:
                continue

            replacement: list[str] | None = None
            for operation, (move_x, move_y) in _MOVES.items():
                target = (x + move_x, y + move_y)
                if target in shed_tiles and target not in occupied:
                    replacement = [operation]
                    break
            if replacement is None:
                continue

            if actor == 0:
                result["farmer"] = replacement
            elif actor - 1 < len(result.get("hands", [])):
                result["hands"][actor - 1] = replacement

        return result


@dataclass(slots=True)
class AdjacentShedReturnController:
    """Run one observable enter-shed/drop transaction, then resume the base policy."""

    minimum_inventory_value: float = 1_000.0
    maximum_interventions_per_game: int = 1
    liquidation_items: tuple[str, ...] = ()
    maximum_market_orders: int = 10
    _pending: dict[int, tuple[int, int]] = field(default_factory=dict, init=False)
    _interventions: dict[int, int] = field(default_factory=dict, init=False)
    _last_step: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._pending.clear()
            self._interventions.clear()
            self._last_step.clear()
            return
        self._pending.pop(player, None)
        self._interventions.pop(player, None)
        self._last_step.pop(player, None)

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        player = int(observation.get("player", 0) or 0)
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step < self._last_step.get(player, -1):
            self.reset(player)
        self._last_step[player] = step

        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        tiles = farm.get("tiles", [])
        board_size = len(tiles) if isinstance(tiles, list) else 0
        if board_size < 2:
            return result

        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        inventories = observation.get("private", {}).get("inventories", [])
        if not isinstance(inventories, list):
            return result
        commands = [result.get("farmer", ["PASS"]), *result.get("hands", [])]
        prices = observation.get("market", {}).get("prices", {})
        half = board_size // 2
        shed_tiles = {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }

        pending = self._pending.get(player)
        if pending is not None:
            actor, started_step = pending
            if step > started_step + 2 or actor >= len(positions):
                self._pending.pop(player, None)
                return result
            position = positions[actor]
            inventory = inventories[actor] if actor < len(inventories) else {}
            if (
                isinstance(position, list)
                and len(position) == 2
                and (int(position[0]), int(position[1])) in shed_tiles
                and isinstance(inventory, dict)
                and self._inventory_value(inventory, prices)
                >= self.minimum_inventory_value
            ):
                self._set_actor(result, actor, ["DROP"])
                self._prepend_projected_sales(result, inventory)
                self._pending.pop(player, None)
                return result

        if self._interventions.get(player, 0) >= self.maximum_interventions_per_game:
            return result

        occupied = {
            (int(position[0]), int(position[1]))
            for position in positions
            if isinstance(position, list) and len(position) == 2
        }
        for actor, (position, inventory, command) in enumerate(
            zip(positions, inventories, commands, strict=False)
        ):
            if (
                not isinstance(position, list)
                or len(position) != 2
                or not isinstance(inventory, dict)
                or self._inventory_value(inventory, prices) < self.minimum_inventory_value
                or not isinstance(command, list)
                or not command
                or command[0] not in _MOVES
            ):
                continue
            x, y = int(position[0]), int(position[1])
            if (x, y) in shed_tiles:
                continue
            dx, dy = _MOVES[command[0]]
            if (x + dx, y + dy) in shed_tiles:
                continue
            for operation, (move_x, move_y) in _MOVES.items():
                target = (x + move_x, y + move_y)
                if target not in shed_tiles or target in occupied:
                    continue
                self._set_actor(result, actor, [operation])
                self._pending[player] = (actor, step)
                self._interventions[player] = self._interventions.get(player, 0) + 1
                return result
        return result

    @staticmethod
    def _inventory_value(inventory: dict[str, Any], prices: dict[str, Any]) -> float:
        return sum(
            max(0, int(quantity or 0)) * float(prices.get(str(item), 0.0) or 0.0)
            for item, quantity in inventory.items()
        )

    def _prepend_projected_sales(
        self,
        action: dict[str, Any],
        inventory: dict[str, Any],
    ) -> None:
        """Sell selected carried goods in the same action that drops them into the shed."""
        if not self.liquidation_items or self.maximum_market_orders <= 0:
            return

        market = list(action.get("market", []))
        scheduled: dict[str, int] = {}
        for order in market:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "SELL"
            ):
                item = str(order[1])
                scheduled[item] = scheduled.get(item, 0) + max(0, int(order[2] or 0))

        projected: list[list[Any]] = []
        for item in self.liquidation_items:
            available = max(0, int(inventory.get(item, 0) or 0)) - scheduled.get(item, 0)
            if available > 0:
                projected.append(["SELL", item, available])

        if projected:
            action["market"] = [*projected, *market][: self.maximum_market_orders]

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[str]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command


@dataclass(slots=True)
class OrphanStructureRecoveryController:
    """Fill a structure left empty across a day boundary after route recovery."""

    minimum_step: int = 193
    start_hour: int = 1
    feed_quantity: int = 4
    maximum_interventions_per_game: int = 1
    maximum_task_steps: int = 20
    _pending: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _interventions: dict[int, int] = field(default_factory=dict, init=False)
    _last_step: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._pending.clear()
            self._interventions.clear()
            self._last_step.clear()
            return
        self._pending.pop(player, None)
        self._interventions.pop(player, None)
        self._last_step.pop(player, None)

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        player = int(observation.get("player", 0) or 0)
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step < self._last_step.get(player, -1):
            self.reset(player)
        self._last_step[player] = step

        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        inventories = observation.get("private", {}).get("inventories", [])
        shed = observation.get("private", {}).get("shed", {})
        if not positions or not isinstance(inventories, list) or not isinstance(shed, dict):
            return result

        task = self._pending.get(player)
        if task is None:
            if (
                step < self.minimum_step
                or step % 24 != self.start_hour
                or self._interventions.get(player, 0)
                >= self.maximum_interventions_per_game
                or len(positions) < 2
            ):
                return result
            target = self._find_orphan(farm, shed)
            if target is None:
                return result
            x, y, animal = target
            task = {
                "actor": len(positions) - 1,
                "target": (x, y),
                "animal": animal,
                "started_step": step,
            }
            self._pending[player] = task
            self._interventions[player] = self._interventions.get(player, 0) + 1

        actor = int(task["actor"])
        if (
            actor >= len(positions)
            or step - int(task["started_step"]) > self.maximum_task_steps
        ):
            self._pending.pop(player, None)
            return result

        position = positions[actor]
        inventory = inventories[actor] if actor < len(inventories) else {}
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not isinstance(inventory, dict)
        ):
            self._pending.pop(player, None)
            return result

        x, y = int(position[0]), int(position[1])
        target_x, target_y = task["target"]
        animal = str(task["animal"])
        tile = farm.get("tiles", [])[target_y][target_x]
        if isinstance(tile, dict) and tile.get("animal") is not None:
            if (x, y) != (target_x, target_y):
                self._pending.pop(player, None)
                return result
            if int(inventory.get("WHEAT", 0) or 0) > 0 and tile.get("fed_today") is not True:
                self._set_actor(result, actor, ["FEED"])
                return result
            if tile.get("cared_today") is not True:
                self._set_actor(result, actor, ["CARE"])
                return result
            self._pending.pop(player, None)
            return result

        if not self._matches_empty_structure(tile, animal):
            self._pending.pop(player, None)
            return result

        shed_tiles = self._shed_tiles(len(farm.get("tiles", [])))
        if int(inventory.get(animal, 0) or 0) <= 0:
            if (x, y) not in shed_tiles or int(shed.get(animal, 0) or 0) <= 0:
                self._pending.pop(player, None)
                return result
            self._set_actor(result, actor, ["PICKUP", animal, 1])
            return result

        if (
            int(inventory.get("WHEAT", 0) or 0) <= 0
            and (x, y) in shed_tiles
            and int(shed.get("WHEAT", 0) or 0) > 0
        ):
            quantity = min(self.feed_quantity, int(shed["WHEAT"]))
            self._set_actor(result, actor, ["PICKUP", "WHEAT", quantity])
            return result

        if x != target_x:
            self._set_actor(result, actor, ["EAST" if x < target_x else "WEST"])
            return result
        if y != target_y:
            self._set_actor(result, actor, ["SOUTH" if y < target_y else "NORTH"])
            return result
        self._set_actor(result, actor, ["PLACE", animal, 1])
        return result

    @staticmethod
    def _shed_tiles(board_size: int) -> set[tuple[int, int]]:
        half = board_size // 2
        return {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }

    @staticmethod
    def _matches_empty_structure(tile: Any, animal: str) -> bool:
        expected = "COOP" if animal == "GOOSE" else "PASTURE"
        return (
            isinstance(tile, dict)
            and tile.get("kind") == expected
            and tile.get("animal") is None
        )

    @classmethod
    def _find_orphan(
        cls,
        farm: dict[str, Any],
        shed: dict[str, Any],
    ) -> tuple[int, int, str] | None:
        choices: list[tuple[int, int, int, str]] = []
        tiles = farm.get("tiles", [])
        board_size = len(tiles)
        shed_tiles = cls._shed_tiles(board_size)
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("animal") is not None:
                    continue
                kind = tile.get("kind")
                animals = ("COW", "SHEEP") if kind == "PASTURE" else ("GOOSE",)
                for animal in animals:
                    if int(shed.get(animal, 0) or 0) <= 0:
                        continue
                    distance = min(abs(x - sx) + abs(y - sy) for sx, sy in shed_tiles)
                    choices.append((distance, y, x, animal))
                    break
        if not choices:
            return None
        _, y, x, animal = min(choices)
        return x, y, animal

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[str]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command


@dataclass(frozen=True, slots=True)
class CapHarvRule:
    """Harvest animal yield now when the midnight shed dump would overflow.

    Overnight, every carried inventory drops into the shed up to
    ``shed_capacity`` and the excess is discarded. When an actor is scheduled
    for CARE or COLLECT_FERTILIZER on a COOP/PASTURE tile that already holds
    harvestable yield, and the projected midnight load (shed stock plus all
    carried goods plus the liftable tile yield) exceeds the shed capacity with
    no HARVEST scheduled anywhere else in this action, the banked-care value
    (+1 future yield, contingent on future feeding and the tile max-held cap
    of 6 cow/sheep or 4 goose) is worth less than the concrete yield already
    on the tile. Replacing the maintenance op with HARVEST banks concrete
    goods into carried inventory with maximal remaining steps for downstream
    DROP/SELL before the midnight dump.
    """

    shed_capacity: int = 100

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        private = observation.get("private", {})
        inventories = private.get("inventories", []) if isinstance(private, dict) else []
        shed = private.get("shed", {}) if isinstance(private, dict) else {}
        if not isinstance(inventories, list) or not isinstance(shed, dict):
            return result
        commands = [result.get("farmer", ["PASS"]), *result.get("hands", [])]
        if any(
            isinstance(command, list) and command and command[0] == "HARVEST"
            for command in commands
        ):
            return result

        shed_total = sum(max(0, int(quantity or 0)) for quantity in shed.values())
        carried_total = sum(
            max(0, int(quantity or 0))
            for inventory in inventories
            if isinstance(inventory, dict)
            for quantity in inventory.values()
        )

        tiles = farm.get("tiles", [])
        for actor, (position, command) in enumerate(
            zip(positions, commands, strict=False)
        ):
            if (
                not isinstance(command, list)
                or not command
                or command[0] not in {"CARE", "COLLECT_FERTILIZER"}
                or not isinstance(position, list)
                or len(position) != 2
            ):
                continue
            tile = self._tile_at(tiles, int(position[0]), int(position[1]))
            if (
                tile is None
                or tile.get("kind") not in {"COOP", "PASTURE"}
                or tile.get("animal") is None
            ):
                continue
            liftable = max(0, int(tile.get("yield_units", 0) or 0))
            if liftable <= 0:
                continue
            if shed_total + carried_total + liftable <= self.shed_capacity:
                continue
            self._set_actor(result, actor, ["HARVEST"])
        return result

    @staticmethod
    def _tile_at(tiles: Any, x: int, y: int) -> dict[str, Any] | None:
        if not isinstance(tiles, list) or not 0 <= y < len(tiles):
            return None
        row = tiles[y]
        if not isinstance(row, list) or not 0 <= x < len(row):
            return None
        tile = row[x]
        return tile if isinstance(tile, dict) else None

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[str]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command


@dataclass(frozen=True, slots=True)
class HourGatedCapHarvRule:
    """CapHarvRule flip restricted to the last ``flip_window_turns`` of the day.

    V103 showed the unconditioned flip over-triggers: early in the day the
    projected "overflow" is usually phantom because the router's normal
    DROP/SELL rhythm has a full day left to clear it, while the sacrificed
    banked-care bonus (+1/day, tile cap 6 cow/sheep, 4 goose) is concrete.
    This wrapper only delegates to :class:`CapHarvRule` when the midnight
    dump is close enough that the projection is actionable.

    ``flip_window_turns = 10`` is engine-grounded
    (``rust/kaggriculture-engine/src/engine.rs``): the dump runs after the
    hour-23 turn (``step`` guard at :338, ``drop_inventories_to_shed`` at
    :1146), so observation hour ``h = step % 24`` leaves ``24 - h`` turns
    including the current one. Harvested goods sit in carried inventory and
    only escape the dump via DROP onto a shed tile plus SELL out of shed
    stock. Walk budget on the 10x10 board to the center 2x2 shed is at most 8
    (corner to nearest shed tile) + 1 DROP turn + 1 slack turn for the
    router's SELL to observe and clear. Flips with more than 10 turns left
    are blocked; ``step`` missing/zero fails closed (hour 0, blocked).
    """

    flip_window_turns: int = 10
    turns_per_day: int = 24
    shed_capacity: int = 100

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        per_day = max(1, int(self.turns_per_day))
        step = int(observation.get("step", 0) or 0)
        hour = step % per_day
        if per_day - hour > self.flip_window_turns:
            return copy.deepcopy(action)
        return CapHarvRule(shed_capacity=self.shed_capacity).apply(observation, action)


# Engine-grounded animal tables for ValueGatedCapHarvRule
# (rust/kaggriculture-engine/src/rules.rs `animal_rule`).
_VALUE_GATE_PRODUCTS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
_VALUE_GATE_INTERVALS = {"GOOSE": 1, "COW": 2, "SHEEP": 3}
_VALUE_GATE_MAX_HELD = {"GOOSE": 4, "COW": 6, "SHEEP": 6}
_VALUE_GATE_FIRST_YIELD = {"GOOSE": 4, "COW": 8, "SHEEP": 6}


@dataclass(frozen=True, slots=True)
class ValueGatedCapHarvRule:
    """Flip CARE/COLLECT to HARVEST only when the banked-bonus EV is beaten.

    V108 removed ~95% of V103's harm with the hour gate but left a tiny
    negative residual (paired diff -0.0014, CI excluding 0): even inside the
    window the unconditional flip still accepts bad trades. This wrapper
    composes the v108 hour gate and the v103 overflow projection
    (:class:`HourGatedCapHarvRule` -> :class:`CapHarvRule`, both untouched)
    and then reverts each flipped actor unless the trade wins on value:

        banked_bonus_value < overflow_waste_avoided   -> keep HARVEST
        banked_bonus_value >= overflow_waste_avoided  -> revert (ties fail closed)

    banked_bonus_value = capped_units * fed_factor / days_to_payout * price
    overflow_waste_avoided = min(tile_yield, projected_overcap) * price

    Engine grounding (read-only; rust/kaggriculture-engine/src):
    - ``rules.rs`` `animal_rule`: GOOSE interval 1 / max_held 4 / product EGG,
      COW 2 / 6 / MILK, SHEEP 3 / 6 / WOOL.
    - ``engine.rs`` `daily_refresh_animals` (:1087): at each midnight refresh a
      tile on its production day banks ``1 + pending_care_bonus`` (only if fed;
      unfed banks 1 + 0), capped at ``max_held``; the bonus then resets to 0.
      CARE banks +1 into the bonus only when ``cared_today && fed`` — a second
      CARE the same day banks nothing, and an unfed CARE banks only if a later
      FEED lands before the refresh.
    - ``engine.rs`` `drop_inventories_to_shed` (:1146): carried inventories
      drop into the shed up to ``shed_capacity`` (100, non-seed); the excess is
      discarded.
    - Hour gate (v108): the dump runs after the hour-23 turn, so ``24 - hour``
      turns remain including the current one; flips with more than
      ``flip_window_turns`` (10) left are blocked.

    Per-term mapping onto observables:
    - ``capped_units = min(units_at_risk, max(0, max_held - tile_yield))``:
      the payout caps total yield at max_held, so bonus units beyond current
      headroom are worthless. ``units_at_risk`` is the observed
      ``pending_care_bonus`` plus 1 when the flipped op is CARE on a not-yet-
      cared tile (today's CARE would have banked); COLLECT_FERTILIZER banks
      nothing, so it adds no +1 — for COLLECT the pending bank is kept as a
      conservative proxy for the tile's near-term production value at stake.
    - ``fed_factor`` (``unfed_discount`` = 0.5 when ``fed_today`` is not True):
      the bank pays only if fed at the refresh; an already-fed tile meets the
      condition, an unfed one needs a future FEED. The 0.5 is a judgment call,
      documented here, not an engine constant.
    - ``days_to_payout`` (>= 1): from ``placed_day`` + first-yield day +
      interval against the observed ``day``; the bonus is a delayed lump while
      tonight's waste is concrete, discounted 1/d. Missing ``placed_day`` or
      an unknown animal fails closed (d = 1, full bonus value / revert).
    - ``price``: current ``market.prices[product]``; both sides scale with the
      same product price, so the comparison reduces to units — the price terms
      keep the units honest and force fail-closed when the price is missing.
    - Fertilizer forgone on a COLLECT flip is ignored (same single-step-view
      limitation as the v103 projection, documented not fixed).
    """

    flip_window_turns: int = 10
    turns_per_day: int = 24
    shed_capacity: int = 100
    unfed_discount: float = 0.5

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        gated = HourGatedCapHarvRule(
            flip_window_turns=self.flip_window_turns,
            turns_per_day=self.turns_per_day,
            shed_capacity=self.shed_capacity,
        ).apply(observation, action)
        base_commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        gated_commands = [gated.get("farmer", ["PASS"]), *gated.get("hands", [])]
        if not any(
            isinstance(new, list)
            and new == ["HARVEST"]
            and isinstance(old, list)
            and old
            and old[0] in {"CARE", "COLLECT_FERTILIZER"}
            for old, new in zip(base_commands, gated_commands, strict=False)
        ):
            return gated

        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return gated
        farm = farms[player]
        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        private = observation.get("private", {})
        inventories = private.get("inventories", []) if isinstance(private, dict) else []
        shed = private.get("shed", {}) if isinstance(private, dict) else {}
        if not isinstance(inventories, list) or not isinstance(shed, dict):
            return gated
        prices = observation.get("market", {}).get("prices", {})
        if not isinstance(prices, dict):
            return gated
        day = observation.get("day", int(observation.get("step", 0) or 0) // 24)
        try:
            day = int(day)
        except (TypeError, ValueError):
            return gated

        shed_total = sum(max(0, int(quantity or 0)) for quantity in shed.values())
        carried_total = sum(
            max(0, int(quantity or 0))
            for inventory in inventories
            if isinstance(inventory, dict)
            for quantity in inventory.values()
        )
        tiles = farm.get("tiles", [])
        result = copy.deepcopy(gated)
        for actor, (position, old, new) in enumerate(
            zip(positions, base_commands, gated_commands, strict=False)
        ):
            if (
                not isinstance(new, list)
                or new != ["HARVEST"]
                or not isinstance(old, list)
                or not old
                or old[0] not in {"CARE", "COLLECT_FERTILIZER"}
                or not isinstance(position, list)
                or len(position) != 2
            ):
                continue
            tile = CapHarvRule._tile_at(tiles, int(position[0]), int(position[1]))
            if tile is None:
                continue
            liftable = max(0, int(tile.get("yield_units", 0) or 0))
            overcap = shed_total + carried_total + liftable - self.shed_capacity
            if not self._flip_wins_on_value(
                tile, old[0], liftable, overcap, day, prices
            ):
                self._set_actor(result, actor, old)
        return result

    def _flip_wins_on_value(
        self,
        tile: dict[str, Any],
        op: str,
        liftable: int,
        overcap: int,
        day: int,
        prices: dict[str, Any],
    ) -> bool:
        animal = tile.get("animal")
        if animal not in _VALUE_GATE_PRODUCTS or liftable <= 0 or overcap <= 0:
            return False
        try:
            price = float(prices.get(_VALUE_GATE_PRODUCTS[animal], 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if price <= 0.0:
            return False
        pending = tile.get("pending_care_bonus", 0)
        try:
            pending = max(0, int(pending or 0))
        except (TypeError, ValueError):
            return False
        units_at_risk = pending
        if op == "CARE" and tile.get("cared_today") is not True:
            units_at_risk += 1
        headroom = max(0, _VALUE_GATE_MAX_HELD[animal] - liftable)
        capped_units = min(units_at_risk, headroom)
        fed_factor = 1.0 if tile.get("fed_today") is True else self.unfed_discount
        days = self._days_to_next_payout(tile, day, animal)
        banked_bonus_value = capped_units * fed_factor / days * price
        overflow_waste_avoided = min(liftable, overcap) * price
        return banked_bonus_value < overflow_waste_avoided

    @staticmethod
    def _days_to_next_payout(tile: dict[str, Any], day: int, animal: str) -> int:
        placed = tile.get("placed_day")
        if placed is None or isinstance(placed, bool):
            return 1
        try:
            placed_day = int(placed)
        except (TypeError, ValueError):
            return 1
        interval = _VALUE_GATE_INTERVALS[animal]
        first = _VALUE_GATE_FIRST_YIELD[animal]
        candidate = day + 1
        while True:
            since_first = candidate - placed_day - first
            if since_first >= 0 and since_first % interval == 0:
                return candidate - day
            candidate += 1

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[str]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command


@dataclass(frozen=True, slots=True)
class LocalWorkFillRule:
    """Replace idle PASS commands with useful work on the current tile."""

    operations: frozenset[str] = frozenset(
        {"HARVEST", "FEED", "CARE", "COLLECT_FERTILIZER", "WATER"}
    )
    minimum_step: int = 0

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        if int(observation.get("step", 0) or 0) < self.minimum_step:
            return result
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        inventories = observation.get("private", {}).get("inventories", [])
        commands = [result.get("farmer", ["PASS"]), *result.get("hands", [])]
        day = int(observation.get("day", int(observation.get("step", 0) or 0) // 24))

        for actor, (position, command) in enumerate(
            zip(positions, commands, strict=False)
        ):
            if command != ["PASS"] or not isinstance(position, list) or len(position) != 2:
                continue
            x, y = int(position[0]), int(position[1])
            tile = farm.get("tiles", [])[y][x]
            if not isinstance(tile, dict):
                continue
            inventory = inventories[actor] if actor < len(inventories) else {}
            replacement = self._replacement(tile, inventory, day)
            if replacement is not None:
                self._set_actor(result, actor, replacement)
        return result

    def _replacement(
        self,
        tile: dict[str, Any],
        inventory: Any,
        day: int,
    ) -> list[str] | None:
        if not isinstance(inventory, dict):
            inventory = {}
        if tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", ""))
            maturity = {
                "WHEAT": 2,
                "CARROT": 2,
                "TOMATO": 8,
                "STRAWBERRY": 10,
                "MELON": 10,
            }.get(crop)
            planted_day = int(tile.get("planted_day", day) or 0)
            if (
                "HARVEST" in self.operations
                and maturity is not None
                and day - planted_day >= maturity
                and int(tile.get("yield_units", 0) or 0) > 0
            ):
                return ["HARVEST"]
            if "WATER" in self.operations and tile.get("watered_today") is not True:
                return ["WATER"]
            return None

        if tile.get("animal") is not None:
            if (
                "HARVEST" in self.operations
                and int(tile.get("yield_units", 0) or 0) > 0
            ):
                return ["HARVEST"]
            if (
                "FEED" in self.operations
                and tile.get("fed_today") is not True
                and int(inventory.get("WHEAT", 0) or 0) > 0
            ):
                return ["FEED"]
            if "CARE" in self.operations and tile.get("cared_today") is not True:
                return ["CARE"]
            if (
                "COLLECT_FERTILIZER" in self.operations
                and tile.get("fertilizer_available") is True
            ):
                return ["COLLECT_FERTILIZER"]
        return None

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[str]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command
