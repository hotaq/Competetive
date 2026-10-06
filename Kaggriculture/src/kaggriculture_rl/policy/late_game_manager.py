"""Late-game manager: closed-loop farm upkeep and sell-through after a handover step.

Evidence (analysis/v97 live losses, 2026-09-05): money is level at day 18 and the games are lost in
days 18-30, where opponents out-produce us in milk and strawberries with the same land while our
open-loop tape keeps a fixed plan that desyncs on bad weed draws. A policy that reads the farm
every step does not care about that.

Engine facts the manager relies on (rust/kaggriculture-engine): hands are re-hired every day at
fibonacci coin prices and spawn on the four shed-access tiles; FEED consumes one WHEAT from the
acting unit's inventory and is required for that day's animal yield (1 + accumulated CARE bonus);
two unfed days kill the animal; WATER adds yield to wheat/carrot/melon inside their window and any
plant unwatered two days becomes a weed; PICKUP/DROP only work on the shed-access tiles; unit
inventories drop into the shed at the end of each day and SELL only sells from the shed; the last
day has no end-of-day drop, so goods must be dropped before the final market.

Daily plan (built at hour 0, kept for the day): hire, top up wheat, every unit picks up the wheat
and fertilizer its tour needs, then walks a fixed nearest-neighbour tour over its share of the
tiles doing feed/care/water/harvest/fertilize/plant, drops at the shed, and helps with leftovers.
Pure standard library so the standalone bundle can inline it.

RESULT 2026-09-05 (negative, kept as a record): handing over from the v97 router at step 432
(day 18) loses to the router's own recorded late game in 120/120 development games (2 opponents
x 60 seeds): median day-18-to-30 gain +46k vs the tape's +62k; wins 65/120 vs 110/120. The
tape's day-18 farm was healthy in every one of those games (52-62 plants, <= 13 weeds), so
there was nothing to rescue either. Remaining gaps after fixing feeding, watering, planting,
animal collection and fertilizing: ~20% fewer strawberry/wheat units, and the tape's dumping of
wheat and fertilizer also depresses the opponent's prices (opponent money +7k against the
manager). Do not ship; a late-game policy has to beat the recording, not merely tend the farm.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

PolicyFn = Callable[[Any, Any], dict[str, Any]]
PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}
SHED_TILES: tuple[tuple[int, int], ...] = ((4, 4), (5, 4), (4, 5), (5, 5))
WINDOW_CROPS = {"WHEAT": (2, 4, 6), "CARROT": (2, 3, 4), "MELON": (10, 12, 6)}
ONGOING_CROPS = {"TOMATO", "STRAWBERRY"}
ANIMALS = ("COW", "SHEEP", "GOOSE")
MAX_ORDERS = 10
LAST_DAY = 29
Pos = tuple[int, int]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def dist(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(position: Pos, target: Pos) -> list[str]:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    if dx == 0 and dy == 0:
        return ["PASS"]
    if abs(dx) >= abs(dy):
        return ["EAST"] if dx > 0 else ["WEST"]
    return ["SOUTH"] if dy > 0 else ["NORTH"]


def nearest_shed_tile(position: Pos) -> Pos:
    return min(SHED_TILES, key=lambda t: dist(t, position))


def quadrant_of(x: int, y: int, board: int = 10) -> str:
    half = board // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def tile_work(tile: Any, day: int, plant_until_day: int, seeds_left: int) -> list[str]:
    """Ordered actions a unit should perform on this tile today (empty list = nothing)."""
    if not isinstance(tile, dict) or not tile:
        # water right after planting so the new plant never starts a day unwatered
        return ["PLANT", "WATER"] if day <= plant_until_day and seeds_left > 0 else []
    if tile.get("animal"):
        work = []
        if int(tile.get("yield_units", 0) or 0) > 0:
            work.append("HARVEST")  # milk/wool/eggs accrue on the tile and must be collected
        if not tile.get("fed_today"):
            work.append("FEED")
        if not tile.get("cared_today"):
            work.append("CARE")
        if tile.get("fertilizer_available"):
            work.append("COLLECT_FERTILIZER")  # input for FERTILIZE, which doubles yields
        return work
    kind = tile.get("kind")
    if kind == "PLANT":
        crop = str(tile.get("crop", ""))
        age = day - int(tile.get("planted_day", day) or 0)
        units = int(tile.get("yield_units", 0) or 0)
        work = []
        unfertilized = int(tile.get("fertilized_until_day", -1) or -1) < day
        if crop in ONGOING_CROPS:
            if units > 0:
                work.append("HARVEST")
            if not tile.get("watered_today"):
                work.append("WATER")
            if unfertilized and day <= LAST_DAY - 1:
                # a watered, fertilized strawberry produces 2 per cycle instead of 1
                work.append("FERTILIZE")
            return work
        _first, last, max_yield = WINDOW_CROPS.get(crop, (2, 4, 6))
        if units > 0 and (units >= max_yield or age >= last or day >= LAST_DAY - 1):
            replant = day <= plant_until_day and seeds_left > 0
            return ["HARVEST", "PLANT", "WATER"] if replant else ["HARVEST"]
        if not tile.get("watered_today"):
            work.append("WATER")
        if unfertilized and age < last and day <= plant_until_day + 1:
            work.append("FERTILIZE")
        return work
    if kind == "WEED" and day <= plant_until_day and seeds_left > 0:
        return ["DIG", "PLANT", "WATER"]
    return []


def work_class(tile: Any, actions: list[str]) -> int:
    """Lower is more valuable per action: animals, harvests, watering, then planting/fertilizing."""
    if isinstance(tile, dict) and tile.get("animal"):
        return 0
    if "HARVEST" in actions or (isinstance(tile, dict) and tile.get("crop") in ONGOING_CROPS):
        return 1
    if "WATER" in actions:
        return 2
    return 3


class LateGameManager:
    """`agent(observation) -> action`; holds a per-day plan, so keep one instance per game."""

    def __init__(self, *, hires_per_day: int = 12, sell_batch: int = 6, plant_until_day: int = 26,
                 wheat_buffer: int = 6) -> None:
        self.hires_per_day = hires_per_day
        self.sell_batch = sell_batch
        self.plant_until_day = plant_until_day
        self.wheat_buffer = wheat_buffer
        self._plan_day: int | None = None
        self._tours: dict[int, list[Pos]] = {}
        self._loaded: set[int] = set()
        self._unit_count = 0

    def reset(self) -> None:
        self._plan_day = None
        self._tours = {}
        self._loaded = set()
        self._unit_count = 0

    # ---- planning ---------------------------------------------------------------------
    def _build_tours(self, units: list[Pos], work_tiles: list[Pos]) -> dict[int, list[Pos]]:
        """Balanced nearest-neighbour tours, filled in the given (value) order.

        `work_tiles` arrives sorted by value class; within a round every unit takes the closest
        unclaimed tile of the current class, so if the day runs out the cheapest work is what is
        left undone.
        """
        tours: dict[int, list[Pos]] = {i: [] for i in range(len(units))}
        heads: dict[int, Pos] = {i: units[i] for i in range(len(units))}
        remaining = list(work_tiles)
        while remaining:
            for i in range(len(units)):
                if not remaining:
                    break
                best = min(remaining, key=lambda t: (dist(heads[i], t), t))
                remaining.remove(best)
                tours[i].append(best)
                heads[i] = best
        # locality: reorder each tour as a nearest-neighbour walk from the unit's start
        for i, tour in tours.items():
            walk: list[Pos] = []
            head = units[i]
            pool = list(tour)
            while pool:
                nxt = min(pool, key=lambda t: (dist(head, t), t))
                pool.remove(nxt)
                walk.append(nxt)
                head = nxt
            tours[i] = walk
        return tours

    # ---- policy -----------------------------------------------------------------------
    def agent(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        del configuration
        try:
            return self._agent(observation)
        except Exception:  # a malformed observation must still yield a legal no-op
            return copy.deepcopy(PASS_ACTION)

    def _agent(self, observation: Any) -> dict[str, Any]:
        player = int(_get(observation, "player", 0) or 0)
        farm = list(_get(observation, "farms", []) or [])[player]
        private = _get(observation, "private", {}) or {}
        day = int(_get(observation, "day", 0) or 0)
        hour = int(_get(observation, "hour", 0) or 0)
        tiles = list(_get(farm, "tiles", []) or [])
        farmer = tuple(int(v) for v in (_get(farm, "farmer", [4, 4]) or [4, 4]))
        hands = [tuple(int(v) for v in h) for h in (_get(farm, "hands", []) or [])]
        units: list[Pos] = [farmer, *hands]
        raw_inv = list(_get(private, "inventories", []) or [])
        inventories = [dict(raw_inv[i]) if i < len(raw_inv) and isinstance(raw_inv[i], dict) else {}
                       for i in range(len(units))]
        shed = {k: int(v or 0) for k, v in dict(_get(private, "shed", {}) or {}).items()}
        seeds = {k: int(v or 0) for k, v in dict(_get(private, "seeds", {}) or {}).items()}
        prices = dict(_get(_get(observation, "market", {}) or {}, "prices", {}) or {})
        last_day = day >= LAST_DAY

        # today's work per tile
        seeds_left = seeds.get("WHEAT", 0)
        unlocked = set(_get(farm, "unlocked_quadrants", []) or ["NW"])
        board = len(tiles) or 10
        work: dict[Pos, list[str]] = {}
        animals = 0
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if quadrant_of(x, y, board) not in unlocked:
                    continue
                if isinstance(tile, dict) and tile.get("animal"):
                    animals += 1
                actions = tile_work(tile, day, self.plant_until_day, seeds_left)
                if actions:
                    if "PLANT" in actions:
                        seeds_left -= 1
                    work[(x, y)] = actions

        # (re)plan at hour 0 or when the unit count changed (hires landed / a plan is missing)
        if self._plan_day != day or self._unit_count != len(units):
            self._plan_day = day
            self._unit_count = len(units)
            ordered = sorted(work, key=lambda t: (work_class(tiles[t[1]][t[0]], work[t]), t))
            budget = len(units) * max(0, 23 - hour)
            kept: list[Pos] = []
            spent = 0.0
            for t in ordered:
                cost = len(work[t]) + 1.0
                # planting is optional: never let it eat into the labour that keeps plants alive
                limit = budget * 0.8 if work_class(tiles[t[1]][t[0]], work[t]) >= 3 else budget
                if spent + cost > limit:
                    if work_class(tiles[t[1]][t[0]], work[t]) >= 3:
                        continue
                    break
                kept.append(t)
                spent += cost
            upkeep = [t for t in kept if work_class(tiles[t[1]][t[0]], work[t]) < 3]
            planting = [t for t in kept if work_class(tiles[t[1]][t[0]], work[t]) >= 3]
            self._tours = self._build_tours(units, upkeep)
            tails = [self._tours[i][-1] if self._tours[i] else units[i] for i in range(len(units))]
            extra = self._build_tours(tails, planting)
            for i in range(len(units)):
                self._tours[i].extend(extra.get(i, []))
            self._loaded = set()

        commands: list[list[Any]] = []
        # the engine blocks every PLANT in a step if the demand exceeds the seeds on hand
        plant_budget = seeds.get("WHEAT", 0)
        for ui, pos in enumerate(units):
            inv = inventories[ui]
            tour = self._tours.get(ui, [])
            # drop everything late on the last day so the final market can sell it
            if last_day and hour >= 10 and sum(inv.values()) > 0:
                shed_tile = nearest_shed_tile(pos)
                commands.append(["DROP"] if pos == shed_tile else step_toward(pos, shed_tile))
                continue
            # morning load: wheat for the animals on the tour, fertilizer for its window crops
            if ui not in self._loaded and pos in SHED_TILES and not last_day:
                need_wheat = sum(1 for t in tour if "FEED" in work.get(t, []))
                need_fert = sum(1 for t in tour if "FERTILIZE" in work.get(t, []))
                have_wheat = int(inv.get("WHEAT", 0) or 0)
                if need_wheat > have_wheat and shed.get("WHEAT", 0) > 0:
                    take = min(need_wheat - have_wheat, shed["WHEAT"])
                    commands.append(["PICKUP", "WHEAT", take])
                    shed["WHEAT"] -= min(need_wheat - have_wheat, shed["WHEAT"])
                    continue
                if need_fert > int(inv.get("FERTILIZER", 0) or 0) and shed.get("FERTILIZER", 0) > 0:
                    take = min(need_fert, shed["FERTILIZER"])
                    commands.append(["PICKUP", "FERTILIZER", take])
                    shed["FERTILIZER"] -= take
                    self._loaded.add(ui)
                    continue
                self._loaded.add(ui)
            # advance the tour past tiles with nothing left to do
            while tour and not work.get(tour[0]):
                tour.pop(0)
            if not tour:
                # help: nearest tile with work that nobody is standing on
                leftovers = [t for t, acts in work.items() if acts and t not in units]
                if leftovers:
                    tour.append(min(leftovers, key=lambda t: dist(pos, t)))
                    self._tours[ui] = tour
                else:
                    shed_tile = nearest_shed_tile(pos)
                    if sum(inv.values()) > 0 and not last_day and hour >= 18:
                        at_shed = pos == shed_tile
                        commands.append(["DROP"] if at_shed else step_toward(pos, shed_tile))
                    else:
                        commands.append(["PASS"])
                    continue
            target = tour[0]
            if pos != target:
                commands.append(step_toward(pos, target))
                continue
            acts = work[target]
            action = acts[0]
            if action == "FEED" and int(inv.get("WHEAT", 0) or 0) <= 0:
                acts = [a for a in acts if a != "FEED"]
                if not acts:
                    tour.pop(0)
                    commands.append(["PASS"])
                    continue
                action = acts[0]
            if action == "FERTILIZE" and int(inv.get("FERTILIZER", 0) or 0) <= 0:
                acts = [a for a in acts if a != "FERTILIZE"]
                if not acts:
                    tour.pop(0)
                    commands.append(["PASS"])
                    continue
                action = acts[0]
            if action == "PLANT":
                if plant_budget <= 0:
                    commands.append(["PASS"])  # wait for seeds; the tile stays on the tour
                    continue
                plant_budget -= 1
                commands.append(["PLANT", "WHEAT"])
            else:
                commands.append([action])
            # keep the remaining actions for this tile; next step's observation says what got done
            work[target] = [a for a in acts if a != action]
            if not work[target]:
                tour.pop(0)

        # market: hire at dawn, keep wheat for feeding, sell the rest steadily
        orders: list[list[Any]] = []
        hires = self.hires_per_day if not last_day else max(6, self.hires_per_day // 2)
        if hour == 0:
            deficit = animals + self.wheat_buffer - shed.get("WHEAT", 0)
            if deficit > 0 and not last_day:
                orders.append(["BUY_PRODUCT", "WHEAT", int(deficit)])
            orders.extend([["HIRE"]] * min(hires, MAX_ORDERS - 1 - len(orders)))
        elif hour == 1:
            if hires > MAX_ORDERS - 2:
                orders.extend([["HIRE"]] * (hires - (MAX_ORDERS - 2)))
            if day <= self.plant_until_day:
                wanted = sum(1 for acts in work.values() if "PLANT" in acts)
                ripe_soon = sum(1 for acts in work.values() if "HARVEST" in acts)
                need = min(wanted + ripe_soon + 10, 60) - seeds.get("WHEAT", 0)
                if need > 0:
                    orders.append(["BUY_SEED", "WHEAT", int(need)])
        reserve = 0 if last_day else animals + self.wheat_buffer
        sellable = []
        for item, qty in shed.items():
            if item in ANIMALS or item not in prices or qty <= 0:
                continue
            if item == "WHEAT":
                qty -= reserve
            if qty > 0:
                sellable.append((int(prices.get(item, 0) or 0) * qty, item, qty))
        sellable.sort(reverse=True)
        batch = self.sell_batch * (3 if last_day and hour >= 12 else 1)
        for _, item, qty in sellable:
            if len(orders) >= MAX_ORDERS:
                break
            orders.append(["SELL", item, int(min(qty, batch))])
        return {"farmer": commands[0], "hands": commands[1:], "market": orders[:MAX_ORDERS]}

    __call__ = agent


class LateGameHandover:
    """Play `opening` (e.g. the shop router) until `start_step`, then the manager."""

    def __init__(self, opening: PolicyFn, manager: LateGameManager | None = None, *,
                 start_step: int = 432) -> None:
        self._opening = opening
        self._manager = manager if manager is not None else LateGameManager()
        self._start_step = start_step

    @property
    def start_step(self) -> int:
        return self._start_step

    def reset(self) -> None:
        self._manager.reset()

    def agent(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        raw = _get(observation, "step")
        if raw is None:
            day = int(_get(observation, "day", 0) or 0)
            step = day * 24 + int(_get(observation, "hour", 0) or 0)
        else:
            step = int(raw)
        if step < self._start_step:
            if step == 0:
                self._manager.reset()
            return self._opening(observation, configuration)
        return self._manager.agent(observation, configuration)

    __call__ = agent
