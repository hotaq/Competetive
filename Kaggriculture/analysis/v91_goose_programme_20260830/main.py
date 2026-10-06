"""V67 plus a goose programme run on hands V67 never hired.

V67 ends its games with 48 empty unlocked tiles, no coop and no goose, while EGG is the
only good in the game whose price does not fall with volume — `analysis/v91_depth_matched_producer_20260830/MILESTONE2.md`
measures it at 36 to 42 a unit from 10 a day to 320 a day.

This overlay never touches V67. It hires additional hands beyond the ones V67 asks for,
and those trailing hands, and only those, run coops. V67's farmer and its own hands keep
their exact commands, and V67's market orders are passed through untouched; ours are
appended after them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2] / (
    "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py")

TURNS_PER_DAY = 24
MAX_MARKET_ORDERS = 10
BOARD = 10
SHED_TILES = ([4, 4], [5, 4], [4, 5], [5, 5])

EXTRA_HANDS_PER_DAY = 1      # hired on top of whatever V67 hires
GEESE_TARGET = 10            # sized to spare labour, not to the whole idle board
# Hiring is priced on a Fibonacci curve for the day. V67 already takes 11 to 15 hands a
# day late on, where the next three cost 987, 1,597 and 2,584. Three extra hands every
# day cost 26,982 a game, which was the whole of this candidate's loss. We only take a
# hand while the day's curve is still cheap.
MAX_HIRES_BEFORE_OURS = 10
GOOSE_COST = 300
CASH_FLOOR = 4_000           # never bid against V67's own purchases
HIRE_FLOOR = 800             # hiring is Fibonacci-cheap, so it needs its own floor
START_DAY = 2                # let V67's opening finish first
CARRY_LIMIT = 8              # eggs a hand carries before returning to the shed


def _load_base():
    spec = importlib.util.spec_from_file_location("v91_base_v67", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_base()


class _State:
    def __init__(self) -> None:
        self.reset(-1)

    def reset(self, step: int) -> None:
        self.last_step = step
        self.day = -1
        self.our_hires_today = 0
        # One flag per hand index, in hire order: True where the hand is ours. V67
        # assigns its hand commands positionally, so its k-th command has to land on
        # its k-th hand, not on whatever index our hires pushed it to.
        # Count of HIRE orders V67 itself has issued today. Ownership is then derived
        # from the hand count the observation reports, not from assuming our own hires
        # succeeded: a hire that fails for want of cash used to desynchronise the map
        # for the rest of the day and push V67's commands onto our hands.
        self.v67_hires_today = 0
        # Hands are dismissed at the end of each day and take whatever they carry with
        # them, so an unplaced bird is money burned. Purchases are capped for the whole
        # episode, and only one bird is ever in flight.
        self.geese_bought = 0


_STATE: dict[int, _State] = {}


def _tile(tiles: list[list[Any]], x: int, y: int) -> Any:
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return "LOCKED"


def _step_toward(position: list[int], target: list[int]) -> list[str]:
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _nearest(position: list[int], targets: list[list[int]]) -> list[int] | None:
    if not targets:
        return None
    return min(targets, key=lambda t: abs(t[0] - position[0]) + abs(t[1] - position[1]))


def _scan(tiles: list[list[Any]]) -> dict[str, list[list[int]]]:
    empty: list[list[int]] = []
    ready: list[list[int]] = []
    bare_coop: list[list[int]] = []
    hungry: list[list[int]] = []
    coops = 0
    for y, row in enumerate(tiles):
        for x, cell in enumerate(row):
            if cell is None:
                empty.append([x, y])
            elif isinstance(cell, dict) and cell.get("kind") == "COOP":
                coops += 1
                if cell.get("animal"):
                    if int(cell.get("yield_units", 0)) > 0:
                        ready.append([x, y])
                    if not cell.get("fed_today"):
                        hungry.append([x, y])
                else:
                    bare_coop.append([x, y])
    return {"empty": empty, "ready": ready, "bare_coop": bare_coop,
            "hungry": hungry, "coops": [[coops, 0]]}


def _hand_action(position: list[int], inventory: dict[str, Any],
                 board: dict[str, list[list[int]]], geese_placed: int,
                 shed: dict[str, Any]) -> list[Any]:
    x, y = position
    here = position in [list(t) for t in SHED_TILES]
    carrying_eggs = int(inventory.get("EGG", 0))
    holding_goose = int(inventory.get("GOOSE", 0))
    wheat = int(inventory.get("WHEAT", 0))

    # 1. collect from a coop we are standing on
    if [x, y] in board["ready"]:
        return ["HARVEST"]
    # 2. feed the bird under us if we brought wheat
    if [x, y] in board["hungry"] and wheat > 0:
        return ["FEED"]
    # 3. put a bird into an empty coop we are standing on
    if holding_goose and [x, y] in board["bare_coop"]:
        return ["PLACE", "GOOSE", 1]
    # 4. turn the tile under us into a coop
    if holding_goose and [x, y] in board["empty"]:
        return ["BUILD_COOP"]

    # 5. bank what we are carrying
    if carrying_eggs >= CARRY_LIMIT or (carrying_eggs and not board["ready"]):
        if here:
            return ["DROP"]
        target = _nearest(position, [list(t) for t in SHED_TILES])
        return _step_toward(position, target)

    # 6. restock at the shed, but only for stock that is actually there. A PICKUP of
    #    something the shed does not hold is a silent no-op, and the first version of
    #    this controller spent 444 of its 800 actions repeating one.
    if here:
        if (holding_goose == 0 and geese_placed < GEESE_TARGET
                and int(shed.get("GOOSE", 0)) > 0):
            return ["PICKUP", "GOOSE", 1]
        if wheat < 4 and board["hungry"] and int(shed.get("WHEAT", 0)) > 0:
            return ["PICKUP", "WHEAT", min(4, int(shed.get("WHEAT", 0)))]

    # 7. otherwise go where the work is
    if board["ready"]:
        target = _nearest(position, board["ready"])
    elif holding_goose:
        target = _nearest(position, board["bare_coop"]) or _nearest(position, board["empty"])
    elif int(shed.get("GOOSE", 0)) > 0 and geese_placed < GEESE_TARGET and not here:
        target = _nearest(position, [list(t) for t in SHED_TILES])
    elif board["ready"]:
        target = _nearest(position, board["ready"])
    elif board["hungry"] and wheat > 0:
        target = _nearest(position, board["hungry"])
    else:
        target = _nearest(position, [list(t) for t in SHED_TILES])
    if target is None:
        return ["PASS"]
    return _step_toward(position, target)


def agent(observation: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    action = _BASE_MODULE.agent(observation, configuration)
    player = int(observation.get("player", 0))
    step = int(observation.get("step", 0))
    state = _STATE.setdefault(player, _State())
    if step <= state.last_step:
        state.reset(step)
    state.last_step = step

    day = step // TURNS_PER_DAY
    turn_of_day = step % TURNS_PER_DAY
    if day != state.day:
        state.day = day
        state.our_hires_today = 0
        state.v67_hires_today = 0
    if day < START_DAY:
        return action

    v67_hires = sum(
        1 for order in (action.get("market") or [])
        if isinstance(order, list) and order and order[0] == "HIRE"
    )
    farm = observation["farms"][player]
    private = observation["private"]
    tiles = farm["tiles"]
    board = _scan(tiles)
    coops = board["coops"][0][0]
    money = float(farm["money"])

    hands = farm["hands"]
    base_hands = list(action.get("hands") or [])

    # Our hires are only ever issued on turns V67 does not hire, so ours are the most
    # recent hands and therefore the trailing indices.
    owned = max(0, len(hands) - state.v67_hires_today)
    owned = min(owned, state.our_hires_today)
    ours = list(range(len(hands) - owned, len(hands)))
    theirs = [i for i in range(len(hands)) if i not in set(ours)]

    # V67's commands are positional: its j-th command belongs on its j-th hand.
    remapped: list[Any] = [["PASS"] for _ in hands]
    for slot, index in enumerate(theirs):
        if slot < len(base_hands):
            remapped[index] = base_hands[slot]

    state.v67_hires_today += v67_hires

    orders = list(action.get("market") or [])
    room = MAX_MARKET_ORDERS - len(orders)

    # sell every egg the shed holds; EGG is the one curve that does not punish volume
    eggs = int(private["shed"].get("EGG", 0))
    if eggs > 0 and room > 0:
        orders.append(["SELL", "EGG", eggs])
        room -= 1

    inventories = private.get("inventories") or []
    carried = sum(
        int(inventories[i + 1].get("GOOSE", 0))
        for i in ours
        if i + 1 < len(inventories) and isinstance(inventories[i + 1], dict)
    )
    placed = coops - len(board["bare_coop"])
    in_stock = int(private["shed"].get("GOOSE", 0))
    # Only ever hold one spare bird per hand of ours; 39 bought and 5 placed was the
    # first version's most expensive mistake.
    wanted = GEESE_TARGET - placed - in_stock - carried
    one_in_flight = in_stock > 0 or carried > 0
    if (wanted > 0 and not one_in_flight and state.geese_bought < GEESE_TARGET
            and room > 0 and money > CASH_FLOOR + GOOSE_COST and ours):
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        state.geese_bought += 1
        room -= 1

    # hire our own hands, after V67's orders so they land at the end of the list
    if (state.our_hires_today < EXTRA_HANDS_PER_DAY and room > 0 and money > HIRE_FLOOR
            and v67_hires == 0 and turn_of_day >= 6
            and int(farm.get("hires_today", 0)) < MAX_HIRES_BEFORE_OURS):
        orders.append(["HIRE"])
        state.our_hires_today += 1
        room -= 1

    if not ours:
        return {**action, "hands": remapped, "market": orders}

    for index in ours:
        inventory = inventories[index + 1] if index + 1 < len(inventories) else {}
        if not isinstance(inventory, dict):
            inventory = {}
        remapped[index] = _hand_action(
            list(hands[index]), inventory, board, placed, private["shed"])

    base_hands = remapped
    return {**action, "hands": base_hands, "market": orders}
