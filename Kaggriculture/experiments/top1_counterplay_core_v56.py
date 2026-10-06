"""Shared v56 counterplay repairs for public mirror-tape opponents."""

from __future__ import annotations

from typing import Any, Callable

from experiments.top1_route_shuffle_v55 import agent as v55_agent


EARLY_PASTURE_REPAIRS = {
    # step: (actor index including farmer, expected coordinate)
    125: (4, (3, 3)),
    126: (5, (2, 4)),
}
LIQUIDATION_START = {
    "MILK": 12,
    "MELON": 15,
    "WOOL": 15,
    "STRAWBERRY": 17,
}
LIQUIDATION_FLOOR = {
    "MILK": 70,
    "MELON": 190,
    "WOOL": 80,
    "STRAWBERRY": 180,
}
_delayed_commands: dict[int, list[Any]] = {}
_last_repair_step = -1
repair_events: list[tuple[int, int, str, tuple[int, int]]] = []


def _set_actor(action: dict[str, list[Any]], actor: int, command: list[Any]) -> None:
    if actor == 0:
        action["farmer"] = command
    else:
        action["hands"][actor - 1] = command


def repair_early_pasture(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    global _last_repair_step
    if step == 0 or step <= _last_repair_step or step % 24 == 0:
        _delayed_commands.clear()
        if step == 0:
            repair_events.clear()
    _last_repair_step = step

    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for actor, pending in list(_delayed_commands.items()):
        if actor >= len(commands):
            _delayed_commands.pop(actor, None)
            continue
        current = commands[actor]
        _set_actor(action, actor, pending)
        if current == ["PASS"]:
            _delayed_commands.pop(actor, None)
        else:
            _delayed_commands[actor] = current

    repair = EARLY_PASTURE_REPAIRS.get(step)
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    if repair is not None:
        actor, expected = repair
        if (
            actor not in _delayed_commands
            and actor < len(positions)
            and actor < len(commands)
        ):
            x, y = positions[actor]
            tile = farm["tiles"][y][x]
            if (
                (x, y) == expected
                and commands[actor] == ["BUILD_PASTURE"]
                and isinstance(tile, dict)
                and tile.get("kind") == "WEED"
            ):
                _set_actor(action, actor, ["DIG"])
                _delayed_commands[actor] = ["BUILD_PASTURE"]
                repair_events.append((step, actor, "BUILD_PASTURE", (x, y)))

    # Random weed growth can invalidate any later plant/build command.  Repair
    # it with the same-worker delayed pipeline while enough day remains to
    # encounter a PASS and resynchronize with the base tape.
    hour = step % 24
    if hour > 21 or (step < 600 and hour < 20):
        return
    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for actor, ((x, y), command) in enumerate(zip(positions, commands)):
        if actor in _delayed_commands or not command:
            continue
        if command[0] not in {"PLANT", "BUILD_PASTURE"}:
            continue
        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        _set_actor(action, actor, ["DIG"])
        _delayed_commands[actor] = command
        repair_events.append((step, actor, str(command[0]), (x, y)))


def append_liquidation(
    obs: dict[str, Any],
    action: dict[str, list[Any]],
    day: int,
    mode: str,
    *,
    lead_gap: float = 250,
    floor_scale: float = 1.0,
    front_run: dict[str, float] | None = None,
) -> None:
    shed = obs.get("private", {}).get("shed", {})
    market = action.setdefault("market", [])
    scheduled: dict[str, int] = {}
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            scheduled[order[1]] = scheduled.get(order[1], 0) + int(order[2])
    for product, start_day in LIQUIDATION_START.items():
        if day < start_day or len(market) >= 10:
            continue
        price = int(obs.get("market", {}).get("prices", {}).get(product, 0) or 0)
        front_run_now = bool(
            front_run
            and product in front_run
            and price >= front_run[product]
        )
        if mode in {"floor", "behind_floor"}:
            if price < LIQUIDATION_FLOOR[product] * floor_scale:
                continue
        if mode == "behind_floor":
            player = int(obs.get("player", 0) or 0)
            farms = obs.get("farms", [])
            own_money = float(farms[player].get("money", 0) or 0)
            rival_money = max(
                (float(farm.get("money", 0) or 0) for index, farm in enumerate(farms) if index != player),
                default=own_money,
            )
            if not front_run_now and rival_money - own_money < lead_gap:
                continue
        available = int(shed.get(product, 0) or 0) - scheduled.get(product, 0)
        if available > 0:
            market.append(["SELL", product, available])


def reorder_market(
    obs: dict[str, Any], action: dict[str, list[Any]], strategy: str
) -> None:
    market = action.setdefault("market", [])
    if strategy == "none" or len(market) < 2:
        return
    if strategy == "reverse":
        market.reverse()
        return

    priorities = {
        "SELL": 0,
        "HIRE": 1,
        "BUY_SEED": 2,
        "BUY_PRODUCT": 3,
        "BUY_ANIMAL": 4,
        "BUY_LAND": 5,
    }
    if strategy == "stable_sell_first":
        market.sort(key=lambda order: priorities.get(order[0], 9))
        return
    if strategy == "value_sell_first":
        prices = obs.get("market", {}).get("prices", {})

        def key(order: list[Any]) -> tuple[float, float]:
            priority = priorities.get(order[0], 9)
            if order[0] != "SELL" or len(order) < 3:
                return float(priority), 0.0
            value = float(prices.get(order[1], 0)) * int(order[2])
            return float(priority), -value

        market.sort(key=key)


def make_agent(
    strategy: str,
    liquidate: bool | str,
    *,
    lead_gap: float = 250,
    floor_scale: float = 1.0,
    front_run: dict[str, float] | None = None,
    activation_gap: float | None = None,
    activation_day: int = 0,
    activation_deadband: tuple[float, float] | None = None,
) -> Callable[..., dict[str, list[Any]]]:
    counterplay_active = activation_gap is None
    counterplay_disabled = False
    activation_regime_checked = False

    def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
        nonlocal counterplay_active, counterplay_disabled, activation_regime_checked
        action = v55_agent(obs, config)
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", step // 24) or 0)
        if step == 0:
            counterplay_active = activation_gap is None
            counterplay_disabled = False
            activation_regime_checked = False
        repair_early_pasture(obs, action, step)
        if not counterplay_active and not counterplay_disabled and day >= activation_day:
            player = int(obs.get("player", 0) or 0)
            farms = obs.get("farms", [])
            own_money = float(farms[player].get("money", 0) or 0)
            rival_money = max(
                (
                    float(farm.get("money", 0) or 0)
                    for index, farm in enumerate(farms)
                    if index != player
                ),
                default=own_money,
            )
            rival_lead = rival_money - own_money
            if not activation_regime_checked:
                activation_regime_checked = True
                if (
                    activation_deadband is not None
                    and activation_deadband[0] <= rival_lead <= activation_deadband[1]
                ):
                    counterplay_disabled = True
            if not counterplay_disabled:
                counterplay_active = rival_lead >= float(activation_gap or 0)
        if counterplay_active and liquidate:
            mode = "aggressive" if liquidate is True else str(liquidate)
            append_liquidation(
                obs,
                action,
                day,
                mode,
                lead_gap=lead_gap,
                floor_scale=floor_scale,
                front_run=front_run,
            )
        if counterplay_active:
            reorder_market(obs, action, strategy)
        return action

    return agent
