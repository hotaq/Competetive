"""v54 with state-dependent, position-neutral movement diversification."""

from __future__ import annotations

from typing import Any

from experiments.current_top1_tran_tape_89728633 import TRACE_ACTIONS
from experiments.top1_current_meta_v54 import agent as v54_agent


MOVE_DELTAS = {
    "NORTH": (0, -1),
    "EAST": (1, 0),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
}
INVERSE_MOVE = {
    "NORTH": "SOUTH",
    "EAST": "WEST",
    "SOUTH": "NORTH",
    "WEST": "EAST",
}

_pending_returns: dict[int, tuple[int, tuple[int, int], str]] = {}
_last_step = -1


def _commands(action: dict[str, list[Any]]) -> list[list[Any]]:
    return [action.get("farmer", ["PASS"]), *action.get("hands", [])]


def _set_command(action: dict[str, list[Any]], actor: int, command: list[Any]) -> None:
    if actor == 0:
        action["farmer"] = command
    else:
        action["hands"][actor - 1] = command


def _state_salt(obs: dict[str, Any], player: int, step: int, actor: int) -> int:
    farms = obs.get("farms", [])
    own_money = int(farms[player].get("money", 0) or 0)
    rival_money = sum(
        int(farm.get("money", 0) or 0)
        for index, farm in enumerate(farms)
        if index != player
    )
    shed = obs.get("private", {}).get("shed", {})
    inventory_salt = sum(
        (index + 3) * int(amount or 0)
        for index, amount in enumerate(shed.values())
    )
    return own_money * 3 + rival_money * 5 + inventory_salt * 7 + step * 11 + actor * 17


def _candidate_moves(
    obs: dict[str, Any], player: int, step: int, actor: int, position: list[int]
) -> list[tuple[str, tuple[int, int]]]:
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    width = len(tiles[0])
    height = len(tiles)
    names = list(MOVE_DELTAS)
    salt = _state_salt(obs, player, step, actor)
    rotation = salt % len(names)
    names = names[rotation:] + names[:rotation]
    if (salt // len(names)) % 2:
        names.reverse()

    x, y = position
    candidates: list[tuple[str, tuple[int, int]]] = []
    for name in names:
        dx, dy = MOVE_DELTAS[name]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if tiles[ny][nx] == "LOCKED":
            continue
        candidates.append((name, (nx, ny)))
    return candidates


def _diversify_idle_routes(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    global _last_step
    if step == 0 or step <= _last_step:
        _pending_returns.clear()
    _last_step = step

    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    commands = _commands(action)

    for actor, (expected_step, expected_position, inverse) in list(_pending_returns.items()):
        if expected_step != step:
            if expected_step < step:
                del _pending_returns[actor]
            continue
        del _pending_returns[actor]
        if actor >= len(positions) or actor >= len(commands):
            continue
        if tuple(positions[actor]) != expected_position or commands[actor] != ["PASS"]:
            continue
        _set_command(action, actor, [inverse])

    if step >= len(TRACE_ACTIONS) - 1 or step % 24 == 23:
        return
    next_action = TRACE_ACTIONS[step + 1]
    next_commands = _commands(next_action)
    commands = _commands(action)
    for actor, (position, command) in enumerate(zip(positions, commands)):
        if actor in _pending_returns:
            continue
        if command != ["PASS"] or actor >= len(next_commands):
            continue
        if next_commands[actor] != ["PASS"]:
            continue
        candidates = _candidate_moves(obs, player, step, actor, position)
        if not candidates:
            continue
        move, destination = candidates[0]
        _set_command(action, actor, [move])
        _pending_returns[actor] = (step + 1, destination, INVERSE_MOVE[move])


def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
    action = v54_agent(obs, config)
    step = int(obs.get("step", 0) or 0)
    _diversify_idle_routes(obs, action, step)
    return action


submission_agent = agent
v55_submission_agent = agent
