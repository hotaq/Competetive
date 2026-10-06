"""Idle-hand overlay: turn a unit's PASS into free upkeep on the tile it is standing on.

Evidence (analysis/v97, live losses 2026-09-05): in days 18-30 our tapes issue ~240 hand PASSes
per game while the opponents that beat us out-produce us in milk and wheat with the same number
of tiles and animals. The engine gives yield for CARE (pending care bonus on the next animal
yield), WATER (non-ongoing crops gain +1/+2 per watered day inside their window; any plant that
misses two waterings becomes a weed) and COLLECT_FERTILIZER, none of which need a resource in
the unit's inventory. FEED needs wheat in that hand, so it is left to the tape.

The overlay never moves a unit and never plants, harvests, builds, digs or places, so tile
occupancy - and therefore the town's shop draw - is exactly what the wrapped tape produces.
It only ever replaces a literal PASS, so a game where the tape never idles is unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

PolicyFn = Callable[[Any, Any], dict[str, Any]]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    return getter(key, default) if callable(getter) else getattr(value, key, default)


def upkeep_for_tile(tile: Any) -> list[str] | None:
    """The best resource-free action for a unit idling on ``tile``, or None."""
    if not isinstance(tile, dict) or not tile:
        return None
    if tile.get("animal"):
        if not tile.get("cared_today"):
            return ["CARE"]
        if tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
        return None
    if tile.get("kind") == "PLANT" and not tile.get("watered_today"):
        return ["WATER"]
    return None


def fill_idle_units(
    action: dict[str, Any], observation: Any, *, start_step: int = 0,
) -> dict[str, Any]:
    step = int(_get(observation, "step", 0) or 0)
    if step < start_step:
        return action
    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    if player >= len(farms):
        return action
    farm = farms[player]
    tiles = _get(farm, "tiles", []) or []
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    commands = [action.get("farmer", ["PASS"]), *list(action.get("hands", []))]
    changed = False
    for index, (position, command) in enumerate(zip(positions, commands, strict=False)):
        if command != ["PASS"] or not position:
            continue
        x, y = int(position[0]), int(position[1])
        if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        replacement = upkeep_for_tile(tiles[y][x])
        if replacement is None:
            continue
        if not changed:
            action = copy.deepcopy(action)
            changed = True
        if index == 0:
            action["farmer"] = replacement
        else:
            action["hands"][index - 1] = replacement
    return action


class IdleHandOverlay:
    """Wrap any ``agent(observation, configuration) -> action`` policy."""

    def __init__(self, inner: PolicyFn, *, start_step: int = 0) -> None:
        self._inner = inner
        self._start_step = start_step

    def agent(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        action = self._inner(observation, configuration)
        return fill_idle_units(action, observation, start_step=self._start_step)

    __call__ = agent
