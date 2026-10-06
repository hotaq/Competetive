"""VT1 tail trim overlay (v112): delete tail labour on days 28-29.

Source: `analysis/v105_tomato_20260922/MORE_EXPERIMENTS.md` §C1 — the
cheapest screen in the ranked list (notebook +3/-0, **S0** Cell 18).
Pool-consistent: day-29 hires dip to 9.82/day (lowest since day 7) while
omar holds 11 hires on days 28-29 (taper gap).

Rules (day = step // 24, 24 turns/day, steps 0..718):
- Day 28 (steps 672..695): CARE banked after the last production is
  worthless, so any unit CARE standing on a SHEEP tile becomes PASS.
  CARE elsewhere (cow/goose tiles) is untouched — fail closed. The
  "expansion" qualifier in C1 is not observable per-tile (tile dicts carry
  no quadrant tag), so the overlay gates on animal == "SHEEP", which covers
  the sheep program including any annex/expansion pasture.
- Day 29 (steps 696..718): hire no hand unless wool waits on tiles (a
  SHEEP tile with yield_units > 0), then keep exactly one HIRE ("one
  harvest-only hand" = the single kept hire; the overlay never rewrites
  unit ops into HARVEST — it only deletes). Without wool waiting, all
  HIRE orders are dropped. All FEED unit ops become PASS (no day-29 feed).
- Every other day: action returned untouched (identity).

Labour-negative by construction: the overlay only replaces unit ops with
PASS and only removes HIRE market orders. It never appends, reorders, or
otherwise touches the market list, and never issues BUY/HIRE/plant/water
ops. Invalid-action risk is nil: PASS and dropped orders are always legal.

Wiring: HIRE/FEED/CARE reads are observation-dependent, so like
`carrot_swap` the screen bakes per-game tapes closed-loop
(`scripts/bake_vt1_tail_trim_tapes.py`) and plays them with `play_batch`.
This module stays dependency-free and seat-agnostic.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

TURNS_PER_DAY = 24
# Engine config {"episodeSteps": 720}; tapes hold 719 actions (turns 0..718,
# LAST_ACT_STEP 718), so day 29 ends at step 718, not 719.
DAY28 = 28
DAY29 = 29
DAY28_LO = DAY28 * TURNS_PER_DAY  # 672
DAY28_HI = DAY29 * TURNS_PER_DAY  # 696 (exclusive)
DAY29_HI = 719  # exclusive upper bound covering LAST_ACT_STEP 718


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def observation_step(observation: Any) -> int:
    """Step index, falling back to day/hour (mirrors `opening_override`)."""
    raw = _get(observation, "step")
    if raw is None:
        step = int(_get(observation, "day", 0) or 0) * TURNS_PER_DAY + int(
            _get(observation, "hour", 0) or 0
        )
    else:
        step = int(raw or 0)
    return max(step, 0)


def observation_day(observation: Any) -> int:
    """Day index derived from step (mirrors `carrot_swap`)."""
    raw = _get(observation, "day")
    if raw is None:
        raw = observation_step(observation) // TURNS_PER_DAY
    return max(int(raw or 0), 0)


def _seat_farm(observation: Any) -> Any:
    player = int(_get(observation, "player", 0) or 0)
    farms = _get(observation, "farms", []) or []
    seat = 1 if player == 1 else 0
    if 0 <= seat < len(farms):
        return farms[seat]
    return {}


def _tile_at(tiles: Any, x: int, y: int) -> Any:
    if not isinstance(tiles, list) or not 0 <= y < len(tiles):
        return None
    row = tiles[y]
    if not isinstance(row, list) or not 0 <= x < len(row):
        return None
    tile = row[x]
    return tile if isinstance(tile, Mapping) else None


def _is_sheep_tile(tile: Any) -> bool:
    return isinstance(tile, Mapping) and tile.get("animal") == "SHEEP"


def wool_waits(observation: Any) -> bool:
    """True when a SHEEP tile on our farm holds harvestable wool."""
    farm = _seat_farm(observation)
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    if not isinstance(tiles, list):
        return False
    for row in tiles:
        if not isinstance(row, list):
            continue
        for tile in row:
            if _is_sheep_tile(tile) and int(tile.get("yield_units", 0) or 0) > 0:
                return True
    return False


def _unit_positions(farm: Any) -> list[tuple[int, int] | None]:
    """[farmer, *hands] positions; None where unknown (fail closed)."""

    def _pos(value: Any) -> tuple[int, int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        return None

    positions: list[tuple[int, int] | None] = []
    positions.append(_pos(farm.get("farmer") if isinstance(farm, Mapping) else None))
    hands = farm.get("hands", []) if isinstance(farm, Mapping) else []
    if isinstance(hands, list):
        for hand in hands:
            positions.append(_pos(hand))
    return positions


def _apply_with_stats(observation: Any, action: Action) -> tuple[Action, dict[str, int]]:
    """Apply the overlay; also return deletion counters.

    Counters: cares_skipped (day-28 sheep CARE->PASS), hires_dropped,
    hires_kept (day-29 HIRE cap), feeds_skipped (day-29 FEED->PASS).
    """
    stats = {"cares_skipped": 0, "hires_dropped": 0, "hires_kept": 0, "feeds_skipped": 0}
    day = observation_day(observation)
    if day not in (DAY28, DAY29):
        return action, stats
    farm = _seat_farm(observation)
    tiles = farm.get("tiles", []) if isinstance(farm, Mapping) else []
    positions = _unit_positions(farm)

    updated = copy.deepcopy(action)
    unit_actions: list[Any] = [
        updated.get("farmer", ["PASS"]),
        *list(updated.get("hands", []) or []),
    ]
    while len(unit_actions) < len(positions):
        unit_actions.append(["PASS"])

    def _set_unit(index: int, op: list[Any]) -> None:
        if index == 0:
            updated["farmer"] = op
        else:
            hand_index = index - 1
            hands = list(updated.get("hands", []) or [])
            if hand_index >= len(hands):
                return  # never extend the crew: fail closed
            hands[hand_index] = op
            updated["hands"] = hands

    def _on_sheep_tile(index: int) -> bool:
        if index >= len(positions):
            return False
        pos = positions[index]
        if pos is None:
            return False
        return _is_sheep_tile(_tile_at(tiles, pos[0], pos[1]))

    if day == DAY28:
        for i, op in enumerate(unit_actions):
            if isinstance(op, list) and op[:1] == ["CARE"] and _on_sheep_tile(i):
                _set_unit(i, ["PASS"])
                unit_actions[i] = ["PASS"]
                stats["cares_skipped"] += 1
        return updated, stats

    # Day 29: HIRE gate + no feed.
    market = list(updated.get("market", []) or [])
    hires = [o for o in market if isinstance(o, list) and o[:1] == ["HIRE"]]
    if wool_waits(observation):
        keep = 1 if hires else 0
        kept = 0
        trimmed: list[Any] = []
        for order in market:
            if isinstance(order, list) and order[:1] == ["HIRE"]:
                if kept < keep:
                    trimmed.append(order)
                    kept += 1
                else:
                    stats["hires_dropped"] += 1
            else:
                trimmed.append(order)
        stats["hires_kept"] = kept
        updated["market"] = trimmed
    else:
        updated["market"] = [
            o
            for o in market
            if not (isinstance(o, list) and o[:1] == ["HIRE"])
        ]
        stats["hires_dropped"] = len(hires)
    for i, op in enumerate(unit_actions):
        if isinstance(op, list) and op[:1] == ["FEED"]:
            _set_unit(i, ["PASS"])
            unit_actions[i] = ["PASS"]
            stats["feeds_skipped"] += 1
    return updated, stats


def apply(observation: Any, action: Action) -> Action:
    """Return `action` with the VT1 tail trim applied when day-gated."""
    updated, _ = _apply_with_stats(observation, action)
    return updated


class Vt1TailTrimOverlay:
    """Wrap a raw-action policy with the VT1 day-28/29 tail trim."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy
        self._stats = {
            "cares_skipped": 0,
            "hires_dropped": 0,
            "hires_kept": 0,
            "feeds_skipped": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def reset(self) -> None:
        for key in self._stats:
            self._stats[key] = 0

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        updated, stats = _apply_with_stats(observation, action)
        for key, value in stats.items():
            self._stats[key] += value
        return updated

    __call__ = agent
