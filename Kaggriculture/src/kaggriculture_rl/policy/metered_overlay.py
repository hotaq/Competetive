"""Metered selling overlay: cap one-day bulk dumps, carry backlog forward.

Evidence: in the yusuke ep102574290 race, V21 sells 145 fertilizer plus
30 wheat on day 10 to finance land, while yusuke sells 8 fertilizer and
3 wheat that day and meters the melon liquidation across days 10-11.
Same goods, worse average prices plus starved later cash flow.

The overlay trims SELL quantities above per-item daily caps and replays
the remainder on following days through free market slots only. It never
touches unit actions and never reorders the wrapped policy own orders.

REJECTED 2026-09-04: V21 plus this overlay went 0/24,
median -52k against the yusuke and kenjo tape pools, far worse than
unmodified V21. The day-10 bulk sale it suppresses is load-bearing
financing. Kept as a negative result. Do not ship without a new design.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

DEFAULT_DAILY_CAPS = {
    "FERTILIZER": 15,
    "WHEAT": 12,
}


def _seat_of(observation: Mapping[str, Any]) -> int:
    return 1 if int(observation.get("player", 0) or 0) == 1 else 0


class MeteredSellingOverlay:
    """Wrap a raw-action policy with per-day per-item sell caps."""

    def __init__(
        self,
        policy: PolicyFn,
        daily_caps: Mapping[str, int] | None = None,
    ) -> None:
        self._policy = policy
        self._caps = dict(DEFAULT_DAILY_CAPS)
        if daily_caps is not None:
            self._caps.update(daily_caps)
        self._sold: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._backlog: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_step: dict[int, int] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._sold.clear()
            self._backlog.clear()
            self._last_step.clear()
            return
        seat = _seat_of({"player": player})
        self._sold.pop(seat, None)
        self._backlog.pop(seat, None)
        self._last_step.pop(seat, None)

    @property
    def backlog_for(self) -> dict[int, dict[str, int]]:
        return {seat: dict(items) for seat, items in self._backlog.items()}

    def _trim_market(
        self, seat: int, day: int, market: list[Any]
    ) -> tuple[list[Any], dict[str, int]]:
        trimmed: list[Any] = []
        deferred: dict[str, int] = defaultdict(int)
        for order in market:
            if not isinstance(order, list) or len(order) < 3 or order[0] != "SELL":
                trimmed.append(order)
                continue
            item = str(order[1])
            quantity = int(order[2] or 0)
            cap = self._caps.get(item)
            if cap is None or quantity <= 0:
                trimmed.append(order)
                continue
            room = max(0, cap - self._sold[seat].get(item, 0))
            kept = min(quantity, room)
            if kept > 0:
                trimmed.append(["SELL", item, kept])
                self._sold[seat][item] = self._sold[seat].get(item, 0) + kept
            if quantity > kept:
                deferred[item] += quantity - kept
        return trimmed, deferred

    def _replay_backlog(self, seat: int, market: list[Any]) -> list[Any]:
        pending = self._backlog.get(seat)
        if not pending:
            return market
        replayed: dict[str, int] = {}
        for item in sorted(pending):
            if len(market) >= 10:
                break
            cap = self._caps.get(item)
            if cap is None:
                continue
            room = max(0, cap - self._sold[seat].get(item, 0))
            quantity = min(int(pending[item]), room)
            if quantity > 0:
                market.append(["SELL", item, quantity])
                self._sold[seat][item] = self._sold[seat].get(item, 0) + quantity
                replayed[item] = quantity
        for item, quantity in replayed.items():
            pending[item] -= quantity
            if pending[item] <= 0:
                del pending[item]
        return market

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        seat = _seat_of(observation)
        step = int(observation.get("step", 0) or 0)
        day = step // 24
        if self._last_step.get(seat) != day:
            self._sold[seat].clear()
            self._last_step[seat] = day
        action = copy.deepcopy(self._policy(observation, configuration))
        market = list(action.get("market", []) or [])
        trimmed, deferred = self._trim_market(seat, day, market)
        for item, quantity in deferred.items():
            self._backlog[seat][item] += quantity
        action["market"] = self._replay_backlog(seat, trimmed)
        return action

    __call__ = agent
