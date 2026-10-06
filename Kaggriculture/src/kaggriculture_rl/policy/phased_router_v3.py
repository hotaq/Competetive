"""Phase router v3: scouted commit with opponent-conditioned bundle pick.

The round-robin showed styles make fights: the yusuke bundle leads the
old pool 19/38 while V21 owns the new Sept tapes 7/8, and V21 goes 0/16
against yusuke and kenjo tapes. No single bundle dominates, so v3 picks
the bundle from the opponent opening instead of the seat.

Protocol: step 0 plays a resource-neutral PASS scout while every
route module warms in shadow. At step 1 the opponent hires separate
the families cleanly, because the heavy family genuinely passes turn 1
while every other family opens with 4 to 6 hires, and the router
commits once, then delegates:

- opponent hires == 0: heavy family (yusuke, kenjo, milan signatures).
  Calibration: yusuke bundle 10/24 against them, V21 only 4/24.
- opponent hires >= 4: Sept-dusta, old-dusta and subramanya signatures.
  V21 leads each branch (7/8, 4/8 tie, 6/6) and stays the pick.

Tape bundles replay from their own schedule start at commit time

REJECTED 2026-09-04, two variants:
v3.0 (moon scout steps 0-1, commit step 2) went 11/46: the scout
opening stacks with the picked trace opening (double hires, double
animals) and classification read the wrong frame index.
v3.1 (PASS scout step 0, commit step 1) went 1/46: the idle scout
leaves V21 with zero hands for a trace that hires once at step 0,
and tape paths re-hire on top of the scout. Lesson: any shared
scout breaks some bundle, because openings commit resources on
turn 1 (hires differ 4/5/6 by family) while observation needs
waiting. Oracle headroom for a perfect picker is about +3 games
over the best blind bundle (27/46 vs 24/46), smaller than any
scout costs measured. Opponent-conditioning stays an open problem;
blind commit at step 0 (v2) stands., so a
picked trace stays internally coherent and only starts late, never
spliced mid-schedule. That is the bundle rule learned from v1 failing
0-8 on a spliced foreign opening.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from kaggriculture_rl.policy.phased_router_v1 import load_vendor
from kaggriculture_rl.policy.phased_router_v2 import (
    BUNDLES,
    YUSUKE_SCHEDULES,
    DustaSchedule,
    _seat_of,
    load_schedule,
)

COMMIT_STEP = 1
SCOUT_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def classify_opponent(hires_today: int, money: float) -> str:
    """Map the opponent day-0 opening to a bundle name. Pure function."""
    del money
    if hires_today == 0:
        return "yusuke"
    return "v21"


class PhasedRouterV3:
    """PASS scout on step 0, commit to a bundle at step 1."""

    def __init__(
        self,
        vendor: ModuleType | None = None,
        dusta: DustaSchedule | None = None,
        yusuke: DustaSchedule | None = None,
        commit_step: int = COMMIT_STEP,
    ) -> None:
        self._vendor = vendor if vendor is not None else load_vendor()
        self._dusta = dusta if dusta is not None else DustaSchedule()
        if yusuke is not None:
            self._yusuke = yusuke
        else:
            self._yusuke = DustaSchedule(
                {seat: load_schedule(path) for seat, path in YUSUKE_SCHEDULES.items()}
            )
        self._commit_step = int(commit_step)
        self._committed: dict[int, str] = {}
        self._signals: dict[int, dict[str, Any]] = {}

    def reset(self, player: int | None = None) -> None:
        vendor = self._vendor
        seats = (0, 1) if player is None else (_seat_of({"player": player}),)
        for seat in seats:
            vendor._ROUTE[seat] = None
            vendor._MARKET_OVERLAY[seat] = False
            vendor._PREVIOUS_OPPONENT_MONEY[seat] = None
            vendor._ROUTE_SELECTION[seat] = {"selected": "", "step": None, "signal": None}
            vendor._PENDING_EARLY[seat] = False
            vendor._PENDING_SHOP_ROUTE[seat] = False
            self._committed.pop(seat, None)
            self._signals.pop(seat, None)

    def committed_for(self, player: int) -> str | None:
        return self._committed.get(_seat_of({"player": player}))

    def route_selection_for(self, player: int) -> dict[str, Any]:
        seat = _seat_of({"player": player})
        if seat not in self._committed:
            return {"selected": "", "step": None, "signal": None}
        return dict(self._signals[seat])

    def _commit(self, seat: int, observation: dict[str, Any]) -> str:
        farms = list(observation.get("farms", []) or [])
        opponent = farms[1 - seat] if len(farms) >= 2 else {}
        try:
            hires = int(opponent.get("hires_today", 0) or 0)
        except (TypeError, ValueError):
            hires = 0
        try:
            money = float(opponent.get("money", 3000) or 0)
        except (TypeError, ValueError):
            money = 3000.0
        bundle = classify_opponent(hires, money)
        if bundle not in BUNDLES:
            bundle = "v21"
        step = int(observation.get("step", 0) or 0)
        self._committed[seat] = bundle
        self._signals[seat] = {
            "selected": bundle,
            "step": step,
            "signal": {"hires_today": hires, "money": money},
        }
        return bundle

    def _trace_action(
        self, trace: DustaSchedule, observation: dict[str, Any], seat: int, step: int
    ) -> dict[str, Any]:
        inner = max(0, step - self._commit_step)
        proxy = dict(observation)
        proxy["step"] = inner
        proxy["player"] = seat
        return trace.agent(proxy, None)

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        seat = _seat_of(observation)
        step = int(observation.get("step", 0) or 0)
        vendor_action = self._vendor.agent(observation, configuration)
        if step < self._commit_step:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        bundle = self._committed.get(seat)
        if bundle is None:
            bundle = self._commit(seat, observation)
        if bundle == "dusta":
            return self._trace_action(self._dusta, observation, seat, step)
        if bundle == "yusuke":
            return self._trace_action(self._yusuke, observation, seat, step)
        return vendor_action

    __call__ = agent
