"""Phase router v2: opening-route bundles committed at step 0.

Lesson from v1, which went 0-8: a route schedule is entangled with the
opening actually played. V21 trace modules assume the boatlee opening from
step 0, so replaying the 16 step Crop Dusta prefix underneath them
collapsed the economy. Bundles therefore commit at step 0, the first step
where the two family openings diverge.

Bundles:
- v21: the vendored boatlee V21-R1 router intact, with its own opening.
- dusta: the full 719 step Crop Dusta schedule per seat, taken from the
  newest tape of each seat in the 2026-09-04 top1 snapshot. One coherent
  trace per seat, never a voted chimera.
- yusuke: the full 719 step Yusuke Hayashi schedule per seat, taken from
  the best realized margin tape of each seat in the 20260829 pool
  (ep102574290 seat 0 at +42785, ep102589974 seat 1 at +3739). Schedule
  entries follow the pool tape convention also used by the harness:
  entry k replays the recorded action the harness would play at that step.

Bundle selection is by seat, the only information available at step 0.
Calibration 2026-09-04: 38-game diverse top-5 pool plus 8-game
Sept top1 tapes, 46 games per bundle.
Totals: yusuke 24, v21 21, dusta 5. Yusuke leads the old pool 19/38
and splits the new tapes 5/8; v21 owns the new tapes 7/8 but goes
0/16 against yusuke and kenjo tapes. Styles make fights: no single
bundle dominates every pool.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from kaggriculture_rl.policy.phased_router_v1 import load_vendor

ROOT = Path(__file__).resolve().parents[3]
DUSTA_SCHEDULES = {
    0: ROOT / "artifacts" / "dusta_schedule_s0.jsonl",
    1: ROOT / "artifacts" / "dusta_schedule_s1.jsonl",
}
YUSUKE_SCHEDULES = {
    0: ROOT / "artifacts" / "yusuke_schedule_s0.jsonl",
    1: ROOT / "artifacts" / "yusuke_schedule_s1.jsonl",
}

BUNDLE_SOURCE = {
    "dusta_s0_episode": 105391557,
    "dusta_s1_episode": 105366473,
    "yusuke_s0_episode": 102574290,
    "yusuke_s1_episode": 102589974,
    "snapshot": "replays/top1_20260904",
    "pool": "artifacts/opponent_tapes_20260829",
}

BUNDLES = ("v21", "dusta", "yusuke")

DEFAULT_BUNDLE_BY_SEAT = {0: "yusuke", 1: "yusuke"}

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _seat_of(observation: dict[str, Any]) -> int:
    return 1 if int(observation.get("player", 0) or 0) == 1 else 0


def load_schedule(path: Path) -> list[dict[str, Any]]:
    actions = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            actions.append(json.loads(line))
    if len(actions) not in (719, 720):
        raise ValueError("schedule must hold 719 or 720 steps, got " + str(len(actions)))
    return actions


class DustaSchedule:
    """Coherent per-seat trace reused as a route."""

    def __init__(self, schedules: dict[int, list[dict[str, Any]]] | None = None) -> None:
        if schedules is None:
            schedules = {seat: load_schedule(path) for seat, path in DUSTA_SCHEDULES.items()}
        self._schedules = {seat: copy.deepcopy(trace) for seat, trace in schedules.items()}

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        seat = _seat_of(observation)
        step = int(observation.get("step", 0) or 0)
        trace = self._schedules[seat]
        if step < 0 or step >= len(trace):
            return copy.deepcopy(PASS_ACTION)
        return copy.deepcopy(trace[step])

    __call__ = agent


class PhasedRouterV2:
    """Seat-committed bundle router. V21 keeps its own opening."""

    def __init__(
        self,
        bundle_by_seat: dict[int, str] | None = None,
        vendor: ModuleType | None = None,
        dusta: DustaSchedule | None = None,
        yusuke: DustaSchedule | None = None,
    ) -> None:
        self._bundles = dict(DEFAULT_BUNDLE_BY_SEAT)
        if bundle_by_seat is not None:
            for seat, name in bundle_by_seat.items():
                if name not in BUNDLES:
                    raise ValueError("unknown bundle " + str(name))
                self._bundles[int(seat)] = name
        self._vendor = vendor if vendor is not None else load_vendor()
        self._dusta = dusta if dusta is not None else DustaSchedule()
        if yusuke is not None:
            self._yusuke = yusuke
        else:
            self._yusuke = DustaSchedule(
                {seat: load_schedule(path) for seat, path in YUSUKE_SCHEDULES.items()}
            )

    def bundle_for(self, player: int) -> str:
        return self._bundles[_seat_of({"player": player})]

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

    def route_selection_for(self, player: int) -> dict[str, Any]:
        seat = _seat_of({"player": player})
        if self._bundles[seat] == "dusta":
            return {"selected": "dusta", "step": 0, "signal": BUNDLE_SOURCE}
        if self._bundles[seat] == "yusuke":
            return {"selected": "yusuke", "step": 0, "signal": BUNDLE_SOURCE}
        return dict(self._vendor._ROUTE_SELECTION[seat])

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        bundle = self._bundles[_seat_of(observation)]
        if bundle == "dusta":
            return self._dusta.agent(observation, configuration)
        if bundle == "yusuke":
            return self._yusuke.agent(observation, configuration)
        return self._vendor.agent(observation, configuration)

    __call__ = agent
