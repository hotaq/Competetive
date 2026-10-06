"""Phase router v1: fixed opening book plus vendored V21 route gates.

Opening steps 0-15 replay the exact 16 step common prefix of Crop Dusta
submission 55929317, the 2026-09-04 top1 snapshot covering 8 games on both
seats. From step 16 the vendored boatlee V21-R1 outer router takes over:
early opponent gate, pre-200 shop sequence lock, step 217 market overlay.

The vendor submodules run in shadow from step 0, so the selected route
state is built from the actual shared prefix observations, never from
future information. That is the implementable version of routing that the
V95 oracle analysis requires.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VENDOR_PATH = ROOT / "artifacts" / "recovered_boatlee_v21_source" / "main.py"
OPENING_PATH = ROOT / "artifacts" / "phased_router_v1_opening.json"

BOOK_SOURCE = {
    "submission_id": 55929317,
    "team": "Crop Dusta",
    "snapshot": "replays/top1_20260904",
    "episodes": [
        105391557,
        105387997,
        105384058,
        105380550,
        105377081,
        105373474,
        105369946,
        105366473,
    ],
    "prefix_steps": 16,
}


def load_vendor(path: Path = VENDOR_PATH) -> ModuleType:
    resolved = Path(path).resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    name = "phased_router_v1_vendor_" + digest
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load vendor module from " + str(resolved))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError(str(resolved) + " does not export a callable agent")
    return module


def load_opening(path: Path = OPENING_PATH) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("opening book must be a non-empty list")
    for action in raw:
        if not isinstance(action, dict):
            raise ValueError("opening book entries must be dicts")
        for key in ("farmer", "hands", "market"):
            if key not in action:
                raise ValueError("opening book entry is missing " + key)
    return copy.deepcopy(raw)


def _seat_of(observation: dict[str, Any]) -> int:
    return 1 if int(observation.get("player", 0) or 0) == 1 else 0


class PhasedRouterV1:
    """Fixed opening book, then vendored V21 phase gates."""

    def __init__(
        self,
        vendor: ModuleType | None = None,
        opening: list[dict[str, Any]] | None = None,
    ) -> None:
        self._vendor = vendor if vendor is not None else load_vendor()
        self._opening = copy.deepcopy(opening) if opening is not None else load_opening()
        self._book_len = len(self._opening)

    @property
    def book_len(self) -> int:
        return self._book_len

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
        return dict(self._vendor._ROUTE_SELECTION[_seat_of({"player": player})])

    def _aligned_book_action(self, step: int, observation: dict[str, Any]) -> dict[str, Any]:
        action = copy.deepcopy(self._opening[step])
        farms = observation.get("farms", [])
        seat = _seat_of(observation)
        hands = []
        if isinstance(farms, list) and len(farms) > seat:
            farm = farms[seat]
            if isinstance(farm, dict):
                hands = farm.get("hands", [])
        expected = len(hands) if isinstance(hands, list) else len(action["hands"])
        current = list(action["hands"])
        if len(current) > expected:
            current = current[:expected]
        while len(current) < expected:
            current.append(["PASS"])
        action["hands"] = current
        return action

    def agent(
        self,
        observation: dict[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        if step < self._book_len:
            self._vendor.agent(observation, configuration)
            return self._aligned_book_action(step, observation)
        return self._vendor.agent(observation, configuration)

    __call__ = agent
