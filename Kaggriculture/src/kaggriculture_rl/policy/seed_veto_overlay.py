"""Seed veto overlay: drop seed purchases for configured day and crop pairs.

Evidence: V21 doubles strawberry 19 to 37 on days 10-11 in every game,
outspending yusuke by about 9k over those two days for plants that never
pay back (final margin about -31k). The veto touches market orders only,
so the wrapped unit trace keeps its geometry and the unplantable PLANT
commands degrade to no-ops under the engine atomic planting rule.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

DEFAULT_VETO = frozenset({(10, "STRAWBERRY"), (11, "STRAWBERRY")})


class SeedVetoOverlay:
    """Drop BUY_SEED orders whose day and crop are vetoed."""

    def __init__(
        self,
        policy: PolicyFn,
        veto: set[tuple[int, str]] | frozenset | None = None,
    ) -> None:
        self._policy = policy
        self._veto = set(DEFAULT_VETO) if veto is None else set(veto)
        self._dropped: dict[int, int] = {}

    @property
    def dropped_total(self) -> int:
        return sum(self._dropped.values(), 0)

    def reset(self) -> None:
        self._dropped.clear()

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        step = int(observation.get("step", 0) or 0)
        action = copy.deepcopy(self._policy(observation, configuration))
        day = step // 24
        market = []
        for order in list(action.get("market", []) or []):
            if (
                isinstance(order, list)
                and len(order) > 1
                and order[0] == "BUY_SEED"
                and (day, str(order[1])) in self._veto
            ):
                self._dropped[(day, str(order[1]))] = self._dropped.get((day, str(order[1])), 0) + 1
                continue
            market.append(order)
        action["market"] = market
        return action

    __call__ = agent
