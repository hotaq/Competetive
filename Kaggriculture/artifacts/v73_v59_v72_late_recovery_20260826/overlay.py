"""Fixed-step splice from V59 economy into a fresh V72 closed-loop policy."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from kaggriculture_rl.policy.closed_loop_town_quota import ClosedLoopTownQuotaPolicy
from submissions.meta_v59_causal_route_selector_b120_local import main as v59

PolicyFactory = Callable[[], Any]


def make_agent(
    branch_step: int,
    *,
    late_policy_factory: PolicyFactory = ClosedLoopTownQuotaPolicy,
):
    if branch_step < 0:
        raise ValueError("branch_step must be non-negative")

    late_policy = late_policy_factory()
    last_step_by_player: dict[int, int] = {}

    def agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        player = int(observation.get("player", 0) or 0)
        previous_step = last_step_by_player.get(player, -1)
        if step == 0 or step < previous_step:
            late_policy.reset(player)
        last_step_by_player[player] = step

        if step < branch_step:
            return copy.deepcopy(v59.agent(observation, configuration))
        return copy.deepcopy(late_policy(observation, configuration))

    return agent

