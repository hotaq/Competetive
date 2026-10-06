"""Local V75: V74 scheduling with maintenance-only cash admission."""

from __future__ import annotations

from kaggriculture_rl.policy.closed_loop_town_quota_v75 import (
    ClosedLoopTownQuotaV75Policy,
)
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _v16_agent

_POLICY = ClosedLoopTownQuotaV75Policy(opening_policy=_v16_agent)


def agent(obs, configuration=None):
    return _POLICY(obs, configuration)


__all__ = ["agent"]
