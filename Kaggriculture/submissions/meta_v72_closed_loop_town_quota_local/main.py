"""Local V72 candidate: pinned V16 opening then closed-loop town quotas.

This candidate intentionally depends on the local Python package and native
Rust binding.  It is an evaluation artifact, not an upload-ready submission.
"""

from __future__ import annotations

from kaggriculture_rl.policy.closed_loop_town_quota import ClosedLoopTownQuotaPolicy
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _v16_agent

_POLICY = ClosedLoopTownQuotaPolicy(opening_policy=_v16_agent)


def agent(obs, configuration=None):
    return _POLICY(obs, configuration)


__all__ = ["agent"]
