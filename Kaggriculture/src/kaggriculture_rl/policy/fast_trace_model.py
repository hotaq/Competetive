"""Low-overhead adapters for deterministic trace policies used as world models.

These adapters are deliberately for local counterfactual rollouts only.  The
real submitted policy keeps its recovery and legality layers; model rollouts
hand raw proposals to the Rust engine's state-aware sanitizer instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kaggriculture_rl.policy.primitive_legalizer import (
    SEMANTIC_RECOVERABLE_UNIT_OPS,
    WeedRecoveryController,
    _semantic_unit_fallback,
)

Agent = Callable[..., dict[str, Any]]
STAY_ACTION_ID = -1


class _RawStandalonePolicy:
    def __init__(self, policy: Any) -> None:
        namespace = type(policy).__call__.__globals__
        self.checkpoint = policy.checkpoint
        self.traces = policy.traces
        self.feature_steps = policy.feature_steps
        self.common_action = policy._common_action
        self.feature_encoder = namespace["_encode_features"]
        self.select_action = namespace["select_action"]
        self.selected: dict[int, int] = {}
        self.snapshots: dict[int, dict[int, np.ndarray]] = {}
        self.last_step: dict[int, int] = {}

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self.last_step.get(player, -1)
        if step < previous_step:
            self.selected.pop(player, None)
            self.snapshots.pop(player, None)
        self.last_step[player] = step

        if step in self.feature_steps:
            self.snapshots.setdefault(player, {})[step] = self.feature_encoder(
                observation
            )

        if step < self.checkpoint.config.branch_step:
            action_id = self.common_action
        elif player not in self.selected:
            snapshots = self.snapshots.setdefault(player, {})
            current = self.feature_encoder(observation)
            temporal = (
                np.concatenate(
                    [
                        snapshots.get(feature_step, current)
                        for feature_step in self.feature_steps
                    ]
                )
                if self.feature_steps
                else current
            )
            action_id, _, _ = self.select_action(self.checkpoint, temporal)
            self.selected[player] = action_id
        else:
            action_id = self.selected[player]

        trace = self.traces[action_id]
        return trace[min(step, len(trace) - 1)]


class _RawStagePolicy:
    def __init__(self, policy: Any, base_agent: Agent) -> None:
        namespace = type(policy).__call__.__globals__
        self.base_agent = base_agent
        self.checkpoint = policy.checkpoint
        self.branch_step = policy.branch_step
        self.feature_steps = policy.feature_steps
        self.feature_encoder = policy.feature_encoder
        self.traces = policy.traces
        self.exact_routes = policy._exact_routes
        self.exact_tolerance = policy._exact_tolerance
        self.select_stage = namespace["select_stage_two_action"]
        self.select_exact = namespace["select_exact_stage_two_action"]
        self.selected: dict[int, int] = {}
        self.snapshots: dict[int, dict[int, np.ndarray]] = {}
        self.last_step: dict[int, int] = {}

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self.last_step.get(player, -1)
        if step < previous_step or (step == 0 and previous_step != 0):
            self.selected.pop(player, None)
            self.snapshots.pop(player, None)
        self.last_step[player] = step

        selected = self.selected.get(player, STAY_ACTION_ID)
        if step > self.branch_step and selected != STAY_ACTION_ID:
            trace = self.traces[selected]
            return trace[min(step, len(trace) - 1)]

        base_action = self.base_agent(observation, configuration)
        if step in self.feature_steps:
            self.snapshots.setdefault(player, {})[step] = self.feature_encoder(
                observation
            )
        if step < self.branch_step:
            return base_action

        if player not in self.selected:
            snapshots = self.snapshots.setdefault(player, {})
            missing = set(self.feature_steps) - snapshots.keys()
            if missing:
                self.selected[player] = STAY_ACTION_ID
            else:
                features = np.concatenate(
                    [snapshots[value] for value in self.feature_steps]
                )
                if self.exact_routes is None:
                    action_id, _, _ = self.select_stage(
                        self.checkpoint, features
                    )
                else:
                    action_id = self.select_exact(
                        self.checkpoint,
                        features,
                        self.exact_routes,
                        tolerance=self.exact_tolerance,
                    )
                self.selected[player] = action_id

        selected = self.selected[player]
        if selected == STAY_ACTION_ID:
            return base_action
        trace = self.traces[selected]
        return trace[min(step, len(trace) - 1)]


def _adapt_staged_v12(namespace: dict[str, Any]) -> Agent | None:
    """Collapse V12's nested finalizers into one raw selector stack."""
    required = (
        "_STAGE144",
        "_STAGE96_RETRIEVER",
        "_STAGE96_AGGREGATE",
        "champion_agent",
    )
    if any(name not in namespace for name in required):
        return None

    champion_agent = namespace["champion_agent"]
    champion_namespace = getattr(champion_agent, "__globals__", {})
    champion_policy = champion_namespace.get("_POLICY")
    if champion_policy is None:
        return None

    champion_raw = _RawStandalonePolicy(champion_policy)
    aggregate = _RawStagePolicy(
        namespace["_STAGE96_AGGREGATE"], champion_raw
    )
    retriever = _RawStagePolicy(namespace["_STAGE96_RETRIEVER"], aggregate)
    stage144 = _RawStagePolicy(namespace["_STAGE144"], retriever)
    return stage144


def adapt_collapsed_staged_model(agent: Agent) -> tuple[Agent, str]:
    """Remove redundant intermediate finalizers but keep live semantics."""
    namespace = getattr(agent, "__globals__", {})
    if not isinstance(namespace, dict):
        return agent, "live"

    required = (
        "_STAGE144",
        "_STAGE96_RETRIEVER",
        "_STAGE96_AGGREGATE",
        "champion_agent",
        "_finalize",
        "_FINAL_RECOVERY",
    )
    if any(name not in namespace for name in required):
        return agent, "live"

    aggregate = _RawStagePolicy(
        namespace["_STAGE96_AGGREGATE"], namespace["champion_agent"]
    )
    retriever = _RawStagePolicy(namespace["_STAGE96_RETRIEVER"], aggregate)
    stage144 = _RawStagePolicy(namespace["_STAGE144"], retriever)
    finalize = namespace["_finalize"]
    recovery = type(namespace["_FINAL_RECOVERY"])()

    def collapsed_agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        proposed = stage144(observation, configuration)
        return finalize(recovery, observation, proposed, configuration)

    return collapsed_agent, "collapsed-staged-v12"


def _recovery_only_agent(
    proposal_agent: Agent,
    recovery: WeedRecoveryController,
    append_terminal: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> Agent:
    def recovery_agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        proposed = proposal_agent(observation, configuration)
        repaired = recovery.repair(observation, proposed)
        return append_terminal(observation, repaired)

    return recovery_agent


def adapt_recovery_trace_model(agent: Agent) -> tuple[Agent, str]:
    """Keep tape recovery while delegating final legality to the Rust engine."""
    namespace = getattr(agent, "__globals__", {})
    if not isinstance(namespace, dict):
        return agent, "live"

    append_terminal = namespace.get("append_terminal_liquidation")
    recovery_template = namespace.get("_RECOVERY")

    staged_required = (
        "_STAGE144",
        "_STAGE96_RETRIEVER",
        "_STAGE96_AGGREGATE",
        "champion_agent",
        "_FINAL_RECOVERY",
    )
    if append_terminal is not None and all(
        name in namespace for name in staged_required
    ):
        aggregate = _RawStagePolicy(
            namespace["_STAGE96_AGGREGATE"], namespace["champion_agent"]
        )
        retriever = _RawStagePolicy(
            namespace["_STAGE96_RETRIEVER"], aggregate
        )
        stage144 = _RawStagePolicy(namespace["_STAGE144"], retriever)
        recovery = type(namespace["_FINAL_RECOVERY"])()
        return (
            _recovery_only_agent(stage144, recovery, append_terminal),
            "recovery-staged-v12",
        )

    policy = namespace.get("_POLICY")
    if (
        policy is not None
        and append_terminal is not None
        and recovery_template is not None
    ):
        raw_policy = _RawStandalonePolicy(policy)
        recovery = type(recovery_template)()
        return (
            _recovery_only_agent(raw_policy, recovery, append_terminal),
            "recovery-policy",
        )

    trace = namespace.get("_TRACE")
    if (
        isinstance(trace, list)
        and trace
        and append_terminal is not None
        and recovery_template is not None
    ):

        def raw_trace(
            observation: dict[str, Any], configuration: Any = None
        ) -> dict[str, Any]:
            del configuration
            step = max(0, int(observation.get("step", 0) or 0))
            return trace[min(step, len(trace) - 1)]

        recovery = type(recovery_template)()
        adapted = _recovery_only_agent(raw_trace, recovery, append_terminal)
        if "recover_rejected_unit_intents" in namespace:
            adapted._recover_rejected_unit_intents = True
        return adapted, "recovery-trace"

    return agent, "live"


def adapt_full_recovery_trace_model(agent: Agent) -> tuple[Agent, str]:
    """Also remove the staged V12 champion's embedded Python legalizer."""
    namespace = getattr(agent, "__globals__", {})
    if not isinstance(namespace, dict):
        return agent, "live"

    required = (
        "_STAGE144",
        "_STAGE96_RETRIEVER",
        "_STAGE96_AGGREGATE",
        "champion_agent",
        "_FINAL_RECOVERY",
        "append_terminal_liquidation",
    )
    if any(name not in namespace for name in required):
        return adapt_recovery_trace_model(agent)

    champion_agent = namespace["champion_agent"]
    champion_namespace = getattr(champion_agent, "__globals__", {})
    champion_policy = champion_namespace.get("_POLICY")
    champion_recovery = champion_namespace.get("_RECOVERY")
    champion_append = champion_namespace.get("append_terminal_liquidation")
    if (
        champion_policy is None
        or champion_recovery is None
        or champion_append is None
    ):
        return adapt_recovery_trace_model(agent)

    champion_raw = _RawStandalonePolicy(champion_policy)
    champion = _recovery_only_agent(
        champion_raw,
        type(champion_recovery)(),
        champion_append,
    )
    aggregate = _RawStagePolicy(namespace["_STAGE96_AGGREGATE"], champion)
    retriever = _RawStagePolicy(namespace["_STAGE96_RETRIEVER"], aggregate)
    stage144 = _RawStagePolicy(namespace["_STAGE144"], retriever)
    outer = _recovery_only_agent(
        stage144,
        type(namespace["_FINAL_RECOVERY"])(),
        namespace["append_terminal_liquidation"],
    )
    return outer, "full-recovery-staged-v12"


def recover_rejected_unit_intents_for_engine(
    observation: dict[str, Any],
    proposed: dict[str, Any],
    legalized: dict[str, Any],
) -> dict[str, Any]:
    """Apply semantic fallbacks; caller performs the final Rust sanitization."""
    recovered = {
        "farmer": list(legalized.get("farmer", ["PASS"])),
        "hands": [list(command) for command in legalized.get("hands", [])],
        "market": [list(order) for order in legalized.get("market", [])],
    }
    proposed_units = [
        proposed.get("farmer", ["PASS"]),
        *proposed.get("hands", []),
    ]
    legal_units = [
        legalized.get("farmer", ["PASS"]),
        *legalized.get("hands", []),
    ]
    changed = False
    for actor, (before, after) in enumerate(
        zip(proposed_units, legal_units, strict=False)
    ):
        if (
            not isinstance(before, list)
            or not before
            or before[0] == "PASS"
            or before[0] not in SEMANTIC_RECOVERABLE_UNIT_OPS
            or after != ["PASS"]
        ):
            continue
        fallback = _semantic_unit_fallback(observation, actor)
        if fallback != ["PASS"]:
            WeedRecoveryController._set_actor(recovered, actor, fallback)
            changed = True
    return recovered if changed else legalized


def adapt_fast_trace_model(agent: Agent) -> tuple[Agent, str]:
    """Return a raw trace proposal agent plus a diagnostic adapter label."""
    namespace = getattr(agent, "__globals__", {})
    if not isinstance(namespace, dict):
        return agent, "live"

    staged = _adapt_staged_v12(namespace)
    if staged is not None:
        return staged, "raw-staged-v12"

    policy = namespace.get("_POLICY")
    if policy is not None:
        return _RawStandalonePolicy(policy), "raw-policy"

    trace = namespace.get("_TRACE")
    if isinstance(trace, list) and trace:

        def raw_trace(
            observation: dict[str, Any], configuration: Any = None
        ) -> dict[str, Any]:
            del configuration
            step = max(0, int(observation.get("step", 0) or 0))
            return trace[min(step, len(trace) - 1)]

        return raw_trace, "raw-trace"

    # V16 contains state-aware market and repair logic on top of its tape.  It
    # is already cheap, and treating its tape as raw would remove the very
    # adaptation the selector is meant to evaluate.
    return agent, "live"
