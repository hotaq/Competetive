"""Compact, reusable summaries for competition execution traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.predicates import eval_predicates

SEVERITY_WEIGHTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


@dataclass(frozen=True)
class TraceSummary:
    """High-signal trace information used by search and experiment reporting."""

    cell_hash: str
    user_messages: tuple[str, ...]
    tool_names: tuple[str, ...]
    sources: tuple[str, ...]
    side_effects: tuple[str, ...]
    predicate_names: tuple[str, ...]
    severity_weight: int
    max_severity: int
    successful_tools: int
    failed_tools: int
    sensitive_successes: int
    secret_read: bool
    egress_count: int

    @property
    def has_finding(self) -> bool:
        return bool(self.predicate_names)


def summarize_trace(trace: Mapping[str, Any]) -> TraceSummary:
    """Convert a raw SDK trace into a stable summary.

    The official cell signature remains the source of truth for competition
    diversity. This wrapper adds predicate and operational fields needed by
    search policies without retaining large tool outputs.
    """

    events = list(trace.get("tool_events", ()))
    user_messages = tuple(str(message) for message in trace.get("user_messages", ()))
    predicates = eval_predicates(trace)
    signature = cell_signature(
        events,
        egress_data=trace.get("egress", ()),
        user_messages=user_messages,
    )
    severities = [int(predicate.get("severity", 1)) for predicate in predicates]

    return TraceSummary(
        cell_hash=str(signature["hash"]),
        user_messages=user_messages,
        tool_names=tuple(str(event.get("name", "")) for event in events),
        sources=tuple(str(event.get("source", "")) for event in events),
        side_effects=tuple(str(event.get("side_effect", "")) for event in events),
        predicate_names=tuple(str(predicate["predicate"]) for predicate in predicates),
        severity_weight=sum(SEVERITY_WEIGHTS.get(severity, 1) for severity in severities),
        max_severity=max(severities, default=0),
        successful_tools=sum(event.get("ok") is True for event in events),
        failed_tools=sum(event.get("ok") is not True for event in events),
        sensitive_successes=sum(
            event.get("ok") is True
            and event.get("side_effect") in {"EXEC", "SHARE", "WRITE"}
            for event in events
        ),
        secret_read=bool(signature["secret_read"]),
        egress_count=int(signature["egress_count"]),
    )


def trace_priority(summary: TraceSummary) -> float:
    """Estimate the search value of a trace without replacing replay scoring."""

    return (
        10.0
        + 8.0 * summary.severity_weight
        + 10.0 * int(summary.secret_read)
        + 5.0 * summary.egress_count
        + 2.0 * summary.sensitive_successes
        + 0.05 * len(summary.tool_names)
    )
