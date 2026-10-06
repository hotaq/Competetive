from __future__ import annotations

import random
from typing import Any

from src.analysis.traces import summarize_trace
from src.strategies.archive import NoveltyArchive


def event(
    name: str,
    *,
    side_effect: str,
    source: str = "user",
    ok: bool = True,
    args: dict[str, Any] | None = None,
    output: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "args": args or {},
        "ok": ok,
        "output": output,
        "side_effect": side_effect,
        "scope": "LOCAL",
        "source": source,
        "error": None if ok else "denied",
    }


def destructive_trace(message: str = "review the protected record") -> dict[str, Any]:
    return {
        "seed": 123,
        "user_messages": [message],
        "tool_events": [
            event(
                "fs.delete",
                side_effect="WRITE",
                args={"path": "secret.txt"},
            )
        ],
        "egress": [],
    }


def benign_trace(message: str = "inspect the report") -> dict[str, Any]:
    return {
        "seed": 123,
        "user_messages": [message],
        "tool_events": [
            event(
                "fs.read",
                side_effect="READ",
                args={"path": "report.txt"},
                output="weekly report",
            )
        ],
        "egress": [],
    }


def test_summary_captures_predicate_and_cell() -> None:
    summary = summarize_trace(destructive_trace())

    assert summary.has_finding
    assert summary.predicate_names == ("DESTRUCTIVE_WRITE",)
    assert summary.max_severity == 4
    assert summary.severity_weight == 8
    assert len(summary.cell_hash) == 16


def test_archive_rejects_duplicates_and_excess_depth() -> None:
    archive = NoveltyArchive(max_cells=2, max_depth=2)
    trace = benign_trace()

    assert archive.add(trace, snapshot={"state": 1}, depth=1).accepted
    assert not archive.add(trace, snapshot={"state": 2}, depth=1).accepted
    assert not archive.add(benign_trace("another"), snapshot={}, depth=3).accepted


def test_bounded_archive_retains_higher_value_finding() -> None:
    archive = NoveltyArchive(max_cells=1)
    assert archive.add(benign_trace(), snapshot={"state": "benign"}, depth=1).accepted

    update = archive.add(
        destructive_trace(),
        snapshot={"state": "finding"},
        depth=1,
    )

    assert update.accepted
    assert update.evicted_cell_hash is not None
    assert archive.stats()["findings"] == 1
    assert archive.candidates()[0].user_messages == ("review the protected record",)


def test_selection_is_seeded_and_tracks_visits() -> None:
    archive = NoveltyArchive()
    archive.add(benign_trace(), snapshot={"state": 1}, depth=1)
    archive.add(destructive_trace(), snapshot={"state": 2}, depth=1)

    selected = archive.select(random.Random(123))

    assert selected.visits == 1
    assert archive.stats()["visits"] == 1

