"""Bounded novelty archive for trace-guided attack exploration."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from aicomp_sdk.attacks import AttackCandidate

from src.analysis.traces import TraceSummary, summarize_trace, trace_priority


@dataclass
class ArchiveEntry:
    """A compact frontier entry with an optional restorable environment state."""

    summary: TraceSummary
    snapshot: Any
    depth: int
    priority: float
    visits: int = 0


@dataclass(frozen=True)
class ArchiveUpdate:
    """Result of attempting to add a trace to the archive."""

    accepted: bool
    novel: bool
    evicted_cell_hash: str | None = None


class NoveltyArchive:
    """Bounded cell archive with deterministic weighted frontier selection."""

    def __init__(self, *, max_cells: int = 5_000, max_depth: int = 20) -> None:
        if max_cells <= 0:
            raise ValueError("max_cells must be positive")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.max_cells = max_cells
        self.max_depth = max_depth
        self._entries: dict[str, ArchiveEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, cell_hash: str) -> ArchiveEntry | None:
        return self._entries.get(cell_hash)

    def entry_for_snapshot(self, snapshot: Any) -> ArchiveEntry | None:
        """Return the entry holding ``snapshot`` by object identity."""

        return next(
            (
                entry
                for entry in self._entries.values()
                if entry.snapshot is snapshot
            ),
            None,
        )

    def add(
        self,
        trace: Mapping[str, Any],
        *,
        snapshot: Any,
        depth: int,
    ) -> ArchiveUpdate:
        """Add a novel cell if it is within depth and memory bounds."""

        if depth <= 0 or depth > self.max_depth:
            return ArchiveUpdate(accepted=False, novel=False)

        summary = summarize_trace(trace)
        if summary.cell_hash in self._entries:
            return ArchiveUpdate(accepted=False, novel=False)

        priority = trace_priority(summary)
        evicted: str | None = None
        if len(self._entries) >= self.max_cells:
            worst = min(
                self._entries.values(),
                key=lambda entry: (
                    entry.priority,
                    -entry.visits,
                    -entry.depth,
                    entry.summary.cell_hash,
                ),
            )
            if priority <= worst.priority:
                return ArchiveUpdate(accepted=False, novel=True)
            evicted = worst.summary.cell_hash
            del self._entries[evicted]

        self._entries[summary.cell_hash] = ArchiveEntry(
            summary=summary,
            snapshot=snapshot,
            depth=depth,
            priority=priority,
        )
        return ArchiveUpdate(accepted=True, novel=True, evicted_cell_hash=evicted)

    def select(self, rng: random.Random) -> ArchiveEntry:
        """Select a frontier, favoring valuable and under-visited cells."""

        if not self._entries:
            raise LookupError("cannot select from an empty archive")

        candidates = sorted(
            self._entries.values(),
            key=lambda entry: entry.summary.cell_hash,
        )
        weights = [
            max(0.01, entry.priority) / (1.0 + entry.visits)
            for entry in candidates
        ]
        selected = rng.choices(candidates, weights=weights, k=1)[0]
        selected.visits += 1
        return selected

    def candidates(self, *, limit: int = 2_000) -> list[AttackCandidate]:
        """Return unique replay candidates ordered by expected value."""

        if limit <= 0:
            return []

        findings = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.summary.has_finding and entry.summary.user_messages
            ),
            key=lambda entry: (
                -entry.priority,
                len(entry.summary.user_messages),
                entry.summary.cell_hash,
            ),
        )
        seen_messages: set[tuple[str, ...]] = set()
        candidates: list[AttackCandidate] = []
        for entry in findings:
            messages = entry.summary.user_messages
            if messages in seen_messages:
                continue
            seen_messages.add(messages)
            candidates.append(AttackCandidate.from_messages(messages))
            if len(candidates) >= limit:
                break
        return candidates

    def stats(self) -> dict[str, int]:
        """Return small operational counters suitable for experiment logs."""

        entries = list(self._entries.values())
        return {
            "cells": len(entries),
            "findings": sum(entry.summary.has_finding for entry in entries),
            "visits": sum(entry.visits for entry in entries),
            "max_depth": max((entry.depth for entry in entries), default=0),
        }
