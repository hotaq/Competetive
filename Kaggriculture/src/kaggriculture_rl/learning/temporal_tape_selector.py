"""Temporal contextual-Q policy over compatible replay continuations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.tape_selector import (
    _load_trace,
    encode_selector_features,
    load_selector_checkpoint,
    select_action,
)


def _agent_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(path)


class TemporalTapeSelectorPolicy:
    """Choose once using a causal sequence of early-game state snapshots."""

    def __init__(self, checkpoint_path: Path, population_manifest: Path) -> None:
        self.checkpoint = load_selector_checkpoint(checkpoint_path)
        raw_steps = self.checkpoint.metadata.get("feature_steps", [])
        self.feature_steps = tuple(int(step) for step in raw_steps)
        if not self.feature_steps:
            raise ValueError("checkpoint does not contain temporal feature steps")
        if self.feature_steps[-1] > self.checkpoint.config.branch_step:
            raise ValueError("temporal feature step exceeds branch step")

        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        traces = {
            int(row["episode_id"]): _load_trace(
                _agent_path(population_manifest, row["agent_file"]), index
            )
            for index, row in enumerate(population["agents"])
        }
        missing = set(self.checkpoint.action_ids) - traces.keys()
        if missing:
            raise ValueError(f"checkpoint actions missing from population: {sorted(missing)}")
        self.traces = {
            action_id: traces[action_id] for action_id in self.checkpoint.action_ids
        }
        self._common_action = self.checkpoint.action_ids[0]
        self._selected: dict[int, int] = {}
        self._snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._snapshots.clear()
        self._last_step.clear()

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step:
            self._selected.pop(player, None)
            self._snapshots.pop(player, None)
        self._last_step[player] = step

        if step in self.feature_steps:
            self._snapshots.setdefault(player, {})[step] = encode_selector_features(
                observation
            )

        if step < self.checkpoint.config.branch_step:
            action_id = self._common_action
        elif player not in self._selected:
            snapshots = self._snapshots.setdefault(player, {})
            current = encode_selector_features(observation)
            temporal = np.concatenate(
                [snapshots.get(feature_step, current) for feature_step in self.feature_steps]
            )
            action_id, _, _ = select_action(self.checkpoint, temporal)
            self._selected[player] = action_id
        else:
            action_id = self._selected[player]

        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])
