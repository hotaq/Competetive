"""Two-stage state-aware tree gate over replay-tape continuations."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)


class HierarchicalTreeTapePolicy:
    """Select once at the first branch and optionally refine one risky branch."""

    def __init__(
        self,
        first_checkpoint: Path,
        second_checkpoint: Path,
        population_manifest: Path,
        *,
        refine_action_id: int,
    ) -> None:
        self.first_model = json.loads(first_checkpoint.read_text(encoding="utf-8"))
        self.second_model = json.loads(second_checkpoint.read_text(encoding="utf-8"))
        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        self.traces = {
            int(agent["episode_id"]): _load_trace(
                Path(agent["agent_file"]), index
            )
            for index, agent in enumerate(population["agents"])
        }
        required = {
            *self.first_model["action_ids"],
            *self.second_model["action_ids"],
        }
        missing = {int(value) for value in required}.difference(self.traces)
        if missing:
            raise ValueError(f"population is missing actions: {sorted(missing)}")
        self.refine_action_id = int(refine_action_id)
        self.first_branch = int(self.first_model["branch_step"])
        self.second_branch = int(self.second_model["branch_step"])
        self.common_action_id = int(self.first_model["action_ids"][0])
        self._first_selected: dict[int, int] = {}
        self._second_selected: dict[int, int] = {}
        self._history: dict[int, dict[int, np.ndarray]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._first_selected.clear()
        self._second_selected.clear()
        self._history.clear()
        self._last_step.clear()

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous = self._last_step.get(player, -1)
        if step < previous or (step == 0 and previous != 0):
            self._first_selected.pop(player, None)
            self._second_selected.pop(player, None)
            self._history.pop(player, None)
        self._last_step[player] = step

        relevant_steps = {
            *self.first_model["feature_steps"],
            *self.second_model["feature_steps"],
        }
        if step in relevant_steps:
            self._history.setdefault(player, {})[step] = encode_retriever_features(
                observation
            )

        if step >= self.first_branch and player not in self._first_selected:
            self._first_selected[player] = _select_action(
                self.first_model, self._history[player]
            )

        first_action = self._first_selected.get(player, self.common_action_id)
        if (
            step >= self.second_branch
            and first_action == self.refine_action_id
            and player not in self._second_selected
        ):
            self._second_selected[player] = _select_action(
                self.second_model, self._history[player]
            )
        action_id = self._second_selected.get(player, first_action)
        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])


def _select_action(model: dict[str, Any], history: dict[int, np.ndarray]) -> int:
    feature_steps = [int(value) for value in model["feature_steps"]]
    missing = [step for step in feature_steps if step not in history]
    if missing:
        raise RuntimeError(f"missing feature snapshots: {missing}")
    features = np.concatenate([history[step] for step in feature_steps])
    node = model["tree"]
    while "action_index" not in node:
        node = (
            node["left"]
            if features[int(node["feature_index"])] <= float(node["threshold"])
            else node["right"]
        )
    return int(model["action_ids"][int(node["action_index"])])


def _load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_hierarchical_tape_{module_index}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tape agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)

