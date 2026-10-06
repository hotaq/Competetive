"""State-aware replay policy that can switch to the nearest compatible tape."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.tape_selector import ITEMS, encode_selector_features

MAX_RETRIEVER_UNITS = 13


class DynamicTapeRetrieverPolicy:
    """Retrieve a joint action from the closest source trajectory at each interval."""

    def __init__(
        self,
        checkpoint_path: Path,
        population_manifest: Path,
        *,
        reselect_interval: int = 1,
        stickiness: float = 0.0,
        scale_floor: float = 0.02,
        fixed_unit_action_id: int | None = None,
        fixed_market_action_id: int | None = None,
    ) -> None:
        if reselect_interval < 1:
            raise ValueError("reselect_interval must be positive")
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")

        with np.load(checkpoint_path, allow_pickle=False) as payload:
            self.prototype_features = payload["prototype_features"].astype(
                np.float64, copy=True
            )
            self.feature_scale = payload["feature_scale"].astype(
                np.float64, copy=True
            )
            self.feature_indices = payload["feature_indices"].astype(
                np.int64, copy=True
            )
            self.action_ids = tuple(int(value) for value in payload["action_ids"])

        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        paths = {
            int(agent["episode_id"]): Path(agent["agent_file"])
            for agent in population["agents"]
        }
        self.traces = tuple(
            _load_trace(paths[action_id], index)
            for index, action_id in enumerate(self.action_ids)
        )
        self.reselect_interval = reselect_interval
        self.stickiness = float(stickiness)
        self.scale_floor = float(scale_floor)
        self.fixed_unit_index = (
            self.action_ids.index(fixed_unit_action_id)
            if fixed_unit_action_id is not None
            else None
        )
        self.fixed_market_index = (
            self.action_ids.index(fixed_market_action_id)
            if fixed_market_action_id is not None
            else None
        )
        self._selected: dict[int, int] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._last_step.clear()

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        if step <= self._last_step.get(player, -1):
            self._selected.pop(player, None)
        self._last_step[player] = step

        selected = self._selected.get(player)
        if selected is None or step % self.reselect_interval == 0:
            encoded = encode_retriever_features(observation)[self.feature_indices]
            prototype_step = min(step, self.prototype_features.shape[1] - 1)
            scale = np.maximum(self.feature_scale, self.scale_floor)
            delta = (self.prototype_features[:, prototype_step] - encoded) / scale
            distances = np.mean(np.square(delta), axis=1)
            if selected is not None and self.stickiness:
                distances[selected] = max(0.0, distances[selected] - self.stickiness)
            selected = int(np.argmin(distances))
            self._selected[player] = selected

        trace = self.traces[selected]
        action = copy.deepcopy(trace[min(step, len(trace) - 1)])
        if self.fixed_unit_index is not None:
            unit_trace = self.traces[self.fixed_unit_index]
            unit_action = unit_trace[min(step, len(unit_trace) - 1)]
            action["farmer"] = copy.deepcopy(unit_action["farmer"])
            action["hands"] = copy.deepcopy(unit_action["hands"])
        if self.fixed_market_index is not None:
            market_trace = self.traces[self.fixed_market_index]
            market_action = market_trace[min(step, len(market_trace) - 1)]
            action["market"] = copy.deepcopy(market_action["market"])
        return action


def _load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_dynamic_tape_{module_index}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tape agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def encode_retriever_features(observation: dict[str, Any]) -> np.ndarray:
    """Encode aggregate state plus unit positions and per-unit inventories."""
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    private = observation.get("private", {})
    inventories = private.get("inventories", [])
    positions = [farm.get("farmer"), *farm.get("hands", [])]
    unit_features: list[float] = []
    for index in range(MAX_RETRIEVER_UNITS):
        if index < len(positions) and positions[index] is not None:
            position = positions[index]
            unit_features.extend(
                [float(position[0]) / 9.0, float(position[1]) / 9.0, 1.0]
            )
            inventory = inventories[index] if index < len(inventories) else {}
            unit_features.extend(
                float(inventory.get(item, 0) or 0) / 50.0 for item in ITEMS
            )
        else:
            unit_features.extend((-1.0, -1.0, 0.0))
            unit_features.extend(0.0 for _ in ITEMS)
    return np.concatenate(
        (
            encode_selector_features(observation),
            np.asarray(unit_features, dtype=np.float64),
            np.asarray([float(player)], dtype=np.float64),
        )
    )
