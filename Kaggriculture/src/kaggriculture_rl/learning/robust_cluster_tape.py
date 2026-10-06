"""Runtime for exact-state robust tape groups with targeted receding replans."""

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


class RobustClusterTapePolicy:
    def __init__(
        self,
        checkpoint: Path,
        raw_population: Path,
        program_population: Path,
        robust_cluster_program: Path,
    ) -> None:
        self.model = json.loads(checkpoint.read_text(encoding="utf-8"))
        traces: dict[int, list[dict[str, Any]]] = {}
        module_index = 0
        for manifest_path in (raw_population, program_population):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for agent in manifest["agents"]:
                action_id = int(agent["episode_id"])
                if action_id not in traces:
                    traces[action_id] = _load_trace(
                        Path(agent["agent_file"]), module_index
                    )
                    module_index += 1
        robust_manifest = json.loads(robust_cluster_program.read_text(encoding="utf-8"))
        for agent in robust_manifest["agents"]:
            traces[int(agent["episode_id"])] = _load_trace(
                Path(agent["agent_file"]), module_index
            )
            module_index += 1
        required = {
            int(self.model["base_action_id"]),
            *(
                int(value)
                for value in self.model["special_action_ids"].values()
            ),
            *(int(value) for value in self.model["branch163"]["first_actions"]),
            *(
                int(value)
                for stage in self.model["stages"].values()
                for value in stage["decisions"]
            ),
        }
        missing = required.difference(traces)
        if missing:
            raise ValueError(f"missing traces: {sorted(missing)}")
        self.traces = traces
        self._selected: dict[int, int] = {}
        self._mode: dict[int, str] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._mode.clear()
        self._last_step.clear()

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous = self._last_step.get(player, -1)
        if step < previous or (step == 0 and previous != 0):
            self._selected.pop(player, None)
            self._mode.pop(player, None)
        self._last_step[player] = step
        self._selected.setdefault(player, int(self.model["base_action_id"]))
        self._mode.setdefault(player, "base")

        features: np.ndarray | None = None
        if step == 97:
            features = encode_retriever_features(observation)
            mode, distance = _nearest(self.model["branch97"], features, "modes")
            if distance < 1e-10 and mode != "base":
                self._mode[player] = str(mode)
                self._selected[player] = int(self.model["special_action_ids"][mode])

        if step == 163 and self._mode[player] == "base":
            features = encode_retriever_features(observation)
            row, _ = _nearest_index(self.model["branch163"], features)
            self._selected[player] = int(
                self.model["branch163"]["first_actions"][row]
            )
            replan = str(self.model["branch163"]["replans"][row])
            if replan:
                self._mode[player] = replan

        mode = self._mode[player]
        if mode == "special_cluster" and step == 216:
            mode = "special_cluster_216"
        stage = self.model["stages"].get(mode)
        if stage is not None and step == int(stage["branch_step"]):
            if features is None:
                features = encode_retriever_features(observation)
            selected, _ = _nearest(stage, features, "decisions")
            self._selected[player] = int(selected)
            self._mode[player] = "committed"

        action_id = self._selected[player]
        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])


def _nearest_index(model: dict[str, Any], features: np.ndarray) -> tuple[int, float]:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (features - mean) / scale
    prototypes = np.asarray(model["features"], dtype=np.float64)
    distances = np.linalg.norm(prototypes - normalized, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _nearest(
    model: dict[str, Any], features: np.ndarray, value_key: str
) -> tuple[Any, float]:
    index, distance = _nearest_index(model, features)
    return model[value_key][index], distance


def _load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_robust_cluster_tape_{module_index}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tape agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)

