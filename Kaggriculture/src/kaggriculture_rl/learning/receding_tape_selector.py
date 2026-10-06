"""State-adaptive retrieval over strong replay-tape continuations."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.tape_selector import (
    _load_trace,
    encode_selector_features,
)


@dataclass(frozen=True, slots=True)
class RecedingTapeConfig:
    """Controls how often the policy can change its retrieved continuation."""

    branch_step: int = 72
    reselect_interval: int = 24
    prior_weight: float = 0.5
    switch_penalty: float = 0.25
    feature_clip: float = 5.0
    ood_threshold: float = float("inf")


def _resolve_agent_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(path)


def _source_rows(manifest_paths: tuple[Path, ...]) -> dict[int, tuple[Path, int]]:
    rows: dict[int, tuple[Path, int]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            episode_id = int(row["episode_id"])
            replay_path = manifest_path.parent / f"{episode_id}.json"
            rows[episode_id] = (replay_path, int(row["top1_seat"]))
    return rows


class RecedingTapePolicy:
    """Retrieve the closest teacher state and periodically switch its suffix."""

    def __init__(
        self,
        population_manifest: Path,
        replay_manifests: tuple[Path, ...],
        *,
        config: RecedingTapeConfig | None = None,
        action_scores: dict[int, float] | None = None,
        allowed_action_ids: set[int] | None = None,
    ) -> None:
        self.config = config or RecedingTapeConfig()
        if self.config.reselect_interval <= 0:
            raise ValueError("reselect_interval must be positive")

        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        rows = population["agents"]
        self.action_ids = tuple(int(row["episode_id"]) for row in rows)
        self.traces = {
            action_id: _load_trace(
                _resolve_agent_path(population_manifest, row["agent_file"]), index
            )
            for index, (action_id, row) in enumerate(
                zip(self.action_ids, rows, strict=True)
            )
        }

        allowed = set(self.action_ids) if allowed_action_ids is None else allowed_action_ids
        missing = allowed - set(self.action_ids)
        if missing:
            raise ValueError(f"unknown allowed action ids: {sorted(missing)}")
        self.allowed_indices = np.asarray(
            [index for index, value in enumerate(self.action_ids) if value in allowed],
            dtype=np.int64,
        )
        if not len(self.allowed_indices):
            raise ValueError("at least one action must be allowed")

        scores = action_scores or {}
        priors = np.asarray(
            [float(scores.get(action_id, 0.0)) for action_id in self.action_ids],
            dtype=np.float64,
        )
        allowed_priors = priors[self.allowed_indices]
        prior_scale = float(allowed_priors.std())
        if prior_scale > 1e-9:
            priors = (priors - float(allowed_priors.mean())) / prior_scale
        else:
            priors = np.zeros_like(priors)
        self.priors = priors

        sources = _source_rows(replay_manifests)
        missing_sources = set(self.action_ids) - sources.keys()
        if missing_sources:
            raise ValueError(f"missing source replays: {sorted(missing_sources)}")

        self.reselect_steps = tuple(
            range(self.config.branch_step, 720, self.config.reselect_interval)
        )
        feature_rows: list[list[np.ndarray]] = []
        for action_id in self.action_ids:
            replay_path, seat = sources[action_id]
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            feature_rows.append(
                [
                    encode_selector_features(replay["steps"][step][seat]["observation"])
                    for step in self.reselect_steps
                ]
            )
        features = np.asarray(feature_rows, dtype=np.float64).transpose(1, 0, 2)
        self.feature_mean = features.mean(axis=1)
        self.feature_scale = features.std(axis=1)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        self.source_features = (features - self.feature_mean[:, None, :]) / (
            self.feature_scale[:, None, :]
        )
        self._step_indices = {
            step: index for index, step in enumerate(self.reselect_steps)
        }
        self._selected: dict[int, int] = {}
        self._last_step: dict[int, int] = {}
        self._selection_history: dict[int, list[tuple[int, int, float]]] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._last_step.clear()
        self._selection_history.clear()

    def _select(self, observation: dict[str, Any], step: int, player: int) -> int:
        checkpoint = self._step_indices[step]
        features = encode_selector_features(observation)
        normalized = (features - self.feature_mean[checkpoint]) / self.feature_scale[
            checkpoint
        ]
        difference = self.source_features[checkpoint] - normalized
        difference = np.clip(
            difference, -self.config.feature_clip, self.config.feature_clip
        )
        distances = np.linalg.norm(difference, axis=1)
        scores = -distances + self.config.prior_weight * self.priors

        previous = self._selected.get(player)
        if previous is not None and self.config.switch_penalty:
            scores = scores.copy()
            for index, action_id in enumerate(self.action_ids):
                if action_id != previous:
                    scores[index] -= self.config.switch_penalty

        allowed_scores = scores[self.allowed_indices]
        local_index = int(np.argmax(allowed_scores))
        selected_index = int(self.allowed_indices[local_index])
        distance = float(distances[selected_index])
        if distance > self.config.ood_threshold:
            selected_index = int(self.allowed_indices[np.argmax(self.priors[self.allowed_indices])])
            distance = float(distances[selected_index])
        action_id = self.action_ids[selected_index]
        self._selection_history.setdefault(player, []).append((step, action_id, distance))
        return action_id

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step:
            self._selected.pop(player, None)
            self._selection_history.pop(player, None)
        self._last_step[player] = step

        if step < self.config.branch_step:
            action_id = self.action_ids[0]
        elif step in self._step_indices or player not in self._selected:
            action_id = self._select(observation, step, player)
            self._selected[player] = action_id
        else:
            action_id = self._selected[player]

        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])

    def selection_history(self, player: int = 0) -> tuple[tuple[int, int, float], ...]:
        """Expose deterministic retrieval decisions for evaluation diagnostics."""

        return tuple(self._selection_history.get(player, ()))
