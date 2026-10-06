"""A conservative second-stage selector over replay-tape continuations."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)
from kaggriculture_rl.learning.spatial_recurrent import (
    encode_spatial_observation,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorCheckpoint,
    encode_selector_features,
    load_selector_checkpoint,
    select_action,
)

Agent = Callable[..., dict[str, Any]]
STAY_ACTION_ID = -1


def encode_spatial_selector_features(
    observation: dict[str, Any],
) -> np.ndarray:
    spatial, global_features = encode_spatial_observation(observation)
    return np.concatenate((spatial.reshape(-1), global_features))


FEATURE_ENCODERS: dict[str, Callable[[dict[str, Any]], np.ndarray]] = {
    "aggregate": encode_selector_features,
    "retriever": encode_retriever_features,
    "spatial": encode_spatial_selector_features,
}


def select_stage_two_action(
    checkpoint: TapeSelectorCheckpoint,
    features: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    """Select with either ordinary Q values or conservative rescue utility."""

    mode = str(checkpoint.metadata.get("selection_mode", "q"))
    if mode == "q":
        return select_action(checkpoint, features)
    if mode != "rescue_utility":
        raise ValueError(f"unknown stage-two selection mode: {mode}")

    try:
        stay_column = checkpoint.action_ids.index(STAY_ACTION_ID)
    except ValueError as error:
        raise ValueError("rescue utility requires the stay action") from error
    normalized = (features - checkpoint.feature_mean) / checkpoint.feature_scale
    distances = np.linalg.norm(
        checkpoint.train_features - normalized,
        axis=1,
    )
    count = min(checkpoint.config.neighbors, len(distances))
    nearest = np.argpartition(distances, count - 1)[:count]
    minimum_distance = float(distances[nearest].min())
    values = np.full(len(checkpoint.action_ids), -np.inf, dtype=np.float64)
    values[stay_column] = 0.0
    if minimum_distance > checkpoint.config.ood_threshold:
        return STAY_ACTION_ID, values, minimum_distance

    weights = 1.0 / (
        distances[nearest] + checkpoint.config.distance_epsilon
    )
    neighbor_values = checkpoint.q_values[nearest]
    stay_values = neighbor_values[:, stay_column, None]
    rescue = (neighbor_values > 0.0) & (stay_values <= 0.0)
    spoil = (neighbor_values <= 0.0) & (stay_values > 0.0)
    loss_penalty = float(checkpoint.metadata.get("loss_penalty", 2.0))
    margin_weight = float(checkpoint.metadata.get("margin_weight", 0.05))
    utility = rescue.astype(np.float64)
    utility -= loss_penalty * spoil.astype(np.float64)
    utility += margin_weight * np.tanh(
        (neighbor_values - stay_values) / 10_000.0
    )
    mean_utility = np.average(utility, axis=0, weights=weights)
    centered = utility - mean_utility
    variance = np.average(np.square(centered), axis=0, weights=weights)
    values = mean_utility - checkpoint.config.risk_coefficient * np.sqrt(
        variance
    )
    values[stay_column] = 0.0
    column = int(np.argmax(values))
    if values[column] < checkpoint.config.switch_threshold:
        column = stay_column
    return checkpoint.action_ids[column], values, minimum_distance


def select_exact_stage_two_action(
    checkpoint: TapeSelectorCheckpoint,
    features: np.ndarray,
    routes: Sequence[tuple[int, np.ndarray]],
    *,
    tolerance: float = 0.0,
) -> int:
    """Return a routed suffix only when the normalized state matches exactly.

    The safe router is deliberately asymmetric: an unknown or merely nearby
    state always keeps the live base policy.  A small explicit tolerance is
    supported for serialization round-off, but production callers default to
    bit-for-bit equality.
    """

    if tolerance < 0.0:
        raise ValueError("exact-route tolerance must be non-negative")
    normalized = (features - checkpoint.feature_mean) / checkpoint.feature_scale
    for action_id, target in routes:
        matched = (
            np.array_equal(normalized, target)
            if tolerance == 0.0
            else bool(np.max(np.abs(normalized - target)) <= tolerance)
        )
        if matched:
            return action_id
    return STAY_ACTION_ID


def _load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_stage_two_tape_{module_index}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tape trace from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def _resolve_agent_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(path)


class StageTwoTapeSelectorPolicy:
    """Keep a live base policy unless a later selector clears its safety gate."""

    def __init__(
        self,
        base_agent: Agent,
        checkpoint_path: Path,
        population_manifest: Path,
        *,
        exact_safe_routes: Mapping[int, int] | None = None,
        exact_tolerance: float = 0.0,
    ) -> None:
        self.base_agent = base_agent
        self.checkpoint: TapeSelectorCheckpoint = load_selector_checkpoint(
            checkpoint_path
        )
        self.branch_step = self.checkpoint.config.branch_step
        self.feature_steps = tuple(
            int(value)
            for value in self.checkpoint.metadata["feature_steps"]
        )
        encoder_name = str(
            self.checkpoint.metadata.get("feature_encoder", "aggregate")
        )
        try:
            self.feature_encoder = FEATURE_ENCODERS[encoder_name]
        except KeyError as error:
            raise ValueError(
                f"unknown stage-two feature encoder: {encoder_name}"
            ) from error

        population = json.loads(
            population_manifest.read_text(encoding="utf-8")
        )
        paths = {
            int(row["episode_id"]): _resolve_agent_path(
                population_manifest, row["agent_file"]
            )
            for row in population["agents"]
        }
        required = set(self.checkpoint.action_ids) - {STAY_ACTION_ID}
        missing = required - paths.keys()
        if missing:
            raise ValueError(
                f"stage-two population is missing actions: {sorted(missing)}"
            )
        self.traces = {
            action_id: _load_trace(paths[action_id], index)
            for index, action_id in enumerate(sorted(required))
        }
        self._exact_tolerance = exact_tolerance
        self._exact_routes = self._resolve_exact_routes(exact_safe_routes)
        self._selected: dict[int, int] = {}
        self._snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._last_step: dict[int, int] = {}

    def _resolve_exact_routes(
        self,
        routes: Mapping[int, int] | None,
    ) -> tuple[tuple[int, np.ndarray], ...] | None:
        if routes is None:
            return None
        if self._exact_tolerance < 0.0:
            raise ValueError("exact-route tolerance must be non-negative")
        context_ids = tuple(
            int(value)
            for value in self.checkpoint.metadata.get("context_episode_ids", ())
        )
        if len(context_ids) != len(self.checkpoint.train_features):
            raise ValueError(
                "exact routes require one context episode id per training row"
            )
        row_by_episode = {
            episode_id: self.checkpoint.train_features[index]
            for index, episode_id in enumerate(context_ids)
        }
        resolved: list[tuple[int, np.ndarray]] = []
        for representative_id, action_id in routes.items():
            if representative_id not in row_by_episode:
                raise ValueError(
                    f"unknown exact-route representative: {representative_id}"
                )
            if action_id not in self.traces:
                raise ValueError(f"unknown exact-route suffix: {action_id}")
            target = row_by_episode[representative_id].copy()
            for existing_action, existing_target in resolved:
                if np.array_equal(target, existing_target):
                    if existing_action != action_id:
                        raise ValueError(
                            "conflicting suffixes for the same exact feature row"
                        )
                    break
            else:
                resolved.append((action_id, target))
        return tuple(resolved)

    def reset(self) -> None:
        self._selected.clear()
        self._snapshots.clear()
        self._last_step.clear()

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step or (step == 0 and previous_step != 0):
            self._selected.pop(player, None)
            self._snapshots.pop(player, None)
        self._last_step[player] = step

        # Once this stage has committed to a suffix, its base policy can no
        # longer affect the episode.  Avoid evaluating the full nested policy
        # stack on every remaining turn; besides being wasteful, that stack is
        # the dominant cost when this policy is used inside model rollouts.
        # Keep the branch turn itself unchanged so the base policy observes the
        # same state and all feature snapshots are collected as before.
        selected = self._selected.get(player, STAY_ACTION_ID)
        if step > self.branch_step and selected != STAY_ACTION_ID:
            trace = self.traces[selected]
            return copy.deepcopy(trace[min(step, len(trace) - 1)])

        base_action = self.base_agent(observation, configuration)
        if step in self.feature_steps:
            self._snapshots.setdefault(player, {})[step] = (
                self.feature_encoder(observation)
            )

        if step < self.branch_step:
            return base_action
        if player not in self._selected:
            snapshots = self._snapshots.setdefault(player, {})
            missing = set(self.feature_steps) - snapshots.keys()
            if missing:
                self._selected[player] = STAY_ACTION_ID
            else:
                features = np.concatenate(
                    [snapshots[value] for value in self.feature_steps]
                )
                if self._exact_routes is None:
                    action_id, _, _ = select_stage_two_action(
                        self.checkpoint, features
                    )
                else:
                    action_id = select_exact_stage_two_action(
                        self.checkpoint,
                        features,
                        self._exact_routes,
                        tolerance=self._exact_tolerance,
                    )
                self._selected[player] = action_id

        selected = self._selected[player]
        if selected == STAY_ACTION_ID:
            return base_action
        trace = self.traces[selected]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])

    def selected_action(self, player: int = 0) -> int | None:
        """Return the current second-stage decision for diagnostics."""

        return self._selected.get(player)
