"""Offline contextual-Q selector for compatible replay-tape continuations."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER", *ANIMALS)


@dataclass(frozen=True, slots=True)
class TapeSelectorConfig:
    branch_step: int = 145
    neighbors: int = 3
    distance_epsilon: float = 1e-6
    ood_threshold: float = float("inf")
    risk_coefficient: float = 0.0
    switch_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class TapeSelectorCheckpoint:
    config: TapeSelectorConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_features: np.ndarray
    q_values: np.ndarray
    global_action: int
    action_ids: tuple[int, ...]
    metadata: dict[str, Any]


def _tile_features(farm: dict[str, Any]) -> list[float]:
    kinds: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    ready = 0
    needs_water = 0
    fertilizer = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "EMPTY"))
            kinds[kind] += 1
            crop = tile.get("crop")
            if crop is not None:
                crops[str(crop)] += 1
            animal = tile.get("animal")
            if animal is not None:
                animals[str(animal)] += 1
            ready += int(float(tile.get("yield_units", 0) or 0) > 0)
            needs_water += int(kind == "PLANT" and not tile.get("watered_today", False))
            fertilizer += int(float(tile.get("fertilizer_available", 0) or 0) > 0)

    return [
        len(farm.get("hands", [])) / 16.0,
        len(farm.get("unlocked_quadrants", [])) / 4.0,
        float(farm.get("hires_today", 0) or 0) / 16.0,
        kinds["WEED"] / 100.0,
        kinds["PASTURE"] / 30.0,
        kinds["COOP"] / 30.0,
        ready / 30.0,
        needs_water / 30.0,
        fertilizer / 30.0,
        *(crops[crop] / 30.0 for crop in CROPS),
        *(animals[animal] / 30.0 for animal in ANIMALS),
    ]


def encode_selector_features(observation: dict[str, Any]) -> np.ndarray:
    """Encode only information available to the candidate at decision time."""
    player = int(observation.get("player", 0))
    opponent = 1 - player
    farms = observation["farms"]
    candidate_farm = farms[player]
    opponent_farm = farms[opponent]
    candidate_money = float(candidate_farm.get("money", 0) or 0)
    opponent_money = float(opponent_farm.get("money", 0) or 0)
    prices = observation.get("market", {}).get("prices", {})
    private = observation.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    inventory: Counter[str] = Counter()
    for unit_inventory in private.get("inventories", []):
        inventory.update(
            {
                str(item): int(quantity)
                for item, quantity in unit_inventory.items()
            }
        )

    features = [
        float(observation.get("step", 0) or 0) / 719.0,
        float(observation.get("day", 0) or 0) / 30.0,
        float(observation.get("hour", 0) or 0) / 24.0,
        candidate_money / 200_000.0,
        opponent_money / 200_000.0,
        (candidate_money - opponent_money) / 200_000.0,
        *(float(prices.get(item, 0) or 0) / 2_000.0 for item in ITEMS),
        *_tile_features(candidate_farm),
        *_tile_features(opponent_farm),
        *(float(shed.get(item, 0) or 0) / 200.0 for item in ITEMS),
        *(float(seeds.get(crop, 0) or 0) / 50.0 for crop in CROPS),
        *(float(inventory[item]) / 100.0 for item in ITEMS),
    ]
    return np.asarray(features, dtype=np.float64)


def fit_selector(
    features: np.ndarray,
    q_values: np.ndarray,
    *,
    action_ids: tuple[int, ...],
    config: TapeSelectorConfig,
    metadata: dict[str, Any] | None = None,
) -> TapeSelectorCheckpoint:
    if features.ndim != 2 or q_values.ndim != 2:
        raise ValueError("features and q_values must be rank-two arrays")
    if features.shape[0] != q_values.shape[0]:
        raise ValueError("features and q_values must have equal rows")
    if q_values.shape[1] != len(action_ids):
        raise ValueError("q_values columns must match action_ids")
    if not 1 <= config.neighbors <= len(features):
        raise ValueError("neighbors must be within the training row count")

    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (features - feature_mean) / feature_scale
    global_action = int(np.argmax(q_values.mean(axis=0)))
    return TapeSelectorCheckpoint(
        config=config,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        train_features=normalized,
        q_values=q_values.astype(np.float64, copy=True),
        global_action=global_action,
        action_ids=action_ids,
        metadata=dict(metadata or {}),
    )


def predict_q(
    checkpoint: TapeSelectorCheckpoint, features: np.ndarray
) -> tuple[np.ndarray, float]:
    normalized = (features - checkpoint.feature_mean) / checkpoint.feature_scale
    distances = np.linalg.norm(checkpoint.train_features - normalized, axis=1)
    count = min(checkpoint.config.neighbors, len(distances))
    nearest = np.argpartition(distances, count - 1)[:count]
    minimum_distance = float(distances[nearest].min())
    if minimum_distance > checkpoint.config.ood_threshold:
        values = np.full(checkpoint.q_values.shape[1], -np.inf)
        values[checkpoint.global_action] = 0.0
        return values, minimum_distance
    weights = 1.0 / (distances[nearest] + checkpoint.config.distance_epsilon)
    neighbor_values = checkpoint.q_values[nearest]
    mean_values = np.average(neighbor_values, axis=0, weights=weights)
    centered = neighbor_values - mean_values
    variance = np.average(np.square(centered), axis=0, weights=weights)
    values = mean_values - checkpoint.config.risk_coefficient * np.sqrt(variance)
    return values, minimum_distance


def select_action(
    checkpoint: TapeSelectorCheckpoint, features: np.ndarray
) -> tuple[int, np.ndarray, float]:
    values, distance = predict_q(checkpoint, features)
    column = int(np.argmax(values))
    global_column = checkpoint.global_action
    if (
        column != global_column
        and values[column]
        < values[global_column] + checkpoint.config.switch_threshold
    ):
        column = global_column
    return checkpoint.action_ids[column], values, distance


def save_selector_checkpoint(
    path: Path, checkpoint: TapeSelectorCheckpoint
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        config_json=np.asarray(json.dumps(asdict(checkpoint.config), sort_keys=True)),
        feature_mean=checkpoint.feature_mean,
        feature_scale=checkpoint.feature_scale,
        train_features=checkpoint.train_features,
        q_values=checkpoint.q_values,
        global_action=np.asarray(checkpoint.global_action),
        action_ids=np.asarray(checkpoint.action_ids, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(checkpoint.metadata, sort_keys=True)),
    )


def load_selector_checkpoint(path: Path) -> TapeSelectorCheckpoint:
    with np.load(path, allow_pickle=False) as payload:
        return TapeSelectorCheckpoint(
            config=TapeSelectorConfig(**json.loads(str(payload["config_json"]))),
            feature_mean=payload["feature_mean"].copy(),
            feature_scale=payload["feature_scale"].copy(),
            train_features=payload["train_features"].copy(),
            q_values=payload["q_values"].copy(),
            global_action=int(payload["global_action"]),
            action_ids=tuple(int(value) for value in payload["action_ids"]),
            metadata=json.loads(str(payload["metadata_json"])),
        )


def _load_trace(path: Path, module_index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_tape_selector_trace_{module_index}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load replay trace from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = getattr(module, "TRACE_ACTIONS", None)
    if not isinstance(trace, list) or not trace:
        raise TypeError(f"{path} does not contain TRACE_ACTIONS")
    return trace


class TapeSelectorPolicy:
    """Stateful inference wrapper that commits only after a shared prefix."""

    def __init__(self, checkpoint_path: Path, population_manifest: Path) -> None:
        self.checkpoint = load_selector_checkpoint(checkpoint_path)
        population = json.loads(population_manifest.read_text(encoding="utf-8"))
        self.traces = {
            int(row["episode_id"]): _load_trace(
                Path(row["agent_file"]), index
            )
            for index, row in enumerate(population["agents"])
        }
        missing = set(self.checkpoint.action_ids) - self.traces.keys()
        if missing:
            raise ValueError(f"population is missing checkpoint actions: {missing}")
        self._common_action = self.checkpoint.action_ids[0]
        self._selected: dict[int, int] = {}
        self._last_step: dict[int, int] = {}
        self.last_decision: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._last_step.clear()
        self.last_decision.clear()

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous = self._last_step.get(player, -1)
        if step < previous or (step == 0 and previous != 0):
            self._selected.pop(player, None)
            self.last_decision.pop(player, None)
        self._last_step[player] = step

        if step < self.checkpoint.config.branch_step:
            action_id = self._common_action
        else:
            if player not in self._selected:
                if self.checkpoint.metadata.get("feature_encoder") == "retriever":
                    from kaggriculture_rl.learning.dynamic_tape_retriever import (
                        encode_retriever_features,
                    )

                    features = encode_retriever_features(observation)
                else:
                    features = encode_selector_features(observation)
                action_id, values, distance = select_action(
                    self.checkpoint, features
                )
                self._selected[player] = action_id
                self.last_decision[player] = {
                    "step": step,
                    "action_id": action_id,
                    "distance": distance,
                    "q_values": values.tolist(),
                }
            action_id = self._selected[player]

        trace = self.traces[action_id]
        trace_step = min(step, len(trace) - 1)
        return copy.deepcopy(trace[trace_step])
