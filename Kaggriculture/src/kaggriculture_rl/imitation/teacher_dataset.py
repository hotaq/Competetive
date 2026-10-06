"""Generate spatial imitation data from a standalone live teacher policy."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.bc import confidence_gate
from kaggriculture_rl.imitation.replay_dataset import (
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    PASS_TOKEN,
    STOP_TOKEN,
)
from kaggriculture_rl.imitation.spatial_dataset import (
    SpatialReplayDataset,
    encode_spatial_observation,
    save_spatial_replay_dataset,
)
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy

DEFAULT_CROP_OPPONENTS = (
    ("WHEAT", 6),
    ("CARROT", 6),
    ("TOMATO", 6),
    ("STRAWBERRY", 6),
    ("MELON", 6),
)
_MODULE_COUNTER = itertools.count()


@dataclass(frozen=True, slots=True)
class _OpponentSpec:
    kind: str
    label: str
    path: Path | None = None
    crop: str | None = None
    plots: int | None = None


class StandaloneFilePolicy:
    """Fresh import of a Kaggle-style `main.py` policy for one episode."""

    def __init__(self, path: Path, *, role: str) -> None:
        self.path = path.resolve()
        self.module_name = f"kaggriculture_{role}_{next(_MODULE_COUNTER)}"
        self.module = self._load_module()
        self.agent_fn = self.module.agent
        parameters = list(inspect.signature(self.agent_fn).parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        self.accepts_configuration = len(positional) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )

    def _load_module(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location(self.module_name, self.path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load standalone policy: {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "agent", None)):
            raise RuntimeError(f"standalone policy has no callable agent(): {self.path}")
        return module

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.accepts_configuration:
            action = self.agent_fn(observation, None)
        else:
            action = self.agent_fn(observation)
        if not isinstance(action, dict):
            raise TypeError(f"standalone policy returned non-dict action: {type(action)!r}")
        return action

    def close(self) -> None:
        sys.modules.pop(self.module_name, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_token(action: Any) -> str:
    if not isinstance(action, list) or not action or not isinstance(action[0], str):
        raise ValueError(f"invalid primitive action: {action!r}")
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _aligned_action(action: dict[str, Any], hands_count: int) -> dict[str, Any]:
    farmer = action.get("farmer")
    farmer = list(farmer) if isinstance(farmer, (list, tuple)) and farmer else ["PASS"]
    hands = [
        list(order) if isinstance(order, (list, tuple)) and order else ["PASS"]
        for order in (action.get("hands") or [])
    ]
    if len(hands) < hands_count:
        hands.extend([["PASS"] for _ in range(hands_count - len(hands))])
    market = [
        list(order)
        for order in (action.get("market") or [])
        if isinstance(order, (list, tuple)) and order
    ]
    return {
        "farmer": farmer,
        "hands": hands[:hands_count],
        "market": market[:MAX_MARKET_ORDERS],
    }


def sanitize_standalone_action(
    engine: Engine,
    seat: int,
    raw_action: dict[str, Any],
    hands_count: int,
) -> dict[str, Any]:
    aligned = _aligned_action(raw_action, hands_count)
    sanitized = engine.sanitize_action(seat, aligned)
    return _aligned_action(sanitized, hands_count)


def _opponent_catalog(
    teacher_path: Path,
    file_opponents: Sequence[Path],
    crop_opponents: Sequence[tuple[str, int]],
    *,
    include_self_play: bool,
) -> tuple[_OpponentSpec, ...]:
    specs: list[_OpponentSpec] = []
    if include_self_play:
        specs.append(
            _OpponentSpec("file", f"self:{teacher_path.name}", path=teacher_path)
        )
    specs.extend(
        _OpponentSpec("file", f"file:{path.name}", path=path.resolve())
        for path in file_opponents
    )
    for crop, plots in crop_opponents:
        if plots <= 0:
            raise ValueError("crop opponent plots must be positive")
        specs.append(
            _OpponentSpec(
                "crop",
                f"crop:{crop.upper()}:{plots}",
                crop=crop.upper(),
                plots=plots,
            )
        )
    if not specs:
        raise ValueError("at least one teacher opponent is required")
    return tuple(specs)


def _label_actions(
    actions: Sequence[dict[str, Any]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    unit_tokens = {PASS_TOKEN}
    market_tokens: set[str] = set()
    for action in actions:
        unit_tokens.add(_canonical_token(action["farmer"]))
        unit_tokens.update(_canonical_token(order) for order in action["hands"])
        market_tokens.update(_canonical_token(order) for order in action["market"])
    unit_vocab = (PASS_TOKEN, *sorted(unit_tokens - {PASS_TOKEN}))
    market_vocab = (STOP_TOKEN, *sorted(market_tokens))
    unit_to_id = {token: index for index, token in enumerate(unit_vocab)}
    market_to_id = {token: index for index, token in enumerate(market_vocab)}

    count = len(actions)
    max_hands = max(MAX_HANDS, max(len(action["hands"]) for action in actions))
    farmer_labels = np.zeros(count, dtype=np.int16)
    hand_labels = np.zeros((count, max_hands), dtype=np.int16)
    hand_mask = np.zeros((count, max_hands), dtype=np.bool_)
    market_labels = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.int16)
    market_mask = np.zeros((count, MAX_MARKET_ORDERS), dtype=np.bool_)
    for row, action in enumerate(actions):
        farmer_labels[row] = unit_to_id[_canonical_token(action["farmer"])]
        for index, order in enumerate(action["hands"]):
            hand_labels[row, index] = unit_to_id[_canonical_token(order)]
            hand_mask[row, index] = True
        for index, order in enumerate(action["market"]):
            market_labels[row, index] = market_to_id[_canonical_token(order)]
            market_mask[row, index] = True
        if len(action["market"]) < MAX_MARKET_ORDERS:
            market_mask[row, len(action["market"])] = True
    return (
        farmer_labels,
        hand_labels,
        hand_mask,
        market_labels,
        market_mask,
        unit_vocab,
        market_vocab,
    )


def build_live_teacher_spatial_dataset(
    teacher_path: Path,
    output_path: Path,
    *,
    episodes: int = 24,
    holdout_episodes: int = 4,
    base_seed: int = 610_000,
    file_opponents: Sequence[Path] = (),
    crop_opponents: Sequence[tuple[str, int]] = DEFAULT_CROP_OPPONENTS,
    include_self_play: bool = True,
    behavior_checkpoint: Path | None = None,
    behavior_confidence_threshold: float = 1.0,
    behavior_start_episode: int = 0,
) -> dict[str, Any]:
    """Roll out a strong live teacher and persist the existing spatial-BC schema."""
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number for seat balancing")
    if holdout_episodes <= 0 or holdout_episodes % 2:
        raise ValueError("holdout_episodes must be a positive even number")
    if holdout_episodes >= episodes:
        raise ValueError("holdout_episodes must be smaller than episodes")
    if not 0.0 <= behavior_confidence_threshold <= 1.0:
        raise ValueError("behavior_confidence_threshold must be within 0..1")
    if not 0 <= behavior_start_episode < episodes:
        raise ValueError("behavior_start_episode must be within the episode range")
    teacher_path = teacher_path.resolve()
    output_path = output_path.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    if behavior_checkpoint is not None:
        behavior_checkpoint = behavior_checkpoint.resolve()
        if not behavior_checkpoint.is_file():
            raise FileNotFoundError(behavior_checkpoint)
    specs = _opponent_catalog(
        teacher_path,
        file_opponents,
        crop_opponents,
        include_self_play=include_self_play,
    )

    spatial_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    action_rows: list[dict[str, Any]] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []
    train_rows: list[bool] = []
    records: list[dict[str, Any]] = []
    sanitation_changes = 0
    behavior_action_differences = 0

    for episode_index in range(episodes):
        pair_index = episode_index // 2
        teacher_seat = episode_index % 2
        opponent_seat = 1 - teacher_seat
        game_seed = base_seed + pair_index
        episode_id = base_seed * 100 + episode_index
        opponent_spec = specs[pair_index % len(specs)]
        engine = Engine(
            {"episodeSteps": 720, "seed": game_seed, "weedSpawnChance": 0.005}
        )
        teacher = StandaloneFilePolicy(teacher_path, role="teacher")
        file_opponent: StandaloneFilePolicy | None = None
        crop_opponent: CropHeuristicPolicy | None = None
        behavior_policy: SpatialRecurrentPolicy | None = None
        if behavior_checkpoint is not None and episode_index >= behavior_start_episode:
            behavior_policy = SpatialRecurrentPolicy(behavior_checkpoint)
        if opponent_spec.kind == "file":
            assert opponent_spec.path is not None
            file_opponent = StandaloneFilePolicy(opponent_spec.path, role="opponent")
        else:
            assert opponent_spec.crop is not None and opponent_spec.plots is not None
            crop_opponent = CropHeuristicPolicy(
                crop=opponent_spec.crop,
                target_quantity=opponent_spec.plots,
                sell_reserve=0,
            )

        try:
            for step in range(719):
                teacher_observation = engine.observation(teacher_seat)
                opponent_observation = engine.observation(opponent_seat)
                spatial, global_features = encode_spatial_observation(
                    teacher_observation
                )
                raw_teacher_action = teacher(teacher_observation)
                teacher_hands = len(
                    teacher_observation["farms"][teacher_seat].get("hands", [])
                )
                teacher_action = sanitize_standalone_action(
                    engine,
                    teacher_seat,
                    raw_teacher_action,
                    teacher_hands,
                )
                sanitation_changes += int(
                    teacher_action != _aligned_action(raw_teacher_action, teacher_hands)
                )
                behavior_action = teacher_action
                if behavior_policy is not None:
                    prediction = behavior_policy.predict_with_confidence(
                        engine,
                        teacher_observation,
                    )
                    proposed_behavior, _ = confidence_gate(
                        prediction,
                        teacher_action,
                        behavior_confidence_threshold,
                    )
                    behavior_action = sanitize_standalone_action(
                        engine,
                        teacher_seat,
                        proposed_behavior,
                        teacher_hands,
                    )
                    behavior_action_differences += int(
                        behavior_action != teacher_action
                    )

                if file_opponent is not None:
                    raw_opponent_action = file_opponent(opponent_observation)
                else:
                    assert crop_opponent is not None
                    opponent_plan = engine.plan(
                        opponent_seat,
                        crop_opponent(
                            {"shed": opponent_observation["private"]["shed"]}
                        ),
                    )
                    raw_opponent_action = opponent_plan["joint_action"]["actions"][
                        opponent_seat
                    ]
                opponent_hands = len(
                    opponent_observation["farms"][opponent_seat].get("hands", [])
                )
                opponent_action = sanitize_standalone_action(
                    engine,
                    opponent_seat,
                    raw_opponent_action,
                    opponent_hands,
                )

                spatial_rows.append(spatial.astype(np.float16))
                global_rows.append(global_features.astype(np.float16))
                action_rows.append(teacher_action)
                episode_rows.append(episode_id)
                step_rows.append(step)
                train_rows.append(episode_index < episodes - holdout_episodes)

                actions = [opponent_action, opponent_action]
                actions[teacher_seat] = behavior_action
                actions[opponent_seat] = opponent_action
                engine.step(actions)
        finally:
            teacher.close()
            if file_opponent is not None:
                file_opponent.close()

        state = engine.state
        teacher_money = float(state["farms"][teacher_seat]["money"])
        opponent_money = float(state["farms"][opponent_seat]["money"])
        records.append(
            {
                "episode_id": episode_id,
                "seed": game_seed,
                "teacher_seat": teacher_seat,
                "opponent": opponent_spec.label,
                "teacher_money": teacher_money,
                "rollout_candidate_money": teacher_money,
                "opponent_money": opponent_money,
                "margin": teacher_money - opponent_money,
            }
        )

    (
        farmer_labels,
        hand_labels,
        hand_mask,
        market_labels,
        market_mask,
        unit_vocab,
        market_vocab,
    ) = _label_actions(action_rows)
    metadata = {
        "schema_version": 3,
        "encoder": "perspective-spatial-v1",
        "source": "live-standalone-teacher-v1",
        "teacher_path": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "episodes": episodes,
        "holdout_episodes": holdout_episodes,
        "base_seed": base_seed,
        "samples": len(action_rows),
        "opponents": [spec.label for spec in specs],
        "teacher_sanitation_changes": sanitation_changes,
        "behavior": (
            {
                "checkpoint": str(behavior_checkpoint),
                "checkpoint_sha256": _sha256(behavior_checkpoint),
                "confidence_threshold": behavior_confidence_threshold,
                "start_episode": behavior_start_episode,
                "action_differences": behavior_action_differences,
            }
            if behavior_checkpoint is not None
            else None
        ),
        "unit_vocab_size": len(unit_vocab),
        "market_vocab_size": len(market_vocab),
        "max_hands": int(hand_labels.shape[1]),
        "episode_records": records,
    }
    dataset = SpatialReplayDataset(
        spatial=np.stack(spatial_rows),
        global_features=np.stack(global_rows),
        farmer_labels=farmer_labels,
        hand_labels=hand_labels,
        hand_mask=hand_mask,
        market_labels=market_labels,
        market_mask=market_mask,
        episode_ids=np.asarray(episode_rows, dtype=np.int64),
        steps=np.asarray(step_rows, dtype=np.int16),
        train_mask=np.asarray(train_rows, dtype=np.bool_),
        unit_vocab=unit_vocab,
        market_vocab=market_vocab,
        metadata=metadata,
    )
    save_spatial_replay_dataset(output_path, dataset)
    return {
        "dataset": str(output_path),
        "samples": len(action_rows),
        "train_samples": int(dataset.train_mask.sum()),
        "holdout_samples": int(dataset.holdout_mask.sum()),
        "unit_vocab_size": len(unit_vocab),
        "market_vocab_size": len(market_vocab),
        "max_hands": int(hand_labels.shape[1]),
        "teacher_sanitation_changes": sanitation_changes,
        "behavior_action_differences": behavior_action_differences,
        "episodes": records,
    }
