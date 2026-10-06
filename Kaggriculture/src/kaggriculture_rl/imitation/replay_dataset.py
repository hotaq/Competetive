"""Episode-safe replay extraction for factorized micro-action imitation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FEATURE_DIM = 2_048
MAX_HANDS = 12
MAX_MARKET_ORDERS = 10
STOP_TOKEN = "__STOP__"
PASS_TOKEN = '["PASS"]'
_HASH_PERSON = b"kg-bc-v1"
_SKIPPED_OBSERVATION_KEYS = frozenset({"remainingOverageTime"})


@dataclass(frozen=True)
class ReplayDataset:
    """Dense features, factorized labels, masks, vocabularies and episode split."""

    features: np.ndarray
    farmer_labels: np.ndarray
    hand_labels: np.ndarray
    hand_mask: np.ndarray
    market_labels: np.ndarray
    market_mask: np.ndarray
    episode_ids: np.ndarray
    steps: np.ndarray
    train_mask: np.ndarray
    unit_vocab: tuple[str, ...]
    market_vocab: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def holdout_mask(self) -> np.ndarray:
        return ~self.train_mask


def _hash_slot(key: str, feature_dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=_HASH_PERSON).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & (1 << 63) else -1.0
    return value % feature_dim, sign


def _numeric_value(path: tuple[str, ...], value: float) -> float:
    joined = "/".join(path)
    leaf = path[-1] if path else ""

    if leaf == "step":
        return float(np.clip(value / 719.0, -1.0, 1.5))
    if leaf == "day":
        return float(np.clip(value / 30.0, -1.0, 1.5))
    if leaf == "hour":
        return float(np.clip(value / 24.0, -1.0, 1.5))
    if "/market/inventory/" in f"/{joined}/":
        return float(np.clip((value - 10_000.0) / 2_000.0, -5.0, 5.0))
    if leaf == "money":
        return math.copysign(math.log1p(abs(value)) / 12.0, value)
    if "/market/prices/" in f"/{joined}/":
        return math.log1p(max(value, 0.0)) / 8.0
    if leaf in {"x", "y"}:
        return float(np.clip(value / 9.0, -1.0, 1.0))
    return math.copysign(math.log1p(abs(value)) / 10.0, value)


def encode_observation(
    observation: dict[str, Any], feature_dim: int = DEFAULT_FEATURE_DIM
) -> np.ndarray:
    """Hash a nested Kaggle observation into a deterministic fixed-width vector."""

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    features = np.zeros(feature_dim, dtype=np.float32)

    def add(key: str, value: float) -> None:
        index, sign = _hash_slot(key, feature_dim)
        features[index] += np.float32(sign * value)

    def visit(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                if key not in _SKIPPED_OBSERVATION_KEYS:
                    visit((*path, str(key)), value[key])
            return
        if isinstance(value, list):
            add("/".join((*path, "__length__")), math.log1p(len(value)))
            for index, item in enumerate(value):
                visit((*path, str(index)), item)
            return
        if isinstance(value, bool):
            add(f"{'/'.join(path)}={str(value).lower()}", 1.0)
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            add(f"{'/'.join(path)}#numeric", _numeric_value(path, float(value)))
            add(f"{'/'.join(path)}#present", 0.25)
            return
        if value is None:
            add(f"{'/'.join(path)}=__NONE__", 0.5)
            return
        add(f"{'/'.join(path)}={value}", 1.0)

    visit((), observation)
    norm = float(np.linalg.norm(features))
    if norm > 0.0:
        features /= norm
    return features


def _canonical_token(action: Any) -> str:
    if not isinstance(action, list) or not action or not isinstance(action[0], str):
        raise ValueError(f"invalid primitive action: {action!r}")
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest must contain at least one episode")
    return manifest, episodes


def _load_replay(root: Path, episode_id: int) -> dict[str, Any]:
    path = root / f"{episode_id}.json"
    replay = json.loads(path.read_text())
    frames = replay.get("steps")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(f"episode {episode_id} has no transition frames")
    return replay


def _action_vocab(
    root: Path, episodes: list[dict[str, Any]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unit_tokens = {PASS_TOKEN}
    market_tokens: set[str] = set()
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        seat = int(episode["top1_seat"])
        replay = _load_replay(root, episode_id)
        for frame in replay["steps"][1:]:
            action = frame[seat].get("action")
            if not isinstance(action, dict):
                raise ValueError(f"episode {episode_id} contains a missing action")
            unit_tokens.add(_canonical_token(action.get("farmer") or ["PASS"]))
            unit_tokens.update(_canonical_token(hand) for hand in (action.get("hands") or []))
            market_tokens.update(_canonical_token(order) for order in (action.get("market") or []))

    ordered_units = (PASS_TOKEN, *sorted(unit_tokens - {PASS_TOKEN}))
    ordered_market = (STOP_TOKEN, *sorted(market_tokens))
    return ordered_units, ordered_market


def build_replay_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    holdout_episodes: int = 2,
) -> dict[str, Any]:
    """Build a compressed dataset with newest episodes held out in full."""

    manifest_path = manifest_path.resolve()
    manifest, episodes = _load_manifest(manifest_path)
    if not 0 < holdout_episodes < len(episodes):
        raise ValueError("holdout_episodes must leave at least one training episode")

    root = manifest_path.parent
    unit_vocab, market_vocab = _action_vocab(root, episodes)
    unit_to_id = {token: index for index, token in enumerate(unit_vocab)}
    market_to_id = {token: index for index, token in enumerate(market_vocab)}
    holdout_ids = {int(episode["episode_id"]) for episode in episodes[:holdout_episodes]}

    feature_rows: list[np.ndarray] = []
    farmer_rows: list[int] = []
    hand_rows: list[np.ndarray] = []
    hand_masks: list[np.ndarray] = []
    market_rows: list[np.ndarray] = []
    market_masks: list[np.ndarray] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []
    train_rows: list[bool] = []

    for episode in episodes:
        episode_id = int(episode["episode_id"])
        seat = int(episode["top1_seat"])
        if seat not in (0, 1):
            raise ValueError(f"episode {episode_id} has invalid top1_seat={seat}")
        replay = _load_replay(root, episode_id)
        frames = replay["steps"]

        for frame_index in range(1, len(frames)):
            previous_frame = frames[frame_index - 1]
            action_frame = frames[frame_index]
            observation = dict(previous_frame[seat]["observation"])
            shared_step = int(previous_frame[0]["observation"].get("step", frame_index - 1))
            observation.setdefault("step", shared_step)
            observation.pop("remainingOverageTime", None)

            action = action_frame[seat].get("action")
            if not isinstance(action, dict):
                raise ValueError(f"episode {episode_id} frame {frame_index} contains no action")
            farmer = _canonical_token(action.get("farmer") or ["PASS"])
            hands = action.get("hands") or []
            market = action.get("market") or []
            if len(hands) > MAX_HANDS:
                raise ValueError(f"episode {episode_id} frame {frame_index} has {len(hands)} hands")
            if len(market) > MAX_MARKET_ORDERS:
                raise ValueError(
                    f"episode {episode_id} frame {frame_index} has {len(market)} orders"
                )

            hand_labels = np.zeros(MAX_HANDS, dtype=np.int16)
            hand_mask = np.zeros(MAX_HANDS, dtype=np.bool_)
            for index, hand in enumerate(hands):
                hand_labels[index] = unit_to_id[_canonical_token(hand)]
                hand_mask[index] = True

            market_labels = np.zeros(MAX_MARKET_ORDERS, dtype=np.int16)
            market_mask = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
            for index, order in enumerate(market):
                market_labels[index] = market_to_id[_canonical_token(order)]
                market_mask[index] = True
            if len(market) < MAX_MARKET_ORDERS:
                market_mask[len(market)] = True

            feature_rows.append(encode_observation(observation, feature_dim))
            farmer_rows.append(unit_to_id[farmer])
            hand_rows.append(hand_labels)
            hand_masks.append(hand_mask)
            market_rows.append(market_labels)
            market_masks.append(market_mask)
            episode_rows.append(episode_id)
            step_rows.append(shared_step)
            train_rows.append(episode_id not in holdout_ids)

    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": source_hash,
        "feature_dim": feature_dim,
        "max_hands": MAX_HANDS,
        "max_market_orders": MAX_MARKET_ORDERS,
        "episode_ids": [int(episode["episode_id"]) for episode in episodes],
        "holdout_episode_ids": sorted(holdout_ids),
        "top1_team": (manifest.get("leader") or {}).get("team_name"),
    }

    dataset = ReplayDataset(
        features=np.stack(feature_rows),
        farmer_labels=np.asarray(farmer_rows, dtype=np.int16),
        hand_labels=np.stack(hand_rows),
        hand_mask=np.stack(hand_masks),
        market_labels=np.stack(market_rows),
        market_mask=np.stack(market_masks),
        episode_ids=np.asarray(episode_rows, dtype=np.int64),
        steps=np.asarray(step_rows, dtype=np.int16),
        train_mask=np.asarray(train_rows, dtype=np.bool_),
        unit_vocab=unit_vocab,
        market_vocab=market_vocab,
        metadata=metadata,
    )
    _save_replay_dataset(output_path, dataset)
    return {
        "output": str(output_path),
        "samples": len(dataset.features),
        "train_samples": int(dataset.train_mask.sum()),
        "holdout_samples": int(dataset.holdout_mask.sum()),
        "episodes": len(episodes),
        "holdout_episode_ids": sorted(holdout_ids),
        "feature_dim": feature_dim,
        "unit_vocab_size": len(unit_vocab),
        "market_vocab_size": len(market_vocab),
    }


def _save_replay_dataset(path: Path, dataset: ReplayDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features,
        farmer_labels=dataset.farmer_labels,
        hand_labels=dataset.hand_labels,
        hand_mask=dataset.hand_mask,
        market_labels=dataset.market_labels,
        market_mask=dataset.market_mask,
        episode_ids=dataset.episode_ids,
        steps=dataset.steps,
        train_mask=dataset.train_mask,
        unit_vocab=np.asarray(dataset.unit_vocab),
        market_vocab=np.asarray(dataset.market_vocab),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True)),
    )


def load_replay_dataset(path: Path) -> ReplayDataset:
    """Load and validate the compressed replay dataset."""

    with np.load(path, allow_pickle=False) as payload:
        dataset = ReplayDataset(
            features=payload["features"],
            farmer_labels=payload["farmer_labels"],
            hand_labels=payload["hand_labels"],
            hand_mask=payload["hand_mask"],
            market_labels=payload["market_labels"],
            market_mask=payload["market_mask"],
            episode_ids=payload["episode_ids"],
            steps=payload["steps"],
            train_mask=payload["train_mask"],
            unit_vocab=tuple(str(value) for value in payload["unit_vocab"]),
            market_vocab=tuple(str(value) for value in payload["market_vocab"]),
            metadata=json.loads(str(payload["metadata_json"])),
        )

    samples = len(dataset.features)
    expected_first_dimensions = {
        "farmer_labels": len(dataset.farmer_labels),
        "hand_labels": len(dataset.hand_labels),
        "hand_mask": len(dataset.hand_mask),
        "market_labels": len(dataset.market_labels),
        "market_mask": len(dataset.market_mask),
        "episode_ids": len(dataset.episode_ids),
        "steps": len(dataset.steps),
        "train_mask": len(dataset.train_mask),
    }
    mismatched = {name: size for name, size in expected_first_dimensions.items() if size != samples}
    if mismatched:
        raise ValueError(f"dataset array length mismatch: {mismatched}")
    if dataset.hand_labels.shape[1:] != (MAX_HANDS,):
        raise ValueError("dataset hand label shape mismatch")
    if dataset.market_labels.shape[1:] != (MAX_MARKET_ORDERS,):
        raise ValueError("dataset market label shape mismatch")
    return dataset
