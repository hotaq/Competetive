"""Spatial, perspective-normalized replay tensors for recurrent policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.imitation.replay_dataset import (
    MAX_HANDS,
    _load_manifest,
    _load_replay,
    load_replay_dataset,
)

BOARD_SIZE = 10
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
SHED_ITEMS = (*PRODUCTS, *ANIMALS)
SHOPS = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
SPATIAL_CHANNELS = 20
GLOBAL_FEATURES = 68


@dataclass(frozen=True)
class SpatialReplayDataset:
    spatial: np.ndarray
    global_features: np.ndarray
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


def _scaled(value: Any, scale: float) -> float:
    numeric = max(float(value or 0.0), 0.0)
    return float(np.clip(math.log1p(numeric) / math.log1p(scale), 0.0, 2.0))


def encode_spatial_observation(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Encode one observation without leaking the opponent's private state."""
    player = int(observation["player"])
    day = int(observation.get("day", 0))
    farms = observation["farms"]
    perspective = (player, 1 - player)
    spatial = np.zeros((2, BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS), dtype=np.float32)

    crop_channels = {crop: 2 + index for index, crop in enumerate(CROPS)}
    animal_channels = {animal: 9 + index for index, animal in enumerate(ANIMALS)}
    for board_index, farm_index in enumerate(perspective):
        farm = farms[farm_index]
        tiles = farm["tiles"]
        if len(tiles) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in tiles):
            raise ValueError("spatial encoder requires a 10x10 board")
        board = spatial[board_index]
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED":
                    continue
                board[y, x, 0] = 1.0
                if not isinstance(tile, dict):
                    continue
                kind = str(tile.get("kind", ""))
                if kind == "WEED":
                    board[y, x, 1] = 1.0
                crop = tile.get("crop")
                if crop in crop_channels:
                    board[y, x, crop_channels[crop]] = 1.0
                if kind == "COOP":
                    board[y, x, 7] = 1.0
                elif kind == "PASTURE":
                    board[y, x, 8] = 1.0
                animal = tile.get("animal")
                if animal in animal_channels:
                    board[y, x, animal_channels[animal]] = 1.0
                board[y, x, 12] = float(tile.get("watered_today") is True)
                board[y, x, 13] = _scaled(tile.get("yield_units", 0), 16.0)
                board[y, x, 14] = float(int(tile.get("fertilized_until_day", -1)) >= day)
                board[y, x, 15] = float(tile.get("fed_today") is True)
                board[y, x, 16] = float(tile.get("cared_today") is True)
                board[y, x, 17] = float(tile.get("fertilizer_available") is True)

        farmer_x, farmer_y = farm["farmer"]
        board[int(farmer_y), int(farmer_x), 18] = 1.0
        for hand_x, hand_y in farm.get("hands", []):
            board[int(hand_y), int(hand_x), 19] += 1.0

    step = int(observation.get("step", day * 24 + int(observation.get("hour", 0))))
    hour = int(observation.get("hour", step % 24))
    angle = 2.0 * math.pi * hour / 24.0
    global_values = [
        step / 719.0,
        day / 30.0,
        hour / 23.0,
        math.sin(angle),
        math.cos(angle),
    ]
    for farm_index in perspective:
        farm = farms[farm_index]
        global_values.extend(
            (
                _scaled(farm.get("money", 0), 200_000.0),
                len(farm.get("hands", [])) / MAX_HANDS,
                int(farm.get("hires_today", 0)) / MAX_HANDS,
                len(farm.get("unlocked_quadrants", [])) / 4.0,
            )
        )

    private = observation["private"]
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    inventories = private.get("inventories", [])
    carried = {
        item: sum(int(inventory.get(item, 0)) for inventory in inventories)
        for item in SHED_ITEMS
    }
    global_values.extend(_scaled(shed.get(item, 0), 100.0) for item in SHED_ITEMS)
    global_values.extend(_scaled(seeds.get(crop, 0), 100.0) for crop in CROPS)
    global_values.extend(_scaled(carried[item], 100.0) for item in SHED_ITEMS)

    market = observation["market"]
    market_inventory = market.get("inventory", {})
    prices = market.get("prices", {})
    global_values.extend(_scaled(market_inventory.get(item, 0), 20_000.0) for item in PRODUCTS)
    global_values.extend(_scaled(prices.get(item, 0), 10_000.0) for item in PRODUCTS)
    unlocked_shops = set(observation.get("town", {}).get("unlocked_shops", []))
    global_values.extend(float(shop in unlocked_shops) for shop in SHOPS)

    global_features = np.asarray(global_values, dtype=np.float32)
    if global_features.shape != (GLOBAL_FEATURES,):
        raise AssertionError(
            f"expected {GLOBAL_FEATURES} global features, got {global_features.shape}"
        )
    return spatial, global_features


def build_spatial_replay_dataset(
    manifest_path: Path,
    base_dataset_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Add spatial observations to an already-aligned replay label dataset."""
    manifest_path = manifest_path.resolve()
    base = load_replay_dataset(base_dataset_path)
    manifest, episodes = _load_manifest(manifest_path)
    root = manifest_path.parent
    spatial_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    episode_rows: list[int] = []
    step_rows: list[int] = []

    for episode in episodes:
        episode_id = int(episode["episode_id"])
        seat = int(episode["top1_seat"])
        replay = _load_replay(root, episode_id)
        frames = replay["steps"]
        for frame_index in range(1, len(frames)):
            previous_frame = frames[frame_index - 1]
            observation = dict(previous_frame[seat]["observation"])
            shared_step = int(previous_frame[0]["observation"].get("step", frame_index - 1))
            observation.setdefault("step", shared_step)
            observation.pop("remainingOverageTime", None)
            spatial, global_features = encode_spatial_observation(observation)
            spatial_rows.append(spatial)
            global_rows.append(global_features)
            episode_rows.append(episode_id)
            step_rows.append(shared_step)

    episode_ids = np.asarray(episode_rows, dtype=np.int64)
    steps = np.asarray(step_rows, dtype=np.int16)
    if not np.array_equal(episode_ids, base.episode_ids) or not np.array_equal(steps, base.steps):
        raise ValueError("spatial observations do not align with the base replay labels")

    metadata = {
        **base.metadata,
        "schema_version": 2,
        "encoder": "perspective-spatial-v1",
        "spatial_shape": [2, BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS],
        "global_features": GLOBAL_FEATURES,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    dataset = SpatialReplayDataset(
        spatial=np.stack(spatial_rows).astype(np.float16),
        global_features=np.stack(global_rows).astype(np.float16),
        farmer_labels=base.farmer_labels,
        hand_labels=base.hand_labels,
        hand_mask=base.hand_mask,
        market_labels=base.market_labels,
        market_mask=base.market_mask,
        episode_ids=base.episode_ids,
        steps=base.steps,
        train_mask=base.train_mask,
        unit_vocab=base.unit_vocab,
        market_vocab=base.market_vocab,
        metadata=metadata,
    )
    _save_spatial_dataset(output_path, dataset)
    return {
        "output": str(output_path),
        "samples": len(dataset.spatial),
        "train_samples": int(dataset.train_mask.sum()),
        "holdout_samples": int(dataset.holdout_mask.sum()),
        "spatial_shape": list(dataset.spatial.shape[1:]),
        "global_features": GLOBAL_FEATURES,
        "top1_team": (manifest.get("leader") or {}).get("team_name"),
    }


def _save_spatial_dataset(path: Path, dataset: SpatialReplayDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        spatial=dataset.spatial,
        global_features=dataset.global_features,
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


def save_spatial_replay_dataset(path: Path, dataset: SpatialReplayDataset) -> None:
    """Persist a spatial dataset produced by replay or a live teacher."""
    _save_spatial_dataset(path, dataset)


def align_spatial_dataset_vocabulary(
    dataset: SpatialReplayDataset,
    target_unit_vocab: tuple[str, ...],
    target_market_vocab: tuple[str, ...],
) -> SpatialReplayDataset:
    """Remap labels into an existing checkpoint vocabulary without dropping data."""
    target_unit_ids = {token: index for index, token in enumerate(target_unit_vocab)}
    target_market_ids = {
        token: index for index, token in enumerate(target_market_vocab)
    }
    missing_units = sorted(set(dataset.unit_vocab) - set(target_unit_vocab))
    missing_market = sorted(set(dataset.market_vocab) - set(target_market_vocab))
    if missing_units or missing_market:
        raise ValueError(
            "target vocabulary cannot represent dataset labels: "
            f"units={missing_units}, market={missing_market}"
        )
    unit_remap = np.asarray(
        [target_unit_ids[token] for token in dataset.unit_vocab],
        dtype=np.int16,
    )
    market_remap = np.asarray(
        [target_market_ids[token] for token in dataset.market_vocab],
        dtype=np.int16,
    )
    metadata = {
        **dataset.metadata,
        "vocabulary_alignment": {
            "source_unit_vocab_size": len(dataset.unit_vocab),
            "source_market_vocab_size": len(dataset.market_vocab),
            "target_unit_vocab_size": len(target_unit_vocab),
            "target_market_vocab_size": len(target_market_vocab),
        },
    }
    return SpatialReplayDataset(
        spatial=dataset.spatial,
        global_features=dataset.global_features,
        farmer_labels=unit_remap[dataset.farmer_labels],
        hand_labels=unit_remap[dataset.hand_labels],
        hand_mask=dataset.hand_mask,
        market_labels=market_remap[dataset.market_labels],
        market_mask=dataset.market_mask,
        episode_ids=dataset.episode_ids,
        steps=dataset.steps,
        train_mask=dataset.train_mask,
        unit_vocab=target_unit_vocab,
        market_vocab=target_market_vocab,
        metadata=metadata,
    )


def load_spatial_replay_dataset(path: Path) -> SpatialReplayDataset:
    with np.load(path, allow_pickle=False) as payload:
        dataset = SpatialReplayDataset(
            spatial=payload["spatial"],
            global_features=payload["global_features"],
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
    if dataset.spatial.shape[1:] != (2, BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS):
        raise ValueError(f"unexpected spatial shape {dataset.spatial.shape}")
    if dataset.global_features.shape[1:] != (GLOBAL_FEATURES,):
        raise ValueError(f"unexpected global feature shape {dataset.global_features.shape}")
    return dataset
