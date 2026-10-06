"""Compact spatial-GRU behavior policy with exact legality-masked decoding."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kaggriculture_rl.imitation.bc import PolicyPrediction, _masked_argmax
from kaggriculture_rl.imitation.replay_dataset import MAX_HANDS, MAX_MARKET_ORDERS, STOP_TOKEN
from kaggriculture_rl.imitation.spatial_dataset import (
    GLOBAL_FEATURES,
    SPATIAL_CHANNELS,
    SpatialReplayDataset,
    encode_spatial_observation,
    load_spatial_replay_dataset,
)

type RecurrentParameters = dict[str, jax.Array]


@dataclass(frozen=True)
class SpatialRecurrentConfig:
    unit_classes: int
    market_classes: int
    conv_channels: int = 32
    embedding_dim: int = 128
    hidden_dim: int = 256
    spatial_channels: int = SPATIAL_CHANNELS
    global_features: int = GLOBAL_FEATURES
    max_hands: int = MAX_HANDS
    max_market_orders: int = MAX_MARKET_ORDERS


class SequenceBatch(NamedTuple):
    spatial: np.ndarray
    global_features: np.ndarray
    farmer_labels: np.ndarray
    hand_labels: np.ndarray
    hand_mask: np.ndarray
    market_labels: np.ndarray
    market_mask: np.ndarray
    valid_mask: np.ndarray


def _glorot(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    fan_in = math.prod(shape[:-1])
    fan_out = shape[-1]
    scale = math.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def initialize_parameters(
    config: SpatialRecurrentConfig,
    *,
    seed: int,
) -> RecurrentParameters:
    keys = iter(jax.random.split(jax.random.key(seed), 16))
    pooled_boards = config.conv_channels * 4
    recurrent_input = config.embedding_dim * 2
    return {
        "conv1_weight": _glorot(
            next(keys), (3, 3, config.spatial_channels, config.conv_channels)
        ),
        "conv1_bias": jnp.zeros(config.conv_channels, dtype=jnp.float32),
        "conv2_weight": _glorot(
            next(keys), (3, 3, config.conv_channels, config.conv_channels)
        ),
        "conv2_bias": jnp.zeros(config.conv_channels, dtype=jnp.float32),
        "board_weight": _glorot(next(keys), (pooled_boards, config.embedding_dim)),
        "board_bias": jnp.zeros(config.embedding_dim, dtype=jnp.float32),
        "global_weight": _glorot(
            next(keys), (config.global_features, config.embedding_dim)
        ),
        "global_bias": jnp.zeros(config.embedding_dim, dtype=jnp.float32),
        "gru_input_weight": _glorot(
            next(keys), (recurrent_input, config.hidden_dim * 3)
        ),
        "gru_hidden_weight": _glorot(
            next(keys), (config.hidden_dim, config.hidden_dim * 3)
        ),
        "gru_bias": jnp.zeros(config.hidden_dim * 3, dtype=jnp.float32),
        "farmer_weight": _glorot(next(keys), (config.hidden_dim, config.unit_classes)),
        "farmer_bias": jnp.zeros(config.unit_classes, dtype=jnp.float32),
        "hand_weight": _glorot(
            next(keys), (config.hidden_dim, config.max_hands * config.unit_classes)
        ),
        "hand_bias": jnp.zeros(
            config.max_hands * config.unit_classes, dtype=jnp.float32
        ),
        "market_weight": _glorot(
            next(keys),
            (config.hidden_dim, config.max_market_orders * config.market_classes),
        ),
        "market_bias": jnp.zeros(
            config.max_market_orders * config.market_classes, dtype=jnp.float32
        ),
    }


def _conv2d(inputs: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    output = jax.lax.conv_general_dilated(
        inputs,
        weight,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return output + bias


def encode_frames(
    parameters: RecurrentParameters,
    spatial: jax.Array,
    global_features: jax.Array,
) -> jax.Array:
    batch, sequence, boards, height, width, channels = spatial.shape
    flat_spatial = spatial.reshape(batch * sequence * boards, height, width, channels)
    convolution = jax.nn.gelu(
        _conv2d(flat_spatial, parameters["conv1_weight"], parameters["conv1_bias"])
    )
    convolution = jax.nn.gelu(
        _conv2d(convolution, parameters["conv2_weight"], parameters["conv2_bias"])
    )
    mean_pool = convolution.mean(axis=(1, 2))
    max_pool = convolution.max(axis=(1, 2))
    pooled = jnp.concatenate((mean_pool, max_pool), axis=-1)
    pooled = pooled.reshape(batch, sequence, boards * pooled.shape[-1])
    board_embedding = jax.nn.gelu(
        pooled @ parameters["board_weight"] + parameters["board_bias"]
    )
    global_embedding = jax.nn.gelu(
        global_features @ parameters["global_weight"] + parameters["global_bias"]
    )
    return jnp.concatenate((board_embedding, global_embedding), axis=-1)


def recurrent_forward(
    parameters: RecurrentParameters,
    spatial: jax.Array,
    global_features: jax.Array,
    initial_hidden: jax.Array,
    config: SpatialRecurrentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    embeddings = encode_frames(parameters, spatial, global_features)

    def step(hidden: jax.Array, embedding: jax.Array) -> tuple[jax.Array, jax.Array]:
        input_projection = embedding @ parameters["gru_input_weight"]
        hidden_projection = hidden @ parameters["gru_hidden_weight"]
        input_reset, input_update, input_candidate = jnp.split(input_projection, 3, axis=-1)
        hidden_reset, hidden_update, hidden_candidate = jnp.split(
            hidden_projection, 3, axis=-1
        )
        bias_reset, bias_update, bias_candidate = jnp.split(
            parameters["gru_bias"], 3, axis=-1
        )
        reset = jax.nn.sigmoid(input_reset + hidden_reset + bias_reset)
        update = jax.nn.sigmoid(input_update + hidden_update + bias_update)
        candidate = jnp.tanh(input_candidate + reset * hidden_candidate + bias_candidate)
        next_hidden = update * hidden + (1.0 - update) * candidate
        return next_hidden, next_hidden

    final_hidden, hidden_sequence = jax.lax.scan(
        step,
        initial_hidden,
        jnp.swapaxes(embeddings, 0, 1),
    )
    hidden_sequence = jnp.swapaxes(hidden_sequence, 0, 1)
    farmer_logits = (
        hidden_sequence @ parameters["farmer_weight"] + parameters["farmer_bias"]
    )
    hand_logits = hidden_sequence @ parameters["hand_weight"] + parameters["hand_bias"]
    hand_logits = hand_logits.reshape(
        *hidden_sequence.shape[:2], config.max_hands, config.unit_classes
    )
    market_logits = (
        hidden_sequence @ parameters["market_weight"] + parameters["market_bias"]
    )
    market_logits = market_logits.reshape(
        *hidden_sequence.shape[:2], config.max_market_orders, config.market_classes
    )
    return final_hidden, farmer_logits, hand_logits, market_logits


def _cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), labels[..., None], axis=-1)[..., 0]


def loss_and_metrics(
    parameters: RecurrentParameters,
    batch: SequenceBatch,
    config: SpatialRecurrentConfig,
) -> tuple[jax.Array, jax.Array]:
    batch_size = batch.spatial.shape[0]
    hidden = jnp.zeros((batch_size, config.hidden_dim), dtype=jnp.float32)
    _, farmer_logits, hand_logits, market_logits = recurrent_forward(
        parameters,
        jnp.asarray(batch.spatial, dtype=jnp.float32),
        jnp.asarray(batch.global_features, dtype=jnp.float32),
        hidden,
        config,
    )
    valid = jnp.asarray(batch.valid_mask, dtype=jnp.float32)
    hand_mask = jnp.asarray(batch.hand_mask, dtype=jnp.float32) * valid[..., None]
    market_mask = jnp.asarray(batch.market_mask, dtype=jnp.float32) * valid[..., None]
    farmer_loss = (_cross_entropy(farmer_logits, batch.farmer_labels) * valid).sum() / jnp.maximum(
        valid.sum(), 1.0
    )
    hand_loss = (_cross_entropy(hand_logits, batch.hand_labels) * hand_mask).sum() / jnp.maximum(
        hand_mask.sum(), 1.0
    )
    market_loss = (
        _cross_entropy(market_logits, batch.market_labels) * market_mask
    ).sum() / jnp.maximum(market_mask.sum(), 1.0)
    total = farmer_loss + hand_loss + market_loss

    farmer_correct = (jnp.argmax(farmer_logits, axis=-1) == batch.farmer_labels) * valid
    hand_correct = (jnp.argmax(hand_logits, axis=-1) == batch.hand_labels) * hand_mask
    market_correct = (jnp.argmax(market_logits, axis=-1) == batch.market_labels) * market_mask
    metrics = jnp.asarray(
        [
            farmer_loss,
            hand_loss,
            market_loss,
            farmer_correct.sum() / jnp.maximum(valid.sum(), 1.0),
            hand_correct.sum() / jnp.maximum(hand_mask.sum(), 1.0),
            market_correct.sum() / jnp.maximum(market_mask.sum(), 1.0),
        ]
    )
    return total, metrics


def make_sequence_chunks(
    dataset: SpatialReplayDataset,
    selected: np.ndarray,
    *,
    sequence_length: int,
) -> SequenceBatch:
    chunks: list[np.ndarray] = []
    for episode_id in dict.fromkeys(dataset.episode_ids[selected].tolist()):
        indices = np.flatnonzero(selected & (dataset.episode_ids == episode_id))
        for start in range(0, len(indices), sequence_length):
            chunks.append(indices[start : start + sequence_length])
    if not chunks:
        raise ValueError("selection contains no replay samples")

    count = len(chunks)
    spatial = np.zeros((count, sequence_length, *dataset.spatial.shape[1:]), dtype=np.float16)
    global_features = np.zeros(
        (count, sequence_length, dataset.global_features.shape[-1]), dtype=np.float16
    )
    farmer_labels = np.zeros((count, sequence_length), dtype=np.int16)
    max_hands = dataset.hand_labels.shape[1]
    hand_labels = np.zeros((count, sequence_length, max_hands), dtype=np.int16)
    hand_mask = np.zeros((count, sequence_length, max_hands), dtype=np.bool_)
    market_labels = np.zeros((count, sequence_length, MAX_MARKET_ORDERS), dtype=np.int16)
    market_mask = np.zeros((count, sequence_length, MAX_MARKET_ORDERS), dtype=np.bool_)
    valid_mask = np.zeros((count, sequence_length), dtype=np.bool_)
    for chunk_index, indices in enumerate(chunks):
        length = len(indices)
        target = slice(0, length)
        spatial[chunk_index, target] = dataset.spatial[indices]
        global_features[chunk_index, target] = dataset.global_features[indices]
        farmer_labels[chunk_index, target] = dataset.farmer_labels[indices]
        hand_labels[chunk_index, target] = dataset.hand_labels[indices]
        hand_mask[chunk_index, target] = dataset.hand_mask[indices]
        market_labels[chunk_index, target] = dataset.market_labels[indices]
        market_mask[chunk_index, target] = dataset.market_mask[indices]
        valid_mask[chunk_index, target] = True
    return SequenceBatch(
        spatial=spatial,
        global_features=global_features,
        farmer_labels=farmer_labels,
        hand_labels=hand_labels,
        hand_mask=hand_mask,
        market_labels=market_labels,
        market_mask=market_mask,
        valid_mask=valid_mask,
    )


def _slice_batch(batch: SequenceBatch, indices: np.ndarray) -> SequenceBatch:
    return SequenceBatch(**{field: getattr(batch, field)[indices] for field in batch._fields})


def train_spatial_recurrent(
    dataset_path: Path,
    output_path: Path,
    *,
    epochs: int = 30,
    batch_size: int = 8,
    sequence_length: int = 48,
    conv_channels: int = 32,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 1729,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    dataset = load_spatial_replay_dataset(dataset_path)
    previous_epochs = 0
    previous_history: list[dict[str, Any]] = []
    continuation: dict[str, Any] | None = None
    if initial_checkpoint is None:
        config = SpatialRecurrentConfig(
            unit_classes=len(dataset.unit_vocab),
            market_classes=len(dataset.market_vocab),
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_hands=dataset.hand_labels.shape[1],
        )
        parameters = initialize_parameters(config, seed=seed)
    else:
        initial_checkpoint = initial_checkpoint.resolve()
        (
            parameters,
            config,
            initial_unit_vocab,
            initial_market_vocab,
            initial_metadata,
        ) = load_checkpoint(initial_checkpoint)
        expected = {
            "unit_classes": len(dataset.unit_vocab),
            "market_classes": len(dataset.market_vocab),
            "max_hands": dataset.hand_labels.shape[1],
            "max_market_orders": dataset.market_labels.shape[1],
            "spatial_channels": dataset.spatial.shape[-1],
            "global_features": dataset.global_features.shape[-1],
        }
        mismatches = {
            name: (getattr(config, name), value)
            for name, value in expected.items()
            if getattr(config, name) != value
        }
        if mismatches:
            raise ValueError(f"initial checkpoint shape mismatch: {mismatches}")
        if initial_unit_vocab != dataset.unit_vocab:
            raise ValueError("initial checkpoint unit vocabulary does not match dataset")
        if initial_market_vocab != dataset.market_vocab:
            raise ValueError("initial checkpoint market vocabulary does not match dataset")
        previous_history = list(initial_metadata.get("history", []))
        previous_epochs = int(initial_metadata.get("epochs", len(previous_history)))
        continuation = {
            "initial_checkpoint": str(initial_checkpoint),
            "initial_checkpoint_sha256": hashlib.sha256(
                initial_checkpoint.read_bytes()
            ).hexdigest(),
            "initial_epochs": previous_epochs,
        }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=1e-5),
    )
    optimizer_state = optimizer.init(parameters)
    train_chunks = make_sequence_chunks(
        dataset, dataset.train_mask, sequence_length=sequence_length
    )
    holdout_chunks = make_sequence_chunks(
        dataset, dataset.holdout_mask, sequence_length=sequence_length
    )

    @jax.jit
    def update(
        current: RecurrentParameters,
        state: optax.OptState,
        batch: SequenceBatch,
    ) -> tuple[RecurrentParameters, optax.OptState, jax.Array, jax.Array]:
        (loss, metrics), gradients = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            current, batch, config
        )
        updates, next_state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, loss, metrics

    evaluate = jax.jit(lambda current, batch: loss_and_metrics(current, batch, config))
    random = np.random.default_rng(seed)
    history = previous_history
    for epoch_offset in range(epochs):
        order = random.permutation(len(train_chunks.spatial))
        train_metrics: list[np.ndarray] = []
        for start in range(0, len(order), batch_size):
            batch = _slice_batch(train_chunks, order[start : start + batch_size])
            parameters, optimizer_state, loss, metrics = update(
                parameters, optimizer_state, batch
            )
            train_metrics.append(np.asarray(jax.device_get(jnp.r_[loss, metrics])))
        holdout_loss, holdout_metrics = evaluate(parameters, holdout_chunks)
        history.append(
            {
                "epoch": previous_epochs + epoch_offset + 1,
                "train": np.mean(train_metrics, axis=0).tolist(),
                "holdout": np.asarray(
                    jax.device_get(jnp.r_[holdout_loss, holdout_metrics])
                ).tolist(),
            }
        )

    metadata = {
        "model": "spatial-gru-bc-v1",
        "dataset": str(dataset_path),
        "dataset_metadata": dataset.metadata,
        "epochs": previous_epochs + epochs,
        "continued_epochs": epochs,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "seed": seed,
        "metric_order": [
            "loss",
            "farmer_loss",
            "hand_loss",
            "market_loss",
            "farmer_accuracy",
            "hand_accuracy",
            "market_accuracy",
        ],
        "history": history,
    }
    if continuation is not None:
        metadata["continuation"] = continuation
    save_checkpoint(output_path, parameters, config, dataset, metadata)
    return {"checkpoint": str(output_path), "config": asdict(config), **metadata}


def save_checkpoint(
    path: Path,
    parameters: RecurrentParameters,
    config: SpatialRecurrentConfig,
    dataset: SpatialReplayDataset,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        f"parameter_{key}": np.asarray(jax.device_get(value))
        for key, value in parameters.items()
    }
    arrays.update(
        {
            "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
            "unit_vocab": np.asarray(dataset.unit_vocab),
            "market_vocab": np.asarray(dataset.market_vocab),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    )
    np.savez_compressed(path, **arrays)


def load_checkpoint(
    path: Path,
) -> tuple[
    RecurrentParameters,
    SpatialRecurrentConfig,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
]:
    with np.load(path, allow_pickle=False) as payload:
        parameters = {
            key.removeprefix("parameter_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("parameter_")
        }
        config = SpatialRecurrentConfig(**json.loads(str(payload["config_json"])))
        unit_vocab = tuple(str(value) for value in payload["unit_vocab"])
        market_vocab = tuple(str(value) for value in payload["market_vocab"])
        metadata = json.loads(str(payload["metadata_json"]))
    return parameters, config, unit_vocab, market_vocab, metadata


class SpatialRecurrentPolicy:
    """Stateful GRU policy; instantiate or reset once per game."""

    def __init__(self, checkpoint: Path) -> None:
        (
            self._parameters,
            self._config,
            self._unit_vocab,
            self._market_vocab,
            self.metadata,
        ) = load_checkpoint(checkpoint)
        self._recurrent_step = jax.jit(
            lambda parameters, spatial, global_features, hidden: recurrent_forward(
                parameters,
                spatial,
                global_features,
                hidden,
                self._config,
            )
        )
        self.reset()

    def reset(self) -> None:
        self._hidden = np.zeros((1, self._config.hidden_dim), dtype=np.float32)
        self._last_step = -1

    def predict_with_confidence(
        self,
        engine: Any,
        observation: dict[str, Any],
    ) -> PolicyPrediction:
        hidden, farmer_logits, hand_logits, market_logits = self._advance(observation)
        del hidden
        return self._decode_masked(
            engine,
            observation,
            farmer_logits,
            hand_logits,
            market_logits,
        )

    def encode_hidden(self, observation: dict[str, Any]) -> np.ndarray:
        """Advance the recurrent state and return its latent representation."""
        hidden, _, _, _ = self._advance(observation)
        return hidden

    def _advance(
        self,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        step = int(observation.get("step", 0))
        if step == 0 or step <= self._last_step:
            self.reset()
        spatial, global_features = encode_spatial_observation(observation)
        hidden, farmer_logits, hand_logits, market_logits = self._recurrent_step(
            self._parameters,
            jnp.asarray(spatial[None, None, ...]),
            jnp.asarray(global_features[None, None, ...]),
            jnp.asarray(self._hidden),
        )
        self._hidden = np.asarray(jax.device_get(hidden))
        self._last_step = step
        return (
            self._hidden[0].copy(),
            np.asarray(jax.device_get(farmer_logits))[0, 0],
            np.asarray(jax.device_get(hand_logits))[0, 0],
            np.asarray(jax.device_get(market_logits))[0, 0],
        )

    def _decode_masked(
        self,
        engine: Any,
        observation: dict[str, Any],
        farmer_logits: np.ndarray,
        hand_logits: np.ndarray,
        market_logits: np.ndarray,
    ) -> PolicyPrediction:
        player = int(observation["player"])
        hands_count = len(observation["farms"][player]["hands"])
        unit_candidates = [json.loads(token) for token in self._unit_vocab]
        prefix: list[list[Any]] = []
        farmer_mask = np.asarray(
            engine.unit_action_mask(player, 0, prefix, unit_candidates), dtype=np.bool_
        )
        farmer_id, farmer_confidence = _masked_argmax(farmer_logits, farmer_mask)
        farmer = unit_candidates[farmer_id]
        prefix.append(farmer)

        hands: list[list[Any]] = []
        hand_confidences: list[float] = []
        for hand_index in range(hands_count):
            mask = np.asarray(
                engine.unit_action_mask(player, hand_index + 1, prefix, unit_candidates),
                dtype=np.bool_,
            )
            token_id, confidence = _masked_argmax(hand_logits[hand_index], mask)
            action = unit_candidates[token_id]
            hands.append(action)
            hand_confidences.append(confidence)
            prefix.append(action)

        units = {"farmer": farmer, "hands": hands, "market": []}
        stop_index = self._market_vocab.index(STOP_TOKEN)
        market_indices = [
            index for index, token in enumerate(self._market_vocab) if token != STOP_TOKEN
        ]
        market_candidates = [json.loads(self._market_vocab[index]) for index in market_indices]
        market: list[list[Any]] = []
        market_confidences: list[float] = []
        for order_index in range(self._config.max_market_orders):
            mask = np.zeros(len(self._market_vocab), dtype=np.bool_)
            mask[stop_index] = True
            mask[np.asarray(market_indices)] = engine.market_order_mask(
                player, units, market, market_candidates
            )
            token_id, confidence = _masked_argmax(market_logits[order_index], mask)
            market_confidences.append(confidence)
            if token_id == stop_index:
                break
            market.append(json.loads(self._market_vocab[token_id]))

        action = {"farmer": farmer, "hands": hands, "market": market}
        if engine.sanitize_action(player, action) != action:
            raise RuntimeError("spatial masked policy emitted an ineffective action")
        return PolicyPrediction(
            action=action,
            farmer_confidence=farmer_confidence,
            hand_confidences=tuple(hand_confidences),
            market_confidences=tuple(market_confidences),
        )

    def __call__(self, engine: Any, observation: dict[str, Any]) -> dict[str, Any]:
        return self.predict_with_confidence(engine, observation).action
