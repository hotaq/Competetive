"""JAX behavior cloning for factorized farmer, hand and market actions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kaggriculture_rl.imitation.replay_dataset import (
    MAX_HANDS,
    MAX_MARKET_ORDERS,
    STOP_TOKEN,
    ReplayDataset,
    encode_observation,
)

type Parameters = dict[str, jax.Array]


@dataclass(frozen=True)
class BCConfig:
    input_dim: int
    hidden_dim: int
    unit_classes: int
    market_classes: int
    max_hands: int = MAX_HANDS
    max_market_orders: int = MAX_MARKET_ORDERS


@dataclass(frozen=True)
class PolicyPrediction:
    action: dict[str, Any]
    farmer_confidence: float
    hand_confidences: tuple[float, ...]
    market_confidences: tuple[float, ...]


def confidence_gate(
    prediction: PolicyPrediction,
    fallback: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Keep confident BC components and use a legal executor action otherwise."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")

    farmer_accepted = prediction.farmer_confidence >= threshold
    hands = [
        predicted if confidence >= threshold else fallback["hands"][index]
        for index, (predicted, confidence) in enumerate(
            zip(
                prediction.action["hands"],
                prediction.hand_confidences,
                strict=True,
            )
        )
    ]
    accepted_hands = sum(confidence >= threshold for confidence in prediction.hand_confidences)
    market_accepted = (
        bool(prediction.market_confidences) and min(prediction.market_confidences) >= threshold
    )
    return (
        {
            "farmer": (prediction.action["farmer"] if farmer_accepted else fallback["farmer"]),
            "hands": hands,
            "market": (prediction.action["market"] if market_accepted else fallback["market"]),
        },
        {
            "farmer": int(farmer_accepted),
            "hands": accepted_hands,
            "market": int(market_accepted),
        },
    )


def _glorot(key: jax.Array, input_size: int, output_size: int) -> jax.Array:
    scale = math.sqrt(2.0 / (input_size + output_size))
    return jax.random.normal(key, (input_size, output_size), dtype=jnp.float32) * scale


def initialize_parameters(config: BCConfig, seed: int) -> Parameters:
    keys = iter(jax.random.split(jax.random.key(seed), 4))
    return {
        "input_weight": _glorot(next(keys), config.input_dim, config.hidden_dim),
        "input_bias": jnp.zeros(config.hidden_dim, dtype=jnp.float32),
        "farmer_weight": _glorot(next(keys), config.hidden_dim, config.unit_classes),
        "farmer_bias": jnp.zeros(config.unit_classes, dtype=jnp.float32),
        "hand_weight": _glorot(
            next(keys),
            config.hidden_dim,
            config.max_hands * config.unit_classes,
        ),
        "hand_bias": jnp.zeros(config.max_hands * config.unit_classes, dtype=jnp.float32),
        "market_weight": _glorot(
            next(keys),
            config.hidden_dim,
            config.max_market_orders * config.market_classes,
        ),
        "market_bias": jnp.zeros(
            config.max_market_orders * config.market_classes, dtype=jnp.float32
        ),
    }


def forward(
    parameters: Parameters, features: jax.Array, config: BCConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hidden = jax.nn.gelu(features @ parameters["input_weight"] + parameters["input_bias"])
    farmer = hidden @ parameters["farmer_weight"] + parameters["farmer_bias"]
    hands = hidden @ parameters["hand_weight"] + parameters["hand_bias"]
    market = hidden @ parameters["market_weight"] + parameters["market_bias"]
    return (
        farmer,
        hands.reshape((-1, config.max_hands, config.unit_classes)),
        market.reshape((-1, config.max_market_orders, config.market_classes)),
    )


def _masked_cross_entropy(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probabilities, labels[..., None], axis=-1).squeeze(-1)
    weights = mask.astype(jnp.float32)
    return -(selected * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def _loss(
    parameters: Parameters, batch: dict[str, jax.Array], config: BCConfig
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    farmer_logits, hand_logits, market_logits = forward(parameters, batch["features"], config)
    farmer_loss = _masked_cross_entropy(
        farmer_logits,
        batch["farmer_labels"],
        jnp.ones_like(batch["farmer_labels"], dtype=jnp.bool_),
    )
    hand_loss = _masked_cross_entropy(hand_logits, batch["hand_labels"], batch["hand_mask"])
    market_loss = _masked_cross_entropy(market_logits, batch["market_labels"], batch["market_mask"])
    total = farmer_loss + hand_loss + market_loss
    return total, (farmer_loss, hand_loss, market_loss)


def _batch(dataset: ReplayDataset, indices: np.ndarray) -> dict[str, jax.Array]:
    return {
        "features": jnp.asarray(dataset.features[indices]),
        "farmer_labels": jnp.asarray(dataset.farmer_labels[indices], dtype=jnp.int32),
        "hand_labels": jnp.asarray(dataset.hand_labels[indices], dtype=jnp.int32),
        "hand_mask": jnp.asarray(dataset.hand_mask[indices]),
        "market_labels": jnp.asarray(dataset.market_labels[indices], dtype=jnp.int32),
        "market_mask": jnp.asarray(dataset.market_mask[indices]),
    }


def evaluate_behavior_cloning(
    parameters: Parameters,
    config: BCConfig,
    dataset: ReplayDataset,
    selection: np.ndarray,
    *,
    batch_size: int = 512,
) -> dict[str, float | int]:
    indices = np.flatnonzero(selection)
    if not len(indices):
        raise ValueError("evaluation selection is empty")

    farmer_correct = 0
    farmer_total = 0
    hand_correct = 0
    hand_total = 0
    market_correct = 0
    market_total = 0
    hands_exact = 0
    market_exact = 0
    joint_exact = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        farmer_logits, hand_logits, market_logits = forward(
            parameters, jnp.asarray(dataset.features[batch_indices]), config
        )
        farmer_prediction = np.asarray(jax.device_get(jnp.argmax(farmer_logits, -1)))
        hand_prediction = np.asarray(jax.device_get(jnp.argmax(hand_logits, -1)))
        market_prediction = np.asarray(jax.device_get(jnp.argmax(market_logits, -1)))

        farmer_labels = dataset.farmer_labels[batch_indices]
        hand_labels = dataset.hand_labels[batch_indices]
        hand_mask = dataset.hand_mask[batch_indices]
        market_labels = dataset.market_labels[batch_indices]
        market_mask = dataset.market_mask[batch_indices]

        farmer_matches = farmer_prediction == farmer_labels
        hand_matches = (hand_prediction == hand_labels) | ~hand_mask
        market_matches = (market_prediction == market_labels) | ~market_mask
        sample_hands_exact = hand_matches.all(axis=1)
        sample_market_exact = market_matches.all(axis=1)

        farmer_correct += int(farmer_matches.sum())
        farmer_total += len(batch_indices)
        hand_correct += int(((hand_prediction == hand_labels) & hand_mask).sum())
        hand_total += int(hand_mask.sum())
        market_correct += int(((market_prediction == market_labels) & market_mask).sum())
        market_total += int(market_mask.sum())
        hands_exact += int(sample_hands_exact.sum())
        market_exact += int(sample_market_exact.sum())
        joint_exact += int((farmer_matches & sample_hands_exact & sample_market_exact).sum())

    samples = len(indices)
    return {
        "samples": samples,
        "farmer_accuracy": farmer_correct / farmer_total,
        "hand_accuracy": hand_correct / max(hand_total, 1),
        "market_accuracy": market_correct / max(market_total, 1),
        "hands_exact_accuracy": hands_exact / samples,
        "market_exact_accuracy": market_exact / samples,
        "joint_exact_accuracy": joint_exact / samples,
    }


def train_behavior_cloning(
    dataset: ReplayDataset,
    *,
    hidden_dim: int = 256,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    seed: int = 1729,
) -> tuple[Parameters, BCConfig, dict[str, Any]]:
    """Train a compact MLP and return weights, config and held-out metrics."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    train_indices = np.flatnonzero(dataset.train_mask)
    if not len(train_indices):
        raise ValueError("training split is empty")

    config = BCConfig(
        input_dim=dataset.features.shape[1],
        hidden_dim=hidden_dim,
        unit_classes=len(dataset.unit_vocab),
        market_classes=len(dataset.market_vocab),
    )
    parameters = initialize_parameters(config, seed)
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    optimizer_state = optimizer.init(parameters)

    @jax.jit
    def train_step(
        current_parameters: Parameters,
        current_optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[
        Parameters,
        optax.OptState,
        jax.Array,
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        (loss, components), gradients = jax.value_and_grad(_loss, has_aux=True)(
            current_parameters, batch, config
        )
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_parameters
        )
        next_parameters = optax.apply_updates(current_parameters, updates)
        return next_parameters, next_optimizer_state, loss, components

    random = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        shuffled = random.permutation(train_indices)
        epoch_loss = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            (
                parameters,
                optimizer_state,
                loss,
                components,
            ) = train_step(parameters, optimizer_state, _batch(dataset, batch_indices))
            epoch_loss += np.asarray(
                [loss, components[0], components[1], components[2]], dtype=np.float64
            )
            batches += 1
        history.append(
            {
                "epoch": epoch,
                "loss": float(epoch_loss[0] / batches),
                "farmer_loss": float(epoch_loss[1] / batches),
                "hand_loss": float(epoch_loss[2] / batches),
                "market_loss": float(epoch_loss[3] / batches),
            }
        )

    report = {
        "config": asdict(config),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "final_losses": history[-1],
        },
        "train_metrics": evaluate_behavior_cloning(parameters, config, dataset, dataset.train_mask),
        "holdout_metrics": evaluate_behavior_cloning(
            parameters, config, dataset, dataset.holdout_mask
        ),
        "history": history,
    }
    return parameters, config, report


def save_checkpoint(
    path: Path,
    parameters: Parameters,
    config: BCConfig,
    dataset: ReplayDataset,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"parameter_{name}": np.asarray(value) for name, value in parameters.items()}
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "dataset_metadata": dataset.metadata,
        "report": report,
    }
    np.savez_compressed(
        path,
        **arrays,
        unit_vocab=np.asarray(dataset.unit_vocab),
        market_vocab=np.asarray(dataset.market_vocab),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_checkpoint(
    path: Path,
) -> tuple[Parameters, BCConfig, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        config = BCConfig(**metadata["config"])
        parameters = {
            key.removeprefix("parameter_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("parameter_")
        }
        unit_vocab = tuple(str(value) for value in payload["unit_vocab"])
        market_vocab = tuple(str(value) for value in payload["market_vocab"])
    return parameters, config, unit_vocab, market_vocab, metadata


class BehaviorCloningPolicy:
    """Greedy factorized policy loaded from a portable NumPy checkpoint."""

    def __init__(self, checkpoint: Path) -> None:
        (
            self._parameters,
            self._config,
            self._unit_vocab,
            self._market_vocab,
            self.metadata,
        ) = load_checkpoint(checkpoint)

    def predict_with_confidence(self, observation: dict[str, Any]) -> PolicyPrediction:
        features = encode_observation(observation, self._config.input_dim)
        farmer_logits, hand_logits, market_logits = forward(
            self._parameters, jnp.asarray(features[None, :]), self._config
        )
        farmer_probabilities = np.asarray(jax.device_get(jax.nn.softmax(farmer_logits, axis=-1)))[0]
        hand_probabilities = np.asarray(jax.device_get(jax.nn.softmax(hand_logits, axis=-1)))[0]
        market_probabilities = np.asarray(jax.device_get(jax.nn.softmax(market_logits, axis=-1)))[0]
        farmer_id = int(farmer_probabilities.argmax())
        hand_ids = hand_probabilities.argmax(axis=-1)
        market_ids = market_probabilities.argmax(axis=-1)

        player = int(observation["player"])
        hands_count = len(observation["farms"][player]["hands"])
        if hands_count > self._config.max_hands:
            raise ValueError(f"policy supports at most {self._config.max_hands} hands")

        hands = [json.loads(self._unit_vocab[int(token_id)]) for token_id in hand_ids[:hands_count]]
        hand_confidences = tuple(
            float(hand_probabilities[index, token_id])
            for index, token_id in enumerate(hand_ids[:hands_count])
        )
        market: list[list[Any]] = []
        market_confidences: list[float] = []
        for index, token_id in enumerate(market_ids):
            token = self._market_vocab[int(token_id)]
            market_confidences.append(float(market_probabilities[index, int(token_id)]))
            if token == STOP_TOKEN:
                break
            market.append(json.loads(token))

        return PolicyPrediction(
            action={
                "farmer": json.loads(self._unit_vocab[farmer_id]),
                "hands": hands,
                "market": market,
            },
            farmer_confidence=float(farmer_probabilities[farmer_id]),
            hand_confidences=hand_confidences,
            market_confidences=tuple(market_confidences),
        )

    def predict_masked_with_confidence(
        self,
        engine: Any,
        observation: dict[str, Any],
    ) -> PolicyPrediction:
        """Decode greedily while querying the exact Rust transition rules.

        The supplied engine must be at the same state as ``observation``. Unit
        masks are conditioned on the already-selected unit prefix; market masks
        are conditioned on both selected units and prior market orders.
        """
        features = encode_observation(observation, self._config.input_dim)
        farmer_logits, hand_logits, market_logits = forward(
            self._parameters,
            jnp.asarray(features[None, :]),
            self._config,
        )
        farmer_logits_numpy = np.asarray(jax.device_get(farmer_logits))[0]
        hand_logits_numpy = np.asarray(jax.device_get(hand_logits))[0]
        market_logits_numpy = np.asarray(jax.device_get(market_logits))[0]

        player = int(observation["player"])
        hands_count = len(observation["farms"][player]["hands"])
        if hands_count > self._config.max_hands:
            raise ValueError(f"policy supports at most {self._config.max_hands} hands")

        unit_candidates = [json.loads(token) for token in self._unit_vocab]
        unit_prefix: list[list[Any]] = []
        farmer_mask = np.asarray(
            engine.unit_action_mask(player, 0, unit_prefix, unit_candidates),
            dtype=np.bool_,
        )
        farmer_id, farmer_confidence = _masked_argmax(farmer_logits_numpy, farmer_mask)
        farmer = unit_candidates[farmer_id]
        unit_prefix.append(farmer)

        hands: list[list[Any]] = []
        hand_confidences: list[float] = []
        for hand_index in range(hands_count):
            mask = np.asarray(
                engine.unit_action_mask(
                    player,
                    hand_index + 1,
                    unit_prefix,
                    unit_candidates,
                ),
                dtype=np.bool_,
            )
            token_id, confidence = _masked_argmax(hand_logits_numpy[hand_index], mask)
            action = unit_candidates[token_id]
            hands.append(action)
            hand_confidences.append(confidence)
            unit_prefix.append(action)

        units = {"farmer": farmer, "hands": hands, "market": []}
        market_candidates = [
            json.loads(token) for token in self._market_vocab if token != STOP_TOKEN
        ]
        market_candidate_indices = [
            index for index, token in enumerate(self._market_vocab) if token != STOP_TOKEN
        ]
        stop_index = self._market_vocab.index(STOP_TOKEN)
        market: list[list[Any]] = []
        market_confidences: list[float] = []
        for order_index in range(self._config.max_market_orders):
            candidate_mask = engine.market_order_mask(
                player,
                units,
                market,
                market_candidates,
            )
            mask = np.zeros(len(self._market_vocab), dtype=np.bool_)
            mask[stop_index] = True
            mask[np.asarray(market_candidate_indices)] = np.asarray(candidate_mask, dtype=np.bool_)
            token_id, confidence = _masked_argmax(market_logits_numpy[order_index], mask)
            market_confidences.append(confidence)
            if token_id == stop_index:
                break
            market.append(json.loads(self._market_vocab[token_id]))

        action = {"farmer": farmer, "hands": hands, "market": market}
        sanitized = engine.sanitize_action(player, action)
        if sanitized != action:
            raise RuntimeError("masked decoder produced an ineffective primitive action")
        return PolicyPrediction(
            action=action,
            farmer_confidence=farmer_confidence,
            hand_confidences=tuple(hand_confidences),
            market_confidences=tuple(market_confidences),
        )

    def predict_masked(self, engine: Any, observation: dict[str, Any]) -> dict[str, Any]:
        return self.predict_masked_with_confidence(engine, observation).action

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.predict_with_confidence(observation).action


def _masked_argmax(logits: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    if logits.shape != mask.shape:
        raise ValueError(f"logit shape {logits.shape} does not match mask shape {mask.shape}")
    if not mask.any():
        raise ValueError("legality mask has no valid candidate")
    masked_logits = np.where(mask, logits, -np.inf)
    token_id = int(masked_logits.argmax())
    shifted = masked_logits - masked_logits[token_id]
    probabilities = np.exp(shifted, where=mask, out=np.zeros_like(shifted, dtype=np.float64))
    confidence = float(probabilities[token_id] / probabilities.sum())
    return token_id, confidence
