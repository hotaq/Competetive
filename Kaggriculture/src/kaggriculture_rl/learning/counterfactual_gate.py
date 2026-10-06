"""Conservative supervised pretraining for the V16 residual market gate."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kaggriculture_rl.learning.counterfactual_residual import (
    CounterfactualResidualDataset,
    load_counterfactual_dataset,
)
from kaggriculture_rl.learning.gated_ppo import (
    GateConfig,
    GateParameters,
    initialize_gate_parameters,
    load_gate_checkpoint,
    save_gate_checkpoint,
)


@dataclass(frozen=True, slots=True)
class CounterfactualGateTrainConfig:
    """Training controls for a frozen-encoder, market-only residual head."""

    epochs: int = 500
    learning_rate: float = 0.02
    l2_anchor: float = 1e-3
    neutral_weight: float = 0.1
    margin_scale: float = 1_000.0
    max_margin_weight: float = 25.0
    seed: int = 1729
    initial_acceptance: float = 0.01
    minimum_positive_seeds_per_split: int = 2

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.l2_anchor < 0.0:
            raise ValueError("l2_anchor must be non-negative")
        if not 0.0 < self.neutral_weight <= 1.0:
            raise ValueError("neutral_weight must be within (0, 1]")
        if self.margin_scale <= 0.0:
            raise ValueError("margin_scale must be positive")
        if self.max_margin_weight < 1.0:
            raise ValueError("max_margin_weight must be at least 1")
        if not 0.0 < self.initial_acceptance < 1.0:
            raise ValueError("initial_acceptance must be within (0, 1)")
        if self.minimum_positive_seeds_per_split <= 0:
            raise ValueError("minimum_positive_seeds_per_split must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _hidden_features(
    parameters: GateParameters,
    features: np.ndarray,
) -> np.ndarray:
    input_weight = np.asarray(parameters["input_weight"], dtype=np.float32)
    input_bias = np.asarray(parameters["input_bias"], dtype=np.float32)
    return np.tanh(features @ input_weight + input_bias).astype(np.float32)


def _example_weights(
    dataset: CounterfactualResidualDataset,
    train_config: CounterfactualGateTrainConfig,
) -> np.ndarray:
    deltas = dataset.delta_margin.astype(np.float64)
    magnitude = np.minimum(
        1.0 + np.abs(deltas) / train_config.margin_scale,
        train_config.max_margin_weight,
    )
    magnitude[deltas == 0.0] *= train_config.neutral_weight

    train = dataset.train_mask
    positives = int(((deltas > 0.0) & train).sum())
    non_positives = int(((deltas <= 0.0) & train).sum())
    if positives:
        magnitude[deltas > 0.0] *= max(non_positives / positives, 1.0)
    return magnitude.astype(np.float32)


def _split_metrics(
    dataset: CounterfactualResidualDataset,
    probabilities: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    selected = mask & (probabilities >= threshold)
    beneficial = dataset.delta_margin > 0.0
    harmful = dataset.delta_margin < 0.0
    neutral = dataset.delta_margin == 0.0
    accepted_deltas = dataset.delta_margin[selected].astype(np.float64)
    positive_seed_count = len(set(dataset.seed[mask & beneficial].tolist()))
    accepted_positive_seed_count = len(
        set(dataset.seed[selected & beneficial].tolist())
    )
    return {
        "rows": int(mask.sum()),
        "positive_rows": int((mask & beneficial).sum()),
        "positive_seeds": positive_seed_count,
        "accepted_rows": int(selected.sum()),
        "accepted_beneficial": int((selected & beneficial).sum()),
        "accepted_harmful": int((selected & harmful).sum()),
        "accepted_neutral": int((selected & neutral).sum()),
        "accepted_positive_seeds": accepted_positive_seed_count,
        "accepted_delta_sum": float(accepted_deltas.sum()),
        "accepted_delta_mean": (
            float(accepted_deltas.mean()) if len(accepted_deltas) else 0.0
        ),
        "worst_accepted_delta": (
            float(accepted_deltas.min()) if len(accepted_deltas) else 0.0
        ),
        "maximum_probability": (
            float(probabilities[mask].max()) if mask.any() else 0.0
        ),
    }


def _select_safe_training_threshold(
    dataset: CounterfactualResidualDataset,
    probabilities: np.ndarray,
) -> float:
    train = dataset.train_mask
    beneficial_probabilities = probabilities[train & (dataset.delta_margin > 0.0)]
    if not len(beneficial_probabilities):
        return 1.0

    candidates = {0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999}
    candidates.update(
        float(np.nextafter(probability, 0.0))
        for probability in beneficial_probabilities
    )
    safe: list[tuple[tuple[float, int, float], float]] = []
    for threshold in sorted(candidates):
        metrics = _split_metrics(dataset, probabilities, train, threshold)
        if metrics["accepted_beneficial"] and not metrics["accepted_harmful"]:
            rank = (
                metrics["accepted_delta_sum"],
                metrics["accepted_beneficial"],
                threshold,
            )
            safe.append((rank, threshold))
    return max(safe)[1] if safe else 1.0


def _train_market_head(
    hidden: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    train_mask: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: float,
    train_config: CounterfactualGateTrainConfig,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    x = jnp.asarray(hidden[train_mask], dtype=jnp.float32)
    y = jnp.asarray(labels[train_mask], dtype=jnp.float32)
    example_weight = jnp.asarray(weights[train_mask], dtype=jnp.float32)
    anchor_weight = jnp.asarray(initial_weight, dtype=jnp.float32)
    anchor_bias = jnp.asarray(initial_bias, dtype=jnp.float32)
    parameters = {
        "weight": anchor_weight,
        "bias": anchor_bias,
    }
    optimizer = optax.adam(train_config.learning_rate)
    optimizer_state = optimizer.init(parameters)

    def loss_fn(head: dict[str, jax.Array]) -> jax.Array:
        logits = x @ head["weight"] + head["bias"]
        losses = optax.sigmoid_binary_cross_entropy(logits, y)
        data_loss = jnp.sum(losses * example_weight) / jnp.sum(example_weight)
        anchor_loss = jnp.mean(jnp.square(head["weight"] - anchor_weight))
        anchor_loss += jnp.square(head["bias"] - anchor_bias)
        return data_loss + train_config.l2_anchor * anchor_loss

    @jax.jit
    def update(
        head: dict[str, jax.Array],
        state: optax.OptState,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        loss, gradients = jax.value_and_grad(loss_fn)(head)
        updates, new_state = optimizer.update(gradients, state, head)
        return optax.apply_updates(head, updates), new_state, loss

    history: list[dict[str, float]] = []
    report_every = max(train_config.epochs // 20, 1)
    for epoch in range(1, train_config.epochs + 1):
        parameters, optimizer_state, loss = update(parameters, optimizer_state)
        if epoch == 1 or epoch == train_config.epochs or epoch % report_every == 0:
            history.append({"epoch": epoch, "loss": float(loss)})

    return (
        np.asarray(parameters["weight"], dtype=np.float32),
        float(parameters["bias"]),
        history,
    )


def train_counterfactual_market_gate(
    dataset_path: Path,
    output_path: Path,
    *,
    initial_checkpoint: Path | None = None,
    model_config: GateConfig | None = None,
    train_config: CounterfactualGateTrainConfig | None = None,
) -> dict[str, Any]:
    """Fit only the market head and save a checkpoint compatible with PPO tools."""
    train_config = train_config or CounterfactualGateTrainConfig()
    train_config.validate()
    dataset = load_counterfactual_dataset(dataset_path)

    initial_report: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        parameters, checkpoint_config, initial_report = load_gate_checkpoint(
            initial_checkpoint
        )
        if model_config is not None and model_config != checkpoint_config:
            raise ValueError("model_config does not match initial checkpoint")
        model_config = checkpoint_config
    else:
        max_hands = int(dataset.metadata.get("max_hands", 14))
        model_config = model_config or GateConfig(action_dim=max_hands + 2)
        parameters = initialize_gate_parameters(
            model_config,
            seed=train_config.seed,
            initial_acceptance=train_config.initial_acceptance,
        )

    if dataset.features.shape[1] != model_config.input_dim:
        raise ValueError(
            "dataset feature width does not match gate input: "
            f"{dataset.features.shape[1]} != {model_config.input_dim}"
        )

    market_index = model_config.action_dim - 1
    initial_policy_weight = np.asarray(parameters["policy_weight"], dtype=np.float32)
    initial_policy_bias = np.asarray(parameters["policy_bias"], dtype=np.float32)
    hidden = _hidden_features(parameters, dataset.features)
    labels = (dataset.delta_margin > 0.0).astype(np.float32)
    weights = _example_weights(dataset, train_config)
    market_weight, market_bias, history = _train_market_head(
        hidden,
        labels,
        weights,
        dataset.train_mask,
        initial_policy_weight[:, market_index],
        float(initial_policy_bias[market_index]),
        train_config,
    )

    raw_logits = hidden @ market_weight + market_bias
    raw_probabilities = _sigmoid(raw_logits)
    selected_threshold = _select_safe_training_threshold(dataset, raw_probabilities)
    if selected_threshold >= 1.0:
        threshold_logit = 100.0
    else:
        bounded = min(max(selected_threshold, 1e-7), 1.0 - 1e-7)
        threshold_logit = math.log(bounded / (1.0 - bounded))

    policy_weight = np.zeros_like(initial_policy_weight)
    policy_bias = np.full_like(initial_policy_bias, -30.0)
    policy_weight[:, market_index] = market_weight
    policy_bias[market_index] = market_bias - threshold_logit
    trained_parameters = dict(parameters)
    trained_parameters["policy_weight"] = jnp.asarray(policy_weight)
    trained_parameters["policy_bias"] = jnp.asarray(policy_bias)

    shifted_probabilities = _sigmoid(raw_logits - threshold_logit)
    train_metrics = _split_metrics(
        dataset, shifted_probabilities, dataset.train_mask, 0.5
    )
    holdout_metrics = _split_metrics(
        dataset, shifted_probabilities, ~dataset.train_mask, 0.5
    )
    minimum_seeds = train_config.minimum_positive_seeds_per_split
    offline_passed = all(
        (
            train_metrics["accepted_beneficial"] > 0,
            train_metrics["accepted_harmful"] == 0,
            train_metrics["accepted_positive_seeds"] >= minimum_seeds,
            holdout_metrics["accepted_beneficial"] > 0,
            holdout_metrics["accepted_harmful"] == 0,
            holdout_metrics["accepted_positive_seeds"] >= minimum_seeds,
        )
    )
    report: dict[str, Any] = {
        "method": "counterfactual_leave_one_out_frozen_encoder_market_head",
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "rows": len(dataset.features),
            "train_rows": int(dataset.train_mask.sum()),
            "holdout_rows": int((~dataset.train_mask).sum()),
        },
        "initial_checkpoint": (
            {
                "path": str(initial_checkpoint.resolve()),
                "sha256": _sha256(initial_checkpoint),
                "embedded_report_present": initial_report is not None,
            }
            if initial_checkpoint is not None
            else None
        ),
        "model_config": {
            "observation_dim": model_config.observation_dim,
            "hidden_dim": model_config.hidden_dim,
            "action_dim": model_config.action_dim,
        },
        "training": asdict(train_config),
        "history": history,
        "selection": {
            "raw_probability_threshold": selected_threshold,
            "threshold_logit_shift": threshold_logit,
            "fixed_runtime_threshold": 0.5,
            "rule": "training threshold must accept positive labels and zero harmful labels",
        },
        "train": train_metrics,
        "holdout": holdout_metrics,
        "promotion": {
            "offline_passed": offline_passed,
            "minimum_positive_seeds_per_split": minimum_seeds,
            "requires_live_paired_evaluation": True,
            "promoted": False,
        },
    }
    save_gate_checkpoint(output_path, trained_parameters, model_config, report)
    return report
