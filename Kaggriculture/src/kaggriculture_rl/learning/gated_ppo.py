"""PPO policy that safely selects BC or deterministic fallback components."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.bc import BehaviorCloningPolicy, PolicyPrediction
from kaggriculture_rl.imitation.replay_dataset import (
    DEFAULT_FEATURE_DIM,
    MAX_HANDS,
    encode_observation,
)
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy

GATE_ACTIONS = 2 + MAX_HANDS
MARKET_GATE_INDEX = GATE_ACTIONS - 1

type GateParameters = dict[str, jax.Array]


@dataclass(frozen=True)
class GateConfig:
    observation_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = 128
    action_dim: int = GATE_ACTIONS

    @property
    def input_dim(self) -> int:
        return self.observation_dim + self.action_dim


@dataclass(frozen=True)
class RolloutBatch:
    features: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    games: tuple[dict[str, Any], ...]


def _glorot(key: jax.Array, input_size: int, output_size: int) -> jax.Array:
    scale = math.sqrt(2.0 / (input_size + output_size))
    return jax.random.normal(key, (input_size, output_size), dtype=jnp.float32) * scale


def initialize_gate_parameters(
    config: GateConfig, *, seed: int, initial_acceptance: float = 0.05
) -> GateParameters:
    if not 0.0 < initial_acceptance < 1.0:
        raise ValueError("initial_acceptance must be between zero and one")
    keys = iter(jax.random.split(jax.random.key(seed), 3))
    policy_bias = math.log(initial_acceptance / (1.0 - initial_acceptance))
    return {
        "input_weight": _glorot(next(keys), config.input_dim, config.hidden_dim),
        "input_bias": jnp.zeros(config.hidden_dim, dtype=jnp.float32),
        "policy_weight": _glorot(next(keys), config.hidden_dim, config.action_dim),
        "policy_bias": jnp.full(config.action_dim, policy_bias, dtype=jnp.float32),
        "value_weight": _glorot(next(keys), config.hidden_dim, 1),
        "value_bias": jnp.zeros(1, dtype=jnp.float32),
    }


def gate_forward(parameters: GateParameters, features: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden = jax.nn.tanh(features @ parameters["input_weight"] + parameters["input_bias"])
    logits = hidden @ parameters["policy_weight"] + parameters["policy_bias"]
    values = (hidden @ parameters["value_weight"] + parameters["value_bias"]).squeeze(-1)
    return logits, values


def prediction_features(
    observation: dict[str, Any],
    prediction: PolicyPrediction,
    observation_dim: int,
) -> np.ndarray:
    confidence = np.zeros(GATE_ACTIONS, dtype=np.float32)
    confidence[0] = prediction.farmer_confidence
    confidence[1 : 1 + len(prediction.hand_confidences)] = prediction.hand_confidences
    confidence[MARKET_GATE_INDEX] = min(prediction.market_confidences, default=0.0)
    return np.concatenate((encode_observation(observation, observation_dim), confidence))


def active_gate_mask(hands_count: int) -> np.ndarray:
    if not 0 <= hands_count <= MAX_HANDS:
        raise ValueError(f"hands_count must be between zero and {MAX_HANDS}")
    mask = np.zeros(GATE_ACTIONS, dtype=np.bool_)
    mask[0] = True
    mask[1 : 1 + hands_count] = True
    mask[MARKET_GATE_INDEX] = True
    return mask


def gate_action(
    prediction: PolicyPrediction,
    fallback: dict[str, Any],
    gates: np.ndarray,
) -> dict[str, Any]:
    """Select BC components when gated on and executor output otherwise."""

    if gates.shape != (GATE_ACTIONS,):
        raise ValueError(f"expected {GATE_ACTIONS} gate actions")
    hands = [
        predicted if gates[index + 1] else fallback["hands"][index]
        for index, predicted in enumerate(prediction.action["hands"])
    ]
    return {
        "farmer": (prediction.action["farmer"] if gates[0] else fallback["farmer"]),
        "hands": hands,
        "market": (prediction.action["market"] if gates[MARKET_GATE_INDEX] else fallback["market"]),
    }


def _bernoulli_log_probability(logits: np.ndarray, actions: np.ndarray, mask: np.ndarray) -> float:
    selected = np.where(
        actions,
        -np.logaddexp(0.0, -logits),
        -np.logaddexp(0.0, logits),
    )
    return float((selected * mask).sum())


def _crop_policy(crop: str, plots: int) -> CropHeuristicPolicy:
    return CropHeuristicPolicy(crop=crop, target_quantity=plots, sell_reserve=0)


def collect_rollouts(
    parameters: GateParameters,
    config: GateConfig,
    bc_policy: BehaviorCloningPolicy,
    *,
    episodes: int,
    base_seed: int,
    opponent_specs: tuple[tuple[str, int], ...],
    fallback_spec: tuple[str, int] = ("MELON", 6),
    stochastic: bool = True,
    random_seed: int = 1729,
    reward_scale: float = 10_000.0,
) -> RolloutBatch:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not opponent_specs:
        raise ValueError("opponent_specs must not be empty")
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")

    random = np.random.default_rng(random_seed)
    fallback_policy = _crop_policy(*fallback_spec)
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    log_probability_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    game_rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        seed = base_seed + episode // 2
        candidate_seat = episode % 2
        opponent_spec = opponent_specs[episode % len(opponent_specs)]
        opponent_policy = _crop_policy(*opponent_spec)
        engine = Engine({"seed": seed})
        previous_margin = 0.0
        accepted_components = 0
        active_components = 0

        for turn in range(719):
            opponent_seat = 1 - candidate_seat
            observation = engine.observation(candidate_seat)
            opponent_observation = engine.observation(opponent_seat)
            prediction = bc_policy.predict_with_confidence(observation)
            features = prediction_features(observation, prediction, config.observation_dim)
            logits, values = gate_forward(parameters, jnp.asarray(features[None, :]))
            logits_numpy = np.asarray(jax.device_get(logits))[0]
            value = float(np.asarray(jax.device_get(values))[0])
            stable_logits = np.clip(logits_numpy, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-stable_logits))
            mask = active_gate_mask(len(prediction.action["hands"]))
            if stochastic:
                gates = random.random(config.action_dim) < probabilities
            else:
                gates = probabilities >= 0.5
            gates &= mask

            fallback = engine.plan(
                candidate_seat,
                fallback_policy({"shed": observation["private"]["shed"]}),
            )["joint_action"]["actions"][candidate_seat]
            candidate_action = gate_action(prediction, fallback, gates)
            opponent_action = engine.plan(
                opponent_seat,
                opponent_policy({"shed": opponent_observation["private"]["shed"]}),
            )["joint_action"]["actions"][opponent_seat]
            actions = [
                {"farmer": ["PASS"], "hands": [], "market": []},
                {"farmer": ["PASS"], "hands": [], "market": []},
            ]
            actions[candidate_seat] = candidate_action
            actions[opponent_seat] = opponent_action
            state = engine.step(actions)

            money = [float(farm["money"]) for farm in state["farms"]]
            margin = money[candidate_seat] - money[opponent_seat]
            feature_rows.append(features)
            action_rows.append(gates)
            mask_rows.append(mask)
            log_probability_rows.append(_bernoulli_log_probability(logits_numpy, gates, mask))
            value_rows.append(value)
            reward_rows.append((margin - previous_margin) / reward_scale)
            done_rows.append(turn == 718)
            previous_margin = margin
            accepted_components += int(gates.sum())
            active_components += int(mask.sum())

        game_rows.append(
            {
                "seed": seed,
                "candidate_seat": candidate_seat,
                "opponent_crop": opponent_spec[0],
                "opponent_plots": opponent_spec[1],
                "candidate_money": money[candidate_seat],
                "opponent_money": money[opponent_seat],
                "margin": previous_margin,
                "accepted_component_rate": accepted_components / max(active_components, 1),
            }
        )

    return RolloutBatch(
        features=np.stack(feature_rows),
        actions=np.stack(action_rows),
        action_mask=np.stack(mask_rows),
        old_log_probabilities=np.asarray(log_probability_rows, dtype=np.float32),
        old_values=np.asarray(value_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        games=tuple(game_rows),
    )


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("reward, value and done lengths must match")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        advantages[index] = delta + gamma * gae_lambda * continuation * next_advantage
        next_value = float(values[index])
        next_advantage = float(advantages[index])
    returns = advantages + values
    return advantages, returns


def ppo_update(
    parameters: GateParameters,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    rollout: RolloutBatch,
    *,
    update_epochs: int,
    batch_size: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    seed: int,
) -> tuple[GateParameters, optax.OptState, dict[str, float]]:
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards, rollout.old_values, rollout.dones
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    @jax.jit
    def update_step(
        current_parameters: GateParameters,
        current_optimizer_state: optax.OptState,
        features: jax.Array,
        actions: jax.Array,
        masks: jax.Array,
        old_log_probabilities: jax.Array,
        batch_advantages: jax.Array,
        batch_returns: jax.Array,
    ) -> tuple[
        GateParameters,
        optax.OptState,
        jax.Array,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        def loss_fn(
            candidate_parameters: GateParameters,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            logits, values = gate_forward(candidate_parameters, features)
            log_probabilities = jnp.where(
                actions,
                -jax.nn.softplus(-logits),
                -jax.nn.softplus(logits),
            )
            joint_log_probability = (log_probabilities * masks).sum(axis=-1)
            ratio = jnp.exp(joint_log_probability - old_log_probabilities)
            unclipped = ratio * batch_advantages
            clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -jnp.minimum(unclipped, clipped).mean()
            value_loss = jnp.square(values - batch_returns).mean()
            probabilities = jax.nn.sigmoid(logits)
            entropy = (
                (
                    -probabilities * jax.nn.log_sigmoid(logits)
                    - (1.0 - probabilities) * jax.nn.log_sigmoid(-logits)
                )
                * masks
            ).sum() / jnp.maximum(masks.sum(), 1.0)
            approximate_kl = (old_log_probabilities - joint_log_probability).mean()
            total = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            return total, (policy_loss, value_loss, entropy, approximate_kl)

        (loss, components), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            current_parameters
        )
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_parameters
        )
        next_parameters = optax.apply_updates(current_parameters, updates)
        return next_parameters, next_optimizer_state, loss, components

    random = np.random.default_rng(seed)
    metrics = np.zeros(5, dtype=np.float64)
    updates = 0
    indices = np.arange(len(rollout.features))
    for _ in range(update_epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            (
                parameters,
                optimizer_state,
                loss,
                components,
            ) = update_step(
                parameters,
                optimizer_state,
                jnp.asarray(rollout.features[batch_indices]),
                jnp.asarray(rollout.actions[batch_indices]),
                jnp.asarray(rollout.action_mask[batch_indices]),
                jnp.asarray(rollout.old_log_probabilities[batch_indices]),
                jnp.asarray(advantages[batch_indices]),
                jnp.asarray(returns[batch_indices]),
            )
            metrics += np.asarray(
                [
                    loss,
                    components[0],
                    components[1],
                    components[2],
                    components[3],
                ],
                dtype=np.float64,
            )
            updates += 1

    metrics /= max(updates, 1)
    return (
        parameters,
        optimizer_state,
        {
            "loss": float(metrics[0]),
            "policy_loss": float(metrics[1]),
            "value_loss": float(metrics[2]),
            "entropy": float(metrics[3]),
            "approximate_kl": float(metrics[4]),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
        },
    )


def summarize_games(games: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    margins = [float(game["margin"]) for game in games]
    return {
        "games": len(games),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "mean_accepted_component_rate": fmean(
            float(game["accepted_component_rate"]) for game in games
        ),
    }


def paired_margin_improvement(
    challenger: tuple[dict[str, Any], ...],
    incumbent: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if len(challenger) != len(incumbent):
        raise ValueError("challenger and incumbent game counts must match")
    deltas: list[float] = []
    for challenger_game, incumbent_game in zip(challenger, incumbent, strict=True):
        if "opponent" in challenger_game and "opponent" in incumbent_game:
            identity = ("seed", "candidate_seat", "opponent")
        else:
            identity = ("seed", "candidate_seat", "opponent_crop", "opponent_plots")
        if any(challenger_game[key] != incumbent_game[key] for key in identity):
            raise ValueError("challenger and incumbent games are not paired")
        deltas.append(float(challenger_game["margin"]) - float(incumbent_game["margin"]))
    return {
        "mean_margin_delta": fmean(deltas),
        "worst_margin_delta": min(deltas),
        "improved_games": sum(delta > 0.0 for delta in deltas),
        "equal_games": sum(delta == 0.0 for delta in deltas),
        "regressed_games": sum(delta < 0.0 for delta in deltas),
    }


def save_gate_checkpoint(
    path: Path,
    parameters: GateParameters,
    config: GateConfig,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"parameter_{name}": np.asarray(value) for name, value in parameters.items()}
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "report": report,
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_gate_checkpoint(
    path: Path,
) -> tuple[GateParameters, GateConfig, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        config = GateConfig(**metadata["config"])
        parameters = {
            key.removeprefix("parameter_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("parameter_")
        }
    return parameters, config, metadata


def train_gated_ppo(
    bc_checkpoint: Path,
    output_checkpoint: Path,
    *,
    iterations: int = 2,
    rollout_episodes: int = 8,
    evaluation_episodes: int = 8,
    base_seed: int = 20_000,
    hidden_dim: int = 128,
    learning_rate: float = 1e-4,
    initial_acceptance: float = 0.05,
    reward_scale: float = 10_000.0,
    seed: int = 1729,
    opponent_specs: tuple[tuple[str, int], ...] = (
        ("MELON", 6),
        ("WHEAT", 6),
        ("CARROT", 6),
        ("STRAWBERRY", 6),
    ),
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    bc_policy = BehaviorCloningPolicy(bc_checkpoint)
    config = GateConfig(hidden_dim=hidden_dim)
    parameters = initialize_gate_parameters(
        config, seed=seed, initial_acceptance=initial_acceptance
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(parameters)
    iterations_report: list[dict[str, Any]] = []
    incumbent_parameters = initialize_gate_parameters(
        config, seed=seed, initial_acceptance=initial_acceptance
    )
    incumbent_parameters["policy_weight"] = jnp.zeros_like(incumbent_parameters["policy_weight"])
    incumbent_parameters["policy_bias"] = jnp.full_like(incumbent_parameters["policy_bias"], -100.0)
    incumbent = collect_rollouts(
        incumbent_parameters,
        config,
        bc_policy,
        episodes=evaluation_episodes,
        base_seed=base_seed + 50_000,
        opponent_specs=opponent_specs,
        stochastic=False,
        random_seed=seed,
        reward_scale=reward_scale,
    )
    incumbent_summary = summarize_games(incumbent.games)

    for iteration in range(iterations):
        rollout = collect_rollouts(
            parameters,
            config,
            bc_policy,
            episodes=rollout_episodes,
            base_seed=base_seed + iteration * 1_000,
            opponent_specs=opponent_specs,
            stochastic=True,
            random_seed=seed + iteration,
            reward_scale=reward_scale,
        )
        parameters, optimizer_state, update_metrics = ppo_update(
            parameters,
            optimizer_state,
            optimizer,
            rollout,
            update_epochs=4,
            batch_size=256,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            seed=seed + iteration,
        )
        evaluation = collect_rollouts(
            parameters,
            config,
            bc_policy,
            episodes=evaluation_episodes,
            base_seed=base_seed + 50_000,
            opponent_specs=opponent_specs,
            stochastic=False,
            random_seed=seed,
            reward_scale=reward_scale,
        )
        evaluation_improvement = paired_margin_improvement(evaluation.games, incumbent.games)
        iterations_report.append(
            {
                "iteration": iteration + 1,
                "rollout": summarize_games(rollout.games),
                "update": update_metrics,
                "evaluation": summarize_games(evaluation.games),
                "evaluation_improvement": evaluation_improvement,
            }
        )

    final_evaluation = iterations_report[-1]["evaluation"]
    final_improvement = iterations_report[-1]["evaluation_improvement"]
    report = {
        "config": asdict(config),
        "training": {
            "iterations": iterations,
            "rollout_episodes": rollout_episodes,
            "evaluation_episodes": evaluation_episodes,
            "base_seed": base_seed,
            "learning_rate": learning_rate,
            "initial_acceptance": initial_acceptance,
            "reward_scale": reward_scale,
            "seed": seed,
            "opponent_specs": opponent_specs,
        },
        "iterations": iterations_report,
        "incumbent_evaluation": incumbent_summary,
        "final_evaluation": final_evaluation,
        "final_improvement": final_improvement,
        "promotion": {
            "passed": (
                final_improvement["mean_margin_delta"] > 0.0
                and final_improvement["worst_margin_delta"] >= 0.0
                and final_evaluation["mean_accepted_component_rate"] > 0.0
            ),
            "rule": (
                "paired mean margin delta > 0, worst delta >= 0, and accepted component rate > 0"
            ),
        },
    }
    save_gate_checkpoint(output_checkpoint, parameters, config, report)
    return report


class GatePolicy:
    """Deterministic inference wrapper for a trained BC/fallback selector."""

    def __init__(self, gate_checkpoint: Path, bc_checkpoint: Path) -> None:
        self._parameters, self._config, self.metadata = load_gate_checkpoint(gate_checkpoint)
        self._bc = BehaviorCloningPolicy(bc_checkpoint)

    def action(
        self,
        observation: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        prediction = self._bc.predict_with_confidence(observation)
        features = prediction_features(observation, prediction, self._config.observation_dim)
        logits, _ = gate_forward(self._parameters, jnp.asarray(features[None, :]))
        gates = np.asarray(jax.device_get(logits))[0] >= 0.0
        gates &= active_gate_mask(len(prediction.action["hands"]))
        return gate_action(prediction, fallback, gates)
