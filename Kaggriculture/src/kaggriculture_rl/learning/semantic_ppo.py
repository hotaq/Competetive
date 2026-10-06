"""PPO over executor-backed farming strategies instead of brittle raw actions."""

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

from kaggriculture_rl.learning.gated_ppo import generalized_advantage_estimate
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PLOT_TARGETS = (2, 4, 6, 8, 10)
STRATEGIES = tuple((crop, plots) for crop in CROPS for plots in PLOT_TARGETS)
INCUMBENT_STRATEGY = STRATEGIES.index(("MELON", 6))

type SemanticParameters = dict[str, jax.Array]


@dataclass(frozen=True)
class SemanticPPOConfig:
    feature_dim: int = 256
    action_dim: int = len(STRATEGIES)
    decision_interval: int = 72
    warmup_steps: int = 72
    observe_every_turn: bool = False


@dataclass(frozen=True)
class SemanticRollout:
    features: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    games: tuple[dict[str, Any], ...]


def strategy_targets(strategy_id: int, observation: dict[str, Any]) -> list[dict[str, Any]]:
    crop, plots = STRATEGIES[strategy_id]
    return CropHeuristicPolicy(crop=crop, target_quantity=plots, sell_reserve=0)(
        {"shed": observation["private"]["shed"]}
    )


def initialize_semantic_parameters(
    config: SemanticPPOConfig,
    *,
    seed: int,
    incumbent_logit: float = 5.0,
) -> SemanticParameters:
    actor_key, value_key = jax.random.split(jax.random.key(seed))
    actor_scale = math.sqrt(2.0 / (config.feature_dim + config.action_dim)) * 0.01
    value_scale = math.sqrt(2.0 / (config.feature_dim + 1))
    actor_bias = jnp.zeros(config.action_dim, dtype=jnp.float32)
    actor_bias = actor_bias.at[INCUMBENT_STRATEGY].set(incumbent_logit)
    return {
        "actor_weight": jax.random.normal(
            actor_key, (config.feature_dim, config.action_dim), dtype=jnp.float32
        )
        * actor_scale,
        "actor_bias": actor_bias,
        "value_weight": jax.random.normal(
            value_key, (config.feature_dim, 1), dtype=jnp.float32
        )
        * value_scale,
        "value_bias": jnp.zeros(1, dtype=jnp.float32),
    }


def semantic_forward(
    parameters: SemanticParameters,
    features: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = features @ parameters["actor_weight"] + parameters["actor_bias"]
    values = (features @ parameters["value_weight"] + parameters["value_bias"]).squeeze(-1)
    return logits, values


def _sample_action(logits: np.ndarray, random: np.random.Generator) -> tuple[int, float]:
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    action = int(random.choice(len(probabilities), p=probabilities))
    return action, float(np.log(probabilities[action] + 1e-12))


def _greedy_action(logits: np.ndarray) -> tuple[int, float]:
    action = int(logits.argmax())
    log_probabilities = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
    return action, float(log_probabilities[action])


def collect_semantic_rollouts(
    parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
    *,
    episodes: int,
    base_seed: int,
    opponent_specs: tuple[tuple[str, int], ...],
    stochastic: bool,
    random_seed: int,
    reward_scale: float = 10_000.0,
) -> SemanticRollout:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not opponent_specs:
        raise ValueError("opponent_specs must not be empty")
    random = np.random.default_rng(random_seed)
    feature_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    log_probability_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    games: list[dict[str, Any]] = []

    for episode in range(episodes):
        seed = base_seed + episode // 2
        candidate_seat = episode % 2
        opponent_seat = 1 - candidate_seat
        opponent_crop, opponent_plots = opponent_specs[(episode // 2) % len(opponent_specs)]
        opponent = CropHeuristicPolicy(
            crop=opponent_crop,
            target_quantity=opponent_plots,
            sell_reserve=0,
        )
        engine = Engine({"seed": seed})
        encoder = SpatialRecurrentPolicy(feature_checkpoint)
        previous_margin = 0.0
        turn = 0
        chosen_strategies: list[int] = []
        while turn < 719:
            learning_decision = turn >= config.warmup_steps
            if learning_decision:
                observation = engine.observation(candidate_seat)
                features = encoder.encode_hidden(observation)
                logits, values = semantic_forward(parameters, jnp.asarray(features[None, :]))
                logits_numpy = np.asarray(jax.device_get(logits))[0]
                value = float(np.asarray(jax.device_get(values))[0])
                if stochastic:
                    strategy_id, log_probability = _sample_action(logits_numpy, random)
                else:
                    strategy_id, log_probability = _greedy_action(logits_numpy)
                chosen_strategies.append(strategy_id)
            else:
                strategy_id = INCUMBENT_STRATEGY

            interval_end = min(turn + config.decision_interval, 719)
            while turn < interval_end:
                candidate_observation = engine.observation(candidate_seat)
                opponent_observation = engine.observation(opponent_seat)
                candidate_plan = engine.plan(
                    candidate_seat,
                    strategy_targets(strategy_id, candidate_observation),
                )
                opponent_plan = engine.plan(
                    opponent_seat,
                    opponent({"shed": opponent_observation["private"]["shed"]}),
                )
                actions = opponent_plan["joint_action"]["actions"]
                candidate_action = candidate_plan["joint_action"]["actions"][candidate_seat]
                sanitized_action = engine.sanitize_action(candidate_seat, candidate_action)
                if sanitized_action != candidate_action:
                    raise RuntimeError("semantic executor emitted an ineffective primitive action")
                actions[candidate_seat] = candidate_action
                state = engine.step(actions)
                turn += 1

            money = [float(farm["money"]) for farm in state["farms"]]
            margin = money[candidate_seat] - money[opponent_seat]
            if learning_decision:
                feature_rows.append(features)
                action_rows.append(strategy_id)
                log_probability_rows.append(log_probability)
                value_rows.append(value)
                reward_rows.append((margin - previous_margin) / reward_scale)
                done_rows.append(turn >= 719)
            previous_margin = margin

        games.append(
            {
                "seed": seed,
                "candidate_seat": candidate_seat,
                "opponent_crop": opponent_crop,
                "opponent_plots": opponent_plots,
                "candidate_money": money[candidate_seat],
                "opponent_money": money[opponent_seat],
                "margin": previous_margin,
                "strategy_histogram": np.bincount(
                    chosen_strategies, minlength=config.action_dim
                ).tolist(),
            }
        )

    return SemanticRollout(
        features=np.stack(feature_rows),
        actions=np.asarray(action_rows, dtype=np.int16),
        old_log_probabilities=np.asarray(log_probability_rows, dtype=np.float32),
        old_values=np.asarray(value_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        games=tuple(games),
    )


def ppo_update(
    parameters: SemanticParameters,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    rollout: SemanticRollout,
    *,
    update_epochs: int,
    batch_size: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    seed: int,
) -> tuple[SemanticParameters, optax.OptState, dict[str, float]]:
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards, rollout.old_values, rollout.dones
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    @jax.jit
    def update_step(
        current: SemanticParameters,
        state: optax.OptState,
        features: jax.Array,
        actions: jax.Array,
        old_log_probabilities: jax.Array,
        batch_advantages: jax.Array,
        batch_returns: jax.Array,
    ) -> tuple[SemanticParameters, optax.OptState, jax.Array]:
        def loss_fn(candidate: SemanticParameters) -> jax.Array:
            logits, values = semantic_forward(candidate, features)
            log_probabilities = jax.nn.log_softmax(logits)
            selected = jnp.take_along_axis(
                log_probabilities, actions[:, None], axis=-1
            ).squeeze(-1)
            ratio = jnp.exp(selected - old_log_probabilities)
            unclipped = ratio * batch_advantages
            clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -jnp.minimum(unclipped, clipped).mean()
            value_loss = jnp.square(values - batch_returns).mean()
            probabilities = jax.nn.softmax(logits)
            entropy = -(probabilities * log_probabilities).sum(axis=-1).mean()
            return policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy

        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, next_state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, loss

    random = np.random.default_rng(seed)
    indices = np.arange(len(rollout.features))
    losses: list[float] = []
    for _ in range(update_epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            parameters, optimizer_state, loss = update_step(
                parameters,
                optimizer_state,
                jnp.asarray(rollout.features[batch_indices]),
                jnp.asarray(rollout.actions[batch_indices]),
                jnp.asarray(rollout.old_log_probabilities[batch_indices]),
                jnp.asarray(advantages[batch_indices]),
                jnp.asarray(returns[batch_indices]),
            )
            losses.append(float(loss))
    return parameters, optimizer_state, {
        "loss": fmean(losses),
        "advantage_mean": float(advantages.mean()),
        "return_mean": float(returns.mean()),
    }


def _rollout_summary(rollout: SemanticRollout) -> dict[str, Any]:
    margins = [float(game["margin"]) for game in rollout.games]
    money = [float(game["candidate_money"]) for game in rollout.games]
    return {
        "games": len(margins),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
    }


def save_semantic_checkpoint(
    path: Path,
    parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
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
            "feature_checkpoint": np.asarray(str(feature_checkpoint)),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    )
    np.savez_compressed(path, **arrays)


def load_semantic_checkpoint(
    path: Path,
) -> tuple[SemanticParameters, SemanticPPOConfig, Path, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        parameters = {
            key.removeprefix("parameter_"): jnp.asarray(payload[key])
            for key in payload.files
            if key.startswith("parameter_")
        }
        config = SemanticPPOConfig(**json.loads(str(payload["config_json"])))
        feature_checkpoint = Path(str(payload["feature_checkpoint"]))
        metadata = json.loads(str(payload["metadata_json"]))
    return parameters, config, feature_checkpoint, metadata


def evaluate_semantic_promotion(
    checkpoint: Path,
    *,
    episodes: int = 30,
    base_seed: int = 90_000,
) -> dict[str, Any]:
    """Compare candidate and MELON:6 on identical seats, seeds, and league."""
    parameters, config, feature_checkpoint, _ = load_semantic_checkpoint(checkpoint)
    league = tuple((crop, plots) for crop in CROPS for plots in (4, 6, 8))
    candidate = collect_semantic_rollouts(
        parameters,
        config,
        feature_checkpoint,
        episodes=episodes,
        base_seed=base_seed,
        opponent_specs=league,
        stochastic=False,
        random_seed=0,
    )
    baseline_parameters = initialize_semantic_parameters(
        config, seed=0, incumbent_logit=100.0
    )
    baseline_parameters["actor_weight"] = jnp.zeros_like(
        baseline_parameters["actor_weight"]
    )
    baseline = collect_semantic_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        episodes=episodes,
        base_seed=base_seed,
        opponent_specs=league,
        stochastic=False,
        random_seed=0,
    )
    margin_deltas = [
        float(candidate_game["margin"] - baseline_game["margin"])
        for candidate_game, baseline_game in zip(candidate.games, baseline.games, strict=True)
    ]
    money_deltas = [
        float(candidate_game["candidate_money"] - baseline_game["candidate_money"])
        for candidate_game, baseline_game in zip(candidate.games, baseline.games, strict=True)
    ]
    promotion = (
        fmean(margin_deltas) > 0.0
        and min(margin_deltas) >= 0.0
        and sum(delta > 0.0 for delta in margin_deltas)
        > sum(delta < 0.0 for delta in margin_deltas)
    )
    return {
        "checkpoint": str(checkpoint),
        "candidate": _rollout_summary(candidate),
        "incumbent": _rollout_summary(baseline),
        "paired_delta": {
            "mean_margin": fmean(margin_deltas),
            "worst_margin": min(margin_deltas),
            "mean_money": fmean(money_deltas),
            "worst_money": min(money_deltas),
            "better": sum(delta > 0.0 for delta in margin_deltas),
            "equal": sum(delta == 0.0 for delta in margin_deltas),
            "worse": sum(delta < 0.0 for delta in margin_deltas),
        },
        "promotion": promotion,
    }


def train_semantic_ppo(
    feature_checkpoint: Path,
    output_path: Path,
    *,
    iterations: int = 12,
    rollout_episodes: int = 8,
    evaluation_episodes: int = 10,
    base_seed: int = 4000,
    learning_rate: float = 3e-4,
    incumbent_logit: float = 2.0,
    seed: int = 1729,
) -> dict[str, Any]:
    config = SemanticPPOConfig()
    parameters = initialize_semantic_parameters(
        config, seed=seed, incumbent_logit=incumbent_logit
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
    )
    optimizer_state = optimizer.init(parameters)
    league = tuple((crop, plots) for crop in CROPS for plots in (4, 6, 8))
    history: list[dict[str, Any]] = []

    for iteration in range(iterations):
        rollout = collect_semantic_rollouts(
            parameters,
            config,
            feature_checkpoint,
            episodes=rollout_episodes,
            base_seed=base_seed + iteration * 100,
            opponent_specs=league,
            stochastic=True,
            random_seed=seed + iteration,
        )
        parameters, optimizer_state, update_metrics = ppo_update(
            parameters,
            optimizer_state,
            optimizer,
            rollout,
            update_epochs=4,
            batch_size=64,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            seed=seed + iteration,
        )
        evaluation = collect_semantic_rollouts(
            parameters,
            config,
            feature_checkpoint,
            episodes=evaluation_episodes,
            base_seed=base_seed + 50_000,
            opponent_specs=league,
            stochastic=False,
            random_seed=seed,
        )
        history.append(
            {
                "iteration": iteration + 1,
                "rollout": _rollout_summary(rollout),
                "evaluation": _rollout_summary(evaluation),
                "update": update_metrics,
            }
        )

    final_evaluation = collect_semantic_rollouts(
        parameters,
        config,
        feature_checkpoint,
        episodes=evaluation_episodes,
        base_seed=base_seed + 60_000,
        opponent_specs=league,
        stochastic=False,
        random_seed=seed,
    )
    metadata = {
        "model": "executor-semantic-ppo-v1",
        "feature_checkpoint": str(feature_checkpoint),
        "strategies": [{"crop": crop, "plots": plots} for crop, plots in STRATEGIES],
        "incumbent_strategy": INCUMBENT_STRATEGY,
        "league": [{"crop": crop, "plots": plots} for crop, plots in league],
        "iterations": iterations,
        "rollout_episodes": rollout_episodes,
        "evaluation_episodes": evaluation_episodes,
        "incumbent_logit": incumbent_logit,
        "history": history,
        "final_evaluation": _rollout_summary(final_evaluation),
        "seed": seed,
    }
    save_semantic_checkpoint(output_path, parameters, config, feature_checkpoint, metadata)
    return {"checkpoint": str(output_path), **metadata}


class SemanticPPOPolicy:
    def __init__(self, checkpoint: Path) -> None:
        self._parameters, self._config, feature_checkpoint, self.metadata = (
            load_semantic_checkpoint(checkpoint)
        )
        self._encoder = SpatialRecurrentPolicy(feature_checkpoint)
        self._strategy_id: int | None = None

    def reset(self) -> None:
        self._encoder.reset()
        self._strategy_id = None

    def targets(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        step = int(observation.get("step", 0))
        if step < self._config.warmup_steps:
            self._strategy_id = INCUMBENT_STRATEGY
        elif self._strategy_id is None or (
            step - self._config.warmup_steps
        ) % self._config.decision_interval == 0:
            features = self._encoder.encode_hidden(observation)
            logits, _ = semantic_forward(self._parameters, jnp.asarray(features[None, :]))
            self._strategy_id = int(np.asarray(jax.device_get(logits))[0].argmax())
        return strategy_targets(self._strategy_id, observation)
