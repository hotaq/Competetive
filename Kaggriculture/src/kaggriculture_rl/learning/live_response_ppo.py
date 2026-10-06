"""Risk-sensitive contextual PPO for the first irreversible live response."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.evaluation.live_profile_league import WarmupProfilePolicy
from kaggriculture_rl.evaluation.live_profile_matrix import (
    ProfileGame,
    play_live_policy_game,
)
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    INCUMBENT_PROFILE,
    ProductionProfile,
    StickyProfileController,
    profile_targets,
)
from kaggriculture_rl.policy.reactive_profile import (
    COW4_BASELINE_PROFILE,
    COW5_ROBUST_PROFILE,
    COW6_ROBUST_PROFILE,
    COW8_LAND1_PROFILE,
    COW10_LAND2_PROFILE,
    ReactiveProfilePolicy,
)

LIVE_RESPONSE_PROFILES = (
    INCUMBENT_PROFILE,
    COW4_BASELINE_PROFILE,
    COW10_LAND2_PROFILE,
    COW8_LAND1_PROFILE,
    COW6_ROBUST_PROFILE,
    COW5_ROBUST_PROFILE,
)
PROFILE_INDEX = {
    profile.name: index for index, profile in enumerate(LIVE_RESPONSE_PROFILES)
}


@dataclass(frozen=True, slots=True)
class LiveResponsePPOConfig:
    feature_dim: int = 12
    hidden_dim: int = 16
    action_dim: int = len(LIVE_RESPONSE_PROFILES)
    observe_through_step: int = 72
    reward_scale: float = 10_000.0
    money_weight: float = 0.20
    negative_margin_penalty: float = 2.0
    money_floor: float = 22_000.0
    money_floor_penalty: float = 1.0
    clip_ratio: float = 0.20
    learning_rate: float = 0.008
    value_coefficient: float = 0.30
    entropy_coefficient: float = 0.01
    teacher_coefficient: float = 2.0
    update_epochs: int = 8

    def __post_init__(self) -> None:
        if self.feature_dim != 12:
            raise ValueError("live response feature_dim must be 12")
        if self.action_dim != len(LIVE_RESPONSE_PROFILES):
            raise ValueError("action_dim must match LIVE_RESPONSE_PROFILES")


@dataclass(slots=True)
class LiveResponseParameters:
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    actor_weights: np.ndarray
    actor_bias: np.ndarray
    value_weights: np.ndarray
    value_bias: np.ndarray


@dataclass(frozen=True, slots=True)
class ResponseDecision:
    features: np.ndarray
    action: int
    log_probability: float
    value: float
    teacher_action: int


def _copy_parameters(
    parameters: LiveResponseParameters,
) -> LiveResponseParameters:
    return LiveResponseParameters(
        hidden_weights=parameters.hidden_weights.copy(),
        hidden_bias=parameters.hidden_bias.copy(),
        actor_weights=parameters.actor_weights.copy(),
        actor_bias=parameters.actor_bias.copy(),
        value_weights=parameters.value_weights.copy(),
        value_bias=parameters.value_bias.copy(),
    )


def initialize_parameters(
    config: LiveResponsePPOConfig,
    *,
    seed: int,
) -> LiveResponseParameters:
    rng = np.random.default_rng(seed)
    hidden_scale = math.sqrt(2.0 / (config.feature_dim + config.hidden_dim))
    actor_scale = math.sqrt(2.0 / (config.hidden_dim + config.action_dim))
    return LiveResponseParameters(
        hidden_weights=rng.normal(
            0.0, hidden_scale, (config.feature_dim, config.hidden_dim)
        ),
        hidden_bias=np.zeros(config.hidden_dim, dtype=np.float64),
        actor_weights=rng.normal(
            0.0, actor_scale * 0.2, (config.hidden_dim, config.action_dim)
        ),
        actor_bias=np.zeros(config.action_dim, dtype=np.float64),
        value_weights=np.zeros(config.hidden_dim, dtype=np.float64),
        value_bias=np.zeros(1, dtype=np.float64),
    )


def _token_count(value: Any, target: str) -> int:
    if isinstance(value, str):
        return int(value == target)
    if isinstance(value, Mapping):
        return sum(_token_count(child, target) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_token_count(child, target) for child in value)
    return 0


def encode_live_response_features(observation: Mapping[str, Any]) -> np.ndarray:
    player = int(observation.get("player", 0))
    farms = observation.get("farms", ())
    if not isinstance(farms, Sequence) or len(farms) < 2:
        raise ValueError("observation must contain two farms")
    own = farms[player]
    opponent = farms[1 - player]
    if not isinstance(own, Mapping) or not isinstance(opponent, Mapping):
        raise ValueError("farm observations must be mappings")
    hands_value = opponent.get("hands", ())
    hands = len(hands_value) if isinstance(hands_value, Sequence) else 0
    own_money = float(own.get("money", 0.0))
    opponent_money = float(opponent.get("money", 0.0))
    opponent_unlocked = opponent.get("unlocked_quadrants", ())
    return np.asarray(
        [
            hands / 8.0,
            float(hands == 0),
            float(0 < hands < 4),
            float(hands >= 4),
            own_money / 5_000.0,
            opponent_money / 5_000.0,
            (own_money - opponent_money) / 5_000.0,
            float((own_money - opponent_money) >= 100.0),
            len(opponent_unlocked) / 4.0,
            _token_count(opponent, "COW") / 8.0,
            _token_count(opponent, "SHEEP") / 8.0,
            _token_count(opponent, "GOOSE") / 8.0,
        ],
        dtype=np.float64,
    )


def _forward(
    parameters: LiveResponseParameters,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.tanh(
        features @ parameters.hidden_weights + parameters.hidden_bias
    )
    logits = hidden @ parameters.actor_weights + parameters.actor_bias
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    values = hidden @ parameters.value_weights + parameters.value_bias[0]
    return hidden, probabilities, values


def _teacher_action(observation: dict[str, Any]) -> int:
    teacher = ReactiveProfilePolicy()
    profile = teacher.selected_profile(observation)
    return PROFILE_INDEX[profile.name]


class LiveResponsePPOPolicy:
    def __init__(
        self,
        parameters_or_checkpoint: LiveResponseParameters | Path | str,
        config: LiveResponsePPOConfig | None = None,
        *,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(parameters_or_checkpoint, (Path, str)):
            parameters, loaded_config = load_live_response_checkpoint(
                Path(parameters_or_checkpoint)
            )
            self.parameters = parameters
            self.config = loaded_config
        else:
            if config is None:
                raise ValueError("config is required with in-memory parameters")
            self.parameters = parameters_or_checkpoint
            self.config = config
        self.deterministic = deterministic
        self.rng = rng or np.random.default_rng()
        self._controller = StickyProfileController()
        self._committed: ProductionProfile | None = None
        self.decision: ResponseDecision | None = None

    def reset(self) -> None:
        self._controller.reset()
        self._committed = None
        self.decision = None

    def _select(self, observation: dict[str, Any]) -> ProductionProfile:
        step = int(observation.get("step", 0))
        if step < self.config.observe_through_step:
            return INCUMBENT_PROFILE
        if step == self.config.observe_through_step:
            return COW4_BASELINE_PROFILE
        if self._committed is None:
            features = encode_live_response_features(observation)
            _, probabilities, values = _forward(
                self.parameters, features[np.newaxis, :]
            )
            distribution = probabilities[0]
            action = (
                int(distribution.argmax())
                if self.deterministic
                else int(self.rng.choice(len(distribution), p=distribution))
            )
            self._committed = LIVE_RESPONSE_PROFILES[action]
            self.decision = ResponseDecision(
                features=features,
                action=action,
                log_probability=float(np.log(distribution[action] + 1e-12)),
                value=float(values[0]),
                teacher_action=_teacher_action(observation),
            )
        return self._committed

    def __call__(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        profile = self._select(observation)
        return profile_targets(self._controller.select(profile), observation)


def save_live_response_checkpoint(
    path: Path,
    parameters: LiveResponseParameters,
    config: LiveResponsePPOConfig,
    *,
    report: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "live-response-contextual-ppo-v2",
        "feature_schema": "public-investment-buckets-v2",
        "config": asdict(config),
        "profiles": [profile.name for profile in LIVE_RESPONSE_PROFILES],
        "report": dict(report or {}),
    }
    np.savez_compressed(
        path,
        hidden_weights=parameters.hidden_weights,
        hidden_bias=parameters.hidden_bias,
        actor_weights=parameters.actor_weights,
        actor_bias=parameters.actor_bias,
        value_weights=parameters.value_weights,
        value_bias=parameters.value_bias,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_live_response_checkpoint(
    path: Path,
) -> tuple[LiveResponseParameters, LiveResponsePPOConfig]:
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"].item()))
        if metadata.get("feature_schema") != "public-investment-buckets-v2":
            raise ValueError("checkpoint uses an incompatible feature schema")
        expected = [profile.name for profile in LIVE_RESPONSE_PROFILES]
        if metadata.get("profiles") != expected:
            raise ValueError("checkpoint profile catalog does not match runtime")
        config = LiveResponsePPOConfig(**metadata["config"])
        parameters = LiveResponseParameters(
            hidden_weights=checkpoint["hidden_weights"].copy(),
            hidden_bias=checkpoint["hidden_bias"].copy(),
            actor_weights=checkpoint["actor_weights"].copy(),
            actor_bias=checkpoint["actor_bias"].copy(),
            value_weights=checkpoint["value_weights"].copy(),
            value_bias=checkpoint["value_bias"].copy(),
        )
    return parameters, config


def collect_decision_observation(
    opponent_profile: ProductionProfile,
    *,
    seed: int,
    candidate_seat: int,
    observe_through_step: int = 72,
) -> dict[str, Any]:
    engine = Engine(
        {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    )
    candidate = WarmupProfilePolicy(COW4_BASELINE_PROFILE)
    opponent = WarmupProfilePolicy(opponent_profile)
    opponent_seat = 1 - candidate_seat
    while True:
        observation = engine.observation(candidate_seat)
        if int(observation["step"]) > observe_through_step:
            return observation
        opponent_observation = engine.observation(opponent_seat)
        candidate_plan = engine.plan(candidate_seat, candidate(observation))
        opponent_plan = engine.plan(
            opponent_seat, opponent(opponent_observation)
        )
        actions = candidate_plan["joint_action"]["actions"]
        actions[opponent_seat] = opponent_plan["joint_action"]["actions"][
            opponent_seat
        ]
        engine.step(actions)


def _behavior_clone(
    parameters: LiveResponseParameters,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    count = len(features)
    targets = np.eye(parameters.actor_bias.size)[labels]
    for _ in range(epochs):
        hidden, probabilities, _ = _forward(parameters, features)
        logits_gradient = (probabilities - targets) / count
        actor_weights_gradient = hidden.T @ logits_gradient
        actor_bias_gradient = logits_gradient.sum(axis=0)
        hidden_gradient = (logits_gradient @ parameters.actor_weights.T) * (
            1.0 - hidden**2
        )
        parameters.actor_weights -= learning_rate * actor_weights_gradient
        parameters.actor_bias -= learning_rate * actor_bias_gradient
        parameters.hidden_weights -= learning_rate * (
            features.T @ hidden_gradient
        )
        parameters.hidden_bias -= learning_rate * hidden_gradient.sum(axis=0)
    _, probabilities, _ = _forward(parameters, features)
    chosen = probabilities[np.arange(count), labels]
    return {
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "cross_entropy": float(-np.log(chosen + 1e-12).mean()),
    }


def _risk_adjusted_reward(
    game: ProfileGame,
    config: LiveResponsePPOConfig,
) -> float:
    reward = game.margin + config.money_weight * game.candidate_money
    if game.margin < 0.0:
        reward += config.negative_margin_penalty * game.margin
    if game.candidate_money < config.money_floor:
        reward -= config.money_floor_penalty * (
            config.money_floor - game.candidate_money
        )
    return reward / config.reward_scale


def _clip_gradients(gradients: list[np.ndarray], max_norm: float = 5.0) -> None:
    norm = math.sqrt(sum(float(np.square(gradient).sum()) for gradient in gradients))
    if norm <= max_norm:
        return
    scale = max_norm / (norm + 1e-12)
    for gradient in gradients:
        gradient *= scale


def _ppo_update(
    parameters: LiveResponseParameters,
    config: LiveResponsePPOConfig,
    *,
    features: np.ndarray,
    actions: np.ndarray,
    old_log_probabilities: np.ndarray,
    returns: np.ndarray,
    old_values: np.ndarray,
    teacher_actions: np.ndarray,
) -> dict[str, float]:
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    count = len(features)
    action_targets = np.eye(config.action_dim)[actions]
    teacher_targets = np.eye(config.action_dim)[teacher_actions]

    for _ in range(config.update_epochs):
        hidden, probabilities, values = _forward(parameters, features)
        chosen = probabilities[np.arange(count), actions]
        log_probabilities = np.log(chosen + 1e-12)
        ratios = np.exp(log_probabilities - old_log_probabilities)
        active = np.ones(count, dtype=bool)
        active[(advantages >= 0.0) & (ratios > 1.0 + config.clip_ratio)] = False
        active[(advantages < 0.0) & (ratios < 1.0 - config.clip_ratio)] = False
        log_probability_gradient = np.where(
            active, -advantages * ratios / count, 0.0
        )
        logits_gradient = log_probability_gradient[:, np.newaxis] * (
            action_targets - probabilities
        )
        logits_gradient += (
            config.teacher_coefficient
            * (probabilities - teacher_targets)
            / count
        )
        entropy = -np.sum(
            probabilities * np.log(probabilities + 1e-12), axis=1
        )
        logits_gradient += (
            config.entropy_coefficient
            * probabilities
            * (np.log(probabilities + 1e-12) + entropy[:, np.newaxis])
            / count
        )
        value_gradient = (
            2.0
            * config.value_coefficient
            * (values - returns)
            / count
        )
        actor_weights_gradient = hidden.T @ logits_gradient
        actor_bias_gradient = logits_gradient.sum(axis=0)
        value_weights_gradient = hidden.T @ value_gradient
        value_bias_gradient = np.asarray([value_gradient.sum()])
        hidden_gradient = (
            logits_gradient @ parameters.actor_weights.T
            + value_gradient[:, np.newaxis]
            * parameters.value_weights[np.newaxis, :]
        ) * (1.0 - hidden**2)
        hidden_weights_gradient = features.T @ hidden_gradient
        hidden_bias_gradient = hidden_gradient.sum(axis=0)
        gradients = [
            hidden_weights_gradient,
            hidden_bias_gradient,
            actor_weights_gradient,
            actor_bias_gradient,
            value_weights_gradient,
            value_bias_gradient,
        ]
        _clip_gradients(gradients)
        for value, gradient in zip(
            (
                parameters.hidden_weights,
                parameters.hidden_bias,
                parameters.actor_weights,
                parameters.actor_bias,
                parameters.value_weights,
                parameters.value_bias,
            ),
            gradients,
            strict=True,
        ):
            value -= config.learning_rate * gradient

    _, probabilities, values = _forward(parameters, features)
    chosen = probabilities[np.arange(count), actions]
    return {
        "entropy": float(
            -np.sum(
                probabilities * np.log(probabilities + 1e-12), axis=1
            ).mean()
        ),
        "teacher_accuracy": float(
            (probabilities.argmax(axis=1) == teacher_actions).mean()
        ),
        "value_mse": float(np.square(values - returns).mean()),
        "mean_probability_of_sampled_action": float(chosen.mean()),
    }


def train_live_response_ppo(
    output: Path,
    *,
    seed: int = 31,
    base_seed: int = 300_000,
    iterations: int = 6,
    seeds_per_opponent: int = 3,
    config: LiveResponsePPOConfig | None = None,
) -> dict[str, Any]:
    if iterations <= 0 or seeds_per_opponent <= 0:
        raise ValueError("iterations and seeds_per_opponent must be positive")
    config = config or LiveResponsePPOConfig()
    parameters = initialize_parameters(config, seed=seed)
    rng = np.random.default_rng(seed + 1)

    context_observations = [
        collect_decision_observation(
            opponent,
            seed=base_seed - 100 + offset,
            candidate_seat=seat,
            observe_through_step=config.observe_through_step,
        )
        for opponent in ADAPTIVE_PROFILES
        for offset in range(4)
        for seat in (0, 1)
    ]
    context_features = np.stack(
        [encode_live_response_features(item) for item in context_observations]
    )
    context_labels = np.asarray(
        [_teacher_action(item) for item in context_observations], dtype=np.int64
    )
    bc_epochs = 0
    while True:
        bc_report = _behavior_clone(
            parameters,
            context_features,
            context_labels,
            epochs=500,
            learning_rate=0.05,
        )
        bc_epochs += 500
        if bc_report["accuracy"] == 1.0:
            break
        if bc_epochs >= 5_000:
            raise RuntimeError("behavior cloning failed its 100% accuracy gate")
    bc_report["epochs"] = bc_epochs
    safe_parameters = _copy_parameters(parameters)
    selected_safe_iteration: int | str = "bc"

    history = []
    for iteration in range(iterations):
        games: list[ProfileGame] = []
        decisions: list[ResponseDecision] = []
        rewards: list[float] = []
        for opponent_index, opponent in enumerate(ADAPTIVE_PROFILES):
            for offset in range(seeds_per_opponent):
                episode_seed = (
                    base_seed + iteration * 100 + opponent_index * 10 + offset
                )
                for seat in (0, 1):
                    policy = LiveResponsePPOPolicy(
                        parameters,
                        config,
                        deterministic=False,
                        rng=rng,
                    )
                    game = play_live_policy_game(
                        "live_response_ppo",
                        policy,
                        opponent,
                        seed=episode_seed,
                        candidate_seat=seat,
                    )
                    if policy.decision is None:
                        raise RuntimeError("policy made no live response decision")
                    games.append(game)
                    decisions.append(policy.decision)
                    rewards.append(_risk_adjusted_reward(game, config))
        update = _ppo_update(
            parameters,
            config,
            features=np.stack([decision.features for decision in decisions]),
            actions=np.asarray(
                [decision.action for decision in decisions], dtype=np.int64
            ),
            old_log_probabilities=np.asarray(
                [decision.log_probability for decision in decisions]
            ),
            returns=np.asarray(rewards),
            old_values=np.asarray([decision.value for decision in decisions]),
            teacher_actions=np.asarray(
                [decision.teacher_action for decision in decisions], dtype=np.int64
            ),
        )
        _, validation_probabilities, _ = _forward(
            parameters, context_features
        )
        validation_teacher_accuracy = float(
            (
                validation_probabilities.argmax(axis=1) == context_labels
            ).mean()
        )
        update["validation_teacher_accuracy"] = validation_teacher_accuracy
        if validation_teacher_accuracy == 1.0:
            safe_parameters = _copy_parameters(parameters)
            selected_safe_iteration = iteration
        margins = [game.margin for game in games]
        money = [game.candidate_money for game in games]
        history.append(
            {
                "iteration": iteration,
                "episodes": len(games),
                "wins": sum(margin > 0.0 for margin in margins),
                "mean_margin": fmean(margins),
                "worst_margin": min(margins),
                "mean_money": fmean(money),
                "worst_money": min(money),
                "mean_reward": fmean(rewards),
                "actions": dict(
                    Counter(
                        LIVE_RESPONSE_PROFILES[decision.action].name
                        for decision in decisions
                    )
                ),
                "update": update,
            }
        )

    parameters = safe_parameters
    report = {
        "model": "live-response-contextual-ppo-v2",
        "seed": seed,
        "base_seed": base_seed,
        "iterations": iterations,
        "seeds_per_opponent": seeds_per_opponent,
        "episodes": iterations
        * len(ADAPTIVE_PROFILES)
        * seeds_per_opponent
        * 2,
        "bc": bc_report,
        "selected_safe_iteration": selected_safe_iteration,
        "history": history,
    }
    save_live_response_checkpoint(output, parameters, config, report=report)
    return report
