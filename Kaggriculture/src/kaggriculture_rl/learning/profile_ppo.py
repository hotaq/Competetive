"""PPO over safe, sticky crop/livestock production profiles."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.semantic_ppo import (
    SemanticParameters,
    SemanticPPOConfig,
    SemanticRollout,
    ppo_update,
    semantic_forward,
)
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    INCUMBENT_PROFILE,
    StickyProfileController,
    profile_targets,
)

DEFAULT_PROFILE_CONFIG = SemanticPPOConfig(action_dim=len(ADAPTIVE_PROFILES))
DEFAULT_LEAGUE = tuple(
    (crop, plots)
    for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    for plots in (4, 6, 8)
)


def _has_pending_profile_decision(turn: int, config: SemanticPPOConfig) -> bool:
    if turn <= config.warmup_steps:
        return True
    elapsed = turn - config.warmup_steps
    remainder = elapsed % config.decision_interval
    next_decision = turn if remainder == 0 else turn + config.decision_interval - remainder
    return next_decision < 719


def initialize_profile_parameters(
    config: SemanticPPOConfig = DEFAULT_PROFILE_CONFIG,
    *,
    seed: int,
) -> SemanticParameters:
    if config.action_dim != len(ADAPTIVE_PROFILES):
        raise ValueError("profile PPO action_dim must match ADAPTIVE_PROFILES")
    actor_key, value_key = jax.random.split(jax.random.key(seed))
    actor_scale = math.sqrt(2.0 / (config.feature_dim + config.action_dim)) * 0.01
    value_scale = math.sqrt(2.0 / (config.feature_dim + 1))
    # The prior is evidence-backed, not a fixed policy: cow2 is the robust
    # profile tournament winner, while every action remains explorable.
    actor_bias = jnp.asarray([0.0, -0.25, 0.75, 0.35, 0.10, -0.25], dtype=jnp.float32)
    return {
        "actor_weight": jax.random.normal(
            actor_key,
            (config.feature_dim, config.action_dim),
            dtype=jnp.float32,
        )
        * actor_scale,
        "actor_bias": actor_bias,
        "value_weight": jax.random.normal(
            value_key,
            (config.feature_dim, 1),
            dtype=jnp.float32,
        )
        * value_scale,
        "value_bias": jnp.zeros(1, dtype=jnp.float32),
    }


def _choose_action(
    logits: np.ndarray,
    random: np.random.Generator,
    *,
    stochastic: bool,
) -> tuple[int, float]:
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    if stochastic:
        action = int(random.choice(len(probabilities), p=probabilities))
    else:
        action = int(probabilities.argmax())
    return action, float(np.log(probabilities[action] + 1e-12))


def collect_profile_rollouts(
    parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
    *,
    episodes: int,
    base_seed: int,
    opponent_specs: tuple[tuple[str, int], ...] = DEFAULT_LEAGUE,
    stochastic: bool,
    random_seed: int,
    reward_scale: float = 10_000.0,
) -> SemanticRollout:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not opponent_specs:
        raise ValueError("opponent_specs must not be empty")
    if config.action_dim != len(ADAPTIVE_PROFILES):
        raise ValueError("profile PPO action_dim must match ADAPTIVE_PROFILES")

    random = np.random.default_rng(random_seed)
    feature_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    log_probability_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    games: list[dict[str, Any]] = []

    for episode in range(episodes):
        candidate_seat = episode % 2
        opponent_seat = 1 - candidate_seat
        opponent_spec = opponent_specs[episode % len(opponent_specs)]
        engine = Engine(
            {
                "episodeSteps": 720,
                "seed": base_seed + episode,
                "weedSpawnChance": 0.005,
            }
        )
        opponent = CropHeuristicPolicy(
            crop=opponent_spec[0],
            target_quantity=opponent_spec[1],
            sell_reserve=0,
        )
        encoder = SpatialRecurrentPolicy(feature_checkpoint)
        encoder.reset()
        controller = StickyProfileController()
        previous_margin = 0.0
        turn = 0
        chosen_profiles: list[int] = []
        blocked_entries = 0
        target_entries = 0
        ineffective_actions = 0
        last_encoded_turn = -1

        while turn < 719:
            learning_decision = turn >= config.warmup_steps
            if learning_decision:
                observation = engine.observation(candidate_seat)
                features = encoder.encode_hidden(observation)
                last_encoded_turn = turn
                logits, values = semantic_forward(
                    parameters,
                    jnp.asarray(features[None, :]),
                )
                logits_numpy = np.asarray(jax.device_get(logits))[0]
                value = float(np.asarray(jax.device_get(values))[0])
                profile_id, log_probability = _choose_action(
                    logits_numpy,
                    random,
                    stochastic=stochastic,
                )
                chosen_profiles.append(profile_id)
            else:
                profile_id = 0

            effective_profile = controller.select(ADAPTIVE_PROFILES[profile_id])
            interval_end = (
                min(config.warmup_steps, 719)
                if not learning_decision
                else min(turn + config.decision_interval, 719)
            )
            while turn < interval_end:
                candidate_observation = engine.observation(candidate_seat)
                if (
                    config.observe_every_turn
                    and interval_end < 719
                    and last_encoded_turn != turn
                ):
                    encoder.encode_hidden(candidate_observation)
                    last_encoded_turn = turn
                opponent_observation = engine.observation(opponent_seat)
                candidate_plan = engine.plan(
                    candidate_seat,
                    profile_targets(effective_profile, candidate_observation),
                )
                opponent_plan = engine.plan(
                    opponent_seat,
                    opponent({"shed": opponent_observation["private"]["shed"]}),
                )
                candidate_action = candidate_plan["joint_action"]["actions"][
                    candidate_seat
                ]
                if engine.sanitize_action(candidate_seat, candidate_action) != candidate_action:
                    ineffective_actions += 1
                entries = candidate_plan["report"]["entries"]
                target_entries += len(entries)
                blocked_entries += sum(entry["status"] == "BLOCKED" for entry in entries)
                actions = opponent_plan["joint_action"]["actions"]
                actions[candidate_seat] = candidate_action
                engine.step(actions)
                turn += 1

            state = engine.state
            margin = float(
                state["farms"][candidate_seat]["money"]
                - state["farms"][opponent_seat]["money"]
            )
            if learning_decision:
                feature_rows.append(features)
                action_rows.append(profile_id)
                log_probability_rows.append(log_probability)
                value_rows.append(value)
                reward_rows.append((margin - previous_margin) / reward_scale)
                done_rows.append(turn >= 719)
            previous_margin = margin

        state = engine.state
        games.append(
            {
                "seed": base_seed + episode,
                "candidate_seat": candidate_seat,
                "opponent": f"{opponent_spec[0]}:{opponent_spec[1]}",
                "candidate_money": float(state["farms"][candidate_seat]["money"]),
                "opponent_money": float(state["farms"][opponent_seat]["money"]),
                "margin": previous_margin,
                "chosen_profiles": chosen_profiles,
                "blocked_rate": blocked_entries / target_entries if target_entries else 0.0,
                "ineffective_actions": ineffective_actions,
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


def _summary(rollout: SemanticRollout) -> dict[str, Any]:
    margins = [float(game["margin"]) for game in rollout.games]
    money = [float(game["candidate_money"]) for game in rollout.games]
    ineffective = sum(int(game["ineffective_actions"]) for game in rollout.games)
    return {
        "games": len(margins),
        "wins": sum(margin > 0.0 for margin in margins),
        "draws": sum(margin == 0.0 for margin in margins),
        "losses": sum(margin < 0.0 for margin in margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "mean_blocked_rate": fmean(float(game["blocked_rate"]) for game in rollout.games),
        "ineffective_actions": ineffective,
        "action_counts": np.bincount(
            rollout.actions,
            minlength=len(ADAPTIVE_PROFILES),
        ).tolist(),
    }


def save_profile_checkpoint(
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


def load_profile_checkpoint(
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


def train_profile_ppo(
    feature_checkpoint: Path,
    output: Path,
    *,
    iterations: int = 10,
    rollout_episodes: int = 10,
    evaluation_episodes: int = 10,
    base_seed: int = 8000,
    learning_rate: float = 3e-4,
    seed: int = 2026,
) -> dict[str, Any]:
    config = DEFAULT_PROFILE_CONFIG
    parameters = initialize_profile_parameters(config, seed=seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    optimizer_state = optimizer.init(parameters)
    history: list[dict[str, Any]] = []

    for iteration in range(iterations):
        rollout = collect_profile_rollouts(
            parameters,
            config,
            feature_checkpoint,
            episodes=rollout_episodes,
            base_seed=base_seed + iteration * 100,
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
        evaluation = collect_profile_rollouts(
            parameters,
            config,
            feature_checkpoint,
            episodes=evaluation_episodes,
            base_seed=base_seed + 50_000 + iteration * evaluation_episodes,
            stochastic=False,
            random_seed=seed,
        )
        history.append(
            {
                "iteration": iteration + 1,
                "rollout": _summary(rollout),
                "evaluation": _summary(evaluation),
                "update": update_metrics,
            }
        )

    metadata = {
        "model": "executor-sticky-profile-ppo-v1",
        "profiles": [asdict(profile) for profile in ADAPTIVE_PROFILES],
        "incumbent_profile": INCUMBENT_PROFILE.name,
        "history": history,
    }
    save_profile_checkpoint(output, parameters, config, feature_checkpoint, metadata)
    return metadata


def evaluate_profile_checkpoint(
    checkpoint: Path,
    *,
    episodes: int = 30,
    base_seed: int = 90_000,
) -> dict[str, Any]:
    parameters, config, feature_checkpoint, metadata = load_profile_checkpoint(checkpoint)
    candidate = collect_profile_rollouts(
        parameters,
        config,
        feature_checkpoint,
        episodes=episodes,
        base_seed=base_seed,
        stochastic=False,
        random_seed=0,
    )
    baseline_parameters = initialize_profile_parameters(config, seed=0)
    baseline_parameters["actor_weight"] = jnp.zeros_like(
        baseline_parameters["actor_weight"]
    )
    baseline_parameters["actor_bias"] = jnp.asarray(
        [100.0] + [0.0] * (config.action_dim - 1),
        dtype=jnp.float32,
    )
    baseline = collect_profile_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        episodes=episodes,
        base_seed=base_seed,
        stochastic=False,
        random_seed=0,
    )
    margin_deltas = [
        float(candidate_game["margin"] - baseline_game["margin"])
        for candidate_game, baseline_game in zip(
            candidate.games,
            baseline.games,
            strict=True,
        )
    ]
    promotion = (
        fmean(margin_deltas) > 0.0
        and min(margin_deltas) >= 0.0
        and sum(delta > 0.0 for delta in margin_deltas)
        > sum(delta < 0.0 for delta in margin_deltas)
        and all(int(game["ineffective_actions"]) == 0 for game in candidate.games)
    )
    return {
        "checkpoint": str(checkpoint),
        "model": metadata["model"],
        "profiles": [profile.name for profile in ADAPTIVE_PROFILES],
        "candidate": _summary(candidate),
        "incumbent": _summary(baseline),
        "paired_delta": {
            "mean_margin": fmean(margin_deltas),
            "worst_margin": min(margin_deltas),
            "better": sum(delta > 0.0 for delta in margin_deltas),
            "equal": sum(delta == 0.0 for delta in margin_deltas),
            "worse": sum(delta < 0.0 for delta in margin_deltas),
        },
        "promotion": promotion,
    }


class ProfilePPOPolicy:
    """Greedy inference wrapper with episode-sticky commitments."""

    def __init__(self, checkpoint: Path) -> None:
        parameters, config, feature_checkpoint, metadata = load_profile_checkpoint(checkpoint)
        self._parameters = parameters
        self._config = config
        self.metadata = metadata
        self._encoder = SpatialRecurrentPolicy(feature_checkpoint)
        self._controller = StickyProfileController()
        self._profile_id = 0

    def reset(self) -> None:
        self._encoder.reset()
        self._controller.reset()
        self._profile_id = 0

    def __call__(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        turn = int(observation.get("step", 0))
        features = None
        if self._config.observe_every_turn and _has_pending_profile_decision(
            turn,
            self._config,
        ):
            features = self._encoder.encode_hidden(observation)
        if turn < self._config.warmup_steps:
            self._profile_id = 0
        elif (turn - self._config.warmup_steps) % self._config.decision_interval == 0:
            if features is None:
                features = self._encoder.encode_hidden(observation)
            logits, _ = semantic_forward(
                self._parameters,
                jnp.asarray(features[None, :]),
            )
            self._profile_id = int(np.asarray(jax.device_get(logits))[0].argmax())
        effective = self._controller.select(ADAPTIVE_PROFILES[self._profile_id])
        return profile_targets(effective, observation)
