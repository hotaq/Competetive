"""PPO gate over spatial-BC components with an exact standalone teacher fallback."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.replay_dataset import encode_observation
from kaggriculture_rl.imitation.teacher_dataset import (
    StandaloneFilePolicy,
    sanitize_standalone_action,
)
from kaggriculture_rl.learning.gated_ppo import (
    GateConfig,
    GateParameters,
    RolloutBatch,
    gate_forward,
    initialize_gate_parameters,
    load_gate_checkpoint,
    paired_margin_improvement,
    ppo_update,
    save_gate_checkpoint,
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


def teacher_gate_features(
    observation: dict[str, Any],
    prediction: Any,
    config: GateConfig,
) -> np.ndarray:
    max_hands = config.action_dim - 2
    confidence = np.zeros(config.action_dim, dtype=np.float32)
    confidence[0] = float(prediction.farmer_confidence)
    hand_confidences = prediction.hand_confidences[:max_hands]
    confidence[1 : 1 + len(hand_confidences)] = hand_confidences
    confidence[-1] = min(prediction.market_confidences, default=0.0)
    return np.concatenate(
        (encode_observation(observation, config.observation_dim), confidence)
    )


def teacher_gate_mask(
    hands_count: int,
    *,
    max_hands: int,
    market_safe: bool = True,
    enabled_components: Sequence[str] = ("farmer", "hands", "market"),
) -> np.ndarray:
    if not 0 <= hands_count <= max_hands:
        raise ValueError(f"hands_count must be within 0..{max_hands}")
    enabled = set(enabled_components)
    unknown = enabled.difference(("farmer", "hands", "market"))
    if unknown:
        raise ValueError(f"unknown gate components: {sorted(unknown)}")
    mask = np.zeros(max_hands + 2, dtype=np.bool_)
    mask[0] = "farmer" in enabled
    if "hands" in enabled:
        mask[1 : 1 + hands_count] = True
    mask[-1] = "market" in enabled and market_safe
    return mask


def teacher_confidence_mask(
    prediction: Any,
    *,
    max_hands: int,
    minimum_confidence: float,
) -> np.ndarray:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be within 0..1")
    mask = np.zeros(max_hands + 2, dtype=np.bool_)
    mask[0] = float(prediction.farmer_confidence) >= minimum_confidence
    hand_confidences = prediction.hand_confidences[:max_hands]
    mask[1 : 1 + len(hand_confidences)] = (
        np.asarray(hand_confidences) >= minimum_confidence
    )
    market_confidence = min(prediction.market_confidences, default=0.0)
    mask[-1] = float(market_confidence) >= minimum_confidence
    return mask


def teacher_difference_mask(
    prediction: Any,
    fallback: dict[str, Any],
    *,
    max_hands: int,
) -> np.ndarray:
    mask = np.zeros(max_hands + 2, dtype=np.bool_)
    mask[0] = prediction.action.get("farmer") != fallback.get("farmer")
    prediction_hands = list(prediction.action.get("hands", ()))[:max_hands]
    fallback_hands = list(fallback.get("hands", ()))[:max_hands]
    for index, (prediction_hand, fallback_hand) in enumerate(
        zip(prediction_hands, fallback_hands, strict=False)
    ):
        mask[1 + index] = prediction_hand != fallback_hand
    mask[-1] = prediction.action.get("market") != fallback.get("market")
    return mask


def teacher_gate_action(
    prediction: Any,
    fallback: dict[str, Any],
    gates: np.ndarray,
    *,
    max_hands: int,
) -> dict[str, Any]:
    if gates.shape != (max_hands + 2,):
        raise ValueError(f"expected {max_hands + 2} gate actions")
    return {
        "farmer": prediction.action["farmer"] if gates[0] else fallback["farmer"],
        "hands": [
            predicted if gates[index + 1] else fallback["hands"][index]
            for index, predicted in enumerate(prediction.action["hands"])
        ],
        "market": prediction.action["market"] if gates[-1] else fallback["market"],
    }


def _bernoulli_log_probability(
    logits: np.ndarray,
    actions: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = np.where(
        actions,
        -np.logaddexp(0.0, -logits),
        -np.logaddexp(0.0, logits),
    )
    return float((selected * mask).sum())


def _market_within_hand_limit(
    prediction: Any,
    *,
    hands_count: int,
    max_hands: int,
) -> bool:
    hires = sum(
        1
        for order in prediction.action.get("market", [])
        if isinstance(order, list) and order and order[0] == "HIRE"
    )
    # A learner HIRE can consume capacity that the V16 tape expects to use later,
    # causing a delayed overflow even when the immediate count looks legal.
    # Keep all hiring under the exact teacher and explore only non-HIRE market edits.
    return hands_count <= max_hands and hires == 0


def _summary(games: Sequence[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(game["margin"]) for game in games]
    money = [float(game["candidate_money"]) for game in games]
    summary: dict[str, Any] = {
        "games": len(games),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "mean_accepted_component_rate": fmean(
            float(game["accepted_component_rate"]) for game in games
        ),
        "mean_action_difference_rate": fmean(
            float(game["action_difference_rate"]) for game in games
        ),
    }
    by_opponent: dict[str, dict[str, Any]] = {}
    for opponent in sorted({str(game["opponent"]) for game in games}):
        opponent_games = [game for game in games if str(game["opponent"]) == opponent]
        opponent_margins = [float(game["margin"]) for game in opponent_games]
        by_opponent[opponent] = {
            "games": len(opponent_games),
            "wins": sum(margin > 0 for margin in opponent_margins),
            "draws": sum(margin == 0 for margin in opponent_margins),
            "losses": sum(margin < 0 for margin in opponent_margins),
            "mean_margin": fmean(opponent_margins),
            "worst_margin": min(opponent_margins),
        }
    summary["by_opponent"] = by_opponent
    return summary


def _opponent_catalog(
    crop_opponents: Sequence[tuple[str, int]],
    file_opponents: Sequence[Path],
) -> tuple[tuple[str, Any], ...]:
    catalog: list[tuple[str, Any]] = [
        (f"crop:{crop}:{plots}", (crop, plots))
        for crop, plots in crop_opponents
    ]
    catalog.extend(
        (f"file:{path.resolve()}", path.resolve()) for path in file_opponents
    )
    if not catalog:
        raise ValueError("at least one PPO opponent is required")
    return tuple(catalog)


def collect_teacher_gate_rollouts(
    parameters: GateParameters,
    config: GateConfig,
    spatial_checkpoint: Path,
    teacher_path: Path,
    *,
    episodes: int,
    base_seed: int,
    crop_opponents: Sequence[tuple[str, int]] = DEFAULT_CROP_OPPONENTS,
    file_opponents: Sequence[Path] = (),
    enabled_components: Sequence[str] = ("farmer", "hands", "market"),
    minimum_confidence: float = 0.0,
    stochastic: bool,
    random_seed: int,
    reward_scale: float = 10_000.0,
) -> RolloutBatch:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number")
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    max_hands = config.action_dim - 2
    catalog = _opponent_catalog(crop_opponents, file_opponents)
    random = np.random.default_rng(random_seed)
    forward = jax.jit(lambda row: gate_forward(parameters, row))
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    log_probability_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    games: list[dict[str, Any]] = []

    for episode in range(episodes):
        pair_index = episode // 2
        candidate_seat = episode % 2
        opponent_seat = 1 - candidate_seat
        seed = base_seed + pair_index
        opponent_label, opponent_payload = catalog[pair_index % len(catalog)]
        engine = Engine(
            {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
        )
        spatial = SpatialRecurrentPolicy(spatial_checkpoint)
        teacher = StandaloneFilePolicy(teacher_path, role="ppo_teacher")
        file_opponent: StandaloneFilePolicy | None = None
        crop_opponent: CropHeuristicPolicy | None = None
        if opponent_label.startswith("file:"):
            file_opponent = StandaloneFilePolicy(opponent_payload, role="ppo_opponent")
        else:
            crop, plots = opponent_payload
            crop_opponent = CropHeuristicPolicy(
                crop=crop,
                target_quantity=plots,
                sell_reserve=0,
            )

        previous_margin = 0.0
        accepted_components = 0
        active_components = 0
        action_differences = 0
        try:
            for turn in range(719):
                observation = engine.observation(candidate_seat)
                opponent_observation = engine.observation(opponent_seat)
                hands_count = len(
                    observation["farms"][candidate_seat].get("hands", [])
                )
                if hands_count > max_hands:
                    raise RuntimeError(
                        f"rollout exceeded checkpoint hand capacity: {hands_count}"
                    )
                fallback = sanitize_standalone_action(
                    engine,
                    candidate_seat,
                    teacher(observation),
                    hands_count,
                )
                prediction = spatial.predict_with_confidence(engine, observation)
                features = teacher_gate_features(observation, prediction, config)
                logits, values = forward(jnp.asarray(features[None, :]))
                logits_numpy = np.asarray(jax.device_get(logits))[0]
                value = float(np.asarray(jax.device_get(values))[0])
                mask = teacher_gate_mask(
                    hands_count,
                    max_hands=max_hands,
                    market_safe=_market_within_hand_limit(
                        prediction,
                        hands_count=hands_count,
                        max_hands=max_hands,
                    ),
                    enabled_components=enabled_components,
                )
                mask &= teacher_confidence_mask(
                    prediction,
                    max_hands=max_hands,
                    minimum_confidence=minimum_confidence,
                )
                mask &= teacher_difference_mask(
                    prediction,
                    fallback,
                    max_hands=max_hands,
                )
                if stochastic:
                    probabilities = 1.0 / (1.0 + np.exp(-logits_numpy))
                    gates = random.random(config.action_dim) < probabilities
                else:
                    gates = logits_numpy >= 0.0
                gates &= mask
                log_probability = _bernoulli_log_probability(
                    logits_numpy,
                    gates,
                    mask,
                )
                candidate_action = teacher_gate_action(
                    prediction,
                    fallback,
                    gates,
                    max_hands=max_hands,
                )
                candidate_action = sanitize_standalone_action(
                    engine,
                    candidate_seat,
                    candidate_action,
                    hands_count,
                )

                if file_opponent is not None:
                    raw_opponent = file_opponent(opponent_observation)
                else:
                    assert crop_opponent is not None
                    plan = engine.plan(
                        opponent_seat,
                        crop_opponent(
                            {"shed": opponent_observation["private"]["shed"]}
                        ),
                    )
                    raw_opponent = plan["joint_action"]["actions"][opponent_seat]
                opponent_hands = len(
                    opponent_observation["farms"][opponent_seat].get("hands", [])
                )
                opponent_action = sanitize_standalone_action(
                    engine,
                    opponent_seat,
                    raw_opponent,
                    opponent_hands,
                )
                actions = [opponent_action, opponent_action]
                actions[candidate_seat] = candidate_action
                state = engine.step(actions)
                money = [float(farm["money"]) for farm in state["farms"]]
                margin = money[candidate_seat] - money[opponent_seat]

                feature_rows.append(features)
                action_rows.append(gates)
                mask_rows.append(mask)
                log_probability_rows.append(log_probability)
                value_rows.append(value)
                reward_rows.append((margin - previous_margin) / reward_scale)
                done_rows.append(turn == 718)
                previous_margin = margin
                accepted_components += int(gates.sum())
                active_components += int(mask.sum())
                action_differences += int(candidate_action != fallback)
        finally:
            teacher.close()
            if file_opponent is not None:
                file_opponent.close()

        games.append(
            {
                "seed": seed,
                "candidate_seat": candidate_seat,
                "opponent": opponent_label,
                "candidate_money": money[candidate_seat],
                "opponent_money": money[opponent_seat],
                "margin": previous_margin,
                "accepted_component_rate": accepted_components
                / max(active_components, 1),
                "action_difference_rate": action_differences / 719,
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
        games=tuple(games),
    )


def train_teacher_gate_ppo(
    spatial_checkpoint: Path,
    teacher_path: Path,
    output_checkpoint: Path,
    *,
    iterations: int = 2,
    rollout_episodes: int = 8,
    evaluation_episodes: int = 8,
    base_seed: int = 700_000,
    max_hands: int = 14,
    hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    initial_acceptance: float = 0.02,
    reward_scale: float = 10_000.0,
    seed: int = 2026,
    crop_opponents: Sequence[tuple[str, int]] = DEFAULT_CROP_OPPONENTS,
    file_opponents: Sequence[Path] = (),
    enabled_components: Sequence[str] = ("farmer", "hands", "market"),
    minimum_confidence: float = 0.0,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    enabled_components = tuple(dict.fromkeys(enabled_components))
    if not enabled_components:
        raise ValueError("at least one gate component must be enabled")
    unknown_components = set(enabled_components).difference(
        ("farmer", "hands", "market")
    )
    if unknown_components:
        raise ValueError(f"unknown gate components: {sorted(unknown_components)}")
    config = GateConfig(hidden_dim=hidden_dim, action_dim=max_hands + 2)
    previous_history: list[dict[str, Any]] = []
    resume: dict[str, Any] | None = None
    if initial_checkpoint is None:
        parameters = initialize_gate_parameters(
            config,
            seed=seed,
            initial_acceptance=initial_acceptance,
        )
    else:
        parameters, checkpoint_config, checkpoint_metadata = load_gate_checkpoint(
            initial_checkpoint
        )
        if checkpoint_config != config:
            raise ValueError(
                "initial checkpoint gate config does not match requested config"
            )
        previous_report = checkpoint_metadata.get("report", {})
        previous_spatial = previous_report.get("spatial_checkpoint")
        if previous_spatial and Path(previous_spatial).resolve() != spatial_checkpoint.resolve():
            raise ValueError("initial checkpoint uses a different spatial checkpoint")
        previous_teacher = previous_report.get("teacher_path")
        if previous_teacher and Path(previous_teacher).resolve() != teacher_path.resolve():
            raise ValueError("initial checkpoint uses a different teacher")
        previous_history = list(previous_report.get("history", []))
        resume = {
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "initial_checkpoint_sha256": hashlib.sha256(
                initial_checkpoint.read_bytes()
            ).hexdigest(),
            "previous_iterations": len(previous_history),
            "optimizer_state_restored": False,
        }
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    optimizer_state = optimizer.init(parameters)

    incumbent_parameters = copy.deepcopy(parameters)
    incumbent_parameters["policy_weight"] = jnp.zeros_like(
        incumbent_parameters["policy_weight"]
    )
    incumbent_parameters["policy_bias"] = jnp.full_like(
        incumbent_parameters["policy_bias"],
        -100.0,
    )
    evaluation_seed = base_seed + 50_000
    incumbent = collect_teacher_gate_rollouts(
        incumbent_parameters,
        config,
        spatial_checkpoint,
        teacher_path,
        episodes=evaluation_episodes,
        base_seed=evaluation_seed,
        crop_opponents=crop_opponents,
        file_opponents=file_opponents,
        enabled_components=enabled_components,
        minimum_confidence=minimum_confidence,
        stochastic=False,
        random_seed=seed,
        reward_scale=reward_scale,
    )
    history = previous_history
    iteration_start = len(previous_history)
    for iteration_offset in range(iterations):
        iteration = iteration_start + iteration_offset
        rollout = collect_teacher_gate_rollouts(
            parameters,
            config,
            spatial_checkpoint,
            teacher_path,
            episodes=rollout_episodes,
            base_seed=base_seed + iteration_offset * (rollout_episodes // 2),
            crop_opponents=crop_opponents,
            file_opponents=file_opponents,
            enabled_components=enabled_components,
            minimum_confidence=minimum_confidence,
            stochastic=True,
            random_seed=seed + iteration,
            reward_scale=reward_scale,
        )
        parameters, optimizer_state, update = ppo_update(
            parameters,
            optimizer_state,
            optimizer,
            rollout,
            update_epochs=4,
            batch_size=256,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.002,
            seed=seed + iteration,
        )
        evaluation = collect_teacher_gate_rollouts(
            parameters,
            config,
            spatial_checkpoint,
            teacher_path,
            episodes=evaluation_episodes,
            base_seed=evaluation_seed,
            crop_opponents=crop_opponents,
            file_opponents=file_opponents,
            enabled_components=enabled_components,
            minimum_confidence=minimum_confidence,
            stochastic=False,
            random_seed=seed,
            reward_scale=reward_scale,
        )
        improvement = paired_margin_improvement(evaluation.games, incumbent.games)
        history.append(
            {
                "iteration": iteration + 1,
                "rollout": _summary(rollout.games),
                "rollout_games": list(rollout.games),
                "update": update,
                "evaluation": _summary(evaluation.games),
                "evaluation_games": list(evaluation.games),
                "improvement": improvement,
            }
        )

    final = history[-1]
    promotion = {
        "passed": (
            final["improvement"]["mean_margin_delta"] > 0.0
            and final["improvement"]["worst_margin_delta"] >= 0.0
            and final["evaluation"]["mean_action_difference_rate"] > 0.0
        ),
        "rule": "positive paired mean, non-negative worst delta, and non-zero behavior",
    }
    report = {
        "model": "spatial-teacher-component-gate-ppo-v1",
        "spatial_checkpoint": str(spatial_checkpoint.resolve()),
        "teacher_path": str(teacher_path.resolve()),
        "config": {
            "gate": {
                "observation_dim": config.observation_dim,
                "hidden_dim": config.hidden_dim,
                "action_dim": config.action_dim,
            },
            "max_hands": max_hands,
            "iterations": iterations,
            "total_iterations": len(history),
            "rollout_episodes": rollout_episodes,
            "evaluation_episodes": evaluation_episodes,
            "base_seed": base_seed,
            "learning_rate": learning_rate,
            "initial_acceptance": initial_acceptance,
            "reward_scale": reward_scale,
            "seed": seed,
            "crop_opponents": list(crop_opponents),
            "file_opponents": [str(path.resolve()) for path in file_opponents],
            "enabled_components": list(enabled_components),
            "minimum_confidence": minimum_confidence,
        },
        "incumbent": _summary(incumbent.games),
        "incumbent_games": list(incumbent.games),
        "history": history,
        "promotion": promotion,
        "resume": resume,
    }
    save_gate_checkpoint(output_checkpoint, parameters, config, report)
    return report


def save_teacher_gate_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
