"""Replay-aware macro BC followed by safety-gated mixed-league PPO."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kaggriculture_rl.learning.profile_ppo import (
    DEFAULT_LEAGUE,
    collect_profile_rollouts,
    load_profile_checkpoint,
    save_profile_checkpoint,
)
from kaggriculture_rl.learning.replay_profile_ppo import (
    Top1Tape,
    collect_tape_rollouts,
    concatenate_rollouts,
    load_top1_tapes,
)
from kaggriculture_rl.learning.semantic_ppo import (
    SemanticParameters,
    SemanticPPOConfig,
    SemanticRollout,
    ppo_update,
)
from kaggriculture_rl.policy.production_profile import ADAPTIVE_PROFILES


def _copy_parameters(parameters: SemanticParameters) -> SemanticParameters:
    return {key: jnp.array(value) for key, value in parameters.items()}


def _force_profile_parameters(
    parameters: SemanticParameters,
    profile_id: int,
) -> SemanticParameters:
    forced = _copy_parameters(parameters)
    forced["actor_weight"] = jnp.zeros_like(forced["actor_weight"])
    forced["actor_bias"] = jnp.zeros_like(forced["actor_bias"]).at[profile_id].set(100.0)
    return forced


def _actor_bc(
    parameters: SemanticParameters,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[SemanticParameters, dict[str, float]]:
    actor = {
        "weight": parameters["actor_weight"],
        "bias": parameters["actor_bias"],
    }
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(actor)
    feature_array = jnp.asarray(features)
    label_array = jnp.asarray(labels)

    @jax.jit
    def update(
        current: dict[str, jax.Array],
        state: optax.OptState,
    ) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
        def loss_fn(candidate: dict[str, jax.Array]) -> jax.Array:
            logits = feature_array @ candidate["weight"] + candidate["bias"]
            return optax.softmax_cross_entropy_with_integer_labels(
                logits,
                label_array,
            ).mean()

        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, next_state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), next_state, loss

    loss = jnp.asarray(float("nan"))
    for _ in range(steps):
        actor, optimizer_state, loss = update(actor, optimizer_state)

    updated = _copy_parameters(parameters)
    updated["actor_weight"] = actor["weight"]
    updated["actor_bias"] = actor["bias"]
    predictions = np.asarray(
        jax.device_get(feature_array @ actor["weight"] + actor["bias"])
    ).argmax(axis=-1)
    return updated, {
        "loss": float(loss),
        "accuracy": float((predictions == labels).mean()),
    }


def _tape_oracle_labels(
    parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
    tapes: tuple[Top1Tape, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_profile: list[SemanticRollout] = []
    for profile_id in range(len(ADAPTIVE_PROFILES)):
        by_profile.append(
            collect_tape_rollouts(
                parameters,
                config,
                feature_checkpoint,
                tapes,
                stochastic=False,
                random_seed=0,
                force_profile_id=profile_id,
            )
        )

    features = by_profile[0].features
    remaining = set(range(len(tapes)))
    groups: list[list[int]] = []
    while remaining:
        anchor = min(remaining)
        group = [
            index
            for index in sorted(remaining)
            if np.allclose(features[anchor], features[index], rtol=1e-6, atol=1e-6)
        ]
        groups.append(group)
        remaining.difference_update(group)

    labels = np.full(len(tapes), 3, dtype=np.int32)
    group_by_index: dict[int, list[int]] = {}
    for group in groups:
        safe_profiles: list[tuple[tuple[float, ...], int]] = []
        for profile_id, rollout in enumerate(by_profile):
            money_deltas = [
                float(
                    rollout.games[index]["candidate_money"]
                    - by_profile[3].games[index]["candidate_money"]
                )
                for index in group
            ]
            margin_deltas = [
                float(
                    rollout.games[index]["margin"]
                    - by_profile[3].games[index]["margin"]
                )
                for index in group
            ]
            if min(money_deltas) >= 0.0 and min(margin_deltas) >= 0.0:
                safe_profiles.append(
                    (
                        (
                            min(margin_deltas),
                            min(money_deltas),
                            fmean(margin_deltas),
                            fmean(money_deltas),
                            float(profile_id == 3),
                        ),
                        profile_id,
                    )
                )
        _, selected_profile = max(safe_profiles)
        labels[group] = selected_profile
        for index in group:
            group_by_index[index] = group

    rows: list[dict[str, Any]] = []
    for tape_index, tape in enumerate(tapes):
        candidates = [
            {
                "profile_id": profile_id,
                "profile": ADAPTIVE_PROFILES[profile_id].name,
                "money": float(rollout.games[tape_index]["candidate_money"]),
                "margin": float(rollout.games[tape_index]["margin"]),
            }
            for profile_id, rollout in enumerate(by_profile)
        ]
        baseline = next(
            candidate for candidate in candidates if candidate["profile_id"] == 3
        )
        best = next(
            candidate
            for candidate in candidates
            if candidate["profile_id"] == int(labels[tape_index])
        )
        candidates.sort(
            key=lambda candidate: (candidate["margin"], candidate["money"]),
            reverse=True,
        )
        rows.append(
            {
                "episode_id": tape.episode_id,
                "best": best,
                "baseline": baseline,
                "equivalence_group": [
                    tapes[index].episode_id for index in group_by_index[tape_index]
                ],
                "candidates": candidates,
            }
        )
    return labels, rows


def _paired_deltas(
    candidate: SemanticRollout,
    baseline: SemanticRollout,
) -> dict[str, Any]:
    money = [
        float(candidate_game["candidate_money"] - baseline_game["candidate_money"])
        for candidate_game, baseline_game in zip(
            candidate.games,
            baseline.games,
            strict=True,
        )
    ]
    margins = [
        float(candidate_game["margin"] - baseline_game["margin"])
        for candidate_game, baseline_game in zip(
            candidate.games,
            baseline.games,
            strict=True,
        )
    ]
    actions = [
        int(game["chosen_profiles"][0])
        for game in candidate.games
        if game["chosen_profiles"]
    ]
    return {
        "games": len(margins),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "better": sum(delta > 0.0 for delta in margins),
        "equal": sum(delta == 0.0 for delta in margins),
        "worse": sum(delta < 0.0 for delta in margins),
        "chosen_profile_ids": actions,
        "distinct_profiles": len(set(actions)),
        "ineffective_actions": sum(
            int(game["ineffective_actions"]) for game in candidate.games
        ),
    }


def _safe_score(tape: dict[str, Any], crop: dict[str, Any]) -> tuple[float, ...]:
    safe = min(tape["worst_money"], tape["worst_margin"], crop["worst_margin"])
    return (
        float(safe),
        float(tape["worst_margin"]),
        float(crop["worst_margin"]),
        float(tape["mean_margin"]),
        float(crop["mean_margin"]),
        float(tape["distinct_profiles"]),
    )


def _evaluate_candidate(
    parameters: SemanticParameters,
    baseline_parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
    tapes: tuple[Top1Tape, ...],
    *,
    crop_episodes: int,
    crop_seed: int,
    baseline_tape: SemanticRollout | None = None,
    baseline_crop: SemanticRollout | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_tape = collect_tape_rollouts(
        parameters,
        config,
        feature_checkpoint,
        tapes,
        stochastic=False,
        random_seed=0,
    )
    if baseline_tape is None:
        baseline_tape = collect_tape_rollouts(
            baseline_parameters,
            config,
            feature_checkpoint,
            tapes,
            stochastic=False,
            random_seed=0,
        )
    candidate_crop = collect_profile_rollouts(
        parameters,
        config,
        feature_checkpoint,
        episodes=crop_episodes,
        base_seed=crop_seed,
        opponent_specs=DEFAULT_LEAGUE,
        stochastic=False,
        random_seed=0,
    )
    if baseline_crop is None:
        baseline_crop = collect_profile_rollouts(
            baseline_parameters,
            config,
            feature_checkpoint,
            episodes=crop_episodes,
            base_seed=crop_seed,
            opponent_specs=DEFAULT_LEAGUE,
            stochastic=False,
            random_seed=0,
        )
    return (
        _paired_deltas(candidate_tape, baseline_tape),
        _paired_deltas(candidate_crop, baseline_crop),
    )


def train_robust_profile_ppo(
    source_checkpoint: Path,
    manifest_path: Path | tuple[Path, ...],
    output: Path,
    *,
    bc_steps: int = 500,
    ppo_iterations: int = 8,
    crop_rollout_episodes: int = 16,
    crop_evaluation_episodes: int = 30,
    crop_training_seed: int = 130_000,
    crop_evaluation_seed: int = 140_000,
    tape_repeat: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 3030,
) -> dict[str, Any]:
    source_parameters, source_config, feature_checkpoint, source_metadata = (
        load_profile_checkpoint(source_checkpoint)
    )
    # Investment profiles are irreversible.  One decision after a 72-turn
    # safe opening is the correct credit-assignment horizon for this head.
    config = replace(
        source_config,
        decision_interval=647,
        observe_every_turn=True,
    )
    baseline_parameters = _force_profile_parameters(source_parameters, 3)
    manifest_paths = (
        (manifest_path,) if isinstance(manifest_path, Path) else manifest_path
    )
    if not manifest_paths:
        raise ValueError("at least one replay manifest is required")
    tapes = tuple(
        tape
        for path in manifest_paths
        for tape in load_top1_tapes(path)
    )
    oracle_labels, oracle_rows = _tape_oracle_labels(
        baseline_parameters,
        config,
        feature_checkpoint,
        tapes,
    )

    tape_probe = collect_tape_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        tapes,
        stochastic=False,
        random_seed=0,
        force_profile_id=0,
    )
    crop_probe = collect_profile_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        episodes=60,
        base_seed=crop_training_seed,
        opponent_specs=DEFAULT_LEAGUE,
        stochastic=False,
        random_seed=0,
    )
    training_features = np.concatenate(
        [
            crop_probe.features,
            np.repeat(tape_probe.features, tape_repeat, axis=0),
        ]
    )
    training_labels = np.concatenate(
        [
            np.full(len(crop_probe.features), 3, dtype=np.int32),
            np.repeat(oracle_labels, tape_repeat),
        ]
    )
    parameters, bc_metrics = _actor_bc(
        baseline_parameters,
        training_features,
        training_labels,
        steps=bc_steps,
        learning_rate=1e-2,
    )

    baseline_tape_rollout = collect_tape_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        tapes,
        stochastic=False,
        random_seed=0,
    )
    baseline_crop_rollout = collect_profile_rollouts(
        baseline_parameters,
        config,
        feature_checkpoint,
        episodes=crop_evaluation_episodes,
        base_seed=crop_evaluation_seed,
        opponent_specs=DEFAULT_LEAGUE,
        stochastic=False,
        random_seed=0,
    )
    baseline_tape_metrics = _paired_deltas(
        baseline_tape_rollout,
        baseline_tape_rollout,
    )
    baseline_crop_metrics = _paired_deltas(
        baseline_crop_rollout,
        baseline_crop_rollout,
    )
    best_parameters = _copy_parameters(baseline_parameters)
    best_tape = baseline_tape_metrics
    best_crop = baseline_crop_metrics
    best_score = _safe_score(best_tape, best_crop)
    best_stage = "source"
    history: list[dict[str, Any]] = []

    bc_tape, bc_crop = _evaluate_candidate(
        parameters,
        baseline_parameters,
        config,
        feature_checkpoint,
        tapes,
        crop_episodes=crop_evaluation_episodes,
        crop_seed=crop_evaluation_seed,
        baseline_tape=baseline_tape_rollout,
        baseline_crop=baseline_crop_rollout,
    )
    bc_score = _safe_score(bc_tape, bc_crop)
    if bc_score > best_score:
        best_parameters = _copy_parameters(parameters)
        best_tape, best_crop, best_score = bc_tape, bc_crop, bc_score
        best_stage = "macro_bc"
    history.append(
        {
            "stage": "macro_bc",
            "bc": bc_metrics,
            "tape": bc_tape,
            "crop": bc_crop,
            "safe_score": list(bc_score),
        }
    )

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    optimizer_state = optimizer.init(parameters)
    for iteration in range(ppo_iterations):
        tape_rollout = collect_tape_rollouts(
            parameters,
            config,
            feature_checkpoint,
            tapes,
            stochastic=True,
            random_seed=seed + iteration,
        )
        crop_rollout = collect_profile_rollouts(
            parameters,
            config,
            feature_checkpoint,
            episodes=crop_rollout_episodes,
            base_seed=crop_training_seed + 10_000 + iteration * crop_rollout_episodes,
            opponent_specs=DEFAULT_LEAGUE,
            stochastic=True,
            random_seed=seed + 100 + iteration,
        )
        rollout = concatenate_rollouts(tape_rollout, crop_rollout)
        parameters, optimizer_state, update_metrics = ppo_update(
            parameters,
            optimizer_state,
            optimizer,
            rollout,
            update_epochs=6,
            batch_size=32,
            clip_ratio=0.15,
            value_coefficient=0.5,
            entropy_coefficient=0.005,
            seed=seed + iteration,
        )
        tape_metrics, crop_metrics = _evaluate_candidate(
            parameters,
            baseline_parameters,
            config,
            feature_checkpoint,
            tapes,
            crop_episodes=crop_evaluation_episodes,
            crop_seed=crop_evaluation_seed,
            baseline_tape=baseline_tape_rollout,
            baseline_crop=baseline_crop_rollout,
        )
        score = _safe_score(tape_metrics, crop_metrics)
        if score > best_score:
            best_parameters = _copy_parameters(parameters)
            best_tape, best_crop, best_score = tape_metrics, crop_metrics, score
            best_stage = f"ppo_{iteration + 1}"
        history.append(
            {
                "stage": f"ppo_{iteration + 1}",
                "update": update_metrics,
                "tape": tape_metrics,
                "crop": crop_metrics,
                "safe_score": list(score),
            }
        )

    promotion = (
        best_tape["worst_money"] >= 0.0
        and best_tape["worst_margin"] >= 0.0
        and best_crop["worst_margin"] >= 0.0
        and best_tape["ineffective_actions"] == 0
        and best_crop["ineffective_actions"] == 0
        and best_tape["distinct_profiles"] >= 2
        and (best_tape["better"] > 0 or best_crop["better"] > 0)
    )
    metadata = {
        "model": "executor-history-equivalence-profile-ppo-v4",
        "source_checkpoint": str(source_checkpoint),
        "source_model": source_metadata["model"],
        "manifests": [str(path) for path in manifest_paths],
        "profiles": [asdict(profile) for profile in ADAPTIVE_PROFILES],
        "oracle": oracle_rows,
        "best_stage": best_stage,
        "best_tape": best_tape,
        "best_crop": best_crop,
        "safe_score": list(best_score),
        "promotion": promotion,
        "history": history,
        "limitation": (
            "Top-1 recordings are open-loop action tapes and the eight tape contexts "
            "are training-visible; promotion is therefore local, not proof against a "
            "live or unseen Top-1 policy."
        ),
    }
    save_profile_checkpoint(output, best_parameters, config, feature_checkpoint, metadata)
    return metadata


def save_robust_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
