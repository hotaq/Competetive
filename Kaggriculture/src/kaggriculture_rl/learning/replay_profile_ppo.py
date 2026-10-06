"""Profile-PPO rollouts against recorded Top-1 action tapes.

The tapes are useful hard opponents, but remain open-loop counterfactuals:
changing our actions changes their state, so later recorded actions can become
stale.  Staleness is measured and surfaced rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.profile_ppo import (
    _choose_action,
    semantic_forward,
)
from kaggriculture_rl.learning.semantic_ppo import (
    SemanticParameters,
    SemanticPPOConfig,
    SemanticRollout,
)
from kaggriculture_rl.learning.spatial_recurrent import SpatialRecurrentPolicy
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    StickyProfileController,
    profile_targets,
)


@dataclass(frozen=True, slots=True)
class Top1Tape:
    episode_id: int
    top1_seat: int
    config: dict[str, Any]
    actions: tuple[dict[str, Any], ...]


def load_top1_tapes(manifest_path: Path) -> tuple[Top1Tape, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tapes: list[Top1Tape] = []
    for row in manifest["episodes"]:
        replay_path = manifest_path.parent / f"{int(row['episode_id'])}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        config = dict(replay.get("configuration") or {})
        config["seed"] = int(replay["info"]["seed"])
        top1_seat = int(row["top1_seat"])
        actions = tuple(
            frame[top1_seat]["action"]
            for frame in replay["steps"][1:]
        )
        if len(actions) != 719:
            raise ValueError(
                f"episode {row['episode_id']} has {len(actions)} transitions, expected 719"
            )
        tapes.append(
            Top1Tape(
                episode_id=int(row["episode_id"]),
                top1_seat=top1_seat,
                config=config,
                actions=actions,
            )
        )
    if not tapes:
        raise ValueError("Top-1 manifest contains no episodes")
    return tuple(tapes)


def collect_tape_rollouts(
    parameters: SemanticParameters,
    config: SemanticPPOConfig,
    feature_checkpoint: Path,
    tapes: tuple[Top1Tape, ...],
    *,
    stochastic: bool,
    random_seed: int,
    reward_scale: float = 10_000.0,
    force_profile_id: int | None = None,
) -> SemanticRollout:
    if not tapes:
        raise ValueError("tapes must not be empty")
    if force_profile_id is not None and not 0 <= force_profile_id < len(ADAPTIVE_PROFILES):
        raise ValueError("force_profile_id is outside the adaptive catalog")

    random = np.random.default_rng(random_seed)
    feature_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    log_probability_rows: list[float] = []
    value_rows: list[float] = []
    reward_rows: list[float] = []
    done_rows: list[bool] = []
    games: list[dict[str, Any]] = []

    for tape in tapes:
        engine = Engine(tape.config)
        candidate_seat = 1 - tape.top1_seat
        encoder = SpatialRecurrentPolicy(feature_checkpoint)
        encoder.reset()
        controller = StickyProfileController()
        previous_margin = 0.0
        turn = 0
        chosen_profiles: list[int] = []
        blocked_entries = 0
        target_entries = 0
        ineffective_actions = 0
        stale_tape_actions = 0
        last_encoded_turn = -1

        while turn < len(tape.actions):
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
                if force_profile_id is None:
                    profile_id, log_probability = _choose_action(
                        logits_numpy,
                        random,
                        stochastic=stochastic,
                    )
                else:
                    profile_id = force_profile_id
                    shifted = logits_numpy - logits_numpy.max()
                    log_probability = float(
                        shifted[profile_id] - np.log(np.exp(shifted).sum())
                    )
                chosen_profiles.append(profile_id)
            else:
                profile_id = 0

            effective_profile = controller.select(ADAPTIVE_PROFILES[profile_id])
            interval_end = (
                min(config.warmup_steps, len(tape.actions))
                if not learning_decision
                else min(turn + config.decision_interval, len(tape.actions))
            )
            while turn < interval_end:
                observation = engine.observation(candidate_seat)
                if (
                    config.observe_every_turn
                    and interval_end < len(tape.actions)
                    and last_encoded_turn != turn
                ):
                    encoder.encode_hidden(observation)
                    last_encoded_turn = turn
                candidate_plan = engine.plan(
                    candidate_seat,
                    profile_targets(effective_profile, observation),
                )
                candidate_action = candidate_plan["joint_action"]["actions"][
                    candidate_seat
                ]
                if engine.sanitize_action(candidate_seat, candidate_action) != candidate_action:
                    ineffective_actions += 1
                tape_action = tape.actions[turn]
                if engine.sanitize_action(tape.top1_seat, tape_action) != tape_action:
                    stale_tape_actions += 1
                entries = candidate_plan["report"]["entries"]
                target_entries += len(entries)
                blocked_entries += sum(entry["status"] == "BLOCKED" for entry in entries)
                actions = candidate_plan["joint_action"]["actions"]
                actions[tape.top1_seat] = tape_action
                engine.step(actions)
                turn += 1

            state = engine.state
            margin = float(
                state["farms"][candidate_seat]["money"]
                - state["farms"][tape.top1_seat]["money"]
            )
            if learning_decision:
                feature_rows.append(features)
                action_rows.append(profile_id)
                log_probability_rows.append(log_probability)
                value_rows.append(value)
                reward_rows.append((margin - previous_margin) / reward_scale)
                done_rows.append(turn >= len(tape.actions))
            previous_margin = margin

        state = engine.state
        games.append(
            {
                "episode_id": tape.episode_id,
                "candidate_seat": candidate_seat,
                "candidate_money": float(state["farms"][candidate_seat]["money"]),
                "opponent_money": float(state["farms"][tape.top1_seat]["money"]),
                "margin": previous_margin,
                "chosen_profiles": chosen_profiles,
                "blocked_rate": blocked_entries / target_entries if target_entries else 0.0,
                "ineffective_actions": ineffective_actions,
                "stale_tape_actions": stale_tape_actions,
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


def concatenate_rollouts(*rollouts: SemanticRollout) -> SemanticRollout:
    if not rollouts:
        raise ValueError("at least one rollout is required")
    return SemanticRollout(
        features=np.concatenate([rollout.features for rollout in rollouts]),
        actions=np.concatenate([rollout.actions for rollout in rollouts]),
        old_log_probabilities=np.concatenate(
            [rollout.old_log_probabilities for rollout in rollouts]
        ),
        old_values=np.concatenate([rollout.old_values for rollout in rollouts]),
        rewards=np.concatenate([rollout.rewards for rollout in rollouts]),
        dones=np.concatenate([rollout.dones for rollout in rollouts]),
        games=tuple(game for rollout in rollouts for game in rollout.games),
    )
