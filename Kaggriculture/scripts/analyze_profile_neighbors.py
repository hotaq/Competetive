"""Inspect latent nearest neighbours for conservative profile switching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_rl.learning.profile_ppo import (
    load_profile_checkpoint,
    semantic_forward,
)
from kaggriculture_rl.learning.replay_profile_ppo import (
    collect_tape_rollouts,
    load_top1_tapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_robust_v3.npz"),
    )
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    return parser.parse_args()


def _normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def main() -> None:
    args = parse_args()
    parameters, config, feature_checkpoint, metadata = load_profile_checkpoint(args.checkpoint)
    train_tapes = tuple(
        tape
        for manifest in args.train_manifest
        for tape in load_top1_tapes(manifest)
    )
    test_tapes = load_top1_tapes(args.test_manifest)
    train_rollout = collect_tape_rollouts(
        parameters,
        config,
        feature_checkpoint,
        train_tapes,
        stochastic=False,
        random_seed=0,
        force_profile_id=0,
    )
    test_rollout = collect_tape_rollouts(
        parameters,
        config,
        feature_checkpoint,
        test_tapes,
        stochastic=False,
        random_seed=0,
        force_profile_id=0,
    )
    oracle = {
        int(row["episode_id"]): int(row["best"]["profile_id"])
        for row in metadata["oracle"]
    }
    train_labels = np.asarray([oracle[tape.episode_id] for tape in train_tapes])
    train_features = _normalize(train_rollout.features)
    test_features = _normalize(test_rollout.features)
    logits, _ = semantic_forward(parameters, jnp.asarray(test_rollout.features))
    logits_array = np.asarray(jax.device_get(logits))
    probabilities = jax.nn.softmax(logits, axis=-1)
    probabilities_array = np.asarray(jax.device_get(probabilities))
    similarities = test_features @ train_features.T

    rows = []
    for index, tape in enumerate(test_tapes):
        order = np.argsort(similarities[index])[::-1]
        predicted = int(logits_array[index].argmax())
        safe_indices = np.flatnonzero(train_labels != 3)
        baseline_indices = np.flatnonzero(train_labels == 3)
        nearest_safe = int(safe_indices[np.argmax(similarities[index, safe_indices])])
        nearest_baseline = int(
            baseline_indices[np.argmax(similarities[index, baseline_indices])]
        )
        rows.append(
            {
                "episode_id": tape.episode_id,
                "predicted_profile_id": predicted,
                "confidence": float(probabilities_array[index, predicted]),
                "nearest_safe": {
                    "episode_id": train_tapes[nearest_safe].episode_id,
                    "profile_id": int(train_labels[nearest_safe]),
                    "cosine": float(similarities[index, nearest_safe]),
                },
                "nearest_baseline": {
                    "episode_id": train_tapes[nearest_baseline].episode_id,
                    "cosine": float(similarities[index, nearest_baseline]),
                },
                "safe_advantage": float(
                    similarities[index, nearest_safe]
                    - similarities[index, nearest_baseline]
                ),
                "neighbors": [
                    {
                        "episode_id": train_tapes[neighbor].episode_id,
                        "profile_id": int(train_labels[neighbor]),
                        "cosine": float(similarities[index, neighbor]),
                    }
                    for neighbor in order[:5]
                ],
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
