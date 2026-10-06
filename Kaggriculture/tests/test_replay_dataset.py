from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kaggriculture_rl.imitation.replay_dataset import (
    build_replay_dataset,
    encode_observation,
    load_replay_dataset,
)


def _observation(player: int, step: int, overage: float) -> dict[str, object]:
    observation: dict[str, object] = {
        "player": player,
        "day": 0,
        "hour": step,
        "farms": [
            {"money": 3_000.0, "farmer": [4, 4], "hands": []},
            {"money": 3_000.0, "farmer": [4, 4], "hands": []},
        ],
        "private": {"shed": {"WHEAT": 0}},
        "remainingOverageTime": overage,
    }
    if player == 0:
        observation["step"] = step
    return observation


def _write_episode(root: Path, episode_id: int, top1_seat: int, verb: str) -> None:
    initial = [
        {"observation": _observation(0, 0, 60.0)},
        {"observation": _observation(1, 0, 59.0)},
    ]
    transition = [
        {
            "action": {"farmer": ["PASS"], "hands": [], "market": []},
            "observation": _observation(0, 1, 58.0),
        },
        {
            "action": {"farmer": ["PASS"], "hands": [], "market": []},
            "observation": _observation(1, 1, 57.0),
        },
    ]
    transition[top1_seat]["action"]["farmer"] = [verb]
    (root / f"{episode_id}.json").write_text(json.dumps({"steps": [initial, transition]}))


def test_encoder_is_deterministic_and_ignores_runtime_time() -> None:
    first = _observation(0, 3, 60.0)
    second = dict(reversed(list(first.items())))
    second["remainingOverageTime"] = 1.0
    np.testing.assert_array_equal(encode_observation(first), encode_observation(second))


def test_dataset_pairs_previous_observation_with_next_action(tmp_path: Path) -> None:
    _write_episode(tmp_path, 20, 1, "NORTH")
    _write_episode(tmp_path, 10, 0, "SOUTH")
    manifest = {
        "leader": {"team_name": "leader"},
        "episodes": [
            {"episode_id": 20, "top1_seat": 1},
            {"episode_id": 10, "top1_seat": 0},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "dataset.npz"

    summary = build_replay_dataset(manifest_path, output, feature_dim=64, holdout_episodes=1)
    dataset = load_replay_dataset(output)

    assert summary["train_samples"] == 1
    assert summary["holdout_samples"] == 1
    assert dataset.episode_ids.tolist() == [20, 10]
    assert dataset.train_mask.tolist() == [False, True]
    assert json.loads(dataset.unit_vocab[dataset.farmer_labels[0]]) == ["NORTH"]
    assert json.loads(dataset.unit_vocab[dataset.farmer_labels[1]]) == ["SOUTH"]

    expected = _observation(1, 0, 0.0)
    expected["step"] = 0
    expected.pop("remainingOverageTime")
    np.testing.assert_allclose(dataset.features[0], encode_observation(expected, 64))
