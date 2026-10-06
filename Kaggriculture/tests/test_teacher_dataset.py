from pathlib import Path

from kaggriculture_rl.imitation.spatial_dataset import (
    align_spatial_dataset_vocabulary,
    load_spatial_replay_dataset,
)
from kaggriculture_rl.imitation.teacher_dataset import (
    _label_actions,
    build_live_teacher_spatial_dataset,
)


def test_live_teacher_dataset_uses_existing_spatial_schema(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.py"
    teacher.write_text(
        "def agent(obs):\n"
        "    seat = int(obs.get('player', 0))\n"
        "    hands = len(obs['farms'][seat].get('hands', []))\n"
        "    return {'farmer': ['PASS'], 'hands': [['PASS']] * hands, 'market': []}\n"
    )
    output = tmp_path / "teacher_dataset.npz"

    report = build_live_teacher_spatial_dataset(
        teacher,
        output,
        episodes=4,
        holdout_episodes=2,
        base_seed=91,
        crop_opponents=(("MELON", 2),),
        include_self_play=False,
    )
    dataset = load_spatial_replay_dataset(output)

    assert report["samples"] == 4 * 719
    assert report["train_samples"] == 2 * 719
    assert report["holdout_samples"] == 2 * 719
    assert dataset.spatial.shape[0] == 4 * 719
    assert dataset.global_features.shape[0] == 4 * 719
    assert dataset.unit_vocab == ('["PASS"]',)
    assert dataset.market_vocab == ("__STOP__",)
    assert dataset.hand_labels.shape[1] == 12
    assert dataset.metadata["source"] == "live-standalone-teacher-v1"
    assert dataset.metadata["teacher_sanitation_changes"] == 0
    assert all(game["opponent_money"] > 3000 for game in report["episodes"])


def test_teacher_labels_expand_beyond_legacy_twelve_hands() -> None:
    action = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(14)],
        "market": [],
    }

    _, hand_labels, hand_mask, _, _, _, _ = _label_actions((action,))

    assert hand_labels.shape == (1, 14)
    assert hand_mask.shape == (1, 14)
    assert hand_mask.all()


def test_spatial_dataset_can_align_to_checkpoint_vocab_superset(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.py"
    teacher.write_text(
        "def agent(obs):\n"
        "    seat = int(obs.get('player', 0))\n"
        "    hands = len(obs['farms'][seat].get('hands', []))\n"
        "    return {'farmer': ['PASS'], 'hands': [['PASS']] * hands, 'market': []}\n"
    )
    output = tmp_path / "source.npz"
    build_live_teacher_spatial_dataset(
        teacher,
        output,
        episodes=4,
        holdout_episodes=2,
        base_seed=101,
        crop_opponents=(("MELON", 2),),
        include_self_play=False,
    )
    source = load_spatial_replay_dataset(output)

    aligned = align_spatial_dataset_vocabulary(
        source,
        ('["PASS"]', '["NORTH"]'),
        ("__STOP__", '["HIRE"]'),
    )

    assert aligned.unit_vocab == ('["PASS"]', '["NORTH"]')
    assert aligned.market_vocab == ("__STOP__", '["HIRE"]')
    assert not aligned.farmer_labels.any()
    assert not aligned.market_labels.any()
