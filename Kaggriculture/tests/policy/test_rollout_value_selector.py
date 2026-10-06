from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rl.policy.rollout_value_selector import RolloutValueSelector

ROOT = Path(__file__).resolve().parents[2]


def _episodes(path: Path) -> dict[int, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["episode_id"]): row for row in payload["episodes"]}


def test_selector_reproduces_chronological_holdout_24_of_24() -> None:
    selector = RolloutValueSelector.from_path(
        ROOT / "rl/checkpoints/v19_rollout_value_selector_v4_safe95.json"
    )
    model_rows = {
        "ryo": _episodes(
            ROOT
            / "rl/evaluations/v19_recovery_fast_h719_holdout_ryo.json"
        ),
        "repaired": _episodes(
            ROOT
            / "rl/evaluations/v19_recovery_fast_h719_holdout_repaired.json"
        ),
        "dual": _episodes(
            ROOT
            / "rl/evaluations/v19_recovery_fast_h719_holdout_dual.json"
        ),
    }
    expected_report = json.loads(
        (
            ROOT / "rl/checkpoints/v19_rollout_value_selector_v4_safe95.json"
        ).read_text(encoding="utf-8")
    )
    expected_choices = {
        int(row["episode_id"]): str(row["choice"])
        for row in expected_report["evaluation"]["episodes"]
    }
    actual_margins = {
        policy: {
            int(row["episode_id"]): float(row["candidate_margin"])
            for row in _episodes(
                ROOT
                / (
                    "rl/evaluations/v19_holdout_v16_top1_20260824_replay24.json"
                    if policy == "v16"
                    else "rl/evaluations/v19_holdout_v12_top1_20260824_replay24.json"
                )
            ).values()
        }
        for policy in ("v16", "v12")
    }

    selected_margins: list[float] = []
    for episode_id in sorted(expected_choices):
        first = model_rows["ryo"][episode_id]
        predictions = {
            reference: rows[episode_id]["predictions"]
            for reference, rows in model_rows.items()
        }
        decision = selector.select(int(first["candidate_seat"]), predictions)
        assert decision.policy == expected_choices[episode_id]
        selected_margins.append(actual_margins[decision.policy][episode_id])

    assert len(selected_margins) == 24
    assert min(selected_margins) == 5_398.0
    assert all(margin > 0.0 for margin in selected_margins)
