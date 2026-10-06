from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_submission_variants import selected_rows, simulate_episode_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = (
    PROJECT_ROOT
    / "submissions"
    / "05_adaptive_scientist"
    / "agent"
    / "skills"
    / "adaptive-scientist"
    / "scripts"
    / "select_plan.py"
)


def test_legacy_selection_never_uses_private_score() -> None:
    candidates = pd.DataFrame(
        [
            {
                "task": "train_x",
                "variant": "legacy",
                "candidate": "public_winner",
                "oof_auc": 0.70,
                "test_public": 0.90,
                "test_private": 0.51,
            },
            {
                "task": "train_x",
                "variant": "legacy",
                "candidate": "private_oracle",
                "oof_auc": 0.99,
                "test_public": 0.80,
                "test_private": 0.99,
            },
        ]
    )

    selected = selected_rows(candidates)

    assert selected.iloc[0]["candidate"] == "public_winner"
    assert selected.iloc[0]["test_private"] == 0.51


def test_adaptive_simulation_finalizes_before_revealing_private(tmp_path: Path) -> None:
    ids = np.arange(100)
    predictions = {
        "anchor_tree": np.linspace(0.01, 0.99, len(ids)),
        "anchor_linear": np.linspace(0.99, 0.01, len(ids)),
        "challenger_meta": np.roll(np.linspace(0.01, 0.99, len(ids)), 23),
    }
    manifest_candidates = []
    rows = []
    evidence = {
        "anchor_tree": ("tree", "anchor", 0.80, 0.85, 0.70),
        "anchor_linear": ("linear", "anchor", 0.79, 0.84, 0.99),
        "challenger_meta": ("meta", "challenger", 0.81, 0.86, 0.60),
    }
    for order, (name, values) in enumerate(predictions.items()):
        filename = f"candidate_{name}.csv"
        pd.DataFrame({"row_id": ids, "target": values}).to_csv(
            tmp_path / filename, index=False
        )
        family, role, oof_auc, public_auc, private_auc = evidence[name]
        manifest_candidates.append(
            {
                "name": name,
                "file": filename,
                "family": family,
                "role": role,
                "oof_auc": oof_auc,
                "fold_auc_std": 0.01,
            }
        )
        rows.append(
            {
                "task": "train_x",
                "variant": "adaptive",
                "candidate_order": order,
                "candidate": name,
                "file": filename,
                "family": family,
                "role": role,
                "oof_auc": oof_auc,
                "fold_auc_std": 0.01,
                "output_dir": str(tmp_path),
                "test_public": public_auc,
                "test_private": private_auc,
                "test_overall": (public_auc + private_auc) / 2,
            }
        )
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps({"candidates": manifest_candidates}), encoding="utf-8"
    )

    episodes = simulate_episode_scores(
        pd.DataFrame(rows), {"adaptive": SELECTOR}, Path(sys.executable)
    )

    episode = episodes.iloc[0]
    assert episode["selection_error"] is None
    assert episode["finalists"] == "challenger_meta,anchor_tree"
    assert episode["final_private_auc"] == 0.70
    assert "anchor_linear" not in episode["finalists"]

