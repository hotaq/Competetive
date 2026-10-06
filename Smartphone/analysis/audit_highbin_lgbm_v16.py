from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "kaggle/highbin_lgbm_v16_output"
ANCHOR_DIR = ROOT / "kaggle/public_rank_logit_anchor_v14_output"
REPORT_DIR = ROOT / "reports/highbin_lgbm_v16_pricing_ranklogit"
PREDICTION_COLUMNS = ("lgb_highbin_raw", "lgb_highbin_freq3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    pricing = json.loads((REPORT_DIR / "pricing.json").read_text())
    oof = pd.read_csv(ARTIFACT_DIR / "oof.csv")
    test = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    train_keys = pd.read_parquet(ANCHOR_DIR / "train_keys.parquet")
    anchor_test = pd.read_csv(ANCHOR_DIR / "submission_alpha070.csv")

    expected_oof_columns = {"id", "addicted_label", *PREDICTION_COLUMNS}
    expected_test_columns = {"id", *PREDICTION_COLUMNS}
    assert set(oof.columns) == expected_oof_columns
    assert set(test.columns) == expected_test_columns
    assert oof["id"].is_unique and test["id"].is_unique
    assert set(oof["addicted_label"].unique()).issubset({0, 1})

    all_predictions = np.concatenate(
        [oof[list(PREDICTION_COLUMNS)].to_numpy(), test[list(PREDICTION_COLUMNS)].to_numpy()]
    )
    assert np.isfinite(all_predictions).all()
    assert ((all_predictions >= 0.0) & (all_predictions <= 1.0)).all()

    oof_sorted = oof.sort_values("id").reset_index(drop=True)
    test_sorted = test.sort_values("id").reset_index(drop=True)
    keys_sorted = train_keys.sort_values("id").reset_index(drop=True)
    anchor_test_sorted = anchor_test.sort_values("id").reset_index(drop=True)
    train_ids_aligned = oof_sorted["id"].equals(keys_sorted["id"])
    labels_aligned = oof_sorted["addicted_label"].equals(
        keys_sorted["addicted_label"]
    )
    test_ids_aligned = test_sorted["id"].equals(anchor_test_sorted["id"])
    assert train_ids_aligned and labels_aligned and test_ids_aligned

    hash_matches = {
        name: sha256(ARTIFACT_DIR / name) == expected
        for name, expected in metrics["sha256"].items()
    }
    assert all(hash_matches.values())
    assert pricing["generated_submissions"] == []

    primary_weight = f"{pricing['protocol']['primary_weight']:.3f}"
    promotion = {
        name: {
            "standalone_auc": result["standalone_auc"],
            "spearman_vs_anchor": result["spearman_vs_anchor"],
            "primary_weight": float(primary_weight),
            "primary_delta": result["weights"][primary_weight]["delta"],
            "primary_fold_wins": result["weights"][primary_weight]["fold_wins"],
            "gate_pass": result["primary_gate_pass"],
        }
        for name, result in pricing["candidates"].items()
    }

    audit = {
        "artifacts": {
            "oof_rows": len(oof),
            "test_rows": len(test),
            "unique_ids": True,
            "finite_and_bounded_predictions": True,
            "train_ids_aligned": train_ids_aligned,
            "labels_aligned": labels_aligned,
            "test_ids_aligned": test_ids_aligned,
            "sha256_matches": hash_matches,
        },
        "anchor_auc": pricing["protocol"]["anchor_auc"],
        "promotion": promotion,
        "generated_submissions": pricing["generated_submissions"],
        "competition_submission_made": False,
    }
    output = REPORT_DIR / "artifact_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
