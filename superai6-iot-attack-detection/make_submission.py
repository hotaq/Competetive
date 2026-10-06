#!/usr/bin/env python3
"""Create and validate the send-ready CatBoost hybrid submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "submissions"
CANDIDATE_COLUMN = "candidate_catboost_hybrid"
ARCHIVE_NAME = "submission_before_catboost_hybrid.csv"
OUTPUT_NAME = "submission_catboost_hybrid.csv"


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    if list(submission.columns) != ["Id", "label"]:
        raise ValueError(f"Invalid columns: {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample submission.")
    if not submission["Id"].equals(sample["Id"]):
        raise ValueError("Submission IDs or row order do not match the sample.")
    if submission["Id"].duplicated().any():
        raise ValueError("Submission IDs are not unique.")
    if submission["label"].isna().any():
        raise ValueError("Submission contains missing labels.")
    if not submission["label"].isin([0, 1]).all():
        raise ValueError("Submission labels must be binary 0/1.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    candidates = pd.read_csv(
        ROOT / "artifacts" / "catboost" / "candidate_predictions.csv",
        usecols=["Id", CANDIDATE_COLUMN],
    )
    if not candidates["Id"].equals(sample["Id"]):
        raise ValueError("Candidate IDs do not match sample submission IDs.")

    current_path = ROOT / "submission.csv"
    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    if current_path.exists() and not archive_path.exists():
        archive_path.write_bytes(current_path.read_bytes())

    submission = sample[["Id"]].copy()
    submission["label"] = candidates[CANDIDATE_COLUMN].astype("int8")
    validate_submission(submission, sample)

    artifact_path = OUTPUT_DIR / OUTPUT_NAME
    submission.to_csv(artifact_path, index=False)
    submission.to_csv(current_path, index=False)

    artifact_bytes = artifact_path.read_bytes()
    root_bytes = current_path.read_bytes()
    if artifact_bytes != root_bytes:
        raise ValueError("Root and archived submission files differ.")
    written = pd.read_csv(current_path)
    validate_submission(written, sample)

    metadata = {
        "model": "catboost_hybrid",
        "candidate_column": CANDIDATE_COLUMN,
        "label_mapping_assumption": {"0": "Normal", "1": "Attack"},
        "rows": len(written),
        "normal_rows": int((written["label"] == 0).sum()),
        "attack_rows": int((written["label"] == 1).sum()),
        "attack_pct": 100 * float(written["label"].mean()),
        "sha256": hashlib.sha256(root_bytes).hexdigest(),
        "root_submission": str(current_path),
        "artifact_submission": str(artifact_path),
        "previous_submission_archive": str(archive_path),
    }
    (OUTPUT_DIR / "submission_catboost_hybrid.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
