#!/usr/bin/env python3
"""Append one validated submission result to the explicit feedback ledger."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite_score(value: str) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _candidate_metadata(manifest_path: Path, candidate_name: str) -> dict:
    if candidate_name == "fallback":
        return {
            "name": "fallback",
            "file": "fallback_submission.csv",
            "family": "fallback",
            "oof_auc": 0.5,
            "fold_auc_std": None,
            "test_rank_correlation_to_best": None,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for candidate in manifest.get("candidates", []):
        if candidate.get("name") == candidate_name:
            return candidate
    raise ValueError(f"Unknown candidate: {candidate_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("candidate_manifest.json"))
    parser.add_argument("--ledger", type=Path, default=Path("feedback_ledger.json"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--public-score", required=True)
    parser.add_argument("--submission-order", type=int, required=True)
    args = parser.parse_args()

    metadata = _candidate_metadata(args.manifest, args.candidate)
    score = _finite_score(args.public_score)
    if args.ledger.is_file():
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict) or not isinstance(ledger.get("submissions", []), list):
            raise ValueError("Invalid feedback ledger")
    else:
        ledger = {"ledger_version": 1, "submissions": []}

    record = {
        "submission_order": args.submission_order,
        "candidate": metadata["name"],
        "file": metadata["file"],
        "family": metadata.get("family", "unknown"),
        "oof_auc": metadata.get("oof_auc"),
        "fold_auc_std": metadata.get("fold_auc_std"),
        "prediction_correlation": metadata.get("test_rank_correlation_to_best"),
        "submission_id": args.submission_id,
        "public_score": score,
        "feedback_valid": score is not None,
    }
    existing_ids = {
        item.get("submission_id")
        for item in ledger["submissions"]
        if isinstance(item, dict)
    }
    if args.submission_id in existing_ids:
        raise ValueError("Submission ID already exists in ledger")
    ledger["submissions"].append(record)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(
        "LEDGER "
        + json.dumps(
            {
                "candidate": args.candidate,
                "submission_id": args.submission_id,
                "feedback_valid": score is not None,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
