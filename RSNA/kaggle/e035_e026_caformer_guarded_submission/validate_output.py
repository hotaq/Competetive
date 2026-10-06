#!/usr/bin/env python3
"""Validate downloaded E035 output before consuming a Kaggle submission slot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
PROTECTED = ["MCL", "Synovitis", "Contusion", "Fracture"]
ANCHOR_SHA256 = "46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271"
HERE = Path(__file__).resolve().parent
LOCAL_ANCHOR = HERE.parent / "e026_safe_window_inference/output_v1/submission.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order(values: np.ndarray) -> np.ndarray:
    return values.argsort(kind="mergesort").argsort(kind="mergesort")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    submission_path = output / "submission.csv"
    summary_path = output / "e035_summary.json"
    audit_path = output / "e035_target_audit.csv"
    errors: list[str] = []

    for path in [submission_path, summary_path, audit_path]:
        if not path.is_file():
            errors.append(f"missing {path.name}")
    if errors:
        raise SystemExit("E035 INVALID:\n- " + "\n- ".join(errors))

    submission = pd.read_csv(submission_path, dtype={"StudyInstanceUID": str})
    summary = json.loads(summary_path.read_text())
    audit = pd.read_csv(audit_path)
    if list(submission.columns) != ["StudyInstanceUID"] + TARGETS:
        errors.append("submission schema")
    values = submission[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        errors.append("non-finite values")
    if ((values < 0) | (values > 1)).any():
        errors.append("values outside [0,1]")
    if summary.get("anchor", {}).get("sha256") != ANCHOR_SHA256:
        errors.append("anchor summary SHA")
    if summary.get("submission_sha256") != sha256_file(submission_path):
        errors.append("submission SHA")
    if summary.get("test", {}).get("failure_count") != 0:
        errors.append("test encoding failures")
    if summary.get("protected_targets") != PROTECTED:
        errors.append("protected target list")
    if set(audit.loc[audit["protected_exact"].astype(bool), "target"]) < set(PROTECTED):
        errors.append("protected target audit")

    if LOCAL_ANCHOR.is_file():
        if sha256_file(LOCAL_ANCHOR) != ANCHOR_SHA256:
            errors.append("local anchor SHA")
        else:
            anchor = pd.read_csv(LOCAL_ANCHOR, dtype={"StudyInstanceUID": str})
            if anchor["StudyInstanceUID"].tolist() != submission["StudyInstanceUID"].tolist():
                errors.append("anchor UID/order")
            else:
                for target in PROTECTED:
                    if not np.array_equal(anchor[target].to_numpy(), submission[target].to_numpy()):
                        errors.append(f"protected target changed: {target}")
                changed = [
                    target
                    for target in TARGETS
                    if not np.array_equal(order(anchor[target].to_numpy()), order(submission[target].to_numpy()))
                ]
                if changed != summary.get("changed_rank_targets"):
                    errors.append("changed ranking summary mismatch")

    if args.require_ready and summary.get("decision") != "READY_TO_SUBMIT_E035":
        errors.append("ready decision required")
    if errors:
        raise SystemExit("E035 INVALID:\n- " + "\n- ".join(errors))
    print("E035 VALID", summary.get("decision"), summary.get("changed_rank_targets"))


if __name__ == "__main__":
    main()
