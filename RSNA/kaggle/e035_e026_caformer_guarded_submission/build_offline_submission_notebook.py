#!/usr/bin/env python3
"""Build the internet-disabled E035 submission wrapper notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "e035-e026-caformer-guarded-submission.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> tuple[Path, str]:
    cells = [
        markdown(
            """# E035 — internet-disabled validated submission wrapper

This notebook is the competition-compatible wrapper for the already completed E035
inference run. The preceding GPU version produced a validated `READY_TO_SUBMIT_E035`
artifact but could not be submitted because that version used internet access to fetch
the pretrained CAFormer backbone.

This wrapper has internet disabled. It reads only SHA-locked artifacts from the private
E035 asset dataset, revalidates schema, UID order, protected targets, and changed ranks,
then writes the exact validated `submission.csv` byte-for-byte.
"""
        ),
        code(
            """from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
PROTECTED = ["MCL", "Synovitis", "Contusion", "Fracture"]
EXPECTED_CHANGED = [
    "ACL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "PF OA", "Baker's",
]
EXPECTED_SHA256 = {
    "e026_anchor_submission.csv": "46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271",
    "e035_summary.json": "791da00bc46add56f13df71b92537a80fb18d6a57e464327b9e21efbd8fa8662",
    "e035_target_audit.csv": "8bbcf9179d79a0b042e6c30d47c90ed13885239e69a40e93ef79468f99bffa55",
    "e035_validated_submission.csv": "d088c389a0e212f41afb3bff7472ec6a84a55994430c629ebda81eccfed5f67d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_asset_root() -> Path:
    preferred = INPUT / "rsna-knee-e035-assets"
    if preferred.is_dir():
        return preferred
    candidates = [
        path for path in INPUT.iterdir()
        if path.is_dir() and (path / "e035_validated_submission.csv").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one E035 asset root, found {candidates}")
    return candidates[0]


def find_competition_root() -> Path:
    for root in [
        INPUT / "competitions/rsna-knee-abnormality-detection",
        INPUT / "rsna-knee-abnormality-detection",
    ]:
        if (root / "sample_submission.csv").is_file():
            return root
    raise FileNotFoundError("competition sample_submission.csv")


def order(values: np.ndarray) -> np.ndarray:
    return values.argsort(kind="mergesort").argsort(kind="mergesort")


asset_root = find_asset_root()
for name, expected_hash in EXPECTED_SHA256.items():
    actual_hash = sha256_file(asset_root / name)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{name} SHA mismatch: {actual_hash}")

anchor = pd.read_csv(asset_root / "e026_anchor_submission.csv", dtype={"StudyInstanceUID": str})
submission = pd.read_csv(
    asset_root / "e035_validated_submission.csv",
    dtype={"StudyInstanceUID": str},
)
summary = json.loads((asset_root / "e035_summary.json").read_text())
audit = pd.read_csv(asset_root / "e035_target_audit.csv")
sample = pd.read_csv(
    find_competition_root() / "sample_submission.csv",
    dtype={"StudyInstanceUID": str},
)

expected_columns = ["StudyInstanceUID"] + TARGETS
if list(submission.columns) != expected_columns or list(sample.columns) != expected_columns:
    raise RuntimeError("submission/sample schema mismatch")
if submission["StudyInstanceUID"].tolist() != sample["StudyInstanceUID"].tolist():
    raise RuntimeError("submission/sample UID order mismatch")
if anchor["StudyInstanceUID"].tolist() != submission["StudyInstanceUID"].tolist():
    raise RuntimeError("anchor/submission UID order mismatch")
values = submission[TARGETS].to_numpy(np.float64)
if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
    raise RuntimeError("invalid submission values")
if summary.get("decision") != "READY_TO_SUBMIT_E035":
    raise RuntimeError("READY_TO_SUBMIT_E035 required")
if summary.get("submission_sha256") != EXPECTED_SHA256["e035_validated_submission.csv"]:
    raise RuntimeError("summary submission SHA mismatch")
if summary.get("changed_rank_targets") != EXPECTED_CHANGED:
    raise RuntimeError("changed-rank summary mismatch")

for target in PROTECTED:
    if not np.array_equal(anchor[target].to_numpy(), submission[target].to_numpy()):
        raise RuntimeError(f"protected target changed: {target}")
changed = [
    target for target in TARGETS
    if not np.array_equal(order(anchor[target].to_numpy()), order(submission[target].to_numpy()))
]
if changed != EXPECTED_CHANGED:
    raise RuntimeError(f"changed ranking mismatch: {changed}")
protected_audit = audit.loc[audit["target"].isin(PROTECTED)]
if len(protected_audit) != len(PROTECTED) or not protected_audit["protected_exact"].astype(bool).all():
    raise RuntimeError("protected audit mismatch")

output_submission = WORK / "submission.csv"
shutil.copyfile(asset_root / "e035_validated_submission.csv", output_submission)
shutil.copyfile(asset_root / "e035_summary.json", WORK / "e035_summary.json")
shutil.copyfile(asset_root / "e035_target_audit.csv", WORK / "e035_target_audit.csv")
if sha256_file(output_submission) != EXPECTED_SHA256["e035_validated_submission.csv"]:
    raise RuntimeError("written submission SHA mismatch")

receipt = {
    "decision": "READY_TO_SUBMIT_E035",
    "internet_required": False,
    "asset_dataset": "chinnaphatkhun/rsna-knee-e035-assets",
    "submission_sha256": sha256_file(output_submission),
    "changed_rank_targets": changed,
    "protected_targets": PROTECTED,
}
(WORK / "e035_offline_receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
print(submission.to_string(index=False))
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    return OUTPUT, digest


if __name__ == "__main__":
    path, digest = build()
    print(path)
    print(digest)
