---
name: reliable-baseline
description: Creates a guaranteed fallback submission and a deterministic cross-validated tabular classifier.
---

# Reliable Baseline

Use these scripts in order. They read the competition files from the current
working directory and print compact machine-readable status messages.

## `scripts/prepare_fallback.py`

Creates `fallback_submission.csv` using the training-set positive-class prior.
Run and submit this file before attempting model training.

## `scripts/train_baseline.py`

Creates `baseline_submission.csv` using stratified cross-validation. It prefers
CatBoost for mixed tabular data and automatically falls back to a scikit-learn
histogram gradient boosting pipeline if CatBoost cannot run.

Do not edit these scripts during evaluation.
