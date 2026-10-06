---
name: robust-portfolio
description: Generates a fallback and a guarded native-categorical portfolio for binary tabular prediction.
---

# Robust Portfolio

Run the supplied scripts in order from the sandbox working directory.

## `scripts/prepare_fallback.py`

Writes `fallback_submission.csv` using the positive-class prior. Submit it before
training so the episode always has a valid prediction.

## `scripts/train_portfolio.py`

Trains fold-safe deterministic CatBoost, native-categorical LightGBM with
fold-local early stopping, ExtraTrees, histogram boosting, logistic regression,
and an optional small-data RBF SVM.

A nested-OOF interaction specialist and cross-fitted logistic stack are admitted
only when explicit gain, stability, and prediction-diversity gates pass. At most
four family-reserved candidates are printed in `CANDIDATES`, ordered by OOF AUC.
