---
name: robust-portfolio
description: Generates a fallback and OOF-ranked diverse model portfolio for binary tabular prediction.
---

# Robust Portfolio

Run the scripts in this order from the sandbox working directory.

## `scripts/prepare_fallback.py`

Writes `fallback_submission.csv` using the positive-class prior. Submit it
before training so the episode always has a valid prediction.

## `scripts/train_portfolio.py`

Trains fold-safe deterministic candidates. When installed, it uses two
CatBoost configurations and LightGBM. It always includes ExtraTrees, histogram
gradient boosting, and logistic regression, plus an RBF SVM for small datasets.
It writes direct-model and OOF-gated rank-blend submissions.

The final stdout line starts with `CANDIDATES` and lists files in descending OOF
AUC order. Submit those files in that order, capped by the system prompt.

Do not modify either script during evaluation.
