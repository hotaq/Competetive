---
name: feature-engineer
description: Generates fold-neutral train/test-aligned missing-value and row-aggregation features.
---

# Feature Engineer

Run `scripts/generate_features.py` only when row-wise numeric aggregates are
plausible. It aligns train/test columns, imputes values using training-only
statistics, and adds robust row-level numeric summaries.

```bash
python skills/feature-engineer/scripts/generate_features.py \
  --train train.csv --test test.csv --target target
```

Outputs: `train_engineered.csv` and `test_engineered.csv`.

Read `resources/leakage_checklist.md` before inventing additional features.
