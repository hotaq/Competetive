---
name: adaptive-scientist
description: Profiles binary tabular data, trains fixed anchors and allowlisted specialists, records feedback, and selects diverse finalists.
---

# Adaptive Scientist

Use only the supplied scripts. They read `train.csv`, `test.csv`, and
`sample_submission.csv`; none reads hidden labels or uses the network.

1. `scripts/prepare_fallback.py` creates `fallback_submission.csv`.
2. `scripts/profile_dataset.py` creates deterministic `dataset_profile.json`.
3. `scripts/train_adaptive.py` validates `initial_plan.json`, trains the three
   available anchor families plus a bounded recipe view, and writes candidates and
   `candidate_manifest.json`.
4. `scripts/record_feedback.py` appends each returned ID/Public score and manifest
   metadata to `feedback_ledger.json`.
5. `scripts/select_plan.py --mode challenger` permits at most one guarded,
   unsubmitted candidate. `--mode final` emits `SELECT_MANIFEST` with no more than
   two diverse valid IDs.

Planner JSON never contains code. Initial recipe IDs are `mixed-safe`, `tiny-safe`,
`numeric-boost`, `categorical-native`, `ordinal-native`, and `missing-robust`.
