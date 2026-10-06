# V83 Allocation-v2 On-Policy Dataset — 2026-08-27

## Result

Status: **PASS_COMPLETE_ON_POLICY_DATASET**.

This is development-only Rust-engine evidence. It is not training, a candidate, a package, a submission, or promotion evidence.

## Per-family paired outcomes

| Family | Games | Available / complete | Benefit / neutral / regress | Wins base→forced (Δ) | Mean base→forced (Δ) | Worst base→forced (Δ) | Paired Δ mean / worst | Regressed? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| robust_cluster_family | 8 | 8 / 8 | 2 / 0 / 6 | 8→6 (-2) | 10938.0→5538.4 (-5399.6) | 1237.0→-2247.0 (-3484.0) | -5399.6 / -17268.0 | YES |
| route_99612231_family | 8 | 6 / 6 | 3 / 2 / 3 | 8→5 (-3) | 9358.2→5916.8 (-3441.5) | 3741.0→-7307.0 (-11048.0) | -3441.5 / -11670.0 | YES |
| ryo_selector_family | 8 | 8 / 8 | 3 / 0 / 5 | 5→6 (+1) | 5618.2→2422.5 (-3195.8) | -21048.0→-21906.0 (-858.0) | -3195.8 / -17986.0 | YES |
| v16_route_family | 8 | 7 / 7 | 0 / 1 / 7 | 5→3 (-2) | 1001.9→-2202.6 (-3204.5) | -3585.0→-7517.0 (-3932.0) | -3204.5 / -6367.0 | YES |

## Benefit verdict

The executor is mechanically valid, but the unconditional forced allocation is rejected as a general policy on this development set. Wins fall 26→20, paired mean delta is -3810.3, and 21 of 32 games regress. Every family has negative mean delta and a worse family worst-case margin.

## Mechanical gates and action scope

- Completed pairs: 32 / 32.
- Forced option starts/completions/partials: 29 / 29 / 0.
- Owned hand4 suffix steps: 377 (expected 377).
- Candidate sanitizer changes: 0.
- Direct executor action-scope violations: 0.

The direct scope gate compares the executor output with the V67 base action generated in the same forced state. Only `market[1]` at step 248 and `hands[4]` at steps 249-261 may differ. Counterfactual ABSTAIN-versus-forced response differences after state divergence are recorded but are not mislabeled as executor mutations.

## Frozen scope

- Four catalog representatives only: V22, V12/Ryo, V44, robust cluster.
- Already-consumed seeds `27100100..27100103`, both candidate seats.
- No new seed, replay, holdout, live Kaggle trace, training, candidate, package, submission, or upload.

## Reproduction

```bash
.venv/bin/python analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py
.venv/bin/ruff check analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py
.venv/bin/ruff format --check analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py
```

Frozen input and per-run action-trace hashes are in `results.json`.
