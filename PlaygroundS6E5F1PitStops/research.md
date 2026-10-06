# Playground Series S6E5 - Predicting F1 Pit Stops

## Current Snapshot

- Competition: `playground-series-s6e5`
- Title: Predicting F1 Pit Stops
- Deadline from Kaggle CLI: 2026-05-31 23:59:00
- Teams at first check: 553
- Leaderboard top at first check on 2026-05-04: `0.95435`
- Metric: ROC-AUC, maximize. Confirmed from pulled public baseline notebook.
- Target: `PitNextLap`, probability of a pit stop on the next lap.

## Local Data

Downloaded into `data/`.

- `train.csv`: 439,140 rows, 16 columns
- `test.csv`: 188,165 rows, 15 columns
- `sample_submission.csv`: 188,165 rows
- Positive target rate: 87,381 / 439,140 = 0.19898
- No missing values in train at first schema check.

Core columns:

- Categoricals: `Driver`, `Compound`, `Race`
- Time/race state: `Year`, `PitStop`, `LapNumber`, `Stint`, `TyreLife`, `RaceProgress`
- Performance state: `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change`

## Public Notebook Signals

Pulled references:

- `references/ravi_baseline`: public baseline; says this is binary classification with ROC-AUC. Uses StratifiedGroupKFold, target encoder, race/driver aggregates.
- `references/roman_encoding`: high-scoring notebook titled `Driver x Race x Year Encoding | 0.95354`. Key insight is driver/race/year strategy history. Reports:
  - LGB OOF AUC around 0.95241
  - XGB OOF AUC around 0.95232
  - CatBoost OOF AUC around 0.95127
  - RealMLP around 0.9526
  - stack/blend around 0.9535

Leaderboard/code-tab signals from Kaggle CLI:

- Top public refs include Svanik lap-by-lap engine, Roman target encoding, Ravi public baseline, Anthony blend, RealMLP + CatBoost + XGBoost ensembles.
- The first strong local path should be CatBoost/encoding baseline, then compare with public OOF/rank-blend ideas.

## Discussion Recon - 2026-05-04

Fetched via Kaggle internal discussion API using `~/.kaggle/kaggle.json` auth plus an XSRF cookie. Raw saved under `references/discussions/`.

Topics read:

- `696012`: tire compound domain meaning and the dry-race two-compound rule.
- `696380`: inconsistencies in the original FastF1-derived dataset.
- `696227`: race/driver sequence fields do not make sense when plotted as full stints.
- `696667`: adversarial validation / covariate shift discussion.
- `696769`: yearly target inconsistency in `PitNextLap`.
- `696347`: CTBoost experimental binary-classification starter.

Actionable takeaways:

- Use compound domain signal as tabular interactions: `Compound x Race`, `Compound x Stint`, compound-normalized tyre age, and degradation ratios.
- Do not over-trust sequence reconstruction. Discussions report broken assumptions for `PitNextLap`, `PitStop`, `Stint`, `TyreLife`, `LapNumber`, `LapTime_Delta`, and `Position_Change`.
- Treat this as tabular row-level modeling first; compare any lag/rolling features with an ablation.
- Year `2023` is anomalous. Verified locally:
  - 2022 positive rate: `26.650520%`
  - 2023 positive rate: `0.960726%`
  - 2024 positive rate: `29.531902%`
  - 2025 positive rate: `28.438866%`
- Adversarial validation around `0.498` was reported, but that only supports train/test `P(X)` similarity; it does not prove target-function stability.
- CTBoost is interesting but would add a dependency and has slow runtime, so keep it as a later Kaggle-only lane unless explicitly requested.

## Local Constraints

- `.venv` has CatBoost, pandas, numpy, scikit-learn.
- XGBoost import currently fails on macOS because `libomp.dylib` is missing.
- LightGBM is not installed.
- So local-first baseline should use CatBoost only unless the environment changes or work moves to Kaggle.

## Starter Command

Quick smoke:

```bash
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/PlaygroundS6E5F1PitStops"
../../.venv/bin/python train_catboost_baseline.py --data-dir data --quick --out submission_quick.csv --oof-out oof_quick.csv
```

Smoke result on 2026-05-04:

- Quick mode used 50,000 sampled training rows, 2 folds, 80 iterations.
- OOF ROC-AUC: `0.926487`
- Wrote `submission_quick.csv` with 188,165 rows and no missing predictions.
- This is only a script/schema check, not a serious leaderboard candidate.

Full-ish local run:

```bash
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/PlaygroundS6E5F1PitStops"
../../.venv/bin/python train_catboost_baseline.py --data-dir data --iterations 1200 --out submission.csv --oof-out oof_catboost.csv
```

## Next Improvement Lanes

1. Add leak-free target encodings for `Driver_Race_Year`, `Driver_Race`, `Race_Year`, `Driver_Compound`, `Race_Compound`, and numeric-binned combos.
2. Add fold diagnostics by `Year`, with a specific check for the 2023 target-rate anomaly.
3. Recreate the public baseline's `StratifiedGroupKFold` + target/frequency encoding using only scikit-learn/pandas, then compare against CatBoost-only OOF.
4. Run an ablation between row-level tabular features and sequence/rolling features because discussion evidence says race sequences may be corrupted.
5. Pull more high-score notebooks and save distilled feature/model ideas in this file before copying code.
6. If running on Kaggle, use LGB/XGB/RealMLP blend path because local macOS does not currently support XGBoost.
7. Submit only after a full OOF run writes a valid `submission.csv`.
