# E003 result: NO-GO

Kaggle kernel: `chinnaphatkhun/rsna-knee-e003-dinov2-diversity-lab`, version 1  
Completed: 2026-08-09  
Leaderboard submission: **none**

## Run integrity

- 4,407/4,407 training studies and 3/3 placeholder test studies encoded;
- 190,849 DICOM slices decoded with zero missing-series, header, or pixel failures;
- 100% study embedding coverage;
- 609 MB frozen train embedding cache retained in Kaggle output;
- five grouped folds × three attention-head seeds completed;
- model work completed in about 4,015 seconds (66.9 minutes);
- pinned label and anchor-OOF hashes passed.

## Predeclared gate

Candidate: 75% anchor rank + 25% E003 rank.

| Metric | Anchor | E003 | Candidate | Candidate delta |
|---|---:|---:|---:|---:|
| Broad weak-label OOF macro AUC | 0.8486 | 0.7672 | 0.8427 | -0.0059 |
| Official-label gold macro AUC (58 studies) | 0.8400 | 0.7623 | 0.8322 | -0.0078 |

- Non-worse targets: 0/12.
- Worst target delta: -0.0207 (Baker's).
- Median per-target Spearman correlation: 0.7570.
- Passed checks: diversity and weak-label identity.
- Failed checks: macro gain, gold direction, target breadth, and target veto.

Decision: **do not submit or blend E003 predictions**.

An exploratory, non-gating sweep confirmed that tiny weights do not rescue the arm:
2% E003 changes broad OOF from 0.84863 to 0.84853 while gold moves from 0.84001
to 0.84129. That trade is too small and fails the broad metric.

## Reusable output

Keep the frozen embedding cache and head checkpoints as a Kaggle kernel source for
cheap head/supervision experiments. The next run should ablate supervision weights on
this cache before spending GPU time on another DICOM pass or larger backbone.
