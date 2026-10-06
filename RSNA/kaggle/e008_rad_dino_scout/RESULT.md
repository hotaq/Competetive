# E008 result: NO-GO

- Kaggle kernel: `chinnaphatkhun/rsna-knee-e008-rad-dino-scout`, version 1
- Status: complete
- Completed: 2026-08-10
- Leaderboard submission: **none**

## Run integrity

- pinned `microsoft/rad-dino` checkpoint SHA-256 passed;
- 4,407/4,407 training studies and 3/3 placeholder test studies encoded;
- train and test feature coverage were both 100%;
- the frozen train cache occupied 5.14 GB on Kaggle, not on the local machine;
- train encoding took 4,796.6 seconds (79.9 minutes);
- five-fold dual-adapter training took 482.6 seconds (8.0 minutes);
- projected 1,300-study hidden-test encoding was 1,414.9 seconds (23.6 minutes);
- OOF artifacts contain finite predictions for all 4,407 studies and 12 targets;
- no execution errors were reported.

## Predeclared 80/20 anchor/RAD-DINO candidate

| Metric | Anchor | RAD-DINO arm | Candidate | Candidate delta |
|---|---:|---:|---:|---:|
| Broad weak-label OOF macro AUC | 0.848630 | 0.736817 | 0.842170 | -0.006459 |
| Official-label gold macro AUC | 0.840009 | 0.733555 | 0.833451 | -0.006558 |

The global slice adapter scored 0.714917 broad OOF and the target-specific spatial
adapter scored 0.737543. Spatial pooling helped the radiology-domain backbone, but
the representation remained far below the anchor.

## Gate

Passed:

- cache coverage;
- projected hidden-test runtime;
- prediction diversity (median Spearman 0.7242);
- exact weak-label identity outside gold rows;
- single-target veto (worst candidate delta -0.01331, above the -0.020 limit).

Failed:

- broad macro gain;
- official-gold direction;
- target breadth: only 1/12 targets was within -0.002 of the anchor.

Every target's predeclared candidate delta was negative. An exploratory blend sweep
also found no broad improvement at 0.5%, 1%, 2%, 3%, or 5% RAD-DINO weight; the
best broad weight was zero. Per-target exploratory gains were at most 0.000032,
which is noise-sized and not submission-worthy.

## Decision

**NO-GO — `stop_rad_dino`.** Do not submit E008, do not blend its predictions into
the 0.899 anchor, and do not spend Kaggle GPU time scaling this chest-X-ray backbone
to RayDINO-L. The clean run indicates modality mismatch, not a runtime or pooling
failure.
