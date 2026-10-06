# E004: cached supervision ablation

E004 mounts E003's Kaggle output and reuses its 609 MB frozen DINOv2 cache. It
performs no DICOM decoding and makes no leaderboard submission.

## Fixed components

- all cached pixels/features and preprocessing;
- five duplicate-report-grouped folds;
- ridge and slice-attention architectures;
- three attention seeds;
- 50/50 ridge/attention arm blend.

## Weight schemes

1. `raw_conf_gold8`: E003 control;
2. `anchor_conf_gold3`: `0.25 + 0.75 × confidence`, official labels weighted 3;
3. `equal_gold8`: all weak cells weighted equally, official labels weighted 8.

Each arm is tested in a predeclared 90% anchor / 10% arm rank blend. A non-control
scheme passes only with at least +0.001 broad OOF macro, acceptable gold direction,
8/12 target breadth, no target loss worse than 0.015, and useful diversity.

If neither alternative passes, discard this frozen-head family and spend no more GPU
on it. Expected output: `e004_gate.json` plus small OOF/test prediction archives.
