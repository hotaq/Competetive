# E009 — exact 0.899 pooling OOF

Purpose: reconstruct the scored E002 `0.899` inference recipe honestly on the
training corpus before using local validation for any new architecture.

## Frozen contract

- source: AADIGUPTA script version `341177035` / our submission `55378950`;
- package: `pilkwang/rsna-knee-weights`, manifest SHA-256
  `496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844`;
- 20 DINOv2-small checkpoints = four seeds x five folds;
- exact pixels: 336 px, 130 mm, six slots, 12 slices sampled from 20%–80%;
- ten overlapping three-slice windows;
- max probability: Fracture, Contusion, both menisci, Baker's;
- top-2 probability mean: ACL, MCL;
- probability mean: remaining targets;
- rank each member within its held-out fold, then average the four members.

## Leakage guard

Fold assignment is the frozen report-MD5 split from the scored source. Every
study is predicted only by manifest members with the same held-out fold. The
recomputed original validation AUC of every checkpoint must match its manifest
score; otherwise preprocessing or fold provenance is considered unproven.

## Outputs

- `e009_gate.json`
- `e009_manifest.json`
- `e009_per_member.csv`
- `e009_per_target.csv`
- `e009_oof_predictions.npz`

This experiment never writes `submission.csv` and never submits to Kaggle.
