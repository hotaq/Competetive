# E003: DINOv2 diversity lab

## Outcome contract

This Kaggle-only experiment tests whether a frozen, 336 px, slice-aware DINOv2
family is complementary to the verified 0.899 ensemble. It does **not** download
the competition DICOM archive locally and does **not** auto-submit a prediction.

The predeclared candidate is a per-target rank blend:

```text
75% anchor OOF + 25% E003 OOF
```

The candidate passes only when all checks hold:

- broad OOF macro gain is at least 0.002;
- the 58-study gold audit falls by no more than 0.005;
- at least 8/12 targets are no worse by more than 0.002;
- no target falls by more than 0.02;
- median per-target Spearman correlation is at most 0.97;
- E003 and the anchor use identical weak targets outside the gold audit; the 58 gold
  rows are evaluated against their official `train.csv` labels.

## Material changes from upstream

Source: `wguesdon/rsna-knee-dinov2-at-meniscus-resolution`

Pinned source SHA-256:
`387c8bedb0c87d8ca40f3d7b2bbe57ff1492681c37124f4aa05a5a8016f4c56b`

E003:

1. attaches `pilkwang/rsna-knee-llm-labels` explicitly;
2. fails closed unless `report_labels_v2.csv` is mounted;
3. uses each report target's confidence in ridge and attention losses;
4. disables partial fine-tuning so the run isolates frozen features and heads;
5. retains the approximately 610 MB embedding cache in Kaggle output;
6. compares aligned OOF predictions with `pilkwang/rsna-knee-weights/oof.npz`;
7. writes machine-readable gate and prediction artifacts.

Pinned small-input hashes:

- `report_labels_v2.csv`: `6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920`;
- anchor `oof.npz`: `62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c`.

## Build and run

Pulling the public notebook retrieves source only, not its datasets:

```bash
rtk mkdir -p /tmp/rsna-wguesdon
rtk kaggle kernels pull wguesdon/rsna-knee-dinov2-at-meniscus-resolution \
  -p /tmp/rsna-wguesdon -m
rtk python build_notebook.py \
  /tmp/rsna-wguesdon/rsna-knee-dinov2-at-meniscus-resolution.ipynb \
  e003-dinov2-diversity-lab.ipynb
rtk kaggle kernels push -p .
```

Expected outputs include:

- `e003_gate.json`;
- `e003_oof_predictions.npz`;
- `e003_test_predictions.npz`;
- 15 `e003_attention_fold*_seed*.pt` deterministic attention-head checkpoints;
- `submission.csv` (valid model-only artifact, not automatically submitted);
- the `emb_train_*` frozen-feature cache files.
