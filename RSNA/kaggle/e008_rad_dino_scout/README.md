# E008: frozen RAD-DINO-B scout

## Outcome contract

This Kaggle-only experiment tests whether a radiology-domain DINOv2-B/14 encoder
adds useful diversity to the verified 0.899 DINOv2 family. It never downloads the
competition DICOM archive locally and does not submit to the leaderboard.

The exact public `microsoft/rad-dino` checkpoint is mounted as a private Kaggle
input and pinned by SHA-256. The backbone remains frozen.

## Representation and adapters

- 336 px grayscale MRI slices copied to RGB, matching RAD-DINO channel semantics;
- five plane/contrast slots and at most 30 slices per complete study;
- an uncompressed CLS + patch-mean + patch-max vector for each slice;
- an 8x8 spatial grid projected from 768 to 192 channels with a pinned map;
- one global slice-query adapter and one spatial target-query adapter;
- five duplicate-report-grouped folds, one seed, and fixed epochs.

The predeclared E008 arm is 25% global / 75% spatial by rank. Its offline candidate
is 80% anchor / 20% E008.

## Gate

Expansion requires all of the following:

- broad OOF macro gain at least 0.002;
- official-gold macro direction no worse than -0.005;
- at least 8/12 targets no worse than -0.002;
- no target worse than -0.020;
- median anchor/arm Spearman correlation at most 0.970;
- at least 99% training-study cache coverage;
- projected 1,300-study hidden-test encoding no more than 9,000 seconds;
- exact weak-label identity outside the official-gold rows.

Expected outputs include `e008_gate.json`, OOF/test prediction archives, five
adapter checkpoints, a reusable frozen-feature cache, and a diagnostic
`submission.csv` that is not automatically submitted.

## Build

```bash
rtk python build_notebook.py \
  ../e003_dino_diversity_lab/e003-dinov2-diversity-lab.ipynb \
  e008-rad-dino-scout.ipynb
rtk kaggle kernels push -p .
```
