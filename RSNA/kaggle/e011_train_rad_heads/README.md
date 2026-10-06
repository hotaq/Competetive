# E011 — self-trained RadImageNet heads

No-submission Kaggle training experiment for a license-clean diversity branch.

- Frozen official RadImageNet ResNet-50, verified SHA-256 and MIT upstream.
- Three fat-suppressed planes, eight 224 px slices per plane.
- Target-query attention head, two seeds per fold, five report-hash folds.
- Target-wise fixed pseudo-label teacher selected by the prior Gold58 audit.
- Official competition labels override weak labels where available.
- Fold-safe OOF and Gold58 gates must pass before E012 inference.

The public CC-BY-NC-SA head package is not used.
