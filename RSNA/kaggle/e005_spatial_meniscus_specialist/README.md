# E005: spatial meniscus specialist

E005 is a Kaggle-only, target-specific pilot for medial and lateral meniscus
abnormalities. Unlike E003/E004, it preserves spatial evidence from DINOv2 patch
tokens instead of reducing every slice to global CLS/mean/max features.

## Representation

- 336 px 2.5D images;
- sagittal-fluid: 12 slices;
- coronal-fluid: 8 slices;
- DINOv2-small 24×24 patch grid;
- adaptive 8×8 spatial regions;
- pinned orthogonal projection from 384 to 192 channels;
- approximately 2.6 GB train cache retained on Kaggle.

## Pilot head

A target-query cross-attention head reads slot, slice position, and 2D region
position. The backbone stays frozen. Five duplicate-report-grouped folds, one seed,
and eight fixed epochs provide full OOF predictions without fold-based early stopping.

## Gate

The specialist receives a predeclared 20% share in the two meniscus columns only.
Expansion requires:

- mean two-target OOF gain at least 0.003;
- lateral-meniscus gain at least 0.005;
- medial-meniscus delta no worse than -0.003;
- official-gold mean direction no worse than -0.020;
- median correlation with the anchor no greater than 0.97;
- at least 99% train cache coverage.

No leaderboard submission is made automatically.
