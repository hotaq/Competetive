# E034: nested CAFormer CV and protected blend

E034 tests whether the E033 frozen CAFormer advantage is reproducible without
reopening Gold58.

## Locked protocol

- Same deterministic 640 non-Gold studies used by E033.
- SHA-locked 4407-row CC0 consensus-label parquet embedded in the notebook;
  this preserves E033 labels while avoiding a repeated 20+ minute rebuild.
- One exact64 cache with 24 safe same-series 2.5D windows per study.
- Frozen CoAtNet and CAFormer features extracted once.
- Five untouched outer folds of 128 studies.
- Two fixed-seed outer heads per backbone; fixed 60 epochs with no outer-fold
  early stopping.
- Within each outer training split, four-fold inner OOF predictions choose
  target-wise alpha from 0, 0.25, 0.5, 0.75, 1.
- Alpha remains zero unless the inner gain is at least 0.003.
- Gold IDs are embedded only to exclude them. Gold labels and images are never
  evaluated or encoded.
- No checkpoint, feature cache, Gold prediction, or submission is exported.

## Promotion gate

CAFormer must show at least +0.010 OOF macro AUC, win at least four outer folds
and eight targets, remain positive on at least six targets in four of five
folds, avoid target loss below -0.030, and pass paired bootstrap gates. The
nested protected blend must independently pass its gain, fold, target, safety,
and bootstrap gates.

## Local preparation

    rtk proxy python kaggle/e034_caformer_cv_protected_blend/build_notebook.py
    rtk proxy python kaggle/e034_caformer_cv_protected_blend/smoke_test.py

Uploading or starting the private Kaggle notebook remains a separate external
action requiring explicit approval of the generated payload SHA256.
