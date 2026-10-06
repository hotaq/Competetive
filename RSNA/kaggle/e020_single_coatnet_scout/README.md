# E020 — single-model CoAtNet scout

## Goal

Replace saturated public-prediction blending with one study-level model. The
reference is Dread Development's public one-checkpoint CoAtNet pipeline, which
reports 0.924 public LB with no TTA or prediction ensemble.

The copied inference source remains attributable to
`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`; E020 changes
only the Kaggle kernel ownership and adds our audit/gating wrapper.

## Fixed strategy

1. Audit small public report-label CSVs against the 58 official labels.
2. Build CC0-only soft-label consensus; keep Barun labels audit-only because
   that dataset currently declares a nonstandard `other` license.
3. Hold all 58 official-label studies out of training.
4. Train one CoAtNet checkpoint on Kaggle using the public 44/64-slice corpus,
   physical-mm crops, 2.5D windows, and per-target attention pooling.
5. Run one fold first. Do not spend five-fold compute unless the scout clears
   the predeclared validation gate.

## Scout gate

- Gold58 macro AUC at least 0.910.
- Report-label OOF macro AUC at least 0.850.
- At least 8/12 Gold58 targets no worse than the public reference audit.
- Complete inference artifact, exact schema, finite values, and no missing UID.
- Submit the one-checkpoint scout only after all offline checks pass.

## Label audit

The audit downloads no DICOM data. It expects the small CSV files under
`/tmp/rsna-label-audit` and writes results plus the training parquet under
`/tmp/rsna-label-audit/results`.

```bash
rtk python kaggle/e020_single_coatnet_scout/label_audit.py
```
