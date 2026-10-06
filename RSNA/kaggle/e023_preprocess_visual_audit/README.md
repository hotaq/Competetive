# E023 — RSNA Knee preprocessing visual audit

Private Kaggle notebook that visualizes the exact E020 CoAtNet input path without
downloading the competition images locally.

It audits:

- raw DICOM ordering and 6–94% sampling;
- five fixed series slots and missing/ambiguous series;
- 2–98% series windowing and the image-centred 140 mm crop;
- the 64-slice E020 inference stack versus the public 44-slice fine-tuning corpus;
- 2.5D windows that cross series boundaries or contain zero-filled missing slots;
- `PixelSpacing[0]`-only crops; and
- 336 → 384 double interpolation.

The notebook emits PNG panels, CSV tables, and
`preprocessing_audit_summary.json`. It never creates `submission.csv`.

Build locally:

```bash
rtk proxy python kaggle/e023_preprocess_visual_audit/build_notebook.py
```

Push to Kaggle:

```bash
rtk proxy kaggle kernels push -p kaggle/e023_preprocess_visual_audit
```
