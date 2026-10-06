# E025 — CoAtNet Gold58 window-safety ablation

Inference-only Kaggle experiment using the exact E020 checkpoint and raw DICOM
preprocessing on all 58 structured-label studies.

It compares exact legacy window centres with a series-safe resample and a
minimal repair that changes only invalid boundary/zero windows. The union of
centres is encoded once per study so the experiment changes no weights or
pixels.

The notebook reports per-target and macro AUC, paired bootstrap deltas,
unsafe-window attention mass, and an explicit hidden-test go/no-go gate. It
never creates a submission.

```bash
rtk proxy python kaggle/e025_gold58_window_safety/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e025_gold58_window_safety
```
