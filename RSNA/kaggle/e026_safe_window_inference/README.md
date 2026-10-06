# E026 — single CoAtNet with series-safe windows

Hidden-test inference candidate promoted by the predeclared E025 Gold58 gate.

Only `_eval_centers` differs from E020. It resamples 42 centers from windows whose
three valid slices stay inside one fixed acquisition slot. Model weights, pixels,
physical crop, normalization, architecture, and rank postprocessing are unchanged.

E025 evidence:

- legacy macro AUC: `0.9130486`
- safe-resample macro AUC: `0.9174299` (`+0.0043812`)
- paired bootstrap win rate: `0.9405`
- 95% bootstrap CI: `[-0.0012379, 0.0109827]`
- target wins/losses: `7 / 4`
- decision: `GO_BUILD_HIDDEN_INFERENCE`

The generated Kaggle script is compiled locally before push. The resulting
`submission.csv` must match the sample schema, contain finite probabilities, and
change at least one target ordering relative to E020 before a submission is spent.

```bash
rtk proxy python kaggle/e026_safe_window_inference/build_kernel.py
rtk proxy kaggle kernels push -p kaggle/e026_safe_window_inference
```
