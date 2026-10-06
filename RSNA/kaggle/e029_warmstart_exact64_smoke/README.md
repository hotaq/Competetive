# E029 — paired exact64 warm-start smoke

First training rung after E028/E028x. Version 1 established the exact K=4
control but failed efficacy because its bags omitted available MRI slots.
Version 2 used slot-balanced K=12 bags and improved the paired delta from
negative to +0.000331, but it still failed the fixed efficacy gates. Version 3
is one final isolated control: keep the same K=12 data path and head learning
rate, freeze the CoAtNet backbone, and train only the existing
attention/classifier head. Gold58 remains evaluation-only. No submission or
large checkpoint is produced, and the +0.001 / bootstrap 0.65 gates are not
weakened.

```bash
rtk proxy python kaggle/e029_warmstart_exact64_smoke/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e029_warmstart_exact64_smoke
rtk proxy python kaggle/e029_warmstart_exact64_smoke/validate_output.py \
  kaggle/e029_warmstart_exact64_smoke/output_v3
```
