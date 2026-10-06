# E041: refined-label student (label-lever A/B)

Idea (Archit Konde, #36): OOF predictions beat the extracted labels on Gold58, so the labels are the bottleneck.
Our own check: on Gold58, `0.6·v4 + 0.4·labels` scores 0.9305, against 0.9187 for the model alone and
0.8921 for the labels alone.

- `build_labels.py` builds the refined targets offline.
  - Formula: `0.6·qmap(v4) + 0.4·E021 consensus`.
  - Teacher: v4 at all windows, taken from E037 A1 and E038 B4, covering 4,349 studies.
  - qmap maps v4 onto the consensus marginal. Prevalence and `pos_weight` then equal the control arm's, so
    only the label content differs.
  - The same formula scores 0.9282 on Gold58.
- `make_trainer.py` writes `assets/train_e041.py` from the E021 trainer.
  - `--labels` is required.
  - It defaults to 224px `coatnet_rmlp_2_rw_224`.
  - It uses fp16 + GradScaler on T4, has a `--max_hours` guard, and its input search skips the DICOM trees.
- `e041-launcher.py` trains **refined** on GPU0 and **control** on GPU1 with the same seed and recipe.
  - `MODE="smoke"` runs 300 studies for 1 epoch, to check throughput and fp16.
  - `MODE="full"` runs 4,349 studies for 16 epochs.
- Assets: private dataset `chinnaphatkhun/rsna-knee-e041-assets`.
- Read-out: refined vs control Gold58 per epoch. v4 (0.9174 at 384px) is the reference. The LB is the final judge.

```bash
python3 build_labels.py && python3 make_trainer.py && python3 smoke_test.py
<venv-with-timm>/bin/python mock_run_test.py
kaggle datasets version -p assets -m "..."   # after changing assets
kaggle kernels push -p .
```
