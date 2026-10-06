# E006b — nested target-selective fusion

CPU-only follow-up to E006. It selects a Pilkwang/Steven-v4 blend independently for
each target using repeated inner validation inside repeated outer folds. No target choice
is hard-coded from E006's Gold58 table.

The pinned Pilkwang file omits one official-gold study, so all paired source metrics use
the 57 gold rows covered by both sources. The missing Pilkwang value is never imputed.

Passing authorizes only an EfficientNet-B3 scout. E007 must repeat label-source selection
inside its own image-model folds; the full-data shares emitted here are not an independent
Gold58 validation result.

```bash
python build_notebook.py e006b_audit.py e006b-nested-target-fusion.ipynb
```
