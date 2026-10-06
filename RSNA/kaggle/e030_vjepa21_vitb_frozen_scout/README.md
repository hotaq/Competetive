# E030 — V-JEPA 2.1 ViT-B frozen MRI scout

Controlled challenger to the exact64 CoAtNet incumbent.

- Official Meta V-JEPA 2.1 repository pinned to commit
  `204698b45b3712590f06245fbfba32d3be539812`.
- Official distilled ViT-B checkpoint downloaded only inside Kaggle, hashed, loaded,
  and deleted before output collection.
- Each acquisition slot is an independent ordered 16-slice grayscale-as-RGB clip
  shaped `[B, 3, T, H, W]`; clips never cross series.
- Frozen four-level V-JEPA features feed a target-specific slot/time MIL head.
- Fixed 512/128 non-Gold train/validation split; Gold58 is evaluated once after
  validation selection.
- Dense PCA/evidence maps and top-vs-random deletion tests are produced.
- No submission, checkpoint, or frozen feature cache is exported.

```bash
rtk proxy python kaggle/e030_vjepa21_vitb_frozen_scout/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e030_vjepa21_vitb_frozen_scout
rtk proxy python kaggle/e030_vjepa21_vitb_frozen_scout/validate_output.py \
  kaggle/e030_vjepa21_vitb_frozen_scout/output_v1
```
