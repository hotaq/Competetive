# E024 — What trained knee MRI models see

Private Kaggle notebook comparing the exact 0.924 CoAtNet checkpoint with one
competition-trained DINOv2-S member on the same held-out gold study.

Outputs:

- native CoAtNet window-attention plot;
- native DINO group/slot-attention plot;
- target-gradient overlays for both models;
- CoAtNet coarse occlusion maps;
- DINO class-token attention (or token-energy fallback);
- CSV/JSON comparison artifacts.

The DINO visualization uses one real member for interpretation and is not a
score-equivalent reproduction of the twenty-member ensemble. No submission is
created.

```bash
rtk proxy python kaggle/e024_model_vision_audit/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e024_model_vision_audit
```
