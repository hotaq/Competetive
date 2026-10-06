# E031 — frozen V-JEPA dual-head blend

Controlled comparison of two independent heads on one frozen V-JEPA feature tensor.

- E030-compatible target-specific slot/time MIL head (`256` hidden features).
- Two-block, eight-head masked Transformer (`192` hidden features) with twelve
  learned abnormality tokens.
- Separate parameters, optimizers, early stopping, and checkpoint selection.
- Probability and per-target rank blends use one global alpha from
  `[0, 0.25, 0.5, 0.75, 1]`, selected only on the fixed non-Gold validation set.
- The selected heads and blend touch Gold58 once.
- The exact E029 CoAtNet zero-step baseline is embedded as a 4.7 KB SHA-verified
  prediction payload, avoiding fragile notebook-output mount discovery.
- MIL dense deletion and Transformer token-attention deletion must both pass.
- Official V-JEPA code/checkpoint are pinned, downloaded only inside Kaggle, and
  removed before output collection.
- No competition submission, checkpoint, or frozen feature cache is exported.

```bash
rtk proxy python kaggle/e031_vjepa21_dualhead_blend/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e031_vjepa21_dualhead_blend
rtk proxy python kaggle/e031_vjepa21_dualhead_blend/validate_output.py \
  kaggle/e031_vjepa21_dualhead_blend/output_v1
```
