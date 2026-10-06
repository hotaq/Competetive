# E028x — baseline CoAtNet Grad-CAM audit

No-training XAI gate for the exact E020 checkpoint and E028 exact64 safe windows.

The notebook saves target-specific window attention, top-three Grad-CAM overlays,
positive/negative fixed cases, deletion faithfulness, classifier-randomization
sensitivity, deterministic reproduction, and same-window target specificity.

It must run only after E028 P0/P1 passes. It never creates an optimizer or a
competition submission.

```bash
rtk proxy python kaggle/e028x_coatnet_gradcam_audit/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e028x_coatnet_gradcam_audit
rtk proxy python kaggle/e028x_coatnet_gradcam_audit/validate_output.py \
  kaggle/e028x_coatnet_gradcam_audit/output_v1
```
