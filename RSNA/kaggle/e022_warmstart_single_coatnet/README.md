# E022 — warm-start the verified 0.924 single CoAtNet

E022 loads the complete CC0 `raptor_ft_coatnet_v4_full.pt` state strictly,
including its per-target attention head, then fine-tunes with the pinned E020
CC0 consensus labels. The 58 official-label studies remain validation-only.

Safety properties:

- evaluate the untouched warm-start checkpoint before any optimizer step;
- keep epoch `-1` as a valid best checkpoint;
- use low learning rates (`5e-6` backbone, `1e-4` head);
- first run is a 40-study smoke with 24 validation windows;
- if fine-tuning worsens Gold58, emitted best weights remain the original model.
