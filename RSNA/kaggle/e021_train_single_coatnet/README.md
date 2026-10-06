# E021 — CC0-label single-CoAtNet training

Source base: `dreaddevelopment/knee-mri-training-the-twelve-finding-model`.
The upstream source was pulled on 2026-08-23. E021 keeps its study-window and
per-target attention implementation, then changes only the experiment contract:

- build the pinned CC0 label consensus inside Kaggle;
- exclude all 58 official-label studies from fitting (upstream behavior);
- use `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` at 384 px;
- use ImageNet normalization, 12 sampled train windows and 42 validation windows;
- start with a 40-study smoke run before enabling the 4,349-study run.

No DICOM archive or 22 GB preprocessed corpus is downloaded locally.
