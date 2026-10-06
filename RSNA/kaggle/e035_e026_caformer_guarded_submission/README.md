# E035 — E026-anchored guarded CAFormer submission

E035 is a bounded leaderboard probe built from the E034 signal.

- Anchor: E026 public LB `0.927`, exact output SHA256
  `46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271`.
  Its three-row CSV is stored byte-for-byte in the private, SHA-locked
  `chinnaphatkhun/rsna-knee-e035-assets` dataset and verified at runtime.
- CAFormer: frozen `caformer_b36.sail_in22k_ft_in1k_384` with two MIL heads trained
  on the exact E034 640-study non-Gold set.
  E035 v3 loads the SHA-locked heads from the private asset dataset after their successful training;
  v2 failed only later when Kaggle omitted the E026 kernel-source mount.
- Guarded targets: ACL, both menisci, all three OA targets, Effusion, and Baker's.
- Protected targets: MCL, Synovitis, Contusion, and Fracture remain exactly E026.

E034 did **not** compare against the fine-tuned E026 checkpoint, so E035 is not a
promotion claim. The notebook must report at least one changed target ranking before a
submission slot is spent.

```bash
rtk proxy python kaggle/e035_e026_caformer_guarded_submission/build_notebook.py
rtk proxy python kaggle/e035_e026_caformer_guarded_submission/smoke_test.py
```
