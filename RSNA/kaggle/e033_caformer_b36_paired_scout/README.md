# E033: CAFormer-B36 paired frozen-backbone scout

E033 compares `caformer_b36.sail_in22k_ft_in1k_384` against
`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` before committing to an expensive
CAFormer fine-tuning run.

## Controlled protocol

- Same fixed 512/128 non-Gold split used by E030-E032.
- Same exact64 reconstruction and 24 safe same-series 2.5D windows.
- Both pretrained backbones are frozen. CoAtNet emits 1024-dimensional native
  embeddings and CAFormer emits 768-dimensional embeddings; each receives its
  own learned projection into the same 512-dimensional MIL space.
- Three identically seeded target-specific Attention-MIL heads are trained for
  each backbone.
- The CAFormer decision is made on the fixed validation split first.
- Gold58 is opened once only when CAFormer gains at least 0.005 validation
  macro AUC, wins at least seven targets, and has no target drop below -0.05.
- A Gold58 pass promotes CAFormer only to a partial/full fine-tuning experiment;
  it does not replace the E020 0.924 CoAtNet model directly.
- No submission, checkpoint, or large feature cache is exported.

## Local preparation

```bash
rtk proxy python kaggle/e033_caformer_b36_paired_scout/build_notebook.py
rtk proxy python kaggle/e033_caformer_b36_paired_scout/smoke_test.py
```

Uploading or starting the private Kaggle kernel remains a separate external
action requiring the user's explicit approval of the generated payload hash.
