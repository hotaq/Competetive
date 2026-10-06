# E032: V-JEPA 2.1 partial fine-tuning scout

E032 follows the failed E031 frozen dual-head experiment with one isolated
hypothesis: the V-JEPA representation may need limited MRI-domain adaptation.

## Controlled protocol

- Identical fixed 512/128 non-Gold split and exact64 reconstruction used by E031.
- Official pinned V-JEPA 2.1 ViT-B/16 checkpoint and commit.
- Frozen target-specific MIL warm-up control.
- Exactly encoder blocks 10 and 11 plus `norms_block.3` are unfrozen.
- Differential learning rates: `1e-5` encoder blocks, `2e-5` output norm,
  `8e-5` MIL head.
- Four maximum partial-fine-tuning epochs, gradient accumulation 4, clipping 1.0.
- Only sequence-consistent intensity augmentation; slice order is unchanged.
- Validation and deletion-XAI gates run before Gold58.
- Gold58 is evaluated once only if every required and pre-Gold gate passes.

## Promotion gates

Before Gold58, the selected model must reach validation macro AUC at least 0.77,
gain at least 0.015 over both the within-run frozen warm-up and exact E031 frozen
MIL reference, improve at least seven targets, avoid a target drop below -0.04,
and pass MIL deletion sanity checks.

Even a Gold58 pass only marks a fine-tuning signal for further controlled work.
The notebook never creates a competition submission, model checkpoint, or large
feature cache.
