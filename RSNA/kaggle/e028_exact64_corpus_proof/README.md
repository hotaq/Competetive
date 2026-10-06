# E028 — P0/P1 exact64 corpus proof

No-training Kaggle proof for the approved CoAtNet fine-tuning ladder.

It locks checkpoint/data identities, excludes Gold58 from the future training
manifest, reproduces the E025 exact64 safe baseline, proves lossless cache parity,
measures series/zero-window safety, and projects full-corpus storage/runtime.

The notebook deletes its temporary cache shards after verification and outputs
only small manifests, metrics, predictions, provenance, and a machine-readable
`PASS`/`STOP` decision. It never creates an optimizer or submission.

```bash
rtk proxy python kaggle/e028_exact64_corpus_proof/build_notebook.py
rtk proxy kaggle kernels push -p kaggle/e028_exact64_corpus_proof
rtk proxy python kaggle/e028_exact64_corpus_proof/validate_output.py \
  kaggle/e028_exact64_corpus_proof/output_v2
```
