# E036 — repaired P2 smoke plus P3 512-study two-seed scout

E029 v1/v2/v3 all stopped at P2. The rung was not measuring what it claimed:
it required a `+0.0010` Gold58 macro-AUC gain while its own paired bootstrap
reported a 95% interval near `±0.005` on 58 studies. The threshold sat an
order of magnitude inside the noise of its own estimator, and 40 studies over
two epochs cannot move a checkpoint that already scores `0.9174`. Three runs
were spent proving a rung that could not have passed.

E036 splits the two jobs P2 was conflating.

## P2' — mechanical smoke (wiring only)

Zero-step reproduction of the `0.9174298724699895` E028 baseline, cache pixel
parity, cross-series and zero-slice window counts, sampler slot coverage,
optimizer wiring, backbone and head parameter updates, loss behaviour, and
peak GPU. These are the only claims a smoke rung can support.

## P3 — efficacy (the real decision)

- Deterministic `SPLIT_SEED=3601` split of the 4,349 non-Gold studies into
  **512 train / 512 validation**, both SHA-recorded.
- CC0 consensus labels rebuilt in-kernel from the three public label datasets,
  binarised at `0.5` for AUC. Nothing derived is uploaded.
- Warm start from the exact `raptor_ft_coatnet_v4_full.pt` checkpoint, full
  backbone at LR `5e-6` and head at `1e-4`, 5 epochs, K=12 slot-balanced
  exact64 bags, seeds `2701` and `2702`.
- Epoch selection is by **validation** macro AUC. The selected weights are held
  in CPU memory and restored before any Gold58 call.

### Promotion gates

| gate | threshold |
|---|---|
| validation gain, both seeds | `>= +0.0050` |
| paired bootstrap win rate, both seeds | `>= 0.90` |
| bootstrap CI lower bound, both seeds | `> 0` |
| worst per-target validation delta | `>= -0.030` |
| both seeds positive | required |
| seed spread | `<= 0.010` |
| Gold58 regression veto | `>= -0.0050` |

At `n=512` the paired estimator is roughly three times tighter than Gold58, so
`+0.005` is resolvable rather than aspirational.

**Gold58 is not the selection metric.** It is opened once per seed, on the
epoch validation already chose, purely as a regression veto. This follows the
E031 finding that validation-selected quantities did not transfer: selection
and confirmation are kept on different data.

No competition submission, no exported checkpoint, no exported feature cache.

## v2 — the T4 memory repair

v1 (2026-08-30) reached the first training step and died there:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 486.00 MiB.
GPU 0 has a total capacity of 14.56 GiB ... 13.64 GiB is allocated by PyTorch
  timm/models/maxxvit.py:230  torch.nn.functional.scaled_dot_product_attention
```

Everything ahead of it passed: cache parity, zero cross-series and zero-slice
windows, sampler coverage, and a zero-step Gold58 of `0.9174298724699895` with a
gap of exactly `0.0`. The P3 zero-step validation baseline is
`0.8900592194391378`.

The training path was the one path never exercised on a GPU. Evaluation runs
under `no_grad` in float16, six windows at a time; a training step holds
`2 x 12 = 24` windows at `384` with a full autograd graph, and v1 ran it under
**bfloat16**. T4 is sm_75 with no native bf16, so neither the flash nor the
memory-efficient attention backend was eligible and PyTorch fell back to the
math backend, which materialises and retains the whole attention matrix. At
stage 2 that is 16 heads over 576 tokens across 14 blocks — gigabytes that the
memory-efficient kernel never allocates.

v2 changes three things and no science:

| change | why |
|---|---|
| training autocast `bfloat16` → `float16` + `GradScaler` | matches the evaluation path and makes the memory-efficient attention backend eligible on sm_75 |
| `enable_grad_checkpointing` on the backbone stages during training | recomputes activations in backward instead of retaining them; off again before evaluation |
| a synthetic one-step probe before the cache build | a memory failure now costs two minutes instead of twenty-six |

`float16` needs loss scaling, and a `GradScaler` legitimately skips the steps
where it backs the scale off. Those iterations carry inf gradients by
construction, so `amp_train_step` reports `step_applied` and the seed record
counts `applied_optimizer_steps` and `skipped_optimizer_steps` separately —
otherwise a healthy scaler back-off would have failed the
`gradient_norms_finite` wiring gate. New mechanical gates:
`training_step_probe_fits`, `amp_dtype_is_float16`,
`grad_checkpointing_enabled`, and a per-seed `scaler_skip_rate <= 0.02`.

Batch, `K`, learning rates, epochs, seeds, split, and every P3 threshold are
unchanged from v1.

## Local preparation

```bash
python kaggle/e036_p3_512_scout/build_notebook.py
python kaggle/e036_p3_512_scout/smoke_test.py
python kaggle/e036_p3_512_scout/mock_run_test.py
kaggle kernels push -p kaggle/e036_p3_512_scout
kaggle kernels output chinnaphatkhun/rsna-knee-e036-p3-512-scout \
  -p kaggle/e036_p3_512_scout/output_v1
python kaggle/e036_p3_512_scout/validate_output.py \
  kaggle/e036_p3_512_scout/output_v1
```

## If P3 passes

P4 trains all 4,349 non-Gold studies with the locked configuration, then fold
ensembling. If P3 stops, the failure is now informative: it means fine-tuning
at 512 studies genuinely does not move this checkpoint, which is a real result
rather than an artefact of an unmeasurable threshold.
