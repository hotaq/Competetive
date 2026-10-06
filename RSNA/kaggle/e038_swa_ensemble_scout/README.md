# E038 — inference-only base+SWA ensemble scout

E037 closed the test-time-augmentation lever: ten arms, best real effect
`+0.001044` from pooling every safe window, which is under the `0.0020` noise
floor, and everything else negative. The obvious next lever is ensembling, so
this experiment asks the narrowest honest version of that question:

> Does averaging the 0.927 checkpoint with the SWA checkpoint sitting next to it
> in the same dataset beat the 0.927 checkpoint alone?

## The binding constraint, stated up front

There is no fold ensemble to build. The attached datasets contain exactly two
usable weight files:

- `raptor_ft_coatnet_v4_full.pt` — the LB 0.927 checkpoint, sha256 `89606f05…`,
  292,829,402 bytes
- `raptor_ft_coatnet_v4_full_swa.pt` — its stochastic-weight-averaged sibling,
  292,831,426 bytes

The 2,024-byte difference is the only evidence so far that the second file is
not a copy of the first; E038 checks it properly, by hash and then parameter by
parameter.

They are not independent models. An SWA checkpoint is an average of points along
one training trajectory, so it shares the base checkpoint's data order, its
initialisation, and most of its errors. The literature effect of ensembling two
such members is small, and it can be zero. **The expected outcome of E038 is a
null result.**

That is not a reason to skip it. It is a reason to build it so the null explains
itself: E038 measures how far apart the two members actually are, in weight
space and in prediction space, so a flat result comes with the number that
caused it rather than a shrug. Nothing else in the current tool set costs two
GPU hours and answers a question this cleanly.

## What is measured

Six arms, one baseline, one shared feature bank per checkpoint per study:

| arm | what it is |
| --- | --- |
| `B0_base` | base checkpoint, the 42 baseline windows — the proven path |
| `B1_swa` | SWA checkpoint, the same 42 windows |
| `B2_ens_prob` | mean of base and SWA in probability space |
| `B3_ens_logit` | mean of base and SWA in logit space |
| `B4_base_all` | base checkpoint pooling every safe window — E037's arm A1 |
| `B5_ens_prob_all` | mean of both checkpoints, each pooling every safe window |

`B2` and `B3` are the two standard ways to combine two members and they are not
the same operator; which one wins, if either does, is itself informative.
`B5` asks whether the two levers stack. `B4` costs nothing extra and is here for
a second reason, below.

## Why the arms are almost free

`RaptorClassifier.head` pools its windows with a softmax attention:

```python
a = torch.softmax(self.att(h), dim=1)
pooled = torch.einsum("bkn,bkf->bnf", a, h)
```

That is permutation invariant and accepts any window count `K`. So for each
study E038 runs each checkpoint's backbone **once** over every safe window,
ordered with the 42 baseline centres first, and then reads both the 42-window
and the all-window logits out of that one bank. The cost is two backbone passes
per study — one per checkpoint — not six.

The bank's first 42 rows are the baseline windows in the baseline order, chunked
by the same `CHUNK = 6`, so `B0` is bit-identical to `predict_studies` from E028
and E036. `mock_run_test.py` asserts that against a copy of the E036 function
rather than trusting it.

## The split is E037's untouched complement

E037 drew 2,048 studies from a 4,349-study labelled non-Gold pool with seed
3701 and recorded the hash of that draw. E038:

1. rebuilds the pool and replays the same seeded permutation,
2. hashes the first 2,048 and **refuses to run** unless the hash equals
   `97de5347…`, the value E037 wrote out,
3. takes the remaining 2,301 studies as its evaluation split,
4. asserts the two sets are disjoint, and records that as a mechanical gate.

This buys two things. The ensemble is judged on studies no arm-selection
decision has ever seen. And `B4_base_all` becomes a genuine out-of-sample
replication of E037's `A1`: same operation, same code path, disjoint studies.
`A1` measured `+0.001044` with a 95% interval of `[+0.00062, +0.00150]` on
n=2048. Whether that survives on n=2301 is reported in `e037_a1_replication` —
**reported, never gated.** E038's decision is about the ensemble; moving the
gain floor to let `A1` through after watching it fail would be moving the
goalposts, so the floor stays at `0.0020`.

## Promotion gates

An arm is promoted only if it clears all six, on the 2,301-study split:

| gate | threshold | why |
| --- | --- | --- |
| `gain` | macro delta ≥ `0.0020` | the measured LB noise floor; unchanged from E037 |
| `bootstrap_win` | ≥ 0.98 of 2,000 paired resamples | the effect is not one lucky draw |
| `ci_low_positive` | 95% interval excludes 0 | the sign is real |
| `no_target_collapse` | worst target ≥ −0.010 | no target is sacrificed for the macro |
| `gold58_sign_agrees` | Gold58 delta > 0 | confirmation on adjudicated labels |
| `gold58_no_target_collapse` | worst Gold58 target ≥ −0.010 | same, per target |

Gold58 confirms; it never selects. The winner is chosen by delta on the 2,301
split. E036 and E037 both showed Gold58 can flip sign at 58 studies — in E037 it
reported `+0.00257` on the zoom arm while the 2,048-study split reported
`−0.00612` — so it is only ever allowed to veto, never to pick.

## The estimator reports its own resolution

Every arm records `ci95_halfwidth` and `estimator_resolves_threshold`
(`halfwidth ≤ MIN_VALID_GAIN`). If no arm can resolve `0.0020`, the verdict is
`STOP_ENSEMBLE_UNDERPOWERED` rather than `STOP_ENSEMBLE_NO_GAIN` — a null result
from a blunt instrument is a different fact from a null result from a sharp one.
E037's paired same-model estimator measured a median half-width of `0.000849` at
n=2048, comfortably inside the threshold, so E038 at n=2301 should resolve it;
the run states whether it did instead of assuming.

A null arm — the baseline bootstrapped against a copy of itself — must report a
delta of exactly zero and a win rate of exactly zero, or the run is a mechanical
failure. That is the estimator proving it reports nothing for nothing.

## The null result is not called a broken run

One design point worth stating. If the two members predict alike, `B1`, `B2`,
`B3` and `B5` all collapse onto `B0`. That is E038's answer, not a wiring bug,
so it is **reported** as `ensemble_divergence.swa_arms_differ_from_baseline` and
is not a mechanical gate. The only distinctness the mechanical gates require is
`base_all_arm_distinct`: `B4` pools more windows from one checkpoint, so it must
differ whatever the SWA file turns out to hold.

Two things do fail hard, because they mean there is no experiment at all:
`swa_sha256_differs_from_base`, and `swa_weights_differ_from_base` from a
parameter-by-parameter comparison. A checkpoint that is byte-identical or
weight-identical to the base is not an ensemble member.

## Stage A runs first so a broken run costs seconds

Nobody has opened `raptor_ft_coatnet_v4_full_swa.pt`. It could be a bare state
dict, a wrapped one, DDP-prefixed, or carry `torch.optim.swa_utils`'
`n_averaged` counter. So the order is:

1. load both checkpoints and diff their parameters — about 20 seconds; a bad SWA
   file dies here, before any pixel is read,
2. cache and score Gold58 (58 studies, ~4 minutes) and check `B0` reproduces
   `0.9174298724699895`,
3. project the full runtime from that measured throughput and refuse to start
   stage B if it exceeds the 420-minute budget, or if scratch cannot hold the
   cache with 25% headroom,
4. only then cache and score the 2,301-study split.

`load_checkpoint_variant` applies at most two normalisations — stripping a
`module.` prefix and dropping `n_averaged` — records exactly which it applied in
the summary, and loads `strict=True` so a partial match is a failure rather than
a silently half-initialised model. `mock_run_test.py` exercises all five
plausible file shapes plus two corrupt ones on CPU.

The base checkpoint is loaded by `load_model`, verbatim from E028/E036, so `B0`
stays on the proven path.

## Cost and safety

Projected from E037's measured throughput of 41.6 windows/second and 52.1 mean
safe candidates per study:

- cache 2,359 studies: ~27 min
- arms, 2 checkpoints × ~52 windows × 2,359 studies: ~98 min
- **total ≈ 2.1 h** against the T4 quota

Cache is ~9.7 GiB and is written to `/kaggle/temp`, created not probed — E037 v1
died because the code fell back to `/kaggle/working` when `/kaggle/temp` did not
yet exist, and a 9 GiB working directory is collected as kernel output. A
scratch path that resolves under the working directory raises. The cache is
removed before exit.

The run trains nothing: `smoke_test.py` rejects `torch.optim`, `.backward(`,
`GradScaler`, `model.train()` and `requires_grad` in the notebook alongside the
standing bans on `torch.save`, `safetensors`, `submission.csv` and
`kaggle competitions submit`. No submission, no exported checkpoint, no exported
feature cache.

## Local preparation

```bash
python build_notebook.py     # regenerate the notebook
python smoke_test.py         # 131 static and behavioural checks
python mock_run_test.py      # 99 checks, full pipeline on a toy corpus, CPU
kaggle kernels push -p .
```

After the run:

```bash
kaggle kernels output chinnaphatkhun/rsna-knee-e038-swa-ensemble-scout -p output_v1
python validate_output.py output_v1
```

## What follows

- `PASS_ENSEMBLE_PROMOTE` — the ensemble is worth a submission slot, and the
  named arm goes into an inference kernel against the leaderboard.
- `STOP_ENSEMBLE_NO_GAIN` — the expected outcome. Ensembling two members off one
  trajectory does not move this model, and `ensemble_divergence` says how
  similar they were. Combined with E036 (fine-tuning), E004/E006/E006b (labels),
  E035 (blending) and E037 (TTA), the cheap levers around the 0.927 checkpoint
  are then all measured and all closed.
- `STOP_ENSEMBLE_UNDERPOWERED` — the split cannot resolve `0.0020`; the result
  is inconclusive, not negative, and that distinction is recorded.
