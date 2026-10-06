# E037 — inference-only test-time augmentation scout

E036 closed the fine-tuning lever. Warm-starting `raptor_ft_coatnet_v4_full.pt`
on 512 non-Gold studies degraded it from the very first epoch on both seeds, and
the reason was measurable rather than mysterious: the CC0 consensus the model
trains on scores `0.8921` on the official image-adjudicated Gold58 labels while
the checkpoint itself scores `0.9174`. The model already beats its supervision
by `+0.025`, so every gradient step on those labels pulls it back down. E004,
E006, and E006b had already shown there is no better public label source to
switch to.

That leaves the levers that need **no new supervision**: test-time augmentation
and fold ensembling. E037 measures the first one. Nothing is trained here — the
checkpoint is loaded once, frozen, and asked the same question several ways.

## Why the arms are almost free

`RaptorClassifier.head` pools windows with `softmax(..., dim=1)` over `K`:

```python
a = torch.softmax(self.att(h), dim=1)
pooled = torch.einsum("bkn,bkf->bnf", a, h)
```

It is permutation invariant and accepts any window count. So backbone features
can be computed **once per window** and reused by every arm that changes only
*which* windows are pooled. Only the two arms that change *pixels* need their
own backbone pass. That also makes "pool more windows" a legitimate TTA arm in
its own right, handled natively by the head.

The identity bank is built in the order `eval_centers + remaining candidates`
and chunked by six, exactly as `predict_studies` does in E028/E036, so the first
seven chunks are byte-identical to the proven path. `mock_run_test.py` asserts
this against a copy of `predict_studies`: the baseline logits come out
**bit-identical**, so `A0` must reproduce `0.9174298724699895` — and the v1 run
confirmed it on Gold58, gap `0.0`, bit exact.

Most studies carry more safe candidates than the 42 the baseline draws (measured
over the real 2,106-study corpus: median 54, mean 52.12, max 54), which is what
gives arms A1/A2 anything extra to pool. A short pool is not an error: v1 assumed
from E036's 512-study sample that every study had at least 42 and hard-failed on
the one study in 2,106 that has 36. `resample_centers` returns exactly 42 centres
whatever the pool size, repeating windows when it must, and the production path
does the same — so the notebook reproduces that and counts it
(`short_pool_studies`) rather than rejecting it. Distinctness is still a hard
requirement whenever the pool can supply it. For such a study `identity/all`
equals `identity/base`, so A1 collapses onto A0; at 1 study in 2,106 (0.05%) the
dilution is negligible, and `validate_output.py` checks the share stays under 5%.

## The ten arms

| arm | definition | backbone cost |
|---|---|---|
| `A0_baseline` | 42 linspace centres, identity pixels — the E028/E036 path | shared bank |
| `A1_windows_all` | every safe candidate window pooled by the attention head | shared bank |
| `A2_windows_jitter3` | probability mean over three slid 42-centre draws | shared bank |
| `A3_hflip` | horizontal flip only — a laterality probe, not a serious candidate | one pass |
| `A4_hflip_avg` | probability mean of identity and flip | free |
| `A5_zoom` | 130 mm crop only | one pass |
| `A6_zoom_avg` | probability mean of the 140 mm and 130 mm crops | free |
| `A7_full_avg` | probability mean of identity, flip, and zoom | free |
| `A8_full_avg_logit` | the same three views averaged in logit space instead | free |
| `A9_all_plus_zoom_avg` | all-window pooling averaged with the 130 mm crop | free |

`jittered_centers(candidates, 42, 0.0)` reproduces `resample_centers` exactly,
so `A2` is the baseline draw plus two neighbours rather than a different
sampler. The zoom crops 12 pixels from each side of the cached 336-pixel image:
`140 mm x 312 / 336 = 130 mm` exactly, checked as a mechanical gate.

### What the flip actually tests

A horizontal flip mirrors medial and lateral on the coronal and axial planes
(on sagittal it mirrors anterior/posterior). Four of the twelve targets name a
side, so `A3` is expected to *hurt* Medial Meniscus, Lateral Meniscus, Medial
OA, and Lateral OA if the model uses laterality at all. The summary therefore
records a `laterality_probe` with the per-target flip deltas and the mean over
the four laterality targets against the mean over the other eight — reported
whether or not the flip helps. A flip that costs nothing on those four targets
would be a finding in its own right: it would mean the model is laterality-blind
and there is headroom in teaching it the side.

## What is being decided

- **Selection** on a deterministic `EVAL_SEED=3701` split of **2,048** labelled
  non-Gold studies, with CC0 consensus labels binarised at `0.5`. AUC is
  rank-based and both sides of every comparison see the same labels, so label
  noise widens the interval without biasing the sign.
- **Confirmation** on Gold58, opened only to check that the sign agrees. E036
  showed Gold58 alone waved through a checkpoint that was genuinely worse, so
  it is a second opinion here and never the selector.
- One bootstrap seed (`3702`) is shared by every arm, so all arms are resampled
  on identical study draws and the arms are paired to each other as well as to
  the baseline.

### Promotion gates

Nine candidate arms are compared against one baseline, so the win-rate
threshold is `0.98` rather than E036's `0.90`, and `family_size` is recorded in
the summary. An arm is promoted only if it clears all six:

| gate | threshold |
|---|---|
| `gain` | `eval_macro_delta >= +0.0020` |
| `bootstrap_win` | `>= 0.98` |
| `ci_low_positive` | 95% interval floor `> 0` |
| `no_target_collapse` | no per-target delta below `-0.010` |
| `gold58_sign_agrees` | Gold58 macro delta `> 0` |
| `gold58_no_target_collapse` | no Gold58 per-target delta below `-0.010` |

### The estimator gates itself

E036 measured a paired macro-AUC interval half-width of `±0.0046` at `n=512` —
barely tighter than Gold58's `±0.005` — because Fracture alone (29 positives)
carried 32% of the bootstrap variance and the three rarest targets carried 62%.
`n=2048` takes Fracture to roughly 116 positives, and paired TTA comparisons
are far more correlated than a fine-tuned-versus-frozen comparison, so the
interval should be much tighter. But that is a prediction, not a fact.

So every arm records `ci95_halfwidth` and `estimator_resolves_threshold`
(`halfwidth <= MIN_VALID_GAIN`), and if **no** arm's estimator can resolve the
threshold it is being asked to judge, the run stops with a distinct decision:

| decision | meaning |
|---|---|
| `PASS_TTA_PROMOTE` | at least one arm cleared every gate; `best_arm` names it |
| `STOP_TTA_NO_GAIN` | the estimator was sharp enough and TTA still did not help |
| `STOP_TTA_UNDERPOWERED` | no arm's interval could resolve `+0.0020`; widen the statistic, not the corpus |
| `STOP_MECHANICAL_FAILED` | reproduction, window safety, or wiring failed |

Splitting the last two is the direct lesson of the P2 rung: a null result on an
estimator that could not have detected the effect is not evidence of absence,
and should not be recorded as if it were.

## Stage A runs first so a broken run costs two minutes

E036 v1 spent twenty-six minutes on the cache and the zero-step baseline before
dying on the first training step. E037 caches **only the 58 Gold studies** first,
runs all ten arms on them, and then does two things before the expensive stage:

1. checks that `A0` reproduces `0.9174298724699895` and raises if it does not;
2. turns the measured throughput into a projected total runtime and refuses to
   continue if it exceeds `RUNTIME_BUDGET_MINUTES = 420`.

## Cost and safety

Projected from the E036 run (`45` windows/s end to end, `2.69` s/study cache
build on 4 workers, `4.42` MB/study):

| stage | estimate |
|---|---|
| Gold58 cache + all arms | ~3 min |
| 2,048-study cache build | ~24 min |
| 2,048-study arms (~136 windows/study) | ~105 min |
| paired bootstrap, 9 arms x 2,000 repeats at n=2048 | ~6 min |
| **total** | **~140 min**, against E036's measured 150 |

The ~8.7 GiB exact64 cache is written to `/kaggle/temp`, which is scratch and is
never collected as kernel output, and is removed before exit. v1 *probed* for
that directory and fell back to `/kaggle/working` when the image did not have
it, which put the whole cache into the collected output; scratch is now created
rather than probed, and a scratch path under the working directory raises. No competition
submission, no exported checkpoint, no exported feature cache. `smoke_test.py`
additionally bans every training token (`torch.optim`, `.backward(`,
`GradScaler`, `model.train()`, `requires_grad`) so the notebook cannot quietly
stop being an inference-only scout.

## Local preparation

```bash
python kaggle/e037_tta_scout/build_notebook.py
python kaggle/e037_tta_scout/smoke_test.py
python kaggle/e037_tta_scout/mock_run_test.py
kaggle kernels push -p kaggle/e037_tta_scout
kaggle kernels output chinnaphatkhun/rsna-knee-e037-tta-scout \
  -p kaggle/e037_tta_scout/output_v1
python kaggle/e037_tta_scout/validate_output.py \
  kaggle/e037_tta_scout/output_v1
```

## What follows

If an arm passes, it is a free inference-time gain on the 0.927 anchor and goes
straight into the submission path — no retraining, no new supervision. If the
run stops, the measured `ci95_halfwidth` at `n=2048` is the number E038 needs to
size the fold-ensembling comparison, and the laterality probe stands on its own
regardless of the decision.
