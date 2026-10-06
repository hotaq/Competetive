# S6E8 live Kaggle insight audit — 2026-08-25

Status: **V16 rejected; alpha-0.70 anchor submitted after explicit approval**

## Executive result

The newest public `0.97127` and `0.97128` notebooks are not new predictive
models. They are micro-blends of already scored submissions. The `0.97127`
submission has a published aligned OOF counterpart, which lets us test it
honestly: the leaderboard-improving direction makes OOF worse.

The only fresh mechanism with controlled 5-fold evidence was LightGBM
high-cardinality binning (`max_bin` near the distinct-value count). The private
V16 Kaggle CPU test completed, but both aligned streams failed the incremental
OOF gate against the stronger alpha-0.70 anchor. The mechanism is therefore not
promoted.

## 1. Fresh mechanism: high-cardinality LightGBM bins

Source: [discussion 737422](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/737422)  
Code: [max_bin notebook](https://www.kaggle.com/code/kitopl/max-bin-raise-it-to-your-distinct-value-count)

The author held folds and LightGBM parameters fixed and changed only features
and `max_bin`:

| feature representation | max_bin 255 | max_bin 2047 | gain |
| --- | ---: | ---: | ---: |
| raw 12 columns | 0.96431 | 0.96672 | +0.0024 |
| 18 columns, including interval/leftover features | 0.96541 | 0.96746 | +0.0020 |
| 21 columns, including three frequency encodings | 0.96717 | 0.96767 | +0.0005 |

Interpretation: high-resolution bins and exact-value/frequency encodings are
mostly substitutes. Both help LightGBM distinguish quantised values that its
default 255-bin sketch merges; their gains are not additive.

The published run's raw sweep rose from `0.96406` at 255 bins to `0.96644` at
2047 bins and then plateaued. Its test predictions have Spearman `0.987214`
against our strong rank-logit anchor, so it is different enough to justify an
OOF pricing experiment but not different enough to blend blindly.

V16 protocol:

- Kaggle CPU only;
- fixed `StratifiedKFold(5, shuffle=True, random_state=42)`;
- fixed 2,100 trees, so outer-validation labels do not select checkpoints;
- raw high-bin member and a secondary high-bin + three label-free frequency
  encodings;
- export aligned OOF/test artifacts; make no submission.

Kernel: [private V16 run](https://www.kaggle.com/code/chinnaphatkhun/smartphone-addiction-high-bin-lightgbm-v16)

## 2. Public LB 0.97127: exactly reproduced, OOF-rejected

Source: [regime-calibrated rank fusion](https://www.kaggle.com/code/atakanaldemir/s6e8-regime-calibrated-rank-fusion-lb-0-97127)

The notebook applies percentile ranks and blends:

- `72.5001%` of the public 205-member rank-logit-regime endpoint;
- `27.4999%` of a separately published V13 diversity anchor;
- then ranks the result again.

Our reconstruction matches the published submission with maximum absolute
difference `1.11e-16`, so the OOF comparison is paired to the exact test file.

| stream | OOF AUC |
| --- | ---: |
| alpha-0.70 rank-logit parent | 0.9704211377 |
| V13 diversity anchor | 0.9701665486 |
| published 72.5/27.5 blend | 0.9704023680 |

Published blend minus parent: **-0.0000187697 OOF**. Fold deltas:

`-0.0000291505, -0.0000256637, -0.0000239726, -0.0000285017, +0.0000186304`

Only **1/5 folds** improves. This is a public-leaderboard gain in the opposite
direction from the stronger OOF evidence, so it is rejected.

## 3. Public LB 0.97128 and "liner" notebooks

The `Top 20 Formula: Dual Master Rank Blend` notebook uses a fixed 45/55 rank
blend of two files already labelled `0.97127` and `0.97128`. Its attached
dataset contains only `blend1.csv` and `blend2.csv`; no OOF counterparts are
published. It therefore cannot establish incremental predictive signal.

The `knock the blender with a liner` notebook contains one effective operation:
copy a precomputed CSV from an attached dataset to `submission.csv`. It exposes
no training, OOF, or reproducible weighting method.

Neither is suitable as model evidence.

## 4. Why public-score micro-selection is unsafe here

Source: [Why every S6E8 notebook above 0.97110 probably overfits](https://www.kaggle.com/code/raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits)

The author reports a student residual whose honest OOF optimum is weight `+0.12`
with `+0.00005267` AUC. Public submissions instead prefer weight `-0.08`, even
though that direction loses `-0.00009056` OOF.

In ten pseudo-public experiments sized like the real public board:

- mean apparent selected-split gain: `+0.00000472`;
- mean delta on unused labels: `-0.00000633`;
- a different weight was selected in `8/10` trials.

This matches our V13 audit: tiny leaderboard improvements can select the wrong
sign at the frontier.

## 5. V16 controlled high-bin result

Kernel:
`chinnaphatkhun/smartphone-addiction-high-bin-lightgbm-v16`

- Kaggle CPU runtime: 1,304.78 seconds.
- Frozen validation: 5-fold stratified, shuffled, random state 42.
- `lgb_highbin_raw` standalone OOF: **0.9664298146**.
- `lgb_highbin_freq3` standalone OOF: **0.9666444507**.
- Frequency features improved the standalone model by **+0.0002146360**.
- Strong alpha-0.70 anchor OOF: **0.9704211377**.

At the predeclared primary 2.5% rank-blend weight:

| candidate | Spearman vs anchor | OOF delta | fold wins | gate |
|---|---:|---:|---:|---|
| `lgb_highbin_raw` | 0.9795637 | -0.0000031302 | 1/5 | fail |
| `lgb_highbin_freq3` | 0.9809987 | -0.0000029422 | 1/5 | fail |

Every tested weight from 1% through 10% reduced pooled anchor OOF for both
candidates. No promoted blend CSV was generated. Downloaded OOF and test IDs,
labels, finite ranges, and stored SHA-256 hashes all passed validation.
The read-only Kaggle history check still shows the latest account submission as
the 2026-08-02 V10 RealMLP blend (public 0.97063), confirming that V16 was not
submitted.

Evidence:

- `kaggle/highbin_lgbm_v16_output/metrics.json`
- `reports/highbin_lgbm_v16_pricing_ranklogit/pricing.json`
- `reports/highbin_lgbm_v16_pricing_ranklogit/artifact_audit.json`

## 6. Promotion rule

Do not copy a high public score. Promote V16 only if a fixed small rank blend:

1. improves pooled OOF against the alpha-0.70 anchor;
2. improves at least 4 of 5 frozen folds;
3. is not dependent on choosing its sign or weight from the public leaderboard;
4. produces aligned IDs, finite predictions, and reproducible hashes.

No Kaggle submission will be made without explicit user approval.

## 7. Approved submission receipt

After explicit user approval, the exact validated alpha-0.70 anchor file was
submitted to `playground-series-s6e8`.

- Kaggle reference: `55772491`
- File: `submission_alpha070.csv`
- SHA-256: `d62fd0792a10bcfb00be6d1ab6487c569fcf4a059868bea83588601dfe345bab`
- Status: `SubmissionStatus.COMPLETE`
- Public score: **0.97125**
- Previous account best: **0.97063**
- V16 submitted: **no**

Receipt: `reports/kaggle_submission_alpha070_2026-08-25.json`
