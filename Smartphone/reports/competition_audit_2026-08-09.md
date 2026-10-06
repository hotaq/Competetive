# Playground Series S6E8 audit — 2026-08-09

This is a dated recovery snapshot produced from the authenticated Kaggle CLI. It is
intended to preserve the project context without relying on leaderboard memory.

## Live competition state

- Competition: `playground-series-s6e8` — Predicting Smartphone Addiction
- Metric: ROC AUC on the predicted probability for `addicted_label`
- Deadline: 2026-08-31 23:59 UTC (2026-09-01 06:59 ICT)
- Limits: 10 submissions/day, 2 final submissions, maximum team size 3
- External data/models and AutoML are allowed when reasonably accessible under the rules
- Train: 691,369 rows, 12 predictors plus `id` and target
- Test: 296,302 rows, 12 predictors plus `id`
- Target positive rate: 0.709424
- Missing values: 870,360 in train and 377,157 in test

## Account and leaderboard state

- Team: `hotaq` (`chinnaphatkhun`)
- Best public score: **0.97063**
- Live rank: **173 / 1,223** (about top 14.1%)
- Public leader: **0.97120**
- Gap to rank 1: **0.00057 AUC**
- Existing submissions: 5; latest submission was 2026-08-02

Existing progression:

| run | offline evidence | public score |
| --- | ---: | ---: |
| HistGB baseline | OOF 0.959061 | 0.96018 |
| LGBM/HistGB blend | OOF 0.964354 | 0.96588 |
| 54-member logit stack | OOF 0.969427 | 0.97061 |
| + public TabNet rank blend | OOF 0.969448 | 0.97063 |
| + independent RealMLP rank blend | OOF 0.969473 | 0.97063 |

The later Danush blend changed OOF by only +0.00000054 and did not improve the rounded
public score.

## Important audit findings

1. **The project uses an old OOF-library snapshot.** The 2026-08-02 run had 54 members.
   The current public dataset has 74 members and adds the most useful independent family,
   a Lookup Transformer, plus wide pair-lattice models.
2. **`pub_ravi` is a level-2 stack inside the current 54-member pool.** Its provenance is
   explicitly `family = L2 stack`. It must not be treated as a level-1 member in a new
   stack without a strict nested lineage audit. The current model gives it a positive
   coefficient, so the old “honest OOF” label is not strict end-to-end evidence.
3. **The public 0.97092 “elite” submission is exactly an equal rank average of four public
   submissions.** Its raw-probability Pearson correlations form apparent clusters around
   0.884, but the correlations relevant to rank averaging are much higher: Spearman
   0.9939–0.9992. It is a useful public benchmark, but its diversity claim is overstated
   and it has no aligned OOF selection evidence of its own.
4. **The strongest reproducible signals are representation changes, not more tuning:**
   fold-safe exact-value target/frequency encoding, decimal-lattice features, wide pair
   lattice keys, and an independently trained Lookup Transformer on the frozen folds.
5. **External real-world rows should not be injected into training.** The local controlled
   experiment degraded every tested weight, and current public evidence agrees.
6. **Leaderboard changes below roughly 0.00007–0.00014 are not reliable selection signals.**
   Development should be driven by paired, fold-level OOF deltas and provenance checks.

## Compute and CLI policy

- Use Kaggle CLI for competition pages, rules, files, datasets, notebook source/output,
  status checks, and submissions.
- Use local work only for lightweight inspection, alignment, hashes, correlations, and
  submission validation; these actions do not require a Kaggle kernel.
- Launch Kaggle CPU only for actual CPU model/meta-model training.
- Launch Kaggle GPU only when the training algorithm benefits from it (Lookup Transformer,
  TabM/RealMLP, or GPU-accelerated tree training). Never reserve a GPU for rank blending,
  file conversion, downloads, or EDA.
- Pin every competition, dataset, and kernel source revision before a scored run.
- Do not submit a candidate until IDs, row count, finite range, provenance, OOF protocol,
  and prediction hashes have been validated.

The historical `tabnet_blend_v9`, `realmlp_blend_v10`, and `danush_blend_v11` metadata
were changed from GPU to CPU because they only fit small logistic meta-models. The
XGBoost residual kernel remains GPU-enabled because it actually trains XGBoost.

## Recommended next experiment

Build a clean V12 stack on the frozen five-fold split:

1. Pin the current public OOF-library revision and create a provenance manifest.
2. Exclude all level-2/blend members and any source whose fold contract cannot be verified.
3. Compare rank-only, logit-only, and dual `[rank, logit]` member representations under
   strict nested meta-validation.
4. Test the missingness-regime mixture using a predeclared weight and paired fold deltas.
5. Only if the clean stack is saturated, train one genuinely independent member—preferably
   the Lookup Transformer—on Kaggle GPU using the same frozen folds.
6. Keep a strict reproducible candidate and a public-ensemble candidate as separate final
   submission options; do not tune their blend on the public leaderboard.

## CLI sources checked

- Competition pages: abstract, evaluation, timeline, data description, rules, prizes
- Competition files, submissions, and full downloaded public leaderboard
- Current discussion topics and public notebook listings
- Public notebook source and output for the 74-model stack, clean/reproducible stack,
  level-2 lineage audit, feature experiments, and 0.97092 rank ensemble
- Current `szymonkapiski/s6e8-oof-library-47-models` README and manifest
