# ROGII Kaggle incremental insights

Research date: 2026-07-31

## Scope and evidence labels

This note records only findings that add something material beyond
[`rogii_alignment_research.md`](./rogii_alignment_research.md). Sources are the
official competition pages, original Kaggle discussion authors, and public
notebook code. A leaderboard or OOF number below is a **reported result** unless
it is explicitly described as reproduced locally.

- **Disclosed fact** means the linked source directly says or implements it.
- **Reported result** means an original participant reports a number, but this
  research pass did not independently reproduce it.
- **Synthesis** means a proposed experiment derived from those facts. It must
  not be attributed to the source author.

Nothing here infers Tucker Arrants's undisclosed architecture. His whole-well,
non-tabular result remains a target and input-scope clue, not a recipe.

## Executive delta

The strongest genuinely new direction is a **leave-one-well-out lateral
reference bank**. Competition host Igor Kuvaev says the lateral GR curve
self-correlates and has higher resolution than the typewell GR. He also says
that interpretations from earlier nearby laterals are sometimes used as
"pseudo-typewells." The project already identifies repeated master/typewell
groups, so other labeled laterals in the same group can become legal,
higher-resolution reference curves—provided the target well is excluded
strictly.

The second important delta is negative evidence about the current spatial
surface direction. A public train-side measurement found nearest-neighbor
last-known TVT correlation near 0.98, but nearest-neighbor in-zone drift
correlation near zero. Spatial neighbors can locate the regional level; they do
not reliably transfer the target tail's slope. The present surface-transfer
candidate should therefore remain a weak, guarded structural prior, not a
full-strength trajectory replacement.

The third delta is methodological. Public candidate rankers and GR heatmaps can
look excellent in-window while failing strict whole-well validation. One
reported ranker had score/error Spearman near -0.92 and top-three rate 0.92
in-window, yet gained only about 0.03 ft under strict grouped OOF; approximately
23% of wells had no good candidate at all. Candidate **coverage** and a nested
well holdout must be measured before tuning a selector.

The best low-cost notebook-code lead is row-level wavelet reconstruction plus
candidate/model disagreement. One original notebook reports about 0.08 ft OOF
from a `db4` decomposition, while several much more elaborate attention and
sequence models remain far behind the frontier. That makes wavelet bands a
feature ablation, not a replacement architecture.

## New versus already known

| Topic | Already established in the alignment note | Incremental finding here |
|---|---|---|
| Validation | Whole-well folds and pooled row RMSE | Two concrete leakage paths in public pipelines: train-only contact lookup and own-well formation/dense imputation |
| GR matching | Heel anchoring, bounded NCC/PF/beam, repeated motifs | The host explicitly recommends lateral-to-lateral self-correlation because lateral GR has higher resolution |
| Typewell groups | Typewell/master-sequence identity matters | The host confirms some supplied typewells are actually earlier interpreted laterals ("pseudo-typewells") |
| Spatial transfer | Regional surfaces are plausible priors | Measured neighbor level correlation is high while neighbor tail-drift correlation is near zero |
| Candidate selection | Prefix pseudo-tails and conservative gates | Strong in-window ranker metrics can vanish under strict grouped OOF; about 23% candidate non-coverage was reported |
| Path solvers | Blind DTW and unrestricted matching fail | Full-state DP improved OOF by 0.46 ft but moved LB by only 0.001 ft; the observation cost, not decoder, was the bottleneck |
| Neural approaches | Whole-well/non-tabular is a promising grain | Public CNN/GRU/cross-attention attempts were not frontier evidence; a 2-D heatmap implementation is useful as a representation template only |
| Signal decomposition | Rolling-median/FFT denoising needs end-to-end proof | Row-level `db4` approximation/detail bands report a small OOF gain; global per-well wavelet summaries and one EMD implementation are unsafe |
| Uncertainty | Posterior spread/mode gaps can help gating | Low ensemble disagreement can still accompany a shared, large structural error; low spread is not proof of correctness |
| Leaderboard | Execution variance exists | Multiple unchanged rerun series put typical public-score standard deviation near 0.03 ft for those pipelines |
| Notebook titles | Public scores and branches require source inspection | A notebook titled 6.213 is an exact code clone; `MHA180` denotes midpoint-hedge alpha 1.8, not multi-head attention |

## Primary-source findings

### 1. Lateral GR should be treated as a reference signal

**Disclosed fact.** Host Igor Kuvaev says:

- lateral-well GR "correlates on itself" and has better resolution than
  typewell GR
  ([comment 3456388](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825#3456388));
- the visible lateral GR before Prediction Start can be a better correlation
  reference than the typewell, especially when the well proceeds in the
  negative TVT direction after Prediction Start
  ([comment 3456361](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825#3456361)); and
- formation dip in nearby wells should be similar
  ([comment 3456391](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825#3456391)).

**Disclosed fact.** The same host explains that geologists sometimes interpret a
new lateral using TVT/GR from a previously interpreted nearby lateral, called a
"pseudo-typewell," and that some competition typewells are such
pseudo-typewells
([comment 3455674](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449#3455674)).

**Synthesis.** Matching the target tail only to its vertical typewell leaves
signal on the table. Build two lateral reference channels:

1. target-tail windows against the target's own visible-prefix GR/TVT; and
2. target-tail windows against full labeled lateral GR/TVT from other wells in
   the same master/typewell group.

Both channels must allow forward and reverse TVT direction and must expose
ambiguity rather than force a match.

### 2. Spatial neighbors transfer level much better than drift

**Reported result.** Connor Tynan fitted per-well
`dTVT ~= a*dZ + b`, found spatial structure in the multimodal `b`
distribution, and reported:

- an initial geometry/cluster method around LB 12.8
  ([topic 711308](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308)); and
- a later spline/kernel/KNN version around LB 10.8 that correlated with his
  local CV
  ([comment 3492376](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308#3492376)).

**Reported train-side measurement.** Georgy Mamarin measured nearest-neighbor
last-known TVT correlation around 0.98 on 265 wells, while nearest-neighbor
in-zone drift correlation was near zero. His interpretation is that neighbors
share a base surface but not the target tail slope
([comment 3484435](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308#3484435)).

**Disclosed structural analysis.** FranticXu reports that the six formation
columns collapse to one base surface plus constant offsets, with typewell and
horizontal-well layer thickness agreeing within 1 ft for 754/773 wells. Seven
ANCC cases are anomalous because the shallow boundary is truncated
([topic 708167](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167)).

**Reported result.** In a separate original-author discussion, a non-optimized
grid interpolation of the geology plane was reported at validation RMSE 11.09;
another participant reported basic kriging near local RMSE 11 and test RMSE
13.5
([comments 3465975-3466050](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465975)).

**Safe conclusion.** Spatial structure is real, but public evidence supports it
as an anchor/low-frequency guide rather than a source of fine tail shape.
Transfer of a neighbor's full residual trajectory is not supported.

### 3. Candidate generation is often the ceiling; selector metrics leak easily

**Reported result.** Sleep3r describes a 2-D MTP heatmap plus CatBoost
pairwise ranker whose in-window diagnostics looked excellent:
score/error Spearman around -0.92 and top-three rate 0.92. Under strict
well-grouped OOF, the best honest gain over the GBM was only about 0.03 ft.
Approximately 23% of wells had no good candidate, so no scorer could rescue
them
([comment 3465111](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465111)).

**Reported negative controls.** On real train-well panels, normal-GR top-ten
heatmap coverage was 25.6%, versus 26.1% after shuffling GR
([comment 3462629](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3462629)).
A sliding-window CNN achieved 93% accuracy on synthetic pretraining data but
failed to transfer to real wells, producing only about 0.03 ft over baseline
([comment 3462538](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3462538)).

**Reported oracle envelope.**

- Ten correctly placed anchors reduced one reported trajectory system to about
  4 ft; twenty reduced it to about 1.7 ft. GR-selected anchors gave essentially
  zero gain because repeated typewell motifs defeated localization
  ([comment 3465161](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465161)).
- A fine per-well offset-grid oracle was about 7.64 RMSE, while a known-prefix
  offset estimate was 37-39 and a fold-safe selector about 14.8
  ([comment 3463219](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3463219)).
- A CNN window likelihood did not beat point-GR likelihood; the reported match
  peak could sit about 200 ft from truth with AUC around 0.7. Replacing the
  likelihood with an oracle made the PF reach about 1 ft, isolating likelihood
  centering as the bottleneck
  ([comment 3465758](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3465758)).

**Safe conclusion.** Always measure the candidate oracle envelope and shuffled
controls before building a more elaborate ranker. A selector cannot fix absent
candidates, and same-window rank metrics are not evidence of test-time
selection skill.

**Disclosed public gate template, not outcome evidence.** I-aztec's tail-guard
notebook uses visible-prefix cuts at 50%, 65%, and 75%, minimum prefix length
140, and a 35-row pseudo-tail. Its aggregate candidate score is median RMSE
plus `0.1 * std`; its balanced gate requires minimum gain 0.55, RMSE at most
12, and consistency at least 0.50; correction alpha is capped at 0.40 and the
packaged correction is only 0.5-1%
([notebook](https://www.kaggle.com/code/iaztec/rogii-geosteering-with-a-tail-guarded-model-packag)).
No exact LB improvement is disclosed, and the candidate pool includes
same-ID/contact paths that can make train-copy pseudo-tails optimistic. Reuse
the thresholds only as a starting safety template after removing those paths.

Blacklions's public well-level GBDT gate is similarly useful as a contract,
not a result: activate only for grouped OOF gain at least 0.010 ft, P90
degradation at most 0.015, and median degradation at most 0.010; its balanced
transfer weight is 0.14 with per-row cap 0.30
([notebook](https://www.kaggle.com/code/blacklions/well-level-gbdt-gate)).
The notebook has no executed output cells, so the thresholds do not prove a
score gain.

### 4. A stronger decoder does not repair a weak observation model

**Reported result.** Matteo Niccoli's deterministic full-state DP/Viterbi
experiment used five stiffness configurations over about 400 TVT states in a
plus/minus 200-ft corridor. Viterbi-derived features ranked first by LightGBM
gain, and OOF improved from 14.806 to 14.346, but public LB moved only from
14.082 to 14.081. Fold variation increased from 0.44 to 0.87, and path-gap
confidence had near-zero gain
([topic 702919](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919)).

The same source records two useful engineering details:

- anchor-centered GR residuals were more informative than NCC-centered ones;
- z-scoring GR without rescaling the movement penalty made every DP path flat,
  because observation costs fell near 0.03 while movement costs remained
  4-35.

Niccoli later says three path finders over the same matching score produced
essentially the same result, again pointing to the score rather than the
decoder
([comment 3499048](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919#3499048)).

**Safe conclusion.** Do not spend the next iteration swapping beam search for
Viterbi. Change the reference signal or observation representation first.

### 5. Public whole-well neural code is a representation template, not score evidence

**Disclosed code.** Hengck23's MTP discussion publishes a 2-D input with five
channels: expanded typewell GR, expanded horizontal GR, their difference,
known horizontal TVT history expressed in the typewell grid, and a visibility
mask. One experiment used a 512-point horizontal window (compression 2) and a
64-point typewell window
([comment 3467727](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3467727)).
A later full-length experiment used compressed 384-by-768 windows plus
sin/cosine features for well dip and direction
([comment 3467823](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853#3467823)).

**Reported negative result.** Georgy Mamarin says his team's CNN, GRU, GBM, and
typewell cross-attention seq2seq models landed around 13-14 under honest
well-grouped CV, versus about 10 for a tuned multi-scale PF
([comment 3479226](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613#3479226)).

Two public architecture notebooks point the same way. A TCN with LayerNorm,
squeeze-excitation, typewell cross-attention, and a global Transformer reports
v6 OOF 13.80 and LB 14.07
([notebook](https://www.kaggle.com/code/medali1992/rogii-tcn-train-with-ddp-layernorm-se-atten)).
An SCA-U2Net reports five-fold GroupKFold 9.5521 and a title LB of 9.130
([notebook](https://www.kaggle.com/code/sangrampatil5150/lb-9-130-1dunet)).
Both downloaded notebooks have zero executed/output cells, so these are
author-stated numbers rather than self-verifying artifacts.

**Safe conclusion.** The cost-volume/SDF channels are worth reusing, but there
is no public evidence that copying one of these neural architectures will reach
the fives. Train a full-well residual/diversity model against the strong
frontier prediction, not a from-scratch TVT replacement.

### 6. Row-level DWT and disagreement are the cleanest cheap features

**Reported result and disclosed parameters.** Mitch Gansemer's original
feature notebook reports:

- an anchor-relative drift target improving OOF from 19.50 to 14.99;
- a 184-feature XGB/CatBoost/HGB NNLS ensemble at OOF 9.85 and public LB 8.905;
- about 0.08 ft additional OOF gain from row-level wavelet features; and
- the worst 30 of 773 wells dominating the remaining error.

The wavelet implementation uses `db4`, five decomposition levels,
reconstructed `A5`, and rolling RMS with window 16 over `D3`. Its matching
features use NCC half-widths 8/15/25, search radius 150, and weights
proportional to `exp(3 * score)`. Estimator disagreement is useful: NNLS gives
the strongest LGB component 0% weight while retaining a weaker HGB component
at 36%, demonstrating value from decorrelated errors
([notebook](https://www.kaggle.com/code/mitchgansemer/gr-features-outlier-detection-rogii-wellbore)).

The same source reports DTW worsening OOF from 10.008 to 10.07 and online
learning worsening the result. The useful wavelet construction is therefore
row-aligned bands, not a generic signal-decomposition claim.

**Code audit warning.** A separate DWT/EMD notebook has no score, summarizes
bands to constants per well, and appears to iterate `imfs[:, i]` even though
PyEMD returns `(n_imfs, n_samples)`, which selects time-position slices across
IMFs rather than IMF time series
([notebook](https://www.kaggle.com/code/gruubyhimself/rogii-signal-decomposition-dwt-emd-features)).
Another notebook labeled "DWT based" actually implements DTW and acknowledges
the title typo
([notebook](https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based)).

**Safe conclusion.** Add `A5`, `D3_RMS16`, and their local gradients as
rowwise features to the existing residual model. Do not add constant
whole-well wavelet summaries or copy the EMD axis handling.

### 7. Exact HMM smoothing is promising but currently unverified

**Prose-only reported result.** Souldrive's exact HMM notebook claims a
five-well pooled comparison of PF 4.90, second-order HMM 5.18, and their 50:50
blend 4.57. It also claims that adding a dip-rate Ornstein-Uhlenbeck state
improved a first-order random walk from 13.0 to 5.2, with posterior standard
deviation versus absolute error correlation as high as 0.92
([notebook](https://www.kaggle.com/code/souldrive/rogii-wellbore-geology-exact-hmm-smoother)).

Disclosed parameters are TVT grid step 0.35, 41 rate states over plus/minus
0.10, `sig_r=0.002`, `sig_p=0.02`, and momentum 0.998. However, the downloaded
notebook contains zero executed code cells and zero output cells. There is no
reproducible artifact for the headline numbers.

**Safe conclusion.** This is worth one bounded, honest multi-cut OOF
reproduction as a decorrelated challenger and uncertainty feature. It is not
evidence to replace the frontier model.

### 8. Notebook titles and acronyms are not method evidence

**Locally verified code identity.** Converted source from all three notebooks
below is byte-identical:

- [ROGII | Luck is all you need](https://www.kaggle.com/code/opencv411/rogii-luck-is-all-you-need)
- [New Strategy [Score: 6.213]](https://www.kaggle.com/code/leonidzaporozhets/new-strategy-score-6-213)
- [ROGII best score](https://www.kaggle.com/code/farhanabidtech786/rogii-best-score-wellbore-geology-prediction)

The 6.213 title is therefore not a disclosed new architecture; it is evidence
that execution/rerun outcome and title can differ for the same code.

**Disclosed acronym meaning.** In the public `det-mha180sep3` source, `MHA`
means **midpoint hedge alpha 1.8**, not multi-head attention. The notebook uses
LGBM/CatBoost delta models, grouped Ridge, a 128-seed likelihood PF, beam
correlation, robust projection, and a bimodal-datum midpoint hedge. It says a
parent alpha-1.6 run scored 6.927 and predicts 6.910 for alpha 1.8
([notebook](https://www.kaggle.com/code/canqiang/rogii-det-mha180sep3)).

Several recent frontier-titled notebooks explicitly call themselves copies,
anchor internally to a weaker score, or emit tiny single-well challengers
rather than proving their title:
[my0705](https://www.kaggle.com/code/my0705/rogii-stacked-ensemble-highscoring-6-391),
[Raunak Dey](https://www.kaggle.com/code/raunakdey07/rogii-ultra-sub-6-rmse),
[prvsiyan](https://www.kaggle.com/code/prvsiyan/rogii-public-score-frontier-ii-visuals),
and
[Blacklions](https://www.kaggle.com/code/blacklions/rogii-wellbore-geology-prediction-final-hierarch).
Treat code, executed outputs, and a direct score statement separately.

### 9. Hard-well risk and autoregressive drift need separate treatment

**Reported recurring failures.** Multiple participants independently list
`86454a6f`, `4c2208f5`, `389ae58f`, `fb03ae90`, and `91db7070` among their
largest-error wells
([topic 723815](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815)).
One author says a segmented sequential method can make one early mistake and
ruin the remaining well
([comment 3494403](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815#3494403)).
Another could classify large-error wells but not the signed direction of the
error
([comment 3494144](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815#3494144)).

**Reported shared blind spot.** OpPrime describes six independently trained
GRU/CNN/SDF-family models agreeing tightly on the same wrong answer over an
approximately 2,600-ft segment, eventually missing truth by roughly 90 ft
([topic 726834](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726834)).

**Safe conclusion.** A risk model should decide how much to trust or shrink a
branch; it should not be expected to infer the signed correction. Model
agreement is not a sufficient safety signal when all models share the same
inputs and blind spot.

### 10. Strict hidden-like validation must close two concrete leaks

**Disclosed implementation audit.** In a leak-free 60-well holdout, souldrive
reports that:

1. a contact-lookup family read a train-only file and could reconstruct a train
   well to thousandths of a foot even though the branch does not exist for a
   hidden well; and
2. formation/dense imputers invoked with `self_wid=None` let a train well see
   its own tops; passing the held-out well ID correctly excludes it.

The same holdout found top-two aggregate candidate-score gaps below 0.05 ft on
62% of wells and below 0.10 ft on 67%. Mixing tied candidates did not provide a
stable pooled gain as the tie band widened; the shipped gate instead required a
margin above 0.10
([comment 3504568](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477#3504568)).

This directly supports the project's recent own-well exclusion fix. Every new
lateral bank, surface, or formation experiment must preserve it.

### 11. Execution variance sets a practical leaderboard threshold

**Reported rerun measurements.** Four unchanged-rerun series gave public-score
spreads of 0.034-0.089 ft; the two four-run series had standard deviations
0.027 and 0.037 ft
([comment 3504227](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477#3504227)).
A fifth fixed-scoring-path pipeline had standard deviation 0.0201 over five
runs. The author notes that fixed random seeds did not eliminate all spread,
suggesting parallel reductions, GPU learners, or brittle argmin decisions can
amplify numeric variation
([comment 3504568](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477#3504568)).

**Operational rule.** On the current base, treat a one-off public delta smaller
than roughly 0.06 ft (about two standard deviations of the cited order-0.03
band) as inconclusive unless repeated or backed by a clear multi-fold OOF gain.

The public claim that multiplying PF GR noise scale `gs` by about 1.3 improved
a notebook does not disclose the score or a whole-well validation result
([topic 728712](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712)).
It is therefore a low-priority sensitivity check, not reproducible frontier
evidence.

## Ranked implementation experiments

| Rank | Experiment | Expected upside | Cost | Promotion rule |
|---:|---|---|---|---|
| 1 | Same-group lateral-reference bank | High | Medium | Real GR must beat shuffled/no-GR controls and improve pooled LOOW RMSE |
| 2 | Row-level `db4` bands plus disagreement | Low-moderate but cheap | Low | Reproduce the reported ~0.08-ft OOF gain under the current folds |
| 3 | Candidate coverage and nested selector audit | High information value | Low-medium | Do not train a selector until the oracle envelope has adequate coverage |
| 4 | Spatial level-versus-drift ablation | Moderate or decisive negative | Low | Keep only if excluded-well drift adds at least 0.05 ft pooled OOF |
| 5 | Full-well residual cost-volume model | High but uncertain | High | Must beat the same frontier base on identical whole-well folds and seeds |
| 6 | Exact HMM/PF uncertainty challenger | Moderate, weak evidence | Medium-high | Reproduce on all outer folds; five-well prose claims are insufficient |
| 7 | Hard-well risk shrinker | Moderate tail-risk reduction | Medium | Improve pooled RMSE and worst-decile error without signed-error guessing |
| 8 | Raw/median/phase-invariant GR channels | Low-moderate | Medium | Keep only with end-to-end OOF gain; no datum-only promotion |
| 9 | `gs * 1.3` or similar PF micro-tuning | Low | Low | Require repeated runs or delta larger than the measured seed band |

### Experiment 1 — leave-one-well-out lateral-reference bank

#### Reference construction

1. Reuse the existing five-fold whole-well assignment. For outer fold `f`,
   construct the bank from wells **outside fold `f` only**. Excluding the
   entire validation fold is stricter than excluding just the target well and
   prevents two validation wells from reading one another's labels.
2. Key references by the existing master/typewell fingerprint. Keep a global
   fallback bank only as a separately measured ablation.
3. Store each source lateral as
   `(well_id, TVT, GR, X, Y, azimuth, MD)`, with raw GR and independently
   median-smoothed GR. Resample GR to a common TVT grid for fast window lookup,
   but preserve the source row mapping for diagnostics.
4. Preselect at most eight distinct source wells per target using group
   identity first, then spatial distance and azimuth compatibility. Never let
   dense rows from one source dominate the vote. If no same-group source
   remains after fold exclusion, emit missing/support-zero features and retain
   the base path rather than silently using the target or an unrelated group.
5. Fit any gain/offset normalization from the target's visible prefix only.
   Source labels and target post-start GR are legal; target post-start TVT is
   not.

#### Matching and features

6. Match target windows at multiple physical scales against:
   - its own visible prefix, whose TVT is known; and
   - other laterals in the same group.
7. Allow both increasing and decreasing TVT orientation.
8. Center the search around the existing frontier path with a bounded corridor;
   do not globally search the full master curve.
9. Aggregate by distinct reference well, not by raw point count. For each row
   retain `lateral_tvt_best`, robust consensus TVT, consensus MAD, best score,
   top-two margin/separation, support-well count, distance, orientation, and
   residual to the frontier prediction.
10. Compute matches at a fixed stride such as 8 or 16 rows with windows
    32/64/128, then interpolate diagnostics back to the original row grid.
    This avoids a naive all-pairs operation over 3.78 million OOF rows.
11. First add these as features or a small guarded residual. Do not hard
   overwrite the frontier trajectory.

#### Exact 3.78-million-row OOF test

Write one compact `float32` feature matrix aligned byte-for-byte with the
existing 3.78-million hidden-tail OOF rows. Eight numeric channels occupy about
121 MB, so per-well chunks plus a memory-mapped final array are sufficient.

For each outer fold:

1. generate reference features using only the other four folds;
2. fit a tiny ridge/one-depth tree residual or a scalar shrinkage weight on
   cross-fitted feature rows from the other four folds;
3. predict fold `f` once; and
4. accumulate squared error across every original tail row before taking one
   RMSE.

Report the exact base and candidate pooled RMSE, all five fold RMSEs,
well-bootstrap confidence interval, worst-decile well RMSE, and error by
distance from Prediction Start. Also report the oracle candidate envelope and
the percentage of wells for which a lateral candidate beats the base.

Run the entire feature path three times with identical folds and candidates:

- real target GR;
- target GR shuffled within each well after preserving its missingness; and
- no lateral GR score, retaining only geometry/group priors.

Promotion requires a real-GR pooled gain of at least 0.05 ft, the same direction
in at least four of five folds, and a clear separation from both negative
controls. If real GR cannot beat shuffled/no-GR, retain only the bank's
uncertainty diagnostics or stop the branch.

Use azimuth/dip primarily to canonicalize orientation or stratify this matcher,
not as an unconditional scalar feature block. A public notebook adding drilling
sin/cos, formation-dip x/y/magnitude, and drill/dip cosine was mostly flat or
worse in fresh GroupKFold runs, with CatBoost variants up to 0.081 ft worse
([notebook](https://www.kaggle.com/code/qujiahui/9-608-rogii-azimuth-dip-features)).

### Experiment 2 — row-level wavelet and disagreement ablation

Reconstruct `db4` `A5` and `D3` at the original row grid, then add:

- `A5`, `D3`, rolling `RMS(D3, 16)`;
- first differences and local robust scale of those channels; and
- disagreement/spread across the existing PF, beam, geometry, and tree
  predictions.

Use the existing whole-well folds and add one feature family at a time. This is
cheap enough that a reproduced gain smaller than the reported 0.08 ft can
still be useful, but the same-fold delta should exceed run-to-run training
variation. Do not summarize the decomposition to one repeated constant per
well.

### Experiment 3 — audit candidate coverage before selection

For every whole-well holdout, record:

- baseline RMSE;
- each candidate's RMSE;
- oracle-best pooled RMSE;
- percentage of wells with at least one candidate below 4, 6, and 10 ft;
- percentage with any candidate better than the base;
- top-two score margin and candidate disagreement; and
- the same table after shuffling GR.

Only then evaluate a selector. Selector training and all candidate generators
must exclude the outer held-out well. Prefix pseudo-tails may tune a target's
inference-time trust, but must not leak the real hidden tail. Prefer a continuous
shrinkage weight or "retain base" decision over hard argmin when margins are
small.

### Experiment 4 — falsify or constrain spatial tail transfer

Work in the formation-surface coordinate already used locally,
`surface = TVT + Z` (the sign follows the dataset's Z convention). Decompose
each held-out tail into:

- surface level at Prediction Start; and
- surface drift away from that anchor.

Measure excluded-neighbor correlations separately by master group, spatial
distance, azimuth difference, and tail horizon. Compare:

1. current full surface transfer;
2. a locally planar regional surface anchored exactly at the visible prefix;
3. the same surface-drift path shrunk by `lambda` in `{0, .02, .05, .10}`; and
4. use of surface diagnostics only, with no direct correction.

The public evidence predicts that intercept/low-frequency structure will be
more stable than drift. If the corrected all-well benchmark does not beat the
base clearly, stop investing in larger spatial KNNs.

### Experiment 5 — a complete-well residual model

Use one well as one sample, but train it to predict the residual to the existing
frontier path rather than absolute TVT. Suggested inputs:

- complete horizontal MD/Z/GR and geometry;
- complete typewell TVT/GR;
- visible TVT with a mask;
- frontier candidate paths and their disagreement;
- the bounded typewell and lateral-reference cost volumes; and
- distance from the last visible anchor.

Predict the whole residual curve in one pass or decode one global path; do not
chain 512-row forecasts whose anchors are prior predictions. Randomize the
visible-prefix cut during training and optimize the exact pooled row loss. A
small 1-D TCN/U-Net or 2-D cost-volume encoder is enough for the first
falsification test. This architecture is a project synthesis, not Tucker's
disclosed model.

### Experiment 6 — exact HMM as a decorrelated challenger

Reimplement only the disclosed state model first: TVT step 0.35, 41 dip-rate
states over plus/minus 0.10, `sig_r=0.002`, `sig_p=0.02`, momentum 0.998.
Run it on every outer held-out well and compare:

1. frontier base;
2. HMM posterior mean;
3. a fixed 50:50 blend;
4. a fold-fitted scalar blend; and
5. frontier plus HMM posterior spread as a gate feature.

Do not tune against the five wells used in the public claim. Promote only if
the all-well pooled result and multiple prefix cuts agree.

### Experiment 7 — predict risk, not signed correction

Train a fold-safe classifier for `candidate likely worse than base` or
`well likely in worst error decile` using only visible-prefix and candidate
diagnostics. Use its output to reduce correction amplitude, cap movement, or
increase ensemble averaging. Do not ask this model to predict the sign of a
90-ft miss; public participants report that risk was detectable before signed
direction was.

## Immediate project-specific decisions

- Keep the recent own-well exclusion in the neighbor index. It closes exactly
  the failure mode documented in the hidden-like holdout.
- Reuse the existing typewell fingerprint as the key for Experiment 1; this
  makes the pseudo-typewell idea testable without relying on PNG OCR at
  inference.
- Keep surface transfer as a guarded candidate. Public evidence argues against
  promoting its full tail shape without a clean excluded-well gain.
- Do not enable the existing prefix-weight path merely because it fits visible
  pseudo-tails. It needs the nested candidate/selector audit above.
- Freeze decoder rewrites until a new observation channel beats shuffled-GR
  controls.

## Local follow-up: lateral-reference experiment result

The project implemented the rank-1 experiment after this research pass and ran it on
all 3,783,989 hidden-tail OOF rows across 773 wells and 54 inferred typewell groups.
Each validation fold used reference windows from the other four folds only. The
matcher used nine-point, 128-row GR signatures at stride 16, forward and reverse
orientation, an eight-distinct-well spatial shortlist, a 50-ft TVT corridor, and
real-GR, within-well-shuffled-GR, and no-GR controls.

Against the postprocessed model-package OOF base:

| Candidate | Cross-validated RMSE | Gain vs base | Improved folds |
|---|---:|---:|---:|
| Base | 10.670223 | — | — |
| Cross-lateral real GR | 10.663820 | 0.006403 | 3/5 |
| Cross-lateral shuffled GR, same real-GR gate | 10.666114 | 0.004109 | 3/5 |
| No-GR control, same gate | 10.670223 | 0.000000 | 0/5 |
| Same-well visible-prefix real GR | 10.672095 | -0.001872 | 0/5 |

This **fails** the predeclared promotion rule of at least 0.05 ft, four of five folds,
and clear separation from both controls. The self-prefix candidate received an
in-sample optimal weight of zero. Therefore no Kaggle submission should be built from
this implementation. The experiment is useful negative evidence: normalized local
GR-window similarity carries a very small real signal, but most apparent gain is
shared with shuffled matching and is too weak for the current frontier.

Implementation and exact reports:

- [`src/rogii/lateral_reference.py`](../src/rogii/lateral_reference.py)
- [`tools/evaluate_lateral_reference.py`](../tools/evaluate_lateral_reference.py)
- [`data/derived/lateral_reference_oof_report.json`](../data/derived/lateral_reference_oof_report.json)
- [`data/derived/lateral_reference_oof_postprocessed_report.json`](../data/derived/lateral_reference_oof_postprocessed_report.json)

## Code-competition submission workflow

The HTTP 400 from uploading a local CSV is consistent with using file-submission
syntax on a code competition. The current Kaggle CLI documents the required
notebook form:

```bash
rtk kaggle competitions submit rogii-wellbore-geology-prediction \
  -k chinnaphatkhun/<kernel-slug> \
  -v <saved-version-number> \
  -f submission.csv \
  -m "<message>"
```

Here `-f` is the **name of the output file produced by the saved notebook**,
`-k` is the notebook reference, and `-v` is the saved version number. A local
`submission.csv` path by itself is not the code-competition handoff.

## Primary-source index

- [Host: lateral GR self-correlation and nearby-well dip](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825)
- [Host: pseudo-typewells from previously interpreted laterals](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449#3455674)
- [Spatial `dZ`/`dTVT` clustering and spline/KNN](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711308)
- [Formation columns collapse to one base surface](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167)
- [MTP/CNN heatmap code, selector leakage, and shuffled-GR controls](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853)
- [DP/Viterbi OOF-versus-LB failure](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919)
- [PF generator/scorer audit and honest neural baselines](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613)
- [Mitch Gansemer: row-level DWT, NCC, and ensemble disagreement](https://www.kaggle.com/code/mitchgansemer/gr-features-outlier-detection-rogii-wellbore)
- [Souldrive: unexecuted exact-HMM proposal](https://www.kaggle.com/code/souldrive/rogii-wellbore-geology-exact-hmm-smoother)
- [Hengck23: public 2-D MTP/CNN representation](https://www.kaggle.com/code/hengck23/cnn-mtp-example)
- [I-aztec: pseudo-tail guard implementation](https://www.kaggle.com/code/iaztec/rogii-geosteering-with-a-tail-guarded-model-packag)
- [Blacklions: well-level GBDT gate contract](https://www.kaggle.com/code/blacklions/well-level-gbdt-gate)
- [Qu Jiahui: azimuth/dip scalar-feature ablation](https://www.kaggle.com/code/qujiahui/9-608-rogii-azimuth-dip-features)
- [Recurring worst wells and sequential drift](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723815)
- [Six architectures sharing one blind spot](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726834)
- [Rerun variance and leak-free hidden-like holdout](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728477)
- [Unquantified public `gs * 1.3` observation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712)
- [Official data warning that visible test wells are replaced](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
