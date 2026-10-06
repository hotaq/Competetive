# ROGII stratigraphic-alignment research

## Executive verdict

The strongest high-upside path is not a single “magic” architecture. It is a guarded stack:

1. reproduce the competition metric with whole-well folds and pooled row-level RMSE;
2. treat a complete well—not an isolated row—as the model sample;
3. build a multi-scale GR-to-typewell cost volume after prefix-only GR calibration;
4. predict only bounded, heel-anchored relative paths and retain uncertainty over plausible paths;
5. use visible-prefix pseudo-tails to select or reject candidate trajectories; and
6. accept denoising, posterior hedges, or late corrections only when they improve end-to-end out-of-fold RMSE.

This conclusion follows the official task definition, the host’s description of the domain operation, and two public first-party implementations. The official data page says the typewell GR is supplied “for correlation,” while host Igor Kuvaev describes the domain task as manually projecting GR onto TVT ([data](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data), [host comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416#3468837)).

Tucker Arrants publicly reported approximately 5 ft pooled per-row cross-validation without cross-well, spatial, or external-top information, and clarified that the entire well was one model sample rather than a row-tabular sample. He did **not** publish enough method detail to identify this as a GR matcher, particle filter, CNN, or any other architecture ([score claim](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478241), [scope clarification](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430), [whole-well/GR clarification](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465), [non-tabular remark](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3486609)). His result is therefore a credible target, not a reproducible recipe. The complete-well matcher proposed below is a synthesis from public evidence, **not** a reconstruction of Tucker’s undisclosed model.

## What the task actually scores

The competition evaluates one RMSE pooled over all prediction rows, rather than averaging a separate RMSE for each well ([official evaluation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation)). Validation must therefore accumulate squared error over all held-out tail rows before taking the square root. Per-well metrics remain useful diagnostics, but optimizing their unweighted mean changes the objective.

Each example pairs a horizontal well with a typewell. The horizontal-well `TVT_input` is known in the prefix and missing in the evaluation interval, while the typewell GR is explicitly provided for correlation ([official data description](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)). The visible test files are examples that the host says will be replaced by roughly 200 hidden wells, so visible train/test identity is not evidence that the hidden test repeats training wells ([official data description](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)).

### Evidence map

| Question | Primary-source evidence | Safe conclusion |
|---|---|---|
| What is scored? | [Official evaluation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation) | Pool all held-out rows for RMSE. |
| What is the intended geological operation? | [Official data page](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data), [Igor Kuvaev](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416#3468837) | GR-to-typewell correlation is central to projecting TVT. |
| Is approximately 5 ft CV plausible using only same-well inputs? | [Tucker score](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478241), [Tucker metric/scope](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430), [whole-well clarification](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465) | Yes as a reported target; the implementation is undisclosed. |
| Is there a public row-level GR matcher? | [Bernardus notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild) | Yes: multi-scale normalized cross-correlation features inside a grouped boosted-tree pipeline. |
| Is bounded, uncertainty-aware alignment public? | [Pilkwang notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering), [author explanation](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#8-tie-wells-use-posterior-mean-not-overconfident-decisions) | Yes, including prefix-only calibration, a bounded shift scan, and posterior averaging. |
| Why not globally register the two GR curves? | [Georgy Mamarin notebook](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever), [discussion](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037) | Repetitive GR yields confident wrong-depth matches; anchor at the heel and search locally in sequence. |
| Does FFT denoising have an end-to-end RMSE result? | [Rotation/noise thread](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367), [Georgy’s diagnostic](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3482433) | No. The reported improvement is a datum-localization diagnostic using a rolling-median proxy, not FFT and not competition RMSE. |
| Why average ambiguous modes? | [Souldrive notebook](https://www.kaggle.com/code/souldrive/the-15-ft-datum-why-honest-cv-bottoms-out), [Georgy diagnostic](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3481297) | Under squared error, a calibrated posterior mean can beat a forced hard mode on ambiguous cases, but only when ambiguity is real. |

## Tucker Arrants: useful benchmark, unknown method

Tucker wrote that a single model produced “CV ~5ft” and might go below 4 ft ([original comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478241)). He later reported 5.4 for one model and 5.0 for an equal-weight ensemble ([discussion 723647](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723647)). He clarified that the score was pooled per-row RMSE and used only per-well data, with no cross-well/spatial structure or external tops ([clarification](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430)).

The most informative follow-up says GR is the signal, the entire well is one model sample rather than one row, and the best and worst folds were approximately 4.5 and 5.3. Tucker also says that roughly 25 wells exceeded 12 ft error and were not specially handled ([discussion 726465](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465)). A later report gives 4.98 CV from five folds and five seeds versus 5.7 on the public leaderboard, while warning that CV/LB correlation deteriorates below about 6 and that training/GPU nondeterminism matters ([discussion 727570](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727570)). In another comment he described the models only as non-tabular ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3486609)).

Those statements disclose the sample grain, input scope, metric, and rough fold behavior. They do **not** disclose feature construction, target transformation, architecture, loss, preprocessing, post-processing, or alignment mechanics. Reviews of the related public threads likewise do not turn the result into an implementation ([discussion 712037](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037), [discussion 726465](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465), [discussion 727570](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727570)). No Tucker-authored public ROGII notebook was identified in this research pass. Consequently:

- do not label Tucker’s method as normalized cross-correlation, particle filtering, sequence matching, or a neural network;
- do not attribute Bernardus’s or Pilkwang’s public mechanics to Tucker; and
- use approximately 5 ft only as a reported validation target with Tucker’s stated scope.

## Why the alignment must be heel-anchored and bounded

[Georgy Mamarin’s original notebook](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever) provides the clearest public falsification test for naive GR registration. Searching for one absolute global shift confidently locks onto repeated motifs at the wrong depth: the notebook reports roughly 31 ft RMSE for that approach, versus roughly 17 ft for flat carry. The correct abstraction is relative, sequential matching initialized by the known heel and constrained to a local corridor—not unrestricted global curve registration.

The same diagnostic separates *matching capacity* from *legal calibration*:

- a shift scan over `np.arange(-20, 20.01, 0.5)` localizes the datum within 2 ft for about 82% of wells when supplied oracle GR calibration;
- a weak legal flat-surface calibration succeeds on only about 8%, below a shuffled control near 17%;
- legal affine gain/offset calibration fitted on the known heel recovers roughly 80%; and
- a rolling-median denoising proxy nudges that diagnostic to roughly 84%.

These figures are localization diagnostics, not competition RMSE, but they identify the bottleneck: the matcher can work if heel calibration exposes a comparable GR signal. They also make unrestricted shift searches an especially poor default ([notebook](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever), [author discussion](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037)).

Georgy’s particle-filter experiment is also more valuable as an uncertainty sensor than as a direct TVT predictor: posterior spread correlated with realized error at about +0.23. Blind DTW washed out local evidence and could invent movement on flat wells, worsening those cases by about 1.2 ft ([notebook](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever)). A separate original-author pattern-matching thread reports attractive candidate/oracle paths around 4.5–6 ft while emphasizing that ambiguous cost maps make *selection* the hard part; strong paths can still ruin flat wells when applied indiscriminately ([discussion 721549](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/721549)).

The modeling implication is concrete: expose the local loss landscape, score margin, alternative modes, and posterior spread to the final model or selector. Do not replace the base trajectory merely because one candidate has the smallest raw matching cost.

## Reproducible public matcher: Bernardus

[Bernardus’s public rebuild notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild) is the clearest public implementation of a per-row GR/typewell matcher. Its title advertises LB 7.295; it must not be conflated with Tucker’s approximately 5 ft CV.

The notebook:

- creates one feature row for each hidden-tail prediction row;
- validates with `GroupKFold` by well and pooled RMSE;
- smooths GR with a rolling mean of width 5;
- matches at half-window sizes 8, 15, and 25 with stride 3;
- computes rowwise normalized cross-correlation against candidate typewell windows;
- aggregates candidate locations with softmax-like weights proportional to `exp(3 * score)`;
- combines the matcher outputs with particle-filter, beam, formation, and geometry signals; and
- fits boosted-tree residual models.

All parameter values and feature construction above are directly inspectable in the [original notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild).

The same notebook applies per-well Savitzky–Golay smoothing with window 61 and polynomial order 3, while noting only about 0.01 ft effect. It also fits a robust low-order projection of `TVT + Z` against normalized MD and downweights branch outliers. Same-well train/test overrides are guarded: at least 50 known-prefix rows, prefix RMSE no greater than 1 ft, MD interpolation, and in-range-only use ([notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild)).

### Minimal matcher to reproduce first

Start with the notebook’s feature-only core before copying its entire ensemble:

1. smooth horizontal and typewell GR identically;
2. for each tail row and each window scale, score candidate typewell windows by normalized cross-correlation;
3. keep the best score, candidate TVT, score margin, weighted mean TVT, and weighted spread;
4. pass these as features to the existing residual learner; and
5. evaluate only in held-out wells with pooled tail-row RMSE.

The window sizes, stride, and exponential weighting should initially match the [public notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild), then be changed only through whole-well out-of-fold ablations.

### A useful independent baseline and its negative results

[Mitch Gansemer’s original notebook](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore) independently supports several building blocks and records valuable failures:

- predicting drift from the last known anchor instead of absolute TVT improved OOF RMSE from 19.50 to 14.99;
- normalized cross-correlation at windows 8, 15, and 25 was useful;
- particle-filter and beam candidates helped more as features than as standalone predictions;
- the final public pipeline used whole-well grouped validation and Savitzky–Golay smoothing with window 17/order 3;
- the best reported result was 9.85 OOF and 8.905 public LB;
- blind DTW slightly worsened OOF;
- BiLSTM/TCN experiments remained around 14.6 OOF; and
- coordinate-overlap tricks did not generalize.

These are reproducible baseline and negative-result numbers from the [original notebook](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore), not frontier claims. Their main value is narrowing the search: relative-to-anchor targets and matching-derived features are useful; generic sequence models, blind DTW, and overlap shortcuts are not sufficient by themselves.

## Bounded and uncertainty-aware alignment: Pilkwang

[Pilkwang Kim’s public notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering) supplies a second reproducible route whose valuable contribution is conservative gating rather than an unconditional override.

### Active alignment mechanics

The implementation fits horizontal-to-typewell GR affine calibration using only the visible heel. Candidate shifts use a robust score with normalized residuals clipped to `[-6, 6]`. The bounded scan covers ±20 ft in 0.5-ft increments. A possible second mode is sought 10–20 ft away and must be at least 8 ft separated; a near tie is within 15% of the best score ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering), [author explanation](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#4-grtypewell-matching-barcode-not-label)).

Prefix reliability prevents overconfident shifts. The scan probability is shrunk toward ignorance:

```text
p_eff = trust * p_scan + (1 - trust) * 0.5
```

For a credible two-mode match, the output is the posterior mean of the two branches, not a hard winner ([author explanation](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#8-tie-wells-use-posterior-mean-not-overconfident-decisions)).

The particle-filter GR noise scale is clipped to 10–60, the state/path is bounded relative to the typewell, and the beam search moves locally with penalties. Visible-prefix and residual corrections are also bounded and gated; when guards fail, the prior trajectory is retained ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering), [PF/residual description](https://pilkwangkim.github.io/posts/ROGII-Target-Free-Stratigraphic-Alignment-for-TVT/#10-pf-residual-gbdt)).

Pilkwang’s related [dual-track notebook](https://www.kaggle.com/code/pilkwang/rogii-dual-track-prefix-calibrated-geosteering) turns the known prefix into a local backtest. It masks the prefix at fractions 0.50, 0.65, and 0.75, then evaluates candidate trajectories on the resulting pseudo-tail. Candidate families include flat/slope continuations, PF/beam paths, a learned path, and projected/smoothed paths. Disagreement gates, capped corrections, and fallback to the preceding valid path prevent one noisy selector decision from replacing a stable prior. The [working-note notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering) follows the same target-free principle: calibration and model selection use only information that will be visible at inference.

This is one of the most transferable ideas in the public work. Candidate generation can be generous; candidate *acceptance* should be conservative and trained on many prefix pseudo-tails.

### Implemented does not mean enabled

The published main/default profiles set:

```text
RUN_BIMODAL_DETECTOR = False
RUN_GR_FFT_DENOISE = False
RUN_SEQ_MATCHER = False
```

Thus the bimodal override, FFT denoiser, and sequence matcher are implemented diagnostics or optional experiments, **not active evidence of submission gain** ([original notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)). Any local port should preserve this distinction and earn each component through out-of-fold ablation.

## Denoising: plausible signal, insufficient score evidence

Shrey identified a periodic GR pattern associated with sensor rotation ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3474176)). Tabish confirmed frequency-domain structure and posted FFT code ([confirmation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3474223), [code](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3475047)); MY0705 suggested exploiting the periodic rotation for noise reduction ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3474890)).

The only quantified result in that thread is narrower: Georgy reports that a rolling-median low-pass proxy raised a datum-localization diagnostic from roughly 80% to roughly 84%, and explicitly says he did not run a real FFT notch ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3482433)). This is neither a four-point RMSE improvement nor evidence that an FFT notch improves the final model.

Recommended experiment:

- compare raw GR, rolling-median/low-pass GR, and FFT-notch GR as three separate inputs;
- rerun the entire matcher and predictor for each input;
- measure pooled hidden-tail out-of-fold RMSE across identical whole-well folds and seeds; and
- retain a filter only if its end-to-end improvement exceeds run-to-run uncertainty.

The motivation and candidate FFT construction come from the [original discussion](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367); the requirement for end-to-end validation follows because that source does not report competition RMSE.

## Ambiguous geology and the posterior-mean hedge

Souldrive’s original [15-ft datum notebook](https://www.kaggle.com/code/souldrive/the-15-ft-datum-why-honest-cv-bottoms-out) and [Eagle Ford analysis](https://www.kaggle.com/code/souldrive/decoding-eagle-ford-why-some-wells-are-hard) describe repeated bedding bundles roughly 10–20 ft apart and a resulting approximately ±15-ft datum ambiguity. About 30% of rows in that analysis had a strong competing peak. Hard top-1 selection was poor, while a soft expectation was directionally better. Under squared-error loss, a calibrated 50/50 posterior over two locations is minimized by their midpoint, not by arbitrarily choosing either mode; more generally, use the probability-weighted mean of the branches.

Georgy independently reports that posterior averaging reduced the worst-decile ambiguous-tail datum error from about 5.8 ft to about 5.1 ft and beat a fixed midpoint ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3481297)). However, hard commitment still won over all wells in that train-only diagnostic—roughly 5.2 versus 5.7—because hedging easy, unimodal wells adds bias. These are datum diagnostics, not overall competition RMSE. The correct lesson is therefore *gated* posterior averaging, with ambiguity thresholds and temperature learned out of fold, not universal smoothing.

A safe implementation is:

1. obtain a bounded candidate-shift loss curve from visible-prefix GR;
2. require two separated local minima and a near tie before declaring ambiguity;
3. convert the cost gap to probability using an out-of-fold calibrated temperature;
4. shrink those weights toward 0.5 when prefix trust is low; and
5. output the branch-weighted TVT mean while exposing mode separation, margin, entropy, and trust as model features.

Pilkwang’s exact bounded scan and trust shrinkage provide a public starting point ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering), [posterior section](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#8-tie-wells-use-posterior-mean-not-overconfident-decisions)).

## Highest-upside model synthesis

The public evidence supports a complete-well, non-tabular matcher as the next serious model. This architecture is an inference from Tucker’s disclosed constraints plus the reproducible alignment diagnostics above; it is **not** Tucker’s disclosed implementation.

### Inputs and representation

For one well, provide:

- the complete horizontal sequence: MD, Z, raw GR, optionally X/Y-derived geometric features;
- the complete paired typewell sequence: TVT and raw GR;
- the known horizontal `TVT_input` prefix plus an explicit visibility mask; and
- normalized distance from the last known anchor.

Fit only an affine GR gain/offset on the known heel. Construct a bounded typewell-candidate corridor around the heel-anchored prior, then compute a multi-scale cost volume using both raw and independently denoised GR. Include normalized correlation, robust clipped residual cost, and local cost differences. A denoised channel is an input ablation, not a substitute for the raw curve.

### Output and constraints

Predict either:

- logits over candidate relative TVT positions at each horizontal step; or
- logits over a compact library of monotone/bounded candidate paths.

Decode a smooth path with bounded local movement and penalties for implausible jumps. Keep the posterior, its entropy, the top-two mode gap/separation, and candidate disagreement. Train on simulated prefix/tail masks and score the exact pooled row RMSE. The relative target should be anchored at the last visible TVT rather than learning absolute TVT from scratch, consistent with the improvement in [Mitch Gansemer’s notebook](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore).

### Selector and ensemble

Generate several candidates: flat/slope, the existing tree baseline, bounded NCC path, PF/beam path, and the learned cost-volume path. On each training well, create pseudo-tails at multiple prefix cuts and train a selector to predict candidate error or regret from prefix-only diagnostics. It should be allowed to retain the base path. Apply posterior-mean hedging only when an out-of-fold ambiguity detector fires. Finally, average whole-well model predictions across folds and seeds; Tucker’s own 5.4-to-5.0 equal-weight ensemble report establishes a useful expectation for variance reduction ([discussion 723647](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723647)).

## Prioritized experiment plan

### P0 — repair validation before tuning alignment

- Mask known wells at their real prefix/tail boundary, train on other complete wells, and score only the simulated hidden-tail rows.
- Use whole-well `GroupKFold`.
- Accumulate pooled squared error over every held-out tail row, then take one square root.
- Report per-well tail RMSE, worst-decile RMSE, error by tail distance, and datum error only as diagnostics.

The pooled objective comes from the [official evaluation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation); grouped validation is the public pattern in [Bernardus’s notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild).

### P1 — add multi-scale NCC features

Reproduce rolling width 5, half-windows 8/15/25, stride 3, rowwise normalized cross-correlation, and `exp(3 * score)` weighting from the [public rebuild](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild). First add the outputs as features rather than overriding the trajectory. Ablate each scale and record pooled out-of-fold change.

### P2 — prefix-only calibration and bounded shift uncertainty

Fit GR gain/offset on the visible prefix only. Scan ±20 ft at 0.5-ft resolution with clipped residuals, retain score margin/spread, and use Pilkwang’s trust shrinkage ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)). Compare:

- no correction;
- hard best shift;
- a trust-shrunk shift;
- posterior mean of two credible modes; and
- shift-landscape features with no direct override.

### P3 — independent denoising ablations

Test rolling-median and FFT-notch paths separately. Do not enable either from the datum-localization result alone because the source explicitly tested a rolling-median proxy, not final RMSE or a real FFT notch ([Georgy](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3482433)).

### P4 — guard all late corrections

Use a pseudo-tail carved from the visible prefix to estimate whether a proposed correction helps before applying it to the real tail. Require minimum prefix support and cap maximum movement. Bernardus’s same-well override already demonstrates minimum-row, prefix-RMSE, interpolation-range guards ([notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild)); Pilkwang’s public implementation preserves the prior when correction guards fail ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)).

### P5 — train the complete-well cost-volume model

Once P0–P4 establish trustworthy components, replace isolated-row prediction with the complete-well formulation above. Start with a small 1-D encoder over the horizontal curve, a second encoder over the typewell, and their bounded pairwise cost volume. Predict a posterior over relative TVT candidates, decode under smoothness constraints, and ensemble folds/seeds. Compare it with the feature baseline under identical masks and pooled OOF scoring. This stage is motivated by Tucker’s verified whole-well/non-tabular sample grain, but the architecture itself is the synthesis in this report ([discussion 726465](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465)).

### P6 — treat exact overlap as optional only

Keep same-well interpolation as a guarded optional path, never as the core strategy. The official page says visible test examples are replaced by hidden wells ([data](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)), and Tucker explicitly scoped his result to per-well information rather than relying on cross-well structure ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430)).

## Anti-patterns

- **Mean of well RMSEs as the main score:** it does not match the [official pooled-row metric](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation).
- **Random row folds:** rows within a trajectory are strongly coupled; the reproducible public implementation groups folds by well ([Bernardus notebook](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild)).
- **Calling Tucker’s method “the matcher” or naming an architecture:** his public statements give score and scope, not the implementation ([712037](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430), [717573](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3486609)).
- **Unrestricted global GR registration:** repetitive motifs confidently select the wrong absolute depth; use heel-anchored relative matching inside a bounded corridor ([Georgy notebook](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever)).
- **Blind DTW as the path solver:** public experiments washed out useful local evidence and could invent motion on flat wells ([Georgy](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever), [Mitch](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore)).
- **Hard argmin on every correlation curve:** repeated motifs can create separated plausible modes ([Souldrive](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711878)).
- **Posterior averaging every well:** it helps genuinely ambiguous tails but biases easy unimodal wells; ambiguity detection and temperature must be calibrated out of fold ([Georgy](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3481297)).
- **Calling the rolling-median diagnostic an FFT or RMSE gain:** Georgy explicitly distinguishes the proxy from a real FFT notch ([comment](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367#3482433)).
- **Assuming code paths are active because they exist:** Pilkwang’s default profiles disable bimodal detection, FFT denoising, and the sequence matcher ([notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)).
- **Trusting visible train/test overlap:** the host says visible examples are replaced by the hidden test ([data](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)).
- **Ungated late overrides:** both public implementations use prefix evidence and bounds before changing the prior ([Bernardus](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild), [Pilkwang](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)).

## Primary-source index

### Official competition sources

- [Evaluation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/overview/evaluation)
- [Data description](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
- [Host Igor Kuvaev on manual GR-to-TVT projection](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416#3468837)

### Original-author implementations

- [Bernardus — ROGII LB 7.295 public rebuild](https://www.kaggle.com/code/bernubritz/rogii-lb7295-public-rebuild)
- [Mitch Gansemer — drift-targeting NCC/tree baseline](https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore)
- [Georgy Mamarin — measure the alignment noise floor](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever)
- [Pilkwang Kim — target-free TVT geosteering notebook](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)
- [Pilkwang Kim — dual-track prefix-calibrated geosteering](https://www.kaggle.com/code/pilkwang/rogii-dual-track-prefix-calibrated-geosteering)
- [Pilkwang Kim — target-free alignment EDA](https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt)
- [Souldrive — the 15-ft datum](https://www.kaggle.com/code/souldrive/the-15-ft-datum-why-honest-cv-bottoms-out)
- [Souldrive — decoding Eagle Ford ambiguity](https://www.kaggle.com/code/souldrive/decoding-eagle-ford-why-some-wells-are-hard)
- [Pilkwang Kim — GR/typewell matching](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#4-grtypewell-matching-barcode-not-label)
- [Pilkwang Kim — posterior mean for tie wells](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering/#8-tie-wells-use-posterior-mean-not-overconfident-decisions)
- [Pilkwang Kim — GR as stratigraphic barcode](https://pilkwangkim.github.io/posts/ROGII-Target-Free-Stratigraphic-Alignment-for-TVT/#3-gr-as-a-stratigraphic-barcode)
- [Pilkwang Kim — PF and residual GBDT](https://pilkwangkim.github.io/posts/ROGII-Target-Free-Stratigraphic-Alignment-for-TVT/#10-pf-residual-gbdt)

### Original competition discussions/comments

- [Tucker Arrants — approximately 5 ft single-model CV](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478241)
- [Tucker Arrants — pooled metric and per-well-only scope](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3478430)
- [Tucker Arrants — 5.4 single model and 5.0 equal-weight ensemble](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/723647)
- [Tucker Arrants — non-tabular models in the fives](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/717573#3486609)
- [Related public discussion 726465](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726465)
- [Related public discussion 727570](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727570)
- [Pattern-matching paths and selector difficulty](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/721549)
- [Shrey/Tabish/MY0705/Georgy — GR rotation and denoising thread](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367)
- [Souldrive — ±15-ft datum ambiguity](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711878)
- [Georgy — posterior-mean ambiguous-tail diagnostic](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037#3481297)
