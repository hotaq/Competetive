# RSNA Knee Kaggle Code and Discussion Scout

_Snapshot: 2026-08-09 UTC. Votes, notebook versions, runtimes, and leaderboard scores are live values and will change._

## Executive recommendation

Use [Pilkwang Kim's RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) unchanged as the first canonical benchmark. It is the clearest public reference point: 236 votes, an author-claimed public score of 0.891 that agrees with the author's Kaggle team score, internet disabled, public weights, and a complete image-only inference path. Its main limitation is that it is a 20-checkpoint ensemble, not a minimal training baseline. If the immediate goal is instead to reproduce the strongest well-audited public artifact, [Beicicc's guarded Renta V5 reproduction](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction) has the best provenance: it pins script versions, model and manifest hashes, a public submission ID, and the 0.897-scoring Renta V5 behavior.

For the first controlled experiment, fork the pinned Pilkwang anchor and change **only Fracture TTA-window pooling from probability mean to probability max**. Save both baseline and variant submissions from the same run. Do not also change Contusion, Lateral Meniscus, preprocessing, model weights, or rank aggregation. This isolates one plausible hypothesis—small focal fractures may be diluted by averaging—without rebuilding the model stack. The [recent pooling-probe fork](https://www.kaggle.com/code/renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe) is useful prior art, but its current source max-pools three targets, so it does not provide a clean Fracture-only ablation.

The current public leaderboard leader is 0.939, while the strongest source-linked public artifact inspected here is 0.897. The gap means a public baseline is useful for pipeline validation, not evidence that the approach is competitive with the leaders. See the live [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

## Scope and evidence rules

This review used only first-party Kaggle material: the [competition Code listing](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code), Kaggle CLI/API notebook metadata, pulled public notebook source, public execution logs, the leaderboard, and Kaggle Discussion messages. I did **not** download the DICOM archive, execute a notebook, or submit predictions.

Runtime means the latest public notebook-version execution log, not a benchmark on the hidden test. The public example has only three studies, so an inference log of one minute does not imply one-minute hidden-test runtime. For training notebooks, the log includes the full visible training run. Kaggle Staff further clarified that Efficiency runtime is the full notebook wall time, including setup, model loading, and DICOM decoding—not just the forward pass ([Efficiency Prize: GPU and RuntimeSeconds](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475)).

Kaggle Code metadata does not expose a cryptographic notebook-to-submission link. Therefore:

- **Kaggle-visible/consistent** means the author's claim agrees with a visible Kaggle public result, but exact notebook attribution is still inferential.
- **Self-reported** means the number appears only in the title, markdown, filename, or source comments.
- **OOF** means a notebook's own out-of-fold diagnostic; it is not a public leaderboard score.
- A team score is not attributed to a notebook unless the public evidence supports that link.

The competition score itself is macro-average ROC AUC across the 12 targets; see the official [Evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

### Official Code score order versus team scores

Kaggle's `kernels list --sort-by scoreDescending` returns notebook order but withholds the numeric notebook score. The competition leaderboard API returns numeric **team-best** scores but not the notebook that produced them. The two must not be joined automatically.

| Code score order | Current notebook | Votes | Related author's/team best | Attribution strength |
|---:|---|---:|---:|---|
| 1 | [Renta pooling probe](https://www.kaggle.com/code/renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe) | 49 | 0.897 | Current notebook is first in Code score order, but the API does not expose its number. Renta V5's 0.897 is independently source-linked by Beicicc; the current source has changed. |
| 2 | [Beicicc guarded Renta V5 reproduction](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction) | 3 | 0.897-class reproduction | **Strongest provenance:** source records Renta V5 script version `341057541`, submission `55362964`, model/manifest hashes, and its own guarded reproduction identifiers. |
| 3 | [Prvsiyan report-first extension](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 92 | 0.894 | Source says V14 scored 0.894 and the team API agrees, but current V34 is not that historical V14; CLI pull of V14 is unavailable. |
| 4 | [Blacklions open anchor](https://www.kaggle.com/code/blacklions/rsna-knee-open-anchor-fm-pool-v1) | 1 | 0.893 | Team score only; source does not prove exact notebook attribution. |
| 5 | `pritom2357/exp-2` | 2 | 0.892 | Team score only; insufficient source provenance for recommendation. |
| 6 | [Pilkwang baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 236 | 0.891 | Source and multiple derivatives identify the 20-member public ensemble as 0.891; team API agrees. |

## Practical baseline ranking

This is a ranking by **usefulness for our next work**, not raw votes. Recency is the last run reported by Kaggle CLI. “GPU” is the notebook setting/source log, and runtimes are rounded.

| Rank | Notebook | Votes; last run | Score evidence | Latest runtime / GPU | Reproducibility and role |
|---:|---|---|---|---|---|
| 1 | [Guarded Renta V5 reproduction](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction) | 3; Aug 9 | Source-linked/corroborated 0.897-class artifact | Up to 8.5 h guard; exactly one T4 | **Safest exact high-score reproduction.** Pins upstream script versions, submission ID, manifest/checkpoint hashes, all 20 members × 10 windows, and fails rather than silently bank a partial result. |
| 2 | [RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 236; Aug 8 | Public 0.891, source-claimed and team-API-consistent | ~63 s model path; ~6.4 min notebook envelope on 3 examples; T4 | **Best canonical starting code.** Public DINOv2-small, label, and 20-checkpoint weight inputs; internet off. Current time guard may reduce windows/members on the full hidden set. |
| 3 | [Public 4-fold DINOv2 v4](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4) | 38; Aug 6 | OOF 0.7719 on the 58 annotated studies; no LB claim | ~101 min; GPU | **Best transparent train-and-infer reference.** No private checkpoint or external label table; complete four-fold run. Gold OOF is far too small for fine model selection. |
| 4 | [DINOv2 at meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | 82; Aug 6 | Gold OOF ridge 0.7788, blend 0.7878; author explicitly says not an LB estimate | ~92.7 min; 2 GPUs | Strong high-resolution experiment with frozen embeddings and a small attention head. Reproduction requires an Opus report-label dataset that the pulled metadata did not correctly declare. |
| 5 | [DINOsaur V2](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2) | 68; Aug 7 | OOF pseudo-label 0.7350 / exact-label 0.7331; no LB claim | ~3 h 22 min; GPU | Complete self-contained train-and-infer pipeline with report-rule labels. Slower, but less dependent on attached community artifacts. |
| 6 | [Read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 92; Aug 9 | V14 source says 0.894 and team API agrees; current V34 is not V14 | ~17.6 min on 3 examples; GPU | Strong frontier ensemble, but target-specific and dependent on several notebooks/caches/specialists. Historical scored source cannot be pulled, lowering reproducibility. |
| 7 | [Fracture TTA pool probe](https://www.kaggle.com/code/renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe) | 49; Aug 9 | Current Code score-order #1; exact numeric notebook score hidden. Renta V5 0.897 is source-linked separately. | ~81 s model path; ~7.6 min envelope on 3 examples; GPU | Timely Pilkwang fork. Current source max-pools Fracture, Contusion, and Lateral Meniscus; it can adaptively reduce work under time pressure, so it is not an exact all-or-fail reproduction. |
| 8 | [RSNA Knee infer](https://www.kaggle.com/code/tonylica/rsna-knee-infer) | 48; Aug 7 | No attributable LB score | ~35 s inference; ~60 s log envelope on 3 examples; T4 | Fast four-fold image inference using a public model bundle. Useful comparison, but training/checkpoint provenance is outside this notebook. |
| 9 | [Plane-Aware 2.5D MIL GPU Baseline](https://www.kaggle.com/code/lucifer19/rsna-knee-plane-aware-2-5d-mil-gpu-baseline) | 12; Aug 8 | Current OOF partial-gold 0.6465; source self-reports an older V7 public 0.782, not current V10 | ~92.9 min; GPU | A relatively compact trainable 2.5D design with no external checkpoint bundle. The small-vote/current-score evidence is weaker than the title suggests. |
| 10 | [Data structure, EDA, baseline](https://www.kaggle.com/code/romanrozen/rsna-knee-data-structure-eda-baseline) | 86; Aug 8 | Team API 0.888 cannot be attributed to current notebook | ~5.4 min on 3 examples; GPU | Good educational walkthrough, but attached Pilkwang weights make the scored path bypass the advertised v6 training changes. It is principally a Pilkwang inference fork. |

Two high-vote entries should not be mistaken for model baselines. [RSNA Knee Abnormalities – Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) has 85 votes but only renders an efficiency leaderboard file. [RSNA Knee Enhanced Ensemble](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble) has 47 votes and writes several rank variants around Pilkwang's weights; “0.891” refers to the imported source baseline, not a demonstrated score for its default hybrid.

The newly posted [2.5D ResNet34 baseline](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34) is suitable only as a pipeline smoke test in its current form: its logged experiment trains on 46 and validates on 12 of the 58 expert-labeled studies, all six validation AUCs remain 0.500, and its internet-on weight download is inconsistent with a final submission notebook.

## What the strongest notebook sources actually do

### 0. Beicicc: the strongest reproducibility guardrail

This notebook does not introduce a new model. It reconstructs the source-linked Renta V5 artifact under a strict contract: Pilkwang V15 and Renta V5 script IDs, submission ID, manifest SHA, 20 checkpoint filenames/IDs/sizes and model fingerprints, exactly one Tesla T4, 12 cached slices, all 10 overlapping three-slice windows, and all 20 DINOv2-small members. Renta V5 uses probability mean for 10 labels and window max for Fracture and Contusion, followed by per-target rank mean across members.

Unlike adaptive notebooks, it refuses a partial ensemble. It promotes the final file only after schema, UID coverage, nonconstant-output, and hash checks; a failure leaves `submission_fallback.csv` rather than a plausible-looking `submission.csv`. This is why a three-vote notebook ranks first for reproducibility even though Pilkwang is the canonical codebase. The cost is brittleness: it requires the exact public weight package, exact T4 architecture, and enough wall time for the full contract.

### 1. Pilkwang: the public inference benchmark

The notebook uses DINOv2-small and concatenates CLS and patch-pooling features. A diagnosis-specific slot-attention head aggregates six acquisition slots:

1. sagittal fluid-sensitive fat-suppressed;
2. coronal fluid-sensitive fat-suppressed;
3. axial fluid-sensitive fat-suppressed;
4. sagittal fluid-sensitive non-fat-suppressed;
5. coronal T1;
6. sagittal T1.

Preprocessing is one of its strongest contributions. It orders slices from DICOM geometry rather than SOP filenames, normalizes laterality, center-crops a physical 130 mm field of view, caches at 336 px, and stacks three adjacent slices as RGB-like input. At inference it evaluates overlapping slice windows and rank-means 20 checkpoint members. The attached package records member configuration so the notebook can reject incompatible preprocessing rather than silently load mismatched weights.

For training fallback, it prefers a mounted table of locally generated report labels, falls back to a multilingual lexicon, overwrites available targets with the 58 official image labels, and gives official labels higher weight. Reports are not used at test inference.

Likely failure modes are hidden-test DICOM I/O at approximately 1,300 studies, the maintenance burden of 20 checkpoints, silent upstream changes to public label/weight datasets, report-label noise, and public-LB overfitting through repeated ensemble/TTA adjustments. The three-study public run is not a realistic runtime estimate.

### 2. Hida v4: the clearest full training baseline

This notebook is a simpler descendant of the same physical-DICOM ideas. It uses 224 px over a 160 mm crop, six recovered acquisition slots, three adjacent slices per input, CLS plus mean-patch DINOv2-small features, and a per-target slot-attention head. The last six transformer blocks are trainable.

It derives weak labels for all 4,407 reports with a multilingual rule system, overwrites them with official labels where available, and weights gold studies 3×. Four folds are grouped by report-text hash to keep identical reports out of both train and validation. It trains 12 epochs with EMA, mild affine/intensity augmentation, and a pairwise ranking term. All four folds completed in about 101 minutes; the log reports 0.7719 macro OOF AUC on the 58 gold cases.

The source is unusually reproducible—competition data plus the public DINOv2 model are sufficient—but the selection criterion still consults roughly 13–15 gold studies per fold. That estimate is noisy enough to reverse small improvements, and rule-based report labels can systematically miss findings that radiology reports do not explicitly address.

### 3. Wguesdon: frozen high-resolution embeddings

This notebook targets small meniscal findings with DINOv2 ViT-S/14 at 336 px and a stated 150 mm field of view. It samples up to 46 slices over five acquisition slots, generates CLS, patch-mean, and patch-max embeddings, and compares a ridge/SVD control with a 12-query multihead-attention head. It uses five GroupKFold splits and three head seeds, then adds a single partial-backbone fine-tune fold and blends it.

Weak labels come from a public Opus-derived report-label table; the 58 official labels override them and receive weight 8. The public run used two GPUs, spent roughly 67 minutes building train embeddings, created about a 7.5 GB train cache, and finished in about 92.7 minutes. Its gold OOF blend of 0.7878 is a diagnostic, explicitly not a leaderboard estimate.

The largest reproducibility problem is metadata: the source expects the Opus label dataset and fails closed if absent, but the pulled notebook metadata did not identify a valid dataset source. A fork must explicitly attach and version `wguesdon/rsna-knee-llm-report-labels-opus`. Other risks are one-fold backbone fine-tuning, cache size, two-GPU assumptions, and tuning against 58 gold studies.

### 4. DINOsaur V2: self-contained multi-seed training

DINOsaur V2 uses DINOv2-small at 224 px with a 160 mm physical crop, six sequence slots, nine cached slices per slot, and a target-specific slot-attention head. It trains three folds × two seeds for 10 epochs, opens the final six transformer blocks, averages gamma TTA at 0.9/1.0/1.1, and rank-ensembles predictions.

Its multilingual rule extractor supplies weak report labels; gold values override with higher weight. The run completes in about 3.37 hours and reports pseudo-label OOF 0.7350 and exact-label OOF 0.7331. Those figures are not public scores. The main tradeoff is slower training for better self-containment; its likely ceiling is dominated by weak-rule label error and noisy gold calibration.

### 5. Prvsiyan: a frontier ensemble, not a baseline

The notebook first builds a stronger nine-language report parser with directional negation, wrapped-line repair, OA inheritance, severity grading, and a synovitis fallback. That parser is training-side only; the hidden test has no reports.

The image inference graph combines the 20-member Pilkwang DINOv2-small package, Tony Li's four-fold legacy DINO bundle, a synovitis specialist, and DINO/radiomics/PCA/linear/tree specialists for selected targets. It uses target-specific TTA and a weighted rank ensemble. Source comments report public 0.894 for the primary route and 0.836 for a legacy bundle; the current execution takes about 17.6 minutes on the three-example public test.

This is valuable evidence about complementary errors, but it is difficult to reproduce or debug: it couples several public datasets, model versions, kernel outputs, caches, and target-specific decisions. It also has a high public-LB-overfitting risk. It should be considered after an exact Pilkwang reproduction, not before it.

### Other useful source patterns

- Tony Li's inference notebook loads a public four-fold DINOv2-small bundle, uses 224 px/160 mm crops, six slots, nine slices per slot, target-specific attention, and per-fold rank averaging. It is fast and clean, but the training recipe is not in the notebook.
- Lucifer's 2.5D MIL uses two positions × three adjacent slices in six plane/contrast slots, DINOv2-small with the last four blocks open, pathology-family experts, and a report TF-IDF/SVD semantic auxiliary **during training only**. Its test path remains image-only.
- Jeki Wan Taufik's compact route selects one preferred series per plane, encodes two triplets per plane with ResNet18, applies plane-aware attention MIL and flip TTA, then blends 88% image rank with 12% protocol metadata. It is efficient but checkpoint-opaque.
- Luka Duvanov's [weak-label notebook](https://www.kaggle.com/code/nekkon/weak-labels-for-all-12-knee-mri-findings) is not a submission baseline, but it is a useful diagnostic. It reports 23% empty weak-label outputs and warns that the 58 labeled studies are a non-random diagnostic set, not a training set.

## High-signal Discussion findings

### Host- or staff-confirmed

1. **The image labels are authoritative, not the reports.** The [Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) says borderline findings are negative; two MSK radiologists label each reference study independently and a third adjudicates. In [Data/Reporting Inconsistencies](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491), challenge co-leader Po-Hao “Howard” Chen explains that clinical reports can be ambiguous or internally inconsistent, whereas challenge labels are generated by separate image review under the stricter rubric. The same image-labeling process is used for test.

2. **A negative label means annotated absent.** In [Possible inconsistencies between reports and labels](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826), Chen confirms that labels override report text and negatives are not “unannotated.” Bilateral studies can occur; released report/DICOM information was adjusted to disambiguate the knee being evaluated.

3. **The hidden test never supplies reports.** Chen states in [Report in real test set](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592) that `Report` will not be available before or after submission. Any report processing must create training supervision or an auxiliary teacher; inference must be image-only.

4. **`PatientSex` was deliberately removed from `train.csv`.** The host says it remains in DICOM headers and that the Data page would be corrected ([Missing PatientSex column](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733423)).

5. **GPU submissions are eligible for the Efficiency Prize, and every second counts.** Kaggle Staff says runtime starts at notebook execution and includes installs, model loading, and DICOM reads ([Efficiency Prize: GPU and RuntimeSeconds](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475)). This favors cached decoding, local offline dependencies, early valid-submission writes, and a strict time guard.

6. **External-data/hosted-LLM permission remains unresolved.** The 12-vote [External datasets and hosted LLM APIs](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) thread has community replies but no host ruling. A duplicate [third-party LLM thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733873) is also unanswered. Until clarified, generate labels locally/offline and do not transmit reports to a hosted commercial API.

### Useful community evidence, not official rulings

- A [DICOM metadata shortcut probe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517) reports macro AUC 0.6516 with random folds but only 0.5981 when grouped by scanner fingerprint; series descriptors alone score 0.5954. The large drop suggests random CV learns site/scanner effects. Use scanner/site grouping when evaluating metadata features.
- [“Not addressed” is a label too](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932) reports that 25.4% of target/report cells were not explicitly addressed. On the 58 gold studies, LLM labels beat regex labels, but treating silence as negative versus unknown is pathology-specific; blanket imputation hurt.
- [58 studies cannot see a 0.01 gain](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876) uses simulation to argue that the 58-study gold set is pathology-enriched and cannot reliably rank small gains. Use broad weak-label CV for coarse ranking and gold primarily for calibration/diagnostics.
- The [official Discord/onboarding thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/730709) says the Discord is public, private data/code sharing is prohibited, and staff/hosts do not monitor it. Questions requiring an authoritative answer belong in Kaggle Discussion.

## The next two-stage Kaggle-GPU baseline

**Stage 1 — establish the anchor.** Run the exact pinned Pilkwang inference branch unchanged and save its output, package fingerprints, full wall time, test-study count, slot coverage, and member/window count. Prefer all 20 members and all intended windows; if its adaptive time guard drops work, treat the result as a different baseline. Beicicc's guarded notebook is the reference for the level of input and output validation we should copy.

**Stage 2 — one controlled change.** Ask whether max-pooling overlapping TTA windows improves Fracture ordering relative to that exact anchor.

Use the exact current Pilkwang notebook, its pinned 20-member weight package, DINOv2-small model, 336 px/130 mm preprocessing, six slots, physical ordering, laterality normalization, and per-member rank mean. In one notebook run:

1. run the unchanged branch and save `submission_baseline.csv`;
2. retain every baseline prediction except Fracture;
3. for Fracture only, take the maximum probability across overlapping TTA windows within each member, then apply the original across-member rank mean;
4. save `submission_fracture_max.csv`;
5. assert identical UID order, 13 columns, finite values, range `[0,1]`, no nulls, and exact equality of the other 11 columns;
6. log DICOM decode time, per-member inference time, full notebook wall time, test-study count, slot coverage, package fingerprint, and selected GPU.

Run the baseline first. If it does not reproduce the expected Kaggle-visible neighborhood, stop and debug version/input drift instead of interpreting the variant. If it does, submit the paired outputs as the only comparison. A max operation can amplify a noisy slice and create false positives, so a public gain is an experiment result—not a general license to max-pool other pathologies.

## Reproduction provenance

Representative read-only commands used for the snapshot:

```bash
kaggle kernels list --competition rsna-knee-abnormality-detection \
  --kernel-type notebook --sort-by voteCount --page-size 50 -v
kaggle kernels pull AUTHOR/SLUG -p /private/tmp/rsna-kernels/NAME -m
kaggle kernels logs AUTHOR/SLUG
kaggle competitions leaderboard rsna-knee-abnormality-detection --show
```

Public source and metadata were pulled for Beicicc, Blacklions, Pilkwang, Prvsiyan, RomanRozen, Wguesdon, RomanTamrazov, Renta, Tony Li, Sakhawat Hossen, Hida, Lucifer, Jeki Wan Taufik, Luka Duvanov, and the self-reported 0.668 DINOsaur fork. No competition image archive or notebook output dataset was downloaded.
