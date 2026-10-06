# RSNA Knee architecture ladder toward a 0.950 stretch target

_Read-only research snapshot: 2026-08-09. Kaggle scores, notebook ordering, votes, and runtimes can change._

## Bottom line

The reproducible public evidence does **not** currently support a credible promise of 0.950. The live public leaderboard leader is 0.940, so 0.950 is a stretch above the current field, not merely a baseline upgrade. Moving from 0.899 to 0.950 requires 0.612 total AUC points across the 12 targets. If only four targets improve, they must gain an average 0.153 AUC each, which is implausible from another pooling tweak.

The practical route is therefore:

1. test the current seven-target public aggregation recipe that claims the 0.899 neighborhood;
2. improve the training signal with target-specific report-label confidence and missingness;
3. build one high-resolution slice-sequence MIL family and a small number of target specialists on a shared DINO cache;
4. blend only with nested grouped-OOF evidence;
5. defer raw 3D and MRI self-supervised adaptation until the cheaper families pass gates.

## What is actually proven on this competition

| Artifact | Evidence | Runtime evidence | What it establishes |
|---|---|---|---|
| [Pilkwang V15](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | Reproduced public 0.891 | Visible model path about 63 s on 3 placeholder studies; hidden runtime cannot be extrapolated | Canonical 20-member DINOv2-small, 336 px, physical crop, 6-slot, 10-window rank ensemble |
| [Beicicc guarded Renta V5](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction) | Strong source-linked 0.897-class reproduction | Strict 8.5 h guard | Max pooling across TTA windows for Fracture and Contusion is the strongest well-audited public delta |
| [AADIGUPTA “0.899 Let me Cook”](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook) | Current Code-order #1; title claims 0.899, but Kaggle hides the numeric notebook score | 70 s model path on 3 placeholders | Same public weights, but seven target overrides: max for Fracture, Contusion, both Menisci, and Baker's; top-2 probability mean for ACL and MCL. This is a leaderboard-tuned inference recipe, not a new architecture |
| [Wguesdon high-resolution DINOv2](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | Gold OOF ridge 0.7788, attention blend 0.7878; author explicitly says this is not an LB estimate | About 92.7 min on 2 GPUs; about 67 min was feature extraction | Cached 336 px slice embeddings plus a target-query head are computationally attractive, but no LB gain is proven |
| [Hierarchical DINOv2 v5](https://www.kaggle.com/code/gourabroy553/knee-mri-hierarchical-dinov2) | Gold OOF 0.7809; no attributable LB | 309 min for 4 folds | Patch-to-slot attention is feasible but not proven superior to the public ensemble |
| [DINOv2-Base physical-scale soup](https://www.kaggle.com/code/fleongg/rsna-knee-dinov2-base-physical-scale-soup) | Best logged selection 0.7192; no LB claim | About 120 min on 2 GPUs | A larger backbone alone is not evidence of improvement here |
| [Report-teacher hierarchical MIL](https://www.kaggle.com/code/blacklions/report-teacher-anatomy-aware-hierarchical-multimod) | Gold OOF 0.7214; no LB claim | About 2.26 h | Elaborate hierarchy/multi-teacher design can still underperform when labels and validation are weak |
| [Prvsiyan report-first extension](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | Historical V14 self-reports 0.894; current source is a later, different version | Current visible path is much more complex | It identifies Lateral Meniscus, Lateral OA, and Synovitis as specialist targets, but its current specialist graph is not source-linked to a gain |

The main conclusion is uncomfortable but useful: every source-linked public score above 0.891 is still the same Pilkwang checkpoint ensemble with target-specific inference pooling. The exact 0.899 number is only a title claim consistent with current Code ordering, not a verifiable notebook score. Publicly visible higher-capacity training notebooks are research references, not leaderboard-proven replacements.

## Target weaknesses and diversity opportunities

Wguesdon's broad weak-label OOF table is not ground truth, but it is the largest per-target diagnostic among the inspected public notebooks. Its attention blend is weakest for Lateral Meniscus (0.7264), Synovitis (0.7272), MCL (0.7498), and Baker's cyst (0.7721); Medial OA (0.8501) and Effusion (0.8427) are strongest. The same notebook reports only 6.9% weak-label prevalence for Fracture and 11.6% for Contusion, making those targets especially sensitive to pooling and label errors.

Use four small specialist families rather than twelve unrelated models:

| Family | Targets | Primary hypothesis | Pooling hypothesis |
|---|---|---|---|
| Focal tear/injury | Medial/Lateral Meniscus, Fracture, Contusion | 336 px or higher, dense sagittal/coronal fluid-sensitive coverage, retain patch/slice tokens | top-k, log-sum-exp, or max; audited 0.897 evidence supports max for Fracture/Contusion, while the title-only 0.899 recipe also maxes both Menisci |
| Ligament | ACL, MCL | Route sagittal evidence to ACL and coronal evidence to MCL; keep structural and fluid views distinct | attention or top-k; the title-only 0.899 recipe uses top-2 probability mean for both |
| Degenerative | Medial/Lateral/PF OA | Wider physical crop, structural/T1 views, axial route for PF OA | attention/mean because disease is spatially broader |
| Fluid/cystic | Effusion, Synovitis, Baker's | Axial and sagittal fluid-sensitive views, separate diffuse fluid from focal cystic evidence | mean/attention for diffuse targets; the title-only 0.899 recipe maxes Baker's, but this remains LB-tuned rather than independently validated |

The original [MRNet paper](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699) supports the broad design pattern, not this competition's score: it trained separate task/plane models, max-pooled features across slices, and then combined three planes. On a different 1,370-exam dataset it reported AUC 0.965 for ACL tears but only 0.847 for meniscal tears. This is useful evidence that task/plane routing and slice-max pooling can matter, while also warning that menisci remain difficult.

## Recommended model graph

### 1. Fix supervision before scaling the backbone

Reports are training-only and the host says the image-derived labels are authoritative. Treat report silence as **unknown**, not automatically negative. Store, per target:

- soft probability;
- explicit positive/negative/uncertain/not-addressed state;
- confidence;
- evidence source/teacher disagreement.

Official labels override every teacher. Train BCE/ranking only where confidence is sufficient, with target-specific masks. Cross-fit any teacher calibration on the 58 gold cases; never fit and evaluate the same gold row. A public community analysis reports that about 25.4% of report-target cells are not addressed and that silence handling is target-specific ([discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)).

### 2. Share one expensive DINO cache

Build 336 px, physical-scale, geometry-ordered, laterality-normalized 2.5D inputs for five or six acquisition slots. Cache per-slice DINOv2-small CLS, patch-mean, and patch-max features plus normalized physical slice position. Train cheap heads on the cache:

`slice tokens -> target query within slot -> target query across slots -> 12 logits`

This is a natural study-level MIL problem: the original [attention MIL paper](https://proceedings.mlr.press/v80/ilse18a.html) defines learnable permutation-invariant bag aggregation for labels attached to bags rather than instances. DINOv2 itself shows that patch features support dense tasks and that fine-tuning improves downstream classification, but also reports that high-resolution training at 416 px costs about 3x 224 px ([DINOv2 paper](https://arxiv.org/abs/2304.07193)). Therefore 336 px plus a small partial fine-tune is the sensible first ceiling test; DINO-Base/416 px is not.

### 3. Add sequence diversity before raw 3D

The second family should operate over the ordered slice sequence with relative position encoding and target queries. It is cheaper than an end-to-end raw-volume model and reuses the shared cache. Only if that family adds OOF diversity should a lightweight raw 3D/Video Swin model be attempted. [Video Swin](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html) provides a locality-biased spatiotemporal architecture and favorable video benchmark efficiency, but there is no public RSNA Knee evidence that it beats 2.5D DINO under these weak labels.

### 4. Ensemble with target-wise ranks, not public-LB hand tuning

For a single target, any monotonic calibration leaves ROC-AUC unchanged. Calibration matters only before combining probability scales. Rank each model per target, then select blend weights inside nested grouped CV from a coarse grid and shrink toward equal weights. A specialist enters the final blend only if it improves paired OOF and is not nearly identical to the anchor.

Checkpoint/model soups can reduce inference cost for same-architecture fine-tunes: the original [Model Soups paper](https://proceedings.mlr.press/v162/wortsman22a.html) shows that averaging compatible fine-tuned weights can improve accuracy and robustness without ensemble inference cost. Soup only checkpoints from the same initialization/basin and validate the result; it does not replace architectural diversity.

### 5. Defer domain-adaptive self-supervision

MRI-domain MAE/DINO adaptation on competition slices is a plausible late-stage idea. [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) provides an efficient masked-patch pretraining recipe, but no inspected Kaggle artifact proves a gain for this competition. Spend 6-8 h on one such run only after the cached MIL family and specialists pass validation gates.

## Staged experiments and gates

Runtime budgets below are planning estimates anchored to public notebook logs, not guarantees on the hidden rerun.

| Stage | Experiment | GPU budget | Go/no-go gate |
|---:|---|---:|---|
| 0 | Reproduce the current seven-target recipe: max Fracture, Contusion, both Menisci, Baker's; top-2 mean ACL/MCL; assert the other 5 columns are identical to the 0.891 anchor | Visible commit is minutes; reserve the competition's full 9 h hidden limit | Treat 0.899 as a claim, not an expectation guarantee. If it does not exceed the audited 0.897 recipe, reject the extra five overrides. Do not tune an eighth target from one LB result |
| 1 | Teacher audit: rules + attached offline LLM labels, explicit not-addressed masks, gold override, grouped cross-fit | CPU or <0.5 h GPU | Cross-fit gold teacher macro improves >=0.02 versus current labels, no target loses >0.05, and broad label prevalence remains plausible |
| 2 | One shared 336 px DINO-small feature cache; ridge control and target-query slice/slot MIL head | ~2-3 accelerator-hours; Wguesdon's full 2-GPU run was 92.7 min | Grouped broad OOF +0.01 over ridge, gold macro not worse by >0.01, and >=8/12 targets non-worse |
| 3 | Partial fine-tune last 2-4 DINO blocks for one fold/seed | ~2-3 h | Same split/seed broad OOF +0.01 and at least three target gains >=0.02; gold audit must not reverse direction |
| 4 | Two specialist heads first: focal injury and fluid/cystic | ~3-5 h total on shared cache | Target OOF +0.02, prediction Spearman correlation with main family <0.97, no gold sign reversal |
| 5 | Expand only the best family to 4 folds x 2 seeds; optionally soup compatible checkpoints | ~6-10 h | Ensemble/soup +0.005 OOF over best single seed and variance falls across folds |
| 6 | Nested-CV per-target rank blend; one Kaggle submission | CPU | Nested OOF +0.01 macro over best component, >=8/12 non-worse, no target drop >0.02 |
| 7 | One optional diversity bet: raw Video Swin/3D **or** MRI-domain MAE, not both in the same week | 6-8 h | Keep only if blend OOF +0.01 and useful targets have correlation <0.95 with the main family |

Reserve 8-9 GPU-hours each week for failed runs and final inference. Do not submit every ablation: one public submission per pre-registered hypothesis family is safer than fitting per-target knobs to the public leaderboard.

## Validation contract

- Group folds by normalized-report hash, repeated patient ID where available, and a coarse scanner/protocol fingerprint.
- Use broad weak-label OOF to select recipes; use the 58 gold studies as an audit/veto, not a precise optimizer.
- Save per-target OOF predictions for every family and seed.
- Compare candidates on identical folds/seeds with paired bootstrap intervals and per-target direction counts.
- Log target prevalence, teacher coverage, slot coverage, prediction correlation, DICOM decode time, training wall time, and full submission wall time.
- Keep test reports and hosted LLM APIs out of inference. Hidden test has no reports, and hosted API permission remains unresolved.

## Realistic score expectations

- **0.897-0.899:** reproducible public inference improvement.
- **0.90-0.92:** plausible with better supervision plus one complementary specialist family, but not guaranteed by current public evidence.
- **0.93-0.94:** requires multiple genuinely complementary models and careful target-wise blending; this is the current leading range.
- **0.950:** a stretch target above the current public leader. It likely requires either unpublished modeling/data advantages or unusually strong private-label generalization. Treat it as an outcome, not an experiment acceptance threshold.
