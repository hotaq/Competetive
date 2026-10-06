# RSNA Knee: staged score-climb plan toward a 0.950 stretch target

_Research snapshot updated 2026-08-09 after E002 completed. Kaggle scores, Code ordering, votes, runtimes, and submission status are live values and will change. Only small CSV/OOF metadata artifacts were inspected locally; the DICOM archive remains on Kaggle._

## Decision

Do **not** plan 0.950 as the expected result of another pooling tweak. As of this snapshot, the [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) is led by **0.940**, and our exact E002 submission `55378950` is verified at **0.899**.

E002 passed its score gate, so freeze it as the inference anchor and stop hand-tuning TTA pooling. E003-E005 subsequently rejected pooled, supervision-reweighted, and spatial frozen-DINO specialist branches. E006 found that the fixed Steven-v4 label blend improves Gold58 macro AUC by 0.0227 but damages Fracture by 0.0783, while a global three-source fusion is substantially worse. The next admissible step is therefore a metadata-only, nested target-selective supervision gate; an EfficientNet-B3 image run is not yet authorized.

The score arithmetic shows why. The official metric is the unweighted mean of 12 per-target ROC AUCs ([evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)):

| Starting score | Gap to 0.950 | Total AUC gain required across 12 targets |
|---:|---:|---:|
| 0.891 | 0.059 | 0.708 |
| 0.897 | 0.053 | 0.636 |
| 0.899 | 0.051 | 0.612 |
| Current leader, 0.940 | 0.010 | 0.120 |

Even from 0.899, improving only four targets would require an average gain of about 0.153 AUC on each. That is not a realistic expectation from more hand-selected TTA pooling.

## Evidence ladder

Evidence grades used below:

- **Verified:** an exact submission/result is visible in our Kaggle account or an artifact records a submission ID and is corroborated by the official leaderboard.
- **Corroborated claim:** title/source claim is consistent with Kaggle Code ordering or an author's team score, but Kaggle does not expose an exact notebook-to-score join.
- **OOF only:** useful design evidence, not a public leaderboard result.

| Rung | Artifact | Score evidence | Exact material change | Role |
|---:|---|---|---|---|
| 0 | [Our Pilkwang V15 anchor](https://www.kaggle.com/code/chinnaphatkhun/rsna-knee-pilkwang-v15-anchor) | **Verified 0.891**, submission `55374779` | Canonical 20-member DINOv2-small ensemble | Frozen baseline |
| 1 | [Beicicc guarded Renta V5 V2](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction?scriptVersionId=341145955) | **Verified/corroborated 0.897-class**; links Renta submission `55362964`, guarded result `55369001`, and official leaderboard entries at 0.897 | Max probability over ten windows for **Fracture and Contusion only**; mean for the other ten targets | Immediate next run |
| 2 | [AADIGUPTA V341177035](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook?scriptVersionId=341177035); our E002 `55378950` | **Verified 0.899** in our Kaggle account | Wider slice band, max for Fracture, Contusion, both Menisci and Baker's; top-2 mean for ACL/MCL | Frozen best inference anchor |
| 3 | [Wguesdon high-resolution DINOv2](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) | **OOF only:** gold ridge 0.7788, blend 0.7878; author says it is not an LB estimate | 336 px frozen DINO features, target-query attention, one partial fine-tune | Efficient architecture template |
| 4 | [Hierarchical DINOv2](https://www.kaggle.com/code/gourabroy553/knee-mri-hierarchical-dinov2) | **OOF only:** about 0.7809; no attributable LB result | Patch-to-slot attention | Research reference, not a proven rung |
| 5 | [DINOv2-Base physical-scale soup](https://www.kaggle.com/code/fleongg/rsna-knee-dinov2-base-physical-scale-soup) | **OOF only:** logged selection about 0.7192; no LB claim | Larger backbone/checkpoint soup | Evidence that size alone is not enough |

The critical distinction is that every source-linked public improvement above 0.891 still uses Pilkwang's same 20 checkpoints. Public higher-capacity notebooks have not demonstrated a leaderboard improvement.

## Archived control: exact guarded Renta V5

E002 reached 0.899, so preserve Renta V5 as an audited reference and spend the next GPU budget on the offline modeling ladder. Run this control only if later provenance isolation becomes necessary.

### Provenance

- Pilkwang V15 upstream: [script version `340906482`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1?scriptVersionId=340906482), notebook SHA-256 `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8`.
- Scored Renta V5: [script version `341057541`](https://www.kaggle.com/code/renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe?scriptVersionId=341057541), linked public submission `55362964 = 0.897`.
- Guarded reproduction: [Beicicc V2, script version `341145955`](https://www.kaggle.com/code/beicicc/rsna-knee-research-renta-v5-safe-reproduction?scriptVersionId=341145955), guarded submission reference `55369001`; its run time and author align with Kun Zhang's official 0.897 leaderboard entry.
- Renta V5 notebook/executable/AST SHA-256: `88dd523c444013b9d8c13dc958d193cb19fd4d53f535356a1d6e36f6bbd9bec1`, `f8e993b9c1950b7baf4ba1838268d516b3e4e1e77275d01905123361a12a0e72`, `574919df4c061dc3f5d972d12dadd6c305b00d5b11c9b4950143e41ec0ec4da5`.
- Guarded V2 notebook/executable/AST SHA-256: `740740e0e6052170150cdc0d0d97ae2477baff57c0dced06002ddaee4fb92f29`, `c05d18066e69e15bc5097850bb35605287e6b5efc0f070417c72f52960e7d7c9`, `7f062d9491ed664a54e6da5ba5935e7332bd9cf6cbe9b687d9c7d397db709c76`.

Kaggle's leaderboard endpoint does not expose a submission-ID-to-notebook join. The evidence is therefore the notebook's pinned provenance plus matching author, score, and timing—not a cryptographic Kaggle API relationship.

### Exact delta from our 0.891 anchor

Everything stays fixed: 20 DINOv2-small checkpoints, six acquisition slots, 336 px, a 130 mm physical crop, 12 cached slices, all ten overlapping three-slice windows, probability pooling, and equal per-target rank averaging across members.

Only the within-member window reducer changes:

```python
TTA_TARGET_POOL = {"Fracture": "max", "Contusion": "max"}
```

The other ten targets retain Pilkwang's arithmetic mean probability. The latest Renta source has since added Lateral Meniscus; that is **not** the scored V5 recipe.

### Reproduction contract

Use the exact historical V2 link rather than blindly forking the latest source. Select a T4, keep internet off, and attach the exact public versions:

- competition `rsna-knee-abnormality-detection`;
- `pilkwang/rsna-knee-weights/1`;
- `pilkwang/rsna-knee-llm-labels/1`;
- `pilkwang/pilkwang-public-dataset-for-notebooks-figures/22`;
- `metaresearch/dinov2/PyTorch/small/1`.

The notebook requires Tesla T4 `sm_75`, pins the weight-manifest SHA, validates the exact 20 files/member identities/configurations/fingerprints, requires 20 members x 10 windows, and atomically promotes `submission.csv` only after schema, UID, range, finiteness, and nonconstant-output checks. It deliberately fails instead of silently dropping members or windows. Do not set `SLOT_SCHEME` or `RSNA_ORDER_CACHE`.

Accept the run only if logs contain all of the following:

1. preflight PASS on T4 `sm_75`;
2. `20 member(s)` and `336px x 12 slices`;
3. every member predicted over `10 window(s)`;
4. `validated candidate = rank mean of 20`;
5. `final atomic gate passed` and `done`.

The visible three-study path completed at 78.7 seconds and notebook conversion around 120 seconds. This does **not** predict the approximately 1,300-study hidden runtime. Budget the notebook's full 8.5-hour safety limit; hidden-size readiness is engineered but not proved by the placeholder run.

**Go:** full contract passes and the displayed score rounds to 0.897 (accept 0.896-0.898 only after checking score rounding and source identity). Freeze it as the new anchor.

**No-go:** score below 0.896, any reduced member/window count, non-T4 execution, changed input version, or guard failure. Debug provenance/runtime instead of interpreting the pooling hypothesis.

## Completed public experiment: E002 at 0.899

The current [AADIGUPTA source](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook?scriptVersionId=341177035) is a real code delta, not a new model. Relative to Pilkwang V15, it changes the slice sampling band from 20%-80% to 10%-90% and overrides window pooling as follows:

- max: Fracture, Contusion, Medial Meniscus, Lateral Meniscus, Baker's;
- mean of the top two windows: ACL, MCL;
- ordinary mean: Medial OA, Lateral OA, PF OA, Effusion, Synovitis.

Its public placeholder log completes all 20 members x 10 windows in 70.1 seconds. We submitted the exact candidate as E002, submission `55378950`, at 2026-08-09 12:40:48 UTC. The authenticated Kaggle CLI now reports `COMPLETE` with public score **0.899**.

**Result: GO.** Freeze E002 and move directly to the offline model ladder.

Do not tune an eighth target from this public result. If later attribution matters, ablate the wider slice band and added pooling choices in frozen OOF, not by repeatedly probing the public leaderboard.

## Architecture path after 0.899

### 1. Repair supervision before enlarging the backbone

The host says challenge labels come from independent image review and override clinical-report wording ([host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491)); hidden test studies never include reports ([host answer](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733592)). Reports should create training supervision only.

For every report-target cell, store:

- soft probability;
- explicit positive, negative, uncertain, or not-addressed state;
- confidence and evidence source;
- disagreement between independent offline labelers.

Official labels always override teachers. Mask low-confidence/not-addressed cells rather than converting all silence to negative. A community analysis reports that 25.4% of report-target cells are not explicitly addressed and that silence handling is pathology-specific ([discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)); this is useful evidence, not a host ruling. Use only local/offline labelers until hosted-LLM permission is explicitly resolved.

### 2. Build one reusable 336 px DINO cache

Use physical-scale, geometry-ordered, laterality-normalized 2.5D inputs for five or six acquisition slots. Cache per-slice DINOv2-small CLS, patch-mean, patch-max, slot identity, and normalized physical slice position. Train cheap heads on the shared cache:

```text
slice/patch tokens -> target query within each slot
                   -> target query across slots
                   -> 12 logits
```

[DINOv2](https://arxiv.org/abs/2304.07193) is the public anchor backbone; [attention MIL](https://proceedings.mlr.press/v80/ilse18a.html) provides the bag-label aggregation pattern. Wguesdon's public notebook demonstrates that a complete 336 px frozen-feature plus attention experiment can finish in about 92.7 minutes on two GPUs, including roughly 67 minutes of feature extraction. Its OOF numbers are diagnostics, not expected leaderboard scores.

### 3. Add target-family specialists, not twelve unrelated models

The broad weak-label OOF in Wguesdon's notebook is weakest for Lateral Meniscus (0.7264), Synovitis (0.7272), MCL (0.7498), and Baker's (0.7721), while Medial OA and Effusion are stronger. Those are weak-label proxy scores, not ground truth, but they identify useful hypotheses.

| Specialist family | Targets | Evidence route | First aggregation hypothesis | Diversity source |
|---|---|---|---|---|
| Focal tear/injury | Both Menisci, Fracture, Contusion | Dense sagittal/coronal fluid-sensitive views; T1/marrow structure for fracture | top-k/log-sum-exp/max | 336 px target-query MIL plus a small 2.5D CNN arm |
| Ligament | ACL, MCL | Sagittal route for ACL; coronal route for MCL; preserve fluid and structural slots | attention/top-k | ordered sequence head with relative slice position |
| Degenerative | Medial/Lateral/PF OA | Wider physical context; coronal structural/T1 and axial PF route | attention/mean | patch-spatial head rather than only CLS features |
| Fluid/cystic | Effusion, Synovitis, Baker's | Axial and sagittal fluid-sensitive coverage; separate diffuse fluid from focal cyst | mean/attention for diffuse findings, focal reducer for Baker's | target-specific queries and a different seed/backbone head |

The original [MRNet study](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002699) supports task/plane routing and slice-max pooling on knee MRI, but it is not competition evidence. It also reported meniscal performance below ACL performance on its different dataset, reinforcing the need for a dedicated meniscus route.

### 4. Use ordered sequence diversity before raw 3D

Add relative slice positions and a small sequence transformer/gated MIL head over cached embeddings before training an end-to-end volume model. Attempt one raw 3D/Video Swin or MRI-domain MAE experiment only after cached heads demonstrate complementary errors. [Video Swin](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html) and [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) make these technically plausible; no inspected RSNA Knee notebook proves a leaderboard gain from either.

## Validation contract

The 58 image-labeled studies are too small to rank 0.005-0.01 improvements reliably. They are an audit/veto set, not the optimizer.

1. **Freeze folds first.** Keep identical normalized reports in one group. Group repeated patient identifiers where available. Record scanner/protocol fingerprints and run a separate scanner-grouped/domain holdout to expose site leakage.
2. **Freeze label states.** Gold overrides all; explicit high-confidence report labels train; uncertain/not-addressed labels are masked or down-weighted. Do not change labels after looking at a candidate's public score.
3. **Use two evaluation layers.** Select coarse recipes on grouped broad OOF using explicit report evidence; audit direction and catastrophic target failures on the 58 gold studies with wide bootstrap intervals.
4. **Compare paired predictions.** Same folds, seeds, cache, and target rows for every ablation. Save every OOF vector.
5. **Measure complementarity.** Record per-target Spearman correlation with the anchor and the nested-OOF gain from adding the specialist.
6. **Nest blend selection.** Rank-normalize predictions per target. Choose coarse blend weights inside inner folds; report only outer-fold performance. Shrink toward equal weights.
7. **Use public LB sparingly.** Submit one pre-registered hypothesis family only after an offline gate passes. The official rules determine winners on the private leaderboard, not the public one ([rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)).

Log target prevalence/teacher coverage, slot coverage, decode time, training wall time, inference wall time, member/window count, package fingerprints, per-target OOF, bootstrap intervals, and prediction correlations.

## 30 GPU-hour experiment ladder

Runtime figures are planning estimates anchored to public logs, not guarantees. The final inference notebook must remain within the official nine-hour GPU limit with internet disabled and output `submission.csv` ([Code Requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements)). Kaggle Staff says measured runtime includes setup, model loading, DICOM reading, and inference ([efficiency clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733475)).

### Week A: establish the public ceiling and the offline lab

| Experiment | GPU reserve | Gate |
|---|---:|---|
| E002 seven-target/wider-band candidate | Complete | Verified submission `55378950 = 0.899`; frozen anchor |
| Exact guarded Renta V5, conditional control | 8.5 h only if provenance isolation is needed | 20 x 10 contract; score around 0.897 |
| E003 336 px embedding cache + confidence-aware ridge/MIL | Complete in 66.9 min; **NO-GO** | Full coverage, but 75/25 anchor blend lost 0.0059 broad and 0.0078 gold macro |
| E004 cached supervision/head ablation | Complete in 21.5 min; **NO-GO** | All three weighting schemes reduced anchor OOF; discard pooled frozen-DINO head family |
| E005 spatial target specialist | Complete in 38.8 min; **NO-GO** | Diverse but weak: 80/20 blend reduced two-target broad OOF by 0.00265 |
| E006 supervision fusion audit | Complete in 30.4 min; **NO-GO** | Steven v4 macro +0.02267 but Fracture -0.07827; global three-source fusion -0.04184 macro |
| E006b nested target-selective fusion | Complete in 3.2 min accepted run; **NO-GO** | Paired Gold57 macro +0.01612, below +0.020 gate; all other checks passed |
| Failure/runtime buffer | 5-7 h | Preserve capacity for one failed commit |

### Week B: train only gated model families

| Stage | Experiment | GPU budget | Go/no-go threshold |
|---:|---|---:|---|
| 1 | Target-state/teacher audit | CPU or <0.5 h | Cross-fitted gold teacher macro +0.02 versus current labels; no target worse by >0.05; plausible prevalence/coverage |
| 2 | Frozen ridge control plus target-query MIL on shared cache | 2-3 h including cache refresh | Grouped broad OOF +0.01 over ridge, gold macro not worse by >0.01, at least 8/12 targets non-worse |
| 3 | Partial fine-tune last 2-4 DINO blocks for one fold/seed | 2-3 h | Same split/seed broad OOF +0.01 and at least three targets +0.02; gold direction not reversed |
| 4 | Focal-injury and fluid/cystic specialists | 3-5 h total | Target OOF +0.02, correlation with main arm <0.97, no gold sign reversal |
| 5 | Expand only the winning arm to four folds x two seeds; optionally soup compatible checkpoints | 6-8 h | Ensemble/soup +0.005 over best single and lower fold variance |
| 6 | Nested per-target rank blend | CPU | Outer OOF +0.01 macro, at least 8/12 non-worse, no target down >0.02 |
| 7 | One combined hidden-test inference submission | Reserve 8-9 h | Complete path and runtime margin; public score should beat frozen anchor before further scaling |

This totals approximately 24-29 GPU hours because failed/non-winning branches stop early and head/blend work uses cached embeddings or CPU.

### Week C and later: one expensive diversity bet

Allocate 6-8 hours to **one** of raw Video Swin/3D or MRI-domain masked pretraining, not both. Keep it only if nested blend OOF improves by at least 0.01 and useful targets have correlation below 0.95 with the DINO MIL family. Reserve another 8-9 hours for one inference submission and leave the rest as failure/retraining buffer.

## Score checkpoints and stop conditions

| Checkpoint | Interpretation |
|---:|---|
| 0.897 | Exact public inference gain reproduced; pipeline/provenance sound |
| 0.899 | Current public-code neighborhood reached; stop hand-tuning pooling |
| 0.905 | First evidence our supervision/modeling adds value beyond public recipes |
| 0.915-0.920 | Plausible result from better supervision plus one complementary family |
| 0.930-0.940 | Requires several genuinely complementary families and careful target-wise blending; current leading range |
| 0.950 | Stretch outcome above today's field, not an experiment acceptance threshold |

If we cannot exceed 0.905 after two independently validated model families, stop scaling the backbone and audit weak labels, folds, slot recovery, and OOF leakage. If a specialist raises public LB but fails nested OOF/gold direction, treat it as public-leaderboard overfit. If a hidden run drops members/windows or approaches nine hours, reduce architecture cost before adding models.

## Explicit non-goals and risks

- Do not promise 0.950 from currently reproducible public code; there is no evidence for it.
- Do not train or select solely on the 58 gold studies.
- Do not tune target pooling repeatedly against the public leaderboard.
- Do not use test reports; they do not exist.
- Do not send competition reports/images to hosted LLM APIs while permission remains unresolved.
- Do not adopt click-through external MRI datasets without a competition-rule and license audit.
- Do not infer hidden runtime by multiplying a three-study placeholder log without preserving a safety buffer.

## Read-only verification commands

Representative first-party queries used for this snapshot:

```bash
kaggle competitions leaderboard rsna-knee-abnormality-detection --show --page-size 200 -v
kaggle kernels list --competition rsna-knee-abnormality-detection \
  --kernel-type notebook --sort-by scoreDescending --page-size 100 -v
kaggle kernels pull AUTHOR/SLUG -p /private/tmp/rsna-audit/NAME -m
kaggle kernels logs AUTHOR/SLUG
kaggle competitions submissions rsna-knee-abnormality-detection -v
```

Only public notebook source/metadata/logs were pulled. No competition image archive or model weight package was downloaded.
