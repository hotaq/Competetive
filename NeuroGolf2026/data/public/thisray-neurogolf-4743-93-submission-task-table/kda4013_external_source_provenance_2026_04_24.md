# KDA4013 External Source Provenance 2026-04-24

## Summary

這份文件只回答一件事：`KDA4013_kda4010_score_max_reference_repair_overlay` 這個最高分版本，最終 `4743.93` public score 的 submission pack 裡，有哪些 task row 直接使用外部公開成果。

結論：

- Final submitted tasks：`394`
- 直接歸因為外部 public source / public dataset 的 final included rows：`236`
- 比例：約 `59.9%` of final submitted tasks
- 主要外部來源是 `aliafzal9323/neurogolf-2026-tiny-onnx-solver`
- 少量 row 來自 `karnakbaevarthur/neurogolf-all-task-logic-complexity-map` 與 `needless090/neurogolf-onnx-v31`
- 另有多個 public notebooks / datasets 在 public harvest 階段被下載、比較或作為上游 alias，但最終 task table 沒有把它們列為 final primary source

重要 caveat：

- `task_by_task_release_table.csv` 是目前 final pack 的 row-level primary attribution source。
- repo 對「外部 source slug / local capture path / public score」記錄完整度高，但對「每個 Kaggle code page 的 exact version URL / license / redistribution status」仍不完整。
- 舊的 `provenance_matrix_template.csv` 已明確把 KDA4013 的 7 個 public fallback replacements 標為 `needs_review` / `blocked_pending_review`，表示正式公開 full artifact 前仍應人工確認 source version 與 redistribution basis。

## Evidence Files

主要依據：

- `docs/02_planning/kaggle_task_by_task_release_2026_04_23/task_by_task_release_table.csv`
- `docs/02_planning/kaggle_task_by_task_release_2026_04_23/release_summary.json`
- `artifacts/kaggle_diagnostics/KDA4013_kda4010_score_max_reference_repair_overlay/manifest.json`
- `docs/04_logs/experiments/S164_public_harvest_preflight_clean.md`
- `docs/04_logs/experiments/S165_all_source_score_max_clean.md`
- `docs/04_logs/experiments/S166_public_dataset_harvest_dryrun.md`
- `docs/02_planning/public-output-harvest-analysis-2026-04-21.md`
- `docs/02_planning/free-lunch-score-sprint-2026-04-21.md`
- `docs/02_planning/leaderboard-gap-research-2026-04-21.md`
- `docs/02_planning/kaggle_public_release_2026_04_23/provenance_matrix_template.csv`

## Direct External Sources In Final Pack

| final source name | final rows | final source layers | URL / slug | repo evidence |
| --- | ---: | --- | --- | --- |
| `public:aliafzal9323_neurogolf-2026-tiny-onnx-solver` | `228` | `public_kernel`, `final_public_fallback_repair` | `https://www.kaggle.com/code/aliafzal9323/neurogolf-2026-tiny-onnx-solver` | local capture path recorded as `artifacts/downloads/public_kernels_20260421/aliafzal9323_neurogolf-2026-tiny-onnx-solver/submission.zip`; S164 records captured public score `4150.70` and local source zip SHA-256 `1aacb14c14af1357cd42200aad7ccb658a3f89519e552a98a6fab5dcd3d3030f` |
| `public:karnakbaevarthur_neurogolf-all-task-logic-complexity-map` | `6` | `public_kernel` | `https://www.kaggle.com/code/karnakbaevarthur/neurogolf-all-task-logic-complexity-map` | public harvest notes say this output matched `svanikkolli/arc-nano-engine`; final task table still uses the Karnak source label for these six rows |
| `public_dataset:needless090_neurogolf-onnx-v31` | `2` | `public_dataset` | `https://www.kaggle.com/datasets/needless090/neurogolf-onnx-v31` | Kaggle CLI read-only dataset lookup confirmed ref `needless090/neurogolf-onnx-v31`, title `NeuroGolf ONNX Solutions v31`, last updated `2026-04-19 06:02:30.047000` |

Total direct external rows：`228 + 6 + 2 = 236`.

## Source 1: Tiny ONNX Solver

Primary label:

```text
public:aliafzal9323_neurogolf-2026-tiny-onnx-solver
```

Kaggle URL:

```text
https://www.kaggle.com/code/aliafzal9323/neurogolf-2026-tiny-onnx-solver
```

Repo evidence:

- `S164_public_harvest_preflight_clean.md` records this as the base source.
- Captured public score：`4150.70`
- Local capture path：`artifacts/downloads/public_kernels_20260421/aliafzal9323_neurogolf-2026-tiny-onnx-solver/submission.zip`
- Local source zip SHA-256：`1aacb14c14af1357cd42200aad7ccb658a3f89519e552a98a6fab5dcd3d3030f`
- `KDA4013` manifest records the same source zip path for public fallback candidates.

Final included count:

- `221` rows as `public_kernel`
- `7` rows as `final_public_fallback_repair`
- total `228`

The 7 KDA4013 final fallback repairs are:

```text
task012, task017, task044, task058, task139, task243, task308
```

These 7 replacements are especially important because KDA4013 intentionally replaced newer local candidates with Tiny public fallback bytes. The local projection dropped by `-27.0345`, but public score improved by `+24.48`.

Tiny-attributed final tasks:

```text
task001, task002, task003, task006, task007, task010, task011, task012, task013, task016, task017, task019,
task020, task022, task026, task027, task028, task030, task031, task032, task036, task037, task038, task039,
task040, task041, task042, task043, task044, task045, task049, task051, task052, task056, task057, task058,
task059, task060, task061, task062, task063, task067, task068, task070, task071, task072, task074, task075,
task078, task082, task085, task087, task088, task091, task092, task093, task094, task095, task097, task098,
task100, task102, task103, task104, task105, task106, task108, task109, task111, task112, task113, task115,
task116, task117, task122, task123, task126, task128, task129, task131, task132, task134, task135, task136,
task137, task139, task140, task141, task143, task144, task146, task147, task150, task154, task155, task156,
task161, task162, task163, task165, task166, task168, task169, task171, task176, task177, task179, task181,
task186, task188, task190, task192, task193, task195, task196, task197, task198, task199, task200, task203,
task204, task211, task212, task215, task217, task221, task222, task223, task224, task226, task229, task230,
task235, task236, task239, task240, task241, task242, task243, task245, task246, task248, task249, task250,
task251, task252, task254, task256, task257, task258, task259, task260, task261, task262, task266, task267,
task270, task272, task273, task274, task276, task278, task281, task283, task287, task288, task289, task290,
task292, task293, task294, task295, task299, task300, task301, task302, task304, task306, task308, task309,
task312, task313, task314, task315, task316, task317, task318, task320, task322, task325, task326, task327,
task329, task331, task332, task334, task335, task337, task338, task341, task342, task345, task346, task348,
task350, task352, task353, task357, task361, task369, task371, task372, task374, task375, task376, task378,
task380, task381, task384, task386, task388, task390, task391, task393, task394, task398, task399, task400
```

## Source 2: Karnak Logic Complexity Map

Primary label:

```text
public:karnakbaevarthur_neurogolf-all-task-logic-complexity-map
```

Kaggle URL:

```text
https://www.kaggle.com/code/karnakbaevarthur/neurogolf-all-task-logic-complexity-map
```

Final included count：`6`

Final tasks:

```text
task083, task142, task152, task210, task311, task389
```

Important upstream note:

- `public-output-harvest-analysis-2026-04-21.md` says this notebook's main new value was primitive / complexity classification, not a new submission generator.
- That note also says its final submission output was exact-same zip as `svanikkolli/arc-nano-engine`.
- Therefore, for human attribution, this is probably a Karnak-labeled final row with an `arc-nano-engine` upstream relationship.

Related likely upstream URL:

```text
https://www.kaggle.com/code/svanikkolli/arc-nano-engine
```

## Source 3: Needless ONNX v31 Dataset

Primary label:

```text
public_dataset:needless090_neurogolf-onnx-v31
```

Kaggle URL:

```text
https://www.kaggle.com/datasets/needless090/neurogolf-onnx-v31
```

Final included count：`2`

Final tasks:

```text
task164, task172
```

Repo evidence:

- `leaderboard-gap-research-2026-04-21.md` records a dataset-level harvest including `needless090/neurogolf-onnx-v31`.
- `S166_public_dataset_harvest_dryrun.md` records `task064`, `task164`, and `task172` as the overlay candidates from this dataset.
- In the final KDA4013 task table, only `task164` and `task172` remain attributed to `public_dataset:needless090_neurogolf-onnx-v31`.
- `task064` is no longer final-attributed to Needless; final source is `S207G_task064_compact24_v1`.

Small inconsistency to keep in mind:

- `S166_public_dataset_harvest_dryrun.md` says `task164` / `task172` replacements had new cost `36000`.
- The final release table records both rows as cost `37440`.
- For public attribution, source label and SHA are more important than this cost discrepancy. If cost is mentioned externally, re-profile or cite only the final table.

## Public Harvest Sources Considered But Not Final Primary Sources

The repo also records these high-score public sources during public-output harvest:

| source | URL | captured / recorded role | final primary source rows |
| --- | --- | --- | ---: |
| `artemnazemtsev/4k-neurogolf-logic-driven-ensembling` | `https://www.kaggle.com/code/artemnazemtsev/4k-neurogolf-logic-driven-ensembling` | public output comparison; second-layer blend source | `0` |
| `jazivxt/infinitesimals` | `https://www.kaggle.com/code/jazivxt/infinitesimals` | public output comparison; Tiny / Artem cleanup route | `0` |
| `sigmaborov/neurogolf-2026-starter` | `https://www.kaggle.com/code/sigmaborov/neurogolf-2026-starter` | public output comparison; dataset wrapper source | `0` |
| `svanikkolli/arc-nano-engine` | `https://www.kaggle.com/code/svanikkolli/arc-nano-engine` | public output comparison; upstream of Karnak-labeled output | `0` as direct label |
| `mirzamilanfarabi/arc-nano-engine-submission` | `https://www.kaggle.com/code/mirzamilanfarabi/arc-nano-engine-submission` | public output comparison; replay of `arc-nano-engine` | `0` |
| `hanifnoerrofiq/the-epic-sax-ensembler` | `https://www.kaggle.com/code/hanifnoerrofiq/the-epic-sax-ensembler` | checked in free-lunch sprint, lower captured score than 4k sources | `0` |

These sources may appear as aliases in KDA4011 / KDA4013 manifest context for public fallback candidates, but they are not final primary attribution labels in `task_by_task_release_table.csv`.

## Not Counted As External Source Rows

These categories exist in the final table but are not counted as external source rows here:

- `local_generated`
- `local_overlay:*`
- `inherited_submission`
- `confirmed_public_reliability_repair`
- `dropped_public_zero_loss`
- `unknown`

The `confirmed_public_reliability_repair` rows are local candidates selected because public evidence beat newer local candidates:

```text
task025 -> CI190_task025_wave14_compression
task205 -> M172_task205_cost_v3
```

They are public-confirmed, but their final source bytes are not attributed to an external public notebook or dataset.

The `dropped_public_zero_loss` rows are not in the final zip:

```text
task157, task158, task368, task370
```

They should not be counted as external source usage in the submitted artifact.

## Current Completeness Assessment

What is clear:

- Final row-level source attribution is clear enough for internal analysis.
- Direct external source counts and task ids are recoverable from the final release table.
- Tiny source is the dominant external source, with `228` final rows.
- The two dataset rows from Needless are explicitly identified.
- Karnak-labeled rows are explicitly identified, with an upstream caveat pointing to `svanikkolli/arc-nano-engine`.

What is not fully complete yet:

- Exact Kaggle code version ids for every public notebook are not recorded in the final task table.
- License / redistribution basis is not complete for external model bytes.
- The KDA4013 provenance template still contains `TBD_MANUAL_VERIFY` for the 7 public fallback replacements.
- Current M1 Kaggle CLI returned `401 Unauthorized` for kernel listing queries, so code-page URL confirmation should be manually checked in a browser before a public attribution post. Dataset lookup for `needless090/neurogolf-onnx-v31` did work.

## Practical Attribution Text

If you need a short internal summary before editing the Kaggle post:

```text
The 4743.93 pack is a mixed-lineage submission. Of the 394 ONNX files in the submitted zip, 236 final rows are directly attributed to public Kaggle sources or datasets in the task table. The dominant source is aliafzal9323/neurogolf-2026-tiny-onnx-solver, which accounts for 228 final rows including the 7 KDA4013 public fallback repairs. Six rows are attributed to karnakbaevarthur/neurogolf-all-task-logic-complexity-map, with repo notes indicating an arc-nano-engine upstream relationship. Two rows are attributed to needless090/neurogolf-onnx-v31. The remaining rows are local generated, local overlays, inherited internal submissions, or other non-external categories.
```

For a public post, do not imply that these external bytes are self-generated. Keep source attribution visible.
