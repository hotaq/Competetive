# Task Table Schema

`task_by_task_release_table.csv` has one row for every official task id from `task001` to `task400`.

## Pack Counts

- Included tasks: `394`
- Dropped after public-confirmed zero-loss checks: `4`
- Not included in the final pack: `2`
- Kaggle result: `SubmissionStatus.COMPLETE`
- Public score: `4743.93`
- Submitted artifact: `submission.zip`

## Columns

| column | meaning |
| --- | --- |
| `task_id` | Official task id |
| `final_pack_status` | Whether the task is included, dropped, or not included |
| `source_layer` | Final byte lineage category |
| `source_name` | Selected source label |
| `zip_member` | ONNX filename inside `submission.zip` for included tasks |
| `cost` | Local profile cost where available |
| `score_points` | Formula-derived local points where available |
| `primary_family` | Family or branch label where available |
| `implementation_pattern` | Short description of how to interpret the selected byte |
| `validation_evidence` | Local or pack-level validation evidence |
| `public_evidence` | Kaggle public result evidence where available |
| `risk_or_caveat` | Public/local, provenance, or exclusion caveat |
| `sha256` | ONNX file hash for included tasks |

## Source Layer Counts

| source layer | rows |
| --- | ---: |
| `public_kernel` | `227` |
| `local_generated` | `85` |
| `inherited_submission` | `44` |
| `final_public_fallback_repair` | `7` |
| `local_overlay:r1` | `5` |
| `local_overlay:color` | `4` |
| `local_overlay:diagonal` | `1` |
| `local_overlay:geometry` | `4` |
| `local_overlay:periodic_template` | `2` |
| `local_overlay:r2` | `3` |
| `local_overlay:recovery` | `3` |
| `local_overlay:subgrid` | `4` |
| `local_overlay:unknown` | `1` |
| `confirmed_public_reliability_repair` | `2` |
| `dropped_public_zero_loss` | `4` |
| `public_dataset` | `2` |
| `unknown` | `2` |

## Final-Lineage Task Groups

Public fallback repairs:

```text
task012, task017, task044, task058, task139, task243, task308
```

Confirmed public reliability repairs:

```text
task025, task205
```

Dropped after public-confirmed zero-loss checks:

```text
task157, task158, task368, task370
```

