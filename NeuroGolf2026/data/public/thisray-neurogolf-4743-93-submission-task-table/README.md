# NeuroGolf 4743.93 Submission Zip And Task Table

This Dataset contains the submitted ONNX zip and task-by-task metadata behind a NeuroGolf submission that reached **4743.93** public score.

The goal is simple: make the pack usable and inspectable. The zip provides the submitted model files. The table shows how each task row is represented in the final pack.

## Files

| file | description |
| --- | --- |
| `submission.zip` | Submitted ONNX pack for the 4743.93 public score |
| `task_by_task_release_table.csv` | 400-row table with final status, source layer, zip member, cost, evidence, and caveats |
| `release_summary.json` | Compact public result summary |
| `validation_method.md` | Validation ladder and final-pack evidence |
| `task_table_schema.md` | Column guide for the task table |
| `known_caveats.md` | Public/local and scorer/profile caveats |
| `artifact_manifest.csv` | Dataset file list |
| `README.md` | This Dataset overview |

## Public result

- Kaggle status: `SubmissionStatus.COMPLETE`
- Public score: `4743.93`
- Public delta versus previous public anchor: `+24.48`
- Submitted artifact: `submission.zip`
- ONNX files in the zip: `394`
- `submission.zip` SHA-256: `28f34d909fd570da918c47d37213349549889748038439a34161d492f7756633`
- Task table rows: `400`
- Included tasks: `394`
- Dropped after public-confirmed zero-loss checks: `4`
- Not included in the final pack: `2`

## Main table

`task_by_task_release_table.csv` has one row for every official task id from `task001` to `task400`.

Key columns:

- `final_pack_status`
- `source_layer`
- `source_name`
- `zip_member`
- `cost`
- `score_points`
- `primary_family`
- `implementation_pattern`
- `validation_evidence`
- `public_evidence`
- `risk_or_caveat`
- `sha256`

## How to use

Use the table to answer questions such as:

- Which tasks are included in the 4743.93 pack?
- Which ONNX file corresponds to each included task?
- Which rows came from public kernels, local generated models, inherited submissions, local overlays, or repairs?
- Which local candidates did not turn into public score?
- Which rows carry reliability or provenance caveats?
- Which source layers dominate the pack?

## Scope

This Dataset does not include raw official task JSON files or local development logs. It is a release bundle for a submitted zip, task-level metadata, validation notes, and caveats.

