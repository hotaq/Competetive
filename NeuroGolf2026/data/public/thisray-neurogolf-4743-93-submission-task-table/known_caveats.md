# Known Caveats

This release is for artifact use and task-table inspection. It is not a claim that every task row is a clean semantic solver.

## Mixed Lineage

The final pack contains multiple source layers:

- public-kernel rows
- local-generated rows
- inherited-submission rows
- local-overlay rows
- public reliability repairs
- dropped rows

The task table keeps these layers visible so the submitted zip can be inspected task by task.

## Public/Local Gap

Local estimates are useful, but they are not complete substitutes for public scoring.

Some candidates can pass visible examples and still be risky:

- visible-example lookup
- scalar signatures tied closely to official examples
- replay-like models
- reference-matcher candidates
- standalone dry-run candidates with unclear semantic generalization

The table keeps these caveats visible instead of hiding them behind a single score.

## Scorer/Profile Interpretation

There is an active discussion around dynamic shapes and `onnx_tool` shape inference behavior:

https://www.kaggle.com/competitions/neurogolf-2026/discussion/693950

This release treats the issue as a scorer/profile caveat, not as an accusation about any participant.

For cost claims, it is useful to keep these separate:

- semantic correctness
- ONNX Runtime compatibility
- `onnx_tool` profile result
- rule and validator compliance

