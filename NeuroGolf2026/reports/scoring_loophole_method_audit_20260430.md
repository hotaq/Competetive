# Scoring Loophole Method Audit - 2026-04-30

## Live leaderboard signal

Downloaded leaderboard:

- `data/live/leaderboard_20260430/neurogolf-2026.zip`
- Snapshot time from Kaggle CSV: `2026-04-30T08:36:40`

Top rows:

| rank | team | score | submissions | note |
| ---: | --- | ---: | ---: | --- |
| 1 | See Discussion | 9975.00 | 5 | exact multiple-of-25 surface |
| 2 | waiting for rescoring | 9850.00 | 261 | exact multiple-of-25 surface |
| 3 | Ali | 9725.00 | 205 | exact multiple-of-25 surface |
| 4 | Jack Su | 9475.00 | 65 | exact multiple-of-25 surface |
| 5 | Jiyang Qiu | 9399.64 | 449 | mixed score |
| 15 | tyuyoshi | 7115.54 | 272 | first non-9k cluster |
| 19 | GPT or Claude? | 6737.48 | 76 | likely public/open blending plus clean/near-clean work |

The 9k+ cluster is not shaped like ordinary clean graph optimization. Scores such as `9975.00` imply about `399` tasks at exactly `25.00`, or a very similar near-perfect scoring surface.

## Method identified from discussion 695638

Thread:

- `https://www.kaggle.com/competitions/neurogolf-2026/discussion/695638`
- Title: `Agent-Written Report: NeuroGolf 2026 Scoring Loophole Disclosure`
- Author/team context: Jiyang Qiu, currently rank 5 in the competition at the time of reading.

The reported pattern:

- Main graph has only a cheap custom-domain wrapper node such as `custom_domain::Where(cond, input, zero) -> output`.
- The model also defines an ONNX `FunctionProto` named `Where` in that custom domain.
- ONNX Runtime executes the function body normally, so the model can still compute the real task.
- The scorer appears to profile only the visible main graph node and does not charge the computation, tensors, or parameters inside the function body.
- Reported single-task probe score: `25.00`.
- Reported local scorer tuple for wrapped existing models: `macs=0`, `memory=0`, `params=1`, cost `1`, points `25.00`.

This explains the leaderboard jump much better than any clean `20+ every task` hypothesis.

## Why this is not a clean path

The competition overview already disallows `Function`. The April 30 organizer follow-up in our notes also says they intend to reject models that use `model.functions` and models whose graph input/output names collide with initializer names.

So the FunctionProto wrapper is a scorer/validator loophole, not a durable clean NeuroGolf method.

## Related loophole families

Previously observed scorer/idiosyncrasy families:

- `Compress` fake-zero shape profiling: `onnx_tool.Tensor.get_numpy()` fabricates zeros for unknown tensors, making value-dependent outputs profile as zero-sized.
- Scalar `volume([]) == 0`.
- Constants not counted before the April 28 metric update.
- Custom-domain / `FunctionProto` body not profiled.
- Model-local functions and graph IO/initializer name collisions are now explicit patch targets.

## Current local guard result

Auditor:

- `tools/audit_onnx_loophole_risks.py`
- output: `reports/onnx_loophole_risk_audit_20260430.csv`

Checked:

- `generated/task004_corner_shift_v86c/submission.zip`
- `generated/metric_v2_v84_plus_latin_formula_v85/submission.zip`
- `data/public/artemnazemtsev-task1-hacked-submission-25-00/submission.zip`

Result across `789` audited ONNX files:

- models with `model.functions`: `0`
- models with custom-domain nodes: `0`
- graph input/output initializer-name collisions: `0`

This keeps the current clean submission out of the newly reported FunctionProto risk class. The older public `task001` hacked zip does not use FunctionProto; it belongs to the earlier `Compress`/fake-zero/value-info family instead.

## Work decision

Do not spend effort trying to make every task `20+` by ordinary graph optimization. The leaderboard evidence says the 9k+ surface is coming from scoring loopholes, especially FunctionProto wrapping.

For stable competition work:

1. Keep the clean lane: public/open-source blend plus task-specific compact ONNX.
2. Use loophole reports only as validator-risk intelligence.
3. If we need to understand a top score, build audit tooling that detects `model.functions`, custom domains, dynamic-shape profile tricks, and IO/initializer collisions.
4. Continue score work around the real public cluster near `6600-7100`, not the temporary exact-25 leaderboard surface.
