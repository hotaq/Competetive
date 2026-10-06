# Official-Zero Live Pool v110 Summary - 2026-05-01

## Live leaderboard context

- Leaderboard snapshot: `data/live/leaderboard_20260501T051649Z/neurogolf-2026.zip` (`neurogolf-2026-publicleaderboard-2026-05-01T05:16:49.csv`).
- Rank 1: `GPT or Claude?` score `6830.80`, last submission `2026-05-01 02:45:10`.
- Rank 2: `Chris Deotte` score `6743.22`, last submission `2026-05-01 05:06:50`.
- Rank 3: `Pavel` score `6374.66`, last submission `2026-05-01 05:15:11`.
- Rank 4: `ymg_aq` score `6272.91`, last submission `2026-05-01 04:23:10`.
- Rank 5: `Hajime Kato` score `6238.67`, last submission `2026-04-29 11:38:29`.
- Current team `why not sup` rank `53`, public score `5038.80`, submissions `100`.

## What changed locally

- Patched `tools/metric_v2_select.py` to match the latest official `neurogolf_utils.py` scorer by default:
  - no extra `Constant` node tensor penalty beyond `onnx_tool` `graph.params`;
  - zero-cost valid graphs score with `25 - log(max(1, cost))`, so `cost=0` gets `25`;
  - optional `--count-constant-node-params` keeps the older conservative local mode.
- Downloaded current public notebook outputs into `data/kernel_outputs_live_20260501/`.

## Downloaded live outputs

| Zip | ONNX task files |
| --- | ---: |
| `data/kernel_outputs_live_20260501/neurogolf-2026-may-1st-update-not-using-exploit/submission.zip` | 399 |
| `data/kernel_outputs_live_20260501/neurogolf-6645-39-public-score-open-solution/submission.zip` | 401 |
| `data/kernel_outputs_live_20260501/neurogolf-logic-driven-ensembe/submission.zip` | 400 |
| `data/kernel_outputs_live_20260501/neurogolf-logic-driven-ensembling-part-4/submission.zip` | 401 |
| `data/kernel_outputs_live_20260501/neurogolf-multi-source-onnx-solver/submission.zip` | 400 |
| `data/kernel_outputs_live_20260501/neurogolf-update-6500-part-2/submission.zip` | 400 |
| `data/kernel_outputs_live_20260501/neurogolf-update-logic-driven-ensembling-part-5/submission.zip` | 401 |
| `data/kernel_outputs_live_20260501/ngc26-constraint-smart-logic-mix-blending/sub4971-98.zip` | 399 |
| `data/kernel_outputs_live_20260501/ngc26-constraint-smart-logic-mix-blending/submission.zip` | 401 |
| `data/kernel_outputs_live_20260501/update-neurogolf-logic-driven-ensembling-part-1/submission.zip` | 401 |

## v110 result

- Current submitted artifact official local rescore: `5508.165922` across `390` tasks.
- v110 official local score: `5568.889305` across `394` tasks.
- Local gain vs current submitted artifact: `+60.723383`.
- Output zip: `generated/official_zero_live_v110/submission.zip`.
- Selected CSV: `reports/official_zero_live_v110_selected.csv`.
- Full audit CSV: `reports/official_zero_live_v110_all.csv`.

## Changed tasks vs current submitted v104

| Task | Old cost | New cost | Old pts | New pts | Delta | Ops |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `task011` | missing | 122418 | 0.000000 | 13.284803 | +13.284803 | `Cast;Constant;Conv;Mul;Pad;ReduceMax;Relu;Slice;Sub;Sum` |
| `task016` | 10 | 10 | 22.697415 | 22.697415 | +0.000000 | `Gather` |
| `task037` | missing | 477987 | 0.000000 | 11.922661 | +11.922661 | `Add;Cast;Concat;Constant;Conv;Mul;Pad;ReduceMax;ReduceSum;Relu;Slice;Sub` |
| `task103` | 44496 | 41095 | 14.296845 | 14.376358 | +0.079513 | `Cast;Constant;Gather;Less;Mul;ReduceSum;Reshape;Slice;Unsqueeze` |
| `task141` | 256391 | 154684 | 12.545541 | 13.050860 | +0.505319 | `Abs;Add;Cast;Equal;Mul;ReduceSum;Slice;Sub` |
| `task179` | 4 | 0 | 23.613706 | 25.000000 | +1.386294 | `Transpose` |
| `task190` | 327300 | 78994 | 12.301368 | 13.722873 | +1.421505 | `Add;Cast;Concat;Conv;Greater;Identity;Mul;Pad;ReduceMax;ReduceSum;Slice` |
| `task220` | 187356 | 91951 | 12.859234 | 13.570989 | +0.711755 | `Add;Cast;Concat;Conv;Gather;Greater;Mul;Pad;ReduceSum;Sub` |
| `task241` | 4 | 0 | 23.613706 | 25.000000 | +1.386294 | `Transpose` |
| `task274` | missing | 158904 | 0.000000 | 13.023944 | +13.023944 | `Cast;Concat;Conv;Equal;Greater;Mul;ReduceSum;Sum` |
| `task285` | 192656 | 192656 | 12.831339 | 12.831339 | +0.000000 | `ArgMax;Cast;Concat;Flatten;Gather;Less;MatMul;Mul;ReduceMax;ReduceSum;Reshape;ScatterElements;Slice;Sub;Unsqueeze` |
| `task314` | 209128 | 18092 | 12.749298 | 15.196775 | +2.447477 | `Add;Concat;Mul;Pad;ReduceSum;Slice;Sub` |
| `task335` | 282639 | 121205 | 12.448074 | 13.294761 | +0.846687 | `Cast;Concat;Conv;Equal;Less;Max;Mul;ReduceSum;Slice;Sub` |
| `task337` | 10 | 10 | 22.697415 | 22.697415 | +0.000000 | `Gather` |
| `task352` | 74805 | 66616 | 13.777360 | 13.893300 | +0.115940 | `Add;Cast;Concat;Conv;Greater;Mul;Slice;Sub` |
| `task374` | missing | 90112 | 0.000000 | 13.591191 | +13.591191 | `Add;Cast;Concat;Constant;Max;Min;Mul;Pad;ReduceMax;Relu;Slice;Sub` |

## Candidate audit reason distribution

| Reason | Count |
| --- | ---: |
| `ok` | 1892 |
| `banned_op` | 1210 |
| `model_functions_disallowed` | 982 |
| `non_expected_io_shape` | 765 |
| `first mismatch example 0` | 551 |
| `error:dynamic_or_missing_shape` | 60 |
| `example_grid_exceeds_30` | 29 |
| `error:only 0-dimensional arrays can be converted to Python scalars` | 12 |
| `error:Graph must be in single static assignment (SSA) form, however 'sh_900' has been used as output names multiple times.` | 5 |
| `error:[ONNXRuntimeError] : 1 : FAIL : /Users/cloudtest/vss/_work/1/s/onnxruntime/core/graph/graph.cc:1494 void onnxruntime::Graph::InitializeStateFromModelFileGraphProto() This is an invalid model. Graph output (output) does n` | 5 |
| `first mismatch example 11` | 4 |
| `first mismatch example 1` | 3 |
| `error:'list' object has no attribute 'reshape'` | 2 |
| `error:sh_900 initializer name is not unique` | 2 |
| `error:The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` | 2 |
| `error:invalid broadcast shapes, dimensions (30, 60) are incompatible` | 1 |
| `error:Nodes in a graph must be topologically sorted, however input 'cc_axis3' of node:  name:  OpType: CumSum  is not output of any previous nodes.` | 1 |
| `error:dw_ones_7_h_sep initializer name is not unique` | 1 |

## Safety audits

- Loophole audit: `reports/official_zero_live_v110_loophole.csv`; `0` functions, custom-domain nodes, and IO/initializer collisions.
- Metric-change audit: `reports/official_zero_live_v110_metric_change_risk.csv`; `0` banned-op tasks, `0` dynamic-shape tasks, `394` clean static-runtime tasks.
- Zip sanity: `394` `taskNNN.onnx` files; file is named `submission.zip` and is about `1.1M`.

## 6800 implication

- v110 is the best local, official-current, no-Function/no-Compress artifact prepared in this pass.
- It is submission-ready locally, but it is still far from `6800` by local score. The current public score is `5038.80`; even if the local `+60.72` transferred perfectly, the public gap to `6800` remains roughly `+1700`.
- Public high-score notebooks downloaded on 2026-05-01 mostly rely on `model.functions` or stale scorer surfaces; the current official scorer returns `model_functions_disallowed`, so they are not a stable direct path.
