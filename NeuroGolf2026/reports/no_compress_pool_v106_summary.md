# No-Compress Pool v106 Summary - 2026-05-01

## Purpose

Rebuild the current local NeuroGolf selection after the 2026-04-30 official scorer update added `COMPRESS` to the banned op list.

## Inputs

- Updated official data zip copied to `data/latest/neurogolf-2026.zip` from `data/live/dataset_20260501T050204Z/neurogolf-2026.zip`.
- Patched local strict scorer banned ops in `tools/metric_v2_select.py` and `tools/audit_metric_change_risk.py` to include `Compress` / `COMPRESS`.
- Candidate pool: `78` valid local public/kernel/dataset submission zips listed in `reports/no_compress_candidate_zips_v106.txt`.
- Base zip: `generated/metric_v2_v104_compress_ban_v105/submission.zip`.

## Results

- Compress-ban base local score: `5507.151361` across `390` tasks.
- Pool-selected local score: `5558.946955` across `394` tasks.
- Local gain vs compress-ban base: `+51.795594`.
- Non-base selected tasks: `5`.
- Output zip: `generated/no_compress_pool_v106/submission.zip`.
- Full audit CSV: `reports/no_compress_pool_v106_all.csv`.
- Selected CSV: `reports/no_compress_pool_v106_selected.csv`.

## Non-base selected tasks

| Task | Source | Cost | Points | Ops |
| --- | --- | ---: | ---: | --- |
| `task011` | `konbu17-neurogolf-2026-4959-61-4-28-update-version` | 124761 | 13.265845 | `Cast;Constant;Conv;Mul;Pad;ReduceMax;Relu;Slice;Sub;Sum` |
| `task037` | `magmacot-4213-neurogolf-blending-v1-new-metric` | 477998 | 11.922638 | `Add;Cast;Concat;Constant;Conv;Mul;Pad;ReduceMax;ReduceSum;Relu;Slice;Sub` |
| `task274` | `konbu17-neurogolf-2026-blended-341` | 158904 | 13.023944 | `Cast;Concat;Conv;Equal;Greater;Mul;ReduceSum;Sum` |
| `task285` | `afr1ste-neurogolf-6323-15-controlled-public-artifact` | 192656 | 12.831339 | `ArgMax;Cast;Concat;Flatten;Gather;Less;MatMul;Mul;ReduceMax;ReduceSum;Reshape;ScatterElements;Slice;Sub;Unsqueeze` |
| `task374` | `konbu17-neurogolf-2026-4959-61-4-28-update-version` | 90838 | 13.583167 | `Add;Cast;Concat;Constant;Max;Min;Mul;Pad;ReduceMax;Relu;Slice;Sub` |

## Selected source distribution

| Source | Tasks | Points |
| --- | ---: | ---: |
| `base` | 389 | 5494.320022 |
| `konbu17-neurogolf-2026-4959-61-4-28-update-version` | 2 | 26.849012 |
| `magmacot-4213-neurogolf-blending-v1-new-metric` | 1 | 11.922638 |
| `konbu17-neurogolf-2026-blended-341` | 1 | 13.023944 |
| `afr1ste-neurogolf-6323-15-controlled-public-artifact` | 1 | 12.831339 |

## Candidate audit reason distribution

| Reason | Count |
| --- | ---: |
| `ok` | 1775 |
| `non_expected_io_shape` | 692 |
| `first mismatch example 0` | 549 |
| `banned_op` | 522 |
| `error:dynamic_or_missing_shape` | 58 |
| `example_grid_exceeds_30` | 27 |
| `error:only 0-dimensional arrays can be converted to Python scalars` | 12 |
| `error:Graph must be in single static assignment (SSA) form, however 'sh_900' has been used as output names multiple times.` | 4 |
| `first mismatch example 11` | 4 |
| `error:non_positive_cost` | 4 |
| `first mismatch example 1` | 3 |
| `error:'list' object has no attribute 'reshape'` | 2 |

## Safety audits

- Loophole audit: `reports/no_compress_pool_v106_loophole.csv`; `0` functions, custom-domain nodes, and IO/initializer collisions.
- Metric-change audit: `reports/no_compress_pool_v106_metric_change_risk.csv`; `0` banned-op tasks, `0` runtime-zero bad tasks, `394` clean static runtime tasks.
- Selected rows contain no `Compress` op.

## 6800 implication

This strict clean static lane is safe but not enough for 6800 by itself. The immediate no-Compress repair recovers the four invalid Compress tasks and raises the strict local score to ~5558.95, but the current public team score remains 5038.80 until a new Kaggle submission is made. Reaching 6800 still requires either a much stronger public-compatible base artifact or many public-validated task-level gains; continue with one-file/small-batch public probes rather than trusting local-only deltas.
