# Metric-v2 20+ Feasibility

- Source selected report: `generated/task004_corner_shift_v86c/metric_v2_selected.csv`
- Scored tasks in current local best: `394`
- Current local metric-v2 score: `5538.983066`
- Average over 400 tasks: `13.847458`
- `20+` hard cost threshold: `cost <= 148`
- `19+` hard cost threshold: `cost <= 403`
- `18+` hard cost threshold: `cost <= 1096`

## Current Counts

- Tasks at `23+`: `2`
- Tasks at `22+`: `6`
- Tasks at `21+`: `14`
- Tasks at `20+`: `14`
- Tasks at `19+`: `15`
- Tasks at `18+`: `19`
- Tasks at `17+`: `30`
- Tasks at `16+`: `51`
- Tasks at `15+`: `78`
- Tasks at `14+`: `142`
- Tasks at `13+`: `272`
- Tasks at `12+`: `357`
- Tasks at `11+`: `394`

## Practical Conclusion

Clean `20+` requires total `params + static-memory + MACs <= 148`. Under the April 28 metric update, one ordinary 30x30 float activation is already far above this. So the only repeatable clean 20+ family is a graph that returns the output almost directly: channel `Gather`, direct geometric `Transpose`, or an equally tiny no-intermediate transform.

For tasks that need object detection, masks, padding, reductions, or convolution, the clean target should be `13-18` first, then `18-19` for tiny crop/pad cases. Trying to force every task to 20+ cleanly is not a good work queue; it implies a new scoring loophole or a radically different representation.

## 20+ Factory Lanes

1. `channel_gather`: same-shape color remap or channel permutation. Current cost floor: `10-30`, score `21.6-22.7`.
2. `direct_transpose`: exact full-grid transpose/rotation-like tasks with no pad/slice body. Current cost: `5`, score `23.39` locally.
3. `tiny_crop_pad`: near miss only. It lands around `18-19.6` because `Pad/Slice` static tensors still count.
4. `logic_heavy`: Conv/Reduce/Where/MatMul tasks. These are not 20+ candidates under clean metric-v2.

## Fresh Direct-Family Rescan

- Single-axis `Gather` scan found `12` tasks; missed by current selected 20+ set: `0`.
- Gather hits: `task016:axis1/n10`, `task053:axis2/n30`, `task113:axis2/n30`, `task116:axis2/n30`, `task164:axis3/n30`, `task172:axis2/n30`, `task210:axis2/n30`, `task276:axis1/n10`, `task309:axis1/n10`, `task311:axis3/n30`, `task337:axis1/n10`, `task385:axis2/n30`
- Exact geometric scan found `7` non-identity tasks.
- Geometric hits: `task087:rot180`, `task140:rot180`, `task150:flip_h`, `task155:flip_v`, `task179:transpose`, `task241:transpose`, `task380:rot270`
- Only direct `transpose` naturally stays output-direct on the padded tensor. Flips and rotations need crop/pad or a small intermediate to keep the valid grid at the top-left, which already pushes cost above the `20+` threshold.

## Lane Counts

- `17to19_crop_or_small_static`: `17`
- `current_20plus_channel_gather`: `12`
- `current_20plus_direct_transpose`: `2`
- `near20_tiny_crop_pad`: `7`
- `not_20plus_logic_heavy`: `322`
- `not_20plus_other`: `34`

## Current 20+ Tasks

| task | cost | points | lane | ops |
| --- | ---: | ---: | --- | --- |
| task179 | 5 | 23.390562 | current_20plus_direct_transpose | `Constant;Transpose` |
| task241 | 5 | 23.390562 | current_20plus_direct_transpose | `Constant;Transpose` |
| task016 | 10 | 22.697415 | current_20plus_channel_gather | `Gather` |
| task276 | 10 | 22.697415 | current_20plus_channel_gather | `Gather` |
| task309 | 10 | 22.697415 | current_20plus_channel_gather | `Gather` |
| task337 | 10 | 22.697415 | current_20plus_channel_gather | `Gather` |
| task053 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task113 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task116 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task164 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task172 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task210 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task311 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |
| task385 | 30 | 21.598803 | current_20plus_channel_gather | `Gather` |

20+ op patterns:
- `Gather`: `12`
- `Constant;Transpose`: `2`

## Closest Numeric Near Misses

| task | cost | points | lane | ops |
| --- | ---: | ---: | --- | --- |
| task326 | 220 | 19.606372 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task135 | 420 | 18.959745 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task087 | 520 | 18.746171 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task140 | 520 | 18.746171 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task380 | 1089 | 18.006985 | near20_tiny_crop_pad | `Gather;Pad;Slice;Transpose` |
| task399 | 1258 | 17.862722 | near20_tiny_crop_pad | `Cast;Concat;Greater;Mul;Pad;ReduceSum;Slice;Sub;Sum` |
| task266 | 1386 | 17.765823 | near20_tiny_crop_pad | `Concat;Pad;Slice;Sub;Sum` |

## Wider 17-19 Static Queue

| task | cost | points | lane | ops |
| --- | ---: | ---: | --- | --- |
| task326 | 220 | 19.606372 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task135 | 420 | 18.959745 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task087 | 520 | 18.746171 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task140 | 520 | 18.746171 | near20_tiny_crop_pad | `Constant;Pad;Slice` |
| task380 | 1089 | 18.006985 | near20_tiny_crop_pad | `Gather;Pad;Slice;Transpose` |
| task399 | 1258 | 17.862722 | near20_tiny_crop_pad | `Cast;Concat;Greater;Mul;Pad;ReduceSum;Slice;Sub;Sum` |
| task266 | 1386 | 17.765823 | near20_tiny_crop_pad | `Concat;Pad;Slice;Sub;Sum` |
| task395 | 1564 | 17.644998 | 17to19_crop_or_small_static | `Add;Mul;Pad;Slice;Sub` |
| task347 | 1618 | 17.611054 | 17to19_crop_or_small_static | `Add;Cast;Greater;Mul;Pad;Slice;Sub` |
| task257 | 1749 | 17.533201 | 17to19_crop_or_small_static | `Concat;Mul;Pad;Slice;Sub` |
| task322 | 1757 | 17.528637 | 17to19_crop_or_small_static | `Cast;Concat;CumSum;Greater;Pad;ReduceMax;Slice;Sub` |
| task373 | 1936 | 17.431621 | 17to19_crop_or_small_static | `Add;Mul;Pad;Slice` |
| task262 | 2108 | 17.346505 | 17to19_crop_or_small_static | `Expand;Mul;Pad;Slice;Sum` |
| task006 | 2368 | 17.230199 | 17to19_crop_or_small_static | `Add;Mul;Pad;ReduceMax;Slice;Sub` |
| task052 | 2388 | 17.221789 | 17to19_crop_or_small_static | `Add;Cast;Constant;Equal;Mul;Pad;ReduceMax;ReduceSum;Slice;Sub` |
| task227 | 2740 | 17.084287 | 17to19_crop_or_small_static | `Add;Mul;Pad;Slice;Sub` |
| task229 | 3042 | 16.979730 | 17to19_crop_or_small_static | `Add;ArgMax;Cast;Constant;Equal;Mul;Pad;ReduceSum;Reshape;Slice;Sub` |
| task386 | 3262 | 16.909904 | 17to19_crop_or_small_static | `Add;Mul;Pad;ReduceMax;Slice;Sub` |
| task223 | 3628 | 16.803563 | 17to19_crop_or_small_static | `Pad;Resize;Slice` |
| task207 | 3771 | 16.764905 | 17to19_crop_or_small_static | `Add;ArgMax;Cast;Concat;Constant;Gather;Greater;Mul;Neg;Pad;ReduceSum;Resh...` |
| task123 | 4334 | 16.625754 | 17to19_crop_or_small_static | `Concat;Gather;Greater;Max;ReduceSum;Slice;Squeeze;Where` |
| task142 | 4440 | 16.601590 | 17to19_crop_or_small_static | `Concat;Constant;Gather;Pad;Slice` |
| task152 | 4440 | 16.601590 | 17to19_crop_or_small_static | `Concat;Constant;Gather;Pad;Slice` |
| task001 | 4673 | 16.550443 | 17to19_crop_or_small_static | `Cast;Concat;Constant;Gather;Or;Pad;ReduceMax;Slice;Tile;Where` |

## Next Work Queue

1. Re-scan all tasks for exact `Gather` channel remaps and replace any missed same-shape color-map tasks.
2. Re-scan geometric tasks for direct output `Transpose`/`Gather` models that avoid `Pad` entirely.
3. Inspect near misses `task326`, `task135`, `task087`, `task140`, and `task380` for removable constants or output-direct rewrites.
4. Keep complex tasks on the clean 13-18 optimization lane; task004-style object logic will not become 20+ without a new primitive.

