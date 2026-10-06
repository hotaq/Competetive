# 23+ Factory Queue

Cost file: `reports/task292_two_stage_compact_task_costs.csv`
Candidate compare: `reports/candidate_compare_after_batch8.csv`

## Score Ceiling Math

- `23+` requires `cost <= exp(2) = 7.389056`, so integer cost must be `<= 7`.
- `20+` requires `cost <= exp(5) = 148.413159`, so integer cost must be `<= 148`.
- Current local estimated total: `6505.131036`.
- Gap to `20 * 400 = 8000`: `1494.869` points.
- Gap to `23 * 400 = 9200`: `2694.869` points.
- Best current static cost: `28` -> `21.668` points.
- Best current any-shape cost: `17` -> `22.167` points.
- Clean `23+` is below the current static floor by `x3.8` cost on the best task.

Interpretation: a clean all-task `23+` push is not a per-file optimization problem. It would require a new sub-8-cost representation for whole families. Anything that reports hundreds of `25.00` tasks via zero-volume tensors or uncounted constants belongs to metric-bug territory, not this clean factory lane.

## Current Coverage

- Total tasks: `400`
- Locally valid rows: `399`
- Scored rows: `399`
- Static `1x10x30x30` scored rows: `287`
- Non-static scored rows: `112`

| Threshold | All Tasks | Static Tasks |
| ---: | ---: | ---: |
| >= 23 | 0 | 0 |
| >= 22 | 30 | 0 |
| >= 21 | 47 | 2 |
| >= 20 | 69 | 8 |
| >= 19 | 82 | 13 |
| >= 18 | 91 | 17 |
| >= 17 | 100 | 19 |
| >= 16 | 114 | 22 |
| >= 15 | 292 | 197 |

## Best Current Costs

| Task | Bucket | Cost | Points | Static | Families | Ops |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task067 | A_exact | 17 | 22.167 | 0 | crop_exact_same_offset;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Div;Greater;Mul;ReduceMax;ReduceSum... |
| task085 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Compress;Constant;Conv;Floor;Greater;Mul;ReduceSum;Reshape;Sub |
| task141 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task161 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Less;Mul;ReduceMax;ReduceSu... |
| task220 | C_medium | 17 | 22.167 | 0 | same_shape_rule;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task224 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task225 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task229 | C_medium | 17 | 22.167 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors | Add;Cast;Compress;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape;Sub |
| task267 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | And;Cast;Compress;Constant;Greater;Less;Mul;ReduceSum;Reshape;Sum |
| task341 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Constant;Greater;Max;MaxPool;Mul;ReduceMax;ReduceSum;... |
| task004 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax;Relu |
| task015 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task073 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color | Constant;Conv;Mul;ReduceMax |
| task081 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task098 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax |
| task120 | C_medium | 19 | 22.056 | 0 | same_shape_rule;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task122 | C_medium | 19 | 22.056 | 0 | same_shape_rule;low_color | Constant;Conv;Mul;ReduceMax;Relu |
| task139 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task147 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task151 | D_hard | 19 | 22.056 | 0 | same_shape_rule;separator_structure;new_output_colors | Constant;Conv;Mul;ReduceMax |

## Family Counts Still Under 20

| Family | Under-20 Tasks | All Tasks |
| --- | ---: | ---: |
| same_shape_rule | 191 | 250 |
| small_grid | 153 | 196 |
| separator_structure | 132 | 157 |
| low_color | 98 | 127 |
| new_output_colors | 94 | 130 |
| reduce_or_crop | 69 | 73 |
| expand_or_tile | 35 | 36 |
| crop_exact_variable_offset | 23 | 23 |
| split_binary_relation | 7 | 7 |
| geometric_exact | 7 | 7 |
| crop_exact_same_offset | 3 | 4 |

## Lane A: Factorable Bank / Predicate Sharing (11)

Use the task292 trick as a generator: split one giant exact-output predicate bank into shared axes such as size, phase, color, orientation, and local pattern. This is the sharpest clean batch lane because it can cut many constant/Compress/Where graphs without changing semantics.

| Task | Bucket | Cost | Points | Static | Families | Ops |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task167 | B_template | 184 | 19.785 | 1 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Cast;Compress;Concat;Constant;Equal;Greater;ReduceSum;Slice |
| task376 | D_hard | 252 | 19.471 | 1 | expand_or_tile | Cast;Compress;Concat;Constant;Reshape;Slice |
| task399 | C_medium | 264 | 19.424 | 1 | small_grid;low_color;new_output_colors | Compress;Concat;Constant;Equal;ReduceSum;Slice |
| task058 | C_medium | 269 | 19.405 | 1 | same_shape_rule;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape;Slice |
| task176 | C_medium | 359 | 19.117 | 1 | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape;Slice |
| task200 | C_medium | 560 | 18.672 | 1 | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Concat;Constant;Reshape;Slice |
| task375 | C_medium | 745 | 18.387 | 1 | same_shape_rule;separator_structure | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task103 | B_template | 944 | 18.150 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Equal;Not;ReduceMin;Slice |
| task373 | B_template | 1080 | 18.015 | 1 | same_shape_rule;separator_structure;small_grid | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task171 | C_medium | 1223 | 17.891 | 1 | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape;Slice |
| task038 | C_medium | 1303 | 17.828 | 1 | reduce_or_crop;small_grid;low_color | Cast;Compress;Concat;Constant;Conv;Equal;ReduceSum;Slice |

## Lane B: Near-20 Factory Targets (7)

These are close enough that a reusable generator should beat manual one-offs. Minimum useful move is `+0.5`, so the new cost must be at most about `60.65%` of current cost.

| Task | Cost | Points | Needed Cost For +0.5 | Families |
| --- | ---: | ---: | ---: | --- |
| task167 | 184 | 19.785 | 111 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task376 | 252 | 19.471 | 152 | expand_or_tile |
| task399 | 264 | 19.424 | 160 | small_grid;low_color;new_output_colors |
| task058 | 269 | 19.405 | 163 | same_shape_rule;low_color;new_output_colors |
| task176 | 359 | 19.117 | 217 | same_shape_rule;small_grid;low_color;new_output_colors |
| task200 | 560 | 18.672 | 339 | same_shape_rule;small_grid;new_output_colors |
| task375 | 745 | 18.387 | 451 | same_shape_rule;separator_structure |

## Lane C: Dynamic Low-Cost Staticization (65)

These are already low-cost public-style models, but not strict static IO. Rebuilding them cleanly may recover 20-22 points, though not 23+ unless the cost drops below 8.

| Task | Bucket | Cost | Points | Static | Families | Ops |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task067 | A_exact | 17 | 22.167 | 0 | crop_exact_same_offset;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Div;Greater;Mul;ReduceMax;ReduceSum... |
| task085 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Compress;Constant;Conv;Floor;Greater;Mul;ReduceSum;Reshape;Sub |
| task141 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task161 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Less;Mul;ReduceMax;ReduceSu... |
| task220 | C_medium | 17 | 22.167 | 0 | same_shape_rule;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task224 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task225 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task229 | C_medium | 17 | 22.167 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors | Add;Cast;Compress;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape;Sub |
| task267 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | And;Cast;Compress;Constant;Greater;Less;Mul;ReduceSum;Reshape;Sum |
| task341 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Constant;Greater;Max;MaxPool;Mul;ReduceMax;ReduceSum;... |
| task004 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax;Relu |
| task015 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task073 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color | Constant;Conv;Mul;ReduceMax |
| task081 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task098 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax |
| task120 | C_medium | 19 | 22.056 | 0 | same_shape_rule;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task122 | C_medium | 19 | 22.056 | 0 | same_shape_rule;low_color | Constant;Conv;Mul;ReduceMax;Relu |
| task139 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task147 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task151 | D_hard | 19 | 22.056 | 0 | same_shape_rule;separator_structure;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task180 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax |
| task181 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color | Add;Constant;Conv;Mul;ReduceMax;Relu |
| task262 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task272 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task282 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task296 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax;Relu |
| task320 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task336 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task344 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task352 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task024 | C_medium | 27 | 21.704 | 0 | same_shape_rule | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task063 | C_medium | 27 | 21.704 | 0 | same_shape_rule;separator_structure;low_color;new_output_colors | Add;Clip;Compress;Concat;Constant;Greater;Max;Mul;ReduceMax;ReduceS... |
| task166 | C_medium | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task254 | B_template | 27 | 21.704 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Concat;Constant;Greater;Min;Mul;ReduceMax;ReduceMin;Re... |
| task288 | C_medium | 27 | 21.704 | 0 | same_shape_rule;small_grid | Add;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task299 | B_template | 27 | 21.704 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Compress;Concat;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape;Sl... |
| task303 | D_hard | 27 | 21.704 | 0 | same_shape_rule;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task323 | B_template | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors | Compress;Constant;ConvTranspose;Greater;Mul;ReduceSum;Reshape;Slice... |
| task032 | C_medium | 28 | 21.668 | 0 | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Relu;Reshape;Sub |
| task278 | C_medium | 33 | 21.503 | 0 | same_shape_rule;low_color;new_output_colors | Cast;Compress;Concat;Constant;Conv;Greater;MaxPool;Mul;ReduceMax;Re... |
| task334 | C_medium | 36 | 21.416 | 0 | reduce_or_crop;separator_structure;small_grid;new_output_colors | Compress;Concat;Constant;Greater;ReduceSum;Slice |
| task016 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | Constant;Gather |
| task276 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task309 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task337 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | Constant;Gather |
| task190 | C_medium | 57 | 20.957 | 0 | same_shape_rule;small_grid | Clip;Compress;Concat;Constant;Conv;Greater;Identity;Max;Mul;ReduceS... |
| task092 | D_hard | 63 | 20.857 | 0 | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul;ReduceMax;Redu... |
| task129 | B_template | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid | Compress;Concat;Constant;Equal;Greater;ReduceSum;Reshape;Slice |
| task132 | D_hard | 63 | 20.857 | 0 | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul;ReduceMax;Redu... |
| task389 | C_medium | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors | Cast;Compress;Concat;Constant;Greater;Mul;ReduceSum;Reshape;Slice |
| task162 | C_medium | 67 | 20.795 | 0 | same_shape_rule;separator_structure;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task192 | D_hard | 67 | 20.795 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task193 | D_hard | 67 | 20.795 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task025 | D_hard | 69 | 20.766 | 0 | same_shape_rule;separator_structure | Add;Cast;Clip;Compress;Constant;Conv;Greater;Max;MaxPool;Mul;Reduce... |
| task226 | B_template | 77 | 20.656 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Cast;Compress;Constant;Conv;Div;Floor;Greater;Mul;ReduceMax;Red... |
| task196 | C_medium | 117 | 20.238 | 0 | same_shape_rule;low_color;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Less;Mul;ReduceSum;Reshape;Sub |
| task230 | C_medium | 117 | 20.238 | 0 | same_shape_rule;low_color;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task298 | C_medium | 122 | 20.196 | 0 | same_shape_rule;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task212 | B_template | 137 | 20.080 | 0 | same_shape_rule;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task050 | C_medium | 147 | 20.010 | 0 | same_shape_rule;low_color;new_output_colors | Add;Cast;Compress;Conv;CumSum;Greater;Max;Mul;ReduceSum;Reshape;Sli... |
| ... | ... | 5 more | ... | ... | ... | ... |

## Lane D: Structural 20+ Study Set (69)

Study these to identify reusable primitives. They are not necessarily next submissions; they are examples of what a compact clean representation currently looks like.

| Task | Bucket | Cost | Points | Static | Families | Ops |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task067 | A_exact | 17 | 22.167 | 0 | crop_exact_same_offset;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Div;Greater;Mul;ReduceMax;ReduceSum... |
| task085 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Compress;Constant;Conv;Floor;Greater;Mul;ReduceSum;Reshape;Sub |
| task141 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task161 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Less;Mul;ReduceMax;ReduceSu... |
| task220 | C_medium | 17 | 22.167 | 0 | same_shape_rule;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task224 | D_hard | 17 | 22.167 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task225 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task229 | C_medium | 17 | 22.167 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors | Add;Cast;Compress;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape;Sub |
| task267 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid | And;Cast;Compress;Constant;Greater;Less;Mul;ReduceSum;Reshape;Sum |
| task341 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Constant;Greater;Max;MaxPool;Mul;ReduceMax;ReduceSum;... |
| task004 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax;Relu |
| task015 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task073 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color | Constant;Conv;Mul;ReduceMax |
| task081 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task098 | D_hard | 19 | 22.056 | 0 | same_shape_rule | Constant;Conv;Mul;ReduceMax |
| task120 | C_medium | 19 | 22.056 | 0 | same_shape_rule;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task122 | C_medium | 19 | 22.056 | 0 | same_shape_rule;low_color | Constant;Conv;Mul;ReduceMax;Relu |
| task139 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task147 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task151 | D_hard | 19 | 22.056 | 0 | same_shape_rule;separator_structure;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task180 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax |
| task181 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color | Add;Constant;Conv;Mul;ReduceMax;Relu |
| task262 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task272 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task282 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task296 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax;Relu |
| task320 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task336 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task344 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task352 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task024 | C_medium | 27 | 21.704 | 0 | same_shape_rule | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task063 | C_medium | 27 | 21.704 | 0 | same_shape_rule;separator_structure;low_color;new_output_colors | Add;Clip;Compress;Concat;Constant;Greater;Max;Mul;ReduceMax;ReduceS... |
| task166 | C_medium | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task254 | B_template | 27 | 21.704 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Concat;Constant;Greater;Min;Mul;ReduceMax;ReduceMin;Re... |
| task288 | C_medium | 27 | 21.704 | 0 | same_shape_rule;small_grid | Add;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task299 | B_template | 27 | 21.704 | 0 | same_shape_rule;small_grid;low_color;new_output_colors | Compress;Concat;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape;Sl... |
| task303 | D_hard | 27 | 21.704 | 0 | same_shape_rule;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task323 | B_template | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors | Compress;Constant;ConvTranspose;Greater;Mul;ReduceSum;Reshape;Slice... |
| task032 | C_medium | 28 | 21.668 | 0 | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Relu;Reshape;Sub |
| task104 | B_template | 28 | 21.668 | 1 | expand_or_tile;small_grid;low_color | Cast;Compress;Concat;Constant;Reshape;Slice |
| task278 | C_medium | 33 | 21.503 | 0 | same_shape_rule;low_color;new_output_colors | Cast;Compress;Concat;Constant;Conv;Greater;MaxPool;Mul;ReduceMax;Re... |
| task334 | C_medium | 36 | 21.416 | 0 | reduce_or_crop;separator_structure;small_grid;new_output_colors | Compress;Concat;Constant;Greater;ReduceSum;Slice |
| task016 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | Constant;Gather |
| task276 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task309 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task337 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | Constant;Gather |
| task056 | C_medium | 44 | 21.216 | 1 | reduce_or_crop;separator_structure;small_grid | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task190 | C_medium | 57 | 20.957 | 0 | same_shape_rule;small_grid | Clip;Compress;Concat;Constant;Conv;Greater;Identity;Max;Mul;ReduceS... |
| task186 | B_template | 61 | 20.889 | 1 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Compress;Concat;Constant;Equal;ReduceSum;Slice |
| task092 | D_hard | 63 | 20.857 | 0 | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul;ReduceMax;Redu... |
| task129 | B_template | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid | Compress;Concat;Constant;Equal;Greater;ReduceSum;Reshape;Slice |
| task132 | D_hard | 63 | 20.857 | 0 | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul;ReduceMax;Redu... |
| task389 | C_medium | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors | Cast;Compress;Concat;Constant;Greater;Mul;ReduceSum;Reshape;Slice |
| task162 | C_medium | 67 | 20.795 | 0 | same_shape_rule;separator_structure;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task192 | D_hard | 67 | 20.795 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum;Res... |
| task193 | D_hard | 67 | 20.795 | 0 | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape;Sub |
| task025 | D_hard | 69 | 20.766 | 0 | same_shape_rule;separator_structure | Add;Cast;Clip;Compress;Constant;Conv;Greater;Max;MaxPool;Mul;Reduce... |
| task226 | B_template | 77 | 20.656 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Cast;Compress;Constant;Conv;Div;Floor;Greater;Mul;ReduceMax;Red... |
| task061 | C_medium | 80 | 20.618 | 1 | same_shape_rule;separator_structure | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape;Slice |
| task266 | B_template | 105 | 20.346 | 1 | same_shape_rule;small_grid;low_color;new_output_colors | Cast;Compress;Concat;Constant;Reshape;Slice |
| ... | ... | 9 more | ... | ... | ... | ... |

## Lane E: Crop / Geometry 15-17 Cleanup (42)

Only worth batch work if one generator applies across many tasks. Do not hand-chase unless expected gain is at least `+0.5`.

| Task | Bucket | Cost | Points | Static | Families | Ops |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task100 | C_medium | 7529 | 16.073 | 1 | reduce_or_crop;separator_structure;small_grid | Cast;Compress;Concat;Constant;Equal;Greater;Max;Mul;ReduceSum;Slice |
| task026 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | Add;Compress;Concat;Constant;Greater;Mul;Pad;ReduceSum;Reshape;Slic... |
| task072 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | Abs;Compress;Concat;Constant;Greater;Pad;ReduceSum;Reshape;Slice;Sub |
| task144 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | Add;Compress;Concat;Constant;Greater;Mul;Pad;ReduceSum;Reshape;Slic... |
| task146 | B_template | 9017 | 15.893 | 1 | crop_exact_variable_offset;separator_structure;small_grid | Abs;Cast;Compress;Constant;Greater;Mul;Pad;ReduceSum;Reshape;Slice;... |
| task153 | C_medium | 9017 | 15.893 | 1 | reduce_or_crop;small_grid | Add;And;Cast;Compress;Constant;Conv;Equal;Greater;Identity;Mul;Redu... |
| task236 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | Abs;Compress;Concat;Constant;Greater;Pad;ReduceSum;Reshape;Slice;Sub |
| task318 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | Add;Compress;Concat;Constant;Greater;Mul;Pad;ReduceSum;Reshape;Slic... |
| task326 | A_exact | 9017 | 15.893 | 1 | crop_exact_same_offset;separator_structure | Compress;Constant;Greater;Pad;ReduceSum;Reshape;Slice |
| task372 | D_hard | 9017 | 15.893 | 1 | reduce_or_crop;separator_structure | Add;Compress;Concat;Constant;Greater;Mul;Pad;ReduceSum;Reshape;Slice |
| task087 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | Compress;Constant;Greater;Pad;ReduceSum;Reshape;Slice |
| task140 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | Compress;Constant;Greater;Pad;ReduceSum;Reshape;Slice |
| task179 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | Compress;Greater;ReduceSum;Reshape;Transpose |
| task241 | A_exact | 9053 | 15.889 | 1 | geometric_exact;small_grid | Compress;Greater;ReduceSum;Reshape;Transpose |
| task274 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;low_color | ArgMax;Cast;Compress;Concat;Constant;Equal;Greater;ReduceSum;Reshap... |
| task325 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;low_color | Add;And;Cast;Compress;Concat;Constant;Conv;Div;Equal;Greater;Mul;Or... |
| task360 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;separator_structure;small_grid | Add;Compress;Constant;Greater;Mul;Pad;ReduceMax;ReduceSum;Reshape;S... |
| task380 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | Compress;Constant;Greater;Pad;ReduceSum;Reshape;Slice;Transpose |
| task121 | D_hard | 9248 | 15.868 | 1 | reduce_or_crop | Add;ArgMax;Cast;Compress;Conv;Gather;Greater;Mul;Pad;ReduceMax;Redu... |
| task149 | C_medium | 9249 | 15.868 | 1 | reduce_or_crop;separator_structure;low_color;new_output_colors | Cast;Compress;Conv;Gather;Greater;Less;Mul;Pad;ReduceSum;Reshape;Su... |
| task247 | D_hard | 9253 | 15.867 | 1 | reduce_or_crop;small_grid | And;ArgMax;Cast;Compress;Gather;Greater;Less;Mul;Pad;ReduceMax;Redu... |
| task184 | D_hard | 9263 | 15.866 | 1 | reduce_or_crop | Cast;Compress;Concat;CumSum;Equal;Greater;Mul;Pad;ReduceMax;ReduceS... |
| task006 | A_exact | 9387 | 15.853 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | Add;Compress;Greater;Mul;Pad;ReduceMax;ReduceSum;Reshape;Slice;Sub |
| task227 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Greater;Mul;Pad;ReduceMax;ReduceSum;Reshape;Slice;Sub |
| task347 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Greater;Max;Mul;Pad;ReduceMax;ReduceSum;Reshape;Slice;Sub |
| task386 | A_exact | 9387 | 15.853 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | Add;Compress;Greater;Mul;Pad;ReduceMax;ReduceSum;Reshape;Slice;Sub |
| task395 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Greater;Mul;Pad;ReduceMax;ReduceSum;Reshape;Slice;Sub |
| task235 | B_template | 9498 | 15.841 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Conv;Greater;Mul;Pad;ReduceSum;Reshape;Slice;Sub;Trans... |
| task321 | C_medium | 9560 | 15.835 | 1 | reduce_or_crop;separator_structure;small_grid | Compress;Conv;Gather;Greater;Max;Mul;Pad;ReduceSum;Reshape;Sub;Sum |
| task130 | C_medium | 9625 | 15.828 | 1 | reduce_or_crop;small_grid | Add;Compress;Conv;Gather;Greater;MaxPool;Mul;Pad;ReduceMax;ReduceSu... |
| task263 | B_template | 10010 | 15.789 | 1 | crop_exact_variable_offset;separator_structure;small_grid | Cast;Compress;Greater;Less;Mul;Pad;ReduceSum;Reshape;Slice;Sub;Sum |
| task355 | B_template | 10125 | 15.777 | 1 | crop_exact_variable_offset;separator_structure | Add;ArgMax;Cast;Compress;Equal;Gather;Greater;Less;MatMul;Mul;Pad;R... |
| task039 | B_template | 10738 | 15.718 | 1 | crop_exact_variable_offset;small_grid | Abs;ArgMax;Cast;Compress;Greater;Less;MatMul;Mul;Pad;ReduceMax;Redu... |
| task189 | C_medium | 10980 | 15.696 | 1 | reduce_or_crop;separator_structure;small_grid | Add;Cast;Compress;Greater;Mul;Pad;ReduceSum;Reshape;Slice;Sub;Sum |
| task029 | B_template | 13098 | 15.520 | 1 | crop_exact_variable_offset;separator_structure | Add;Cast;Compress;Gather;Greater;Less;Max;Mul;Pad;ReduceMax;ReduceS... |
| task046 | D_hard | 13458 | 15.493 | 1 | reduce_or_crop;separator_structure;small_grid | Abs;Add;Cast;Compress;Concat;Conv;Gather;Greater;Less;MatMul;MaxPoo... |
| task264 | D_hard | 13729 | 15.473 | 1 | reduce_or_crop | Compress;Gather;Greater;Mul;Pad;ReduceSum;Reshape;Slice;Sub;Sum |
| task150 | A_exact | 13827 | 15.466 | 1 | geometric_exact;separator_structure;small_grid | Abs;Add;Cast;Clip;Compress;Greater;Less;MatMul;Mul;ReduceSum;Reshap... |
| task155 | A_exact | 13827 | 15.466 | 1 | geometric_exact;separator_structure;small_grid | Abs;Add;Cast;Clip;Compress;Greater;Less;MatMul;Mul;ReduceSum;Reshap... |
| task065 | B_template | 13832 | 15.465 | 1 | crop_exact_variable_offset;separator_structure | Abs;Add;ArgMax;Cast;Clip;Compress;Greater;Identity;Less;MatMul;Mul;... |
| task346 | B_template | 14022 | 15.452 | 1 | crop_exact_variable_offset;separator_structure | Cast;Compress;Conv;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task177 | D_hard | 15130 | 15.376 | 1 | reduce_or_crop | Add;ArgMax;Concat;Constant;ConstantOfShape;Conv;ReduceMax;Reshape;S... |

## Public Candidate Big-Cuts Still Worth Checking (43)

| Task | Delta | Base Cost | Candidate Cost | Source | Zip |
| --- | ---: | ---: | ---: | --- | --- |
| task001 | 8.549 | 5161 | 1 | artemnazemtsev-neuro-golf-6263 | `data/public/artemnazemtsev-neuro-golf-6263/submission.zip` |
| task001 | 8.549 | 5161 | 1 | artemnazemtsev-neurogolf-logic-driven-ensembling-part-2 | `data/public/artemnazemtsev-neurogolf-logic-driven-ensembling-part-2/submission.zip` |
| task001 | 8.549 | 5161 | 1 | artemnazemtsev-neurogolf-logic-driven-ensembling-part-3 | `data/public/artemnazemtsev-neurogolf-logic-driven-ensembling-part-3/submission.zip` |
| task001 | 8.549 | 5161 | 1 | artemnazemtsev-neurogolf-logic-driven-ensembling-part-4 | `data/public/artemnazemtsev-neurogolf-logic-driven-ensembling-part-4/submission.zip` |
| task001 | 8.549 | 5161 | 1 | jonathanchan-ngc26-constraint-smart-logic-mix-blending | `data/public/jonathanchan-ngc26-constraint-smart-logic-mix-blending/submission.zip` |
| task001 | 8.549 | 5161 | 1 | ldausl-the-2026-neurogolf-championship | `data/public/ldausl-the-2026-neurogolf-championship/submission.zip` |
| task076 | 4.855 | 12344719 | 96126 | needless090-neurogolf-4250 | `data/public/needless090-neurogolf-4250/submission.zip` |
| task076 | 4.855 | 12344719 | 96162 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | afr1ste-neurogolf-6225-51-public-score-open-solution | `data/public/afr1ste-neurogolf-6225-51-public-score-open-solution/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | artemnazemtsev-neuro-golf-6263 | `data/public/artemnazemtsev-neuro-golf-6263/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | artemnazemtsev-neurogolf-logic-driven-ensembling-part-3 | `data/public/artemnazemtsev-neurogolf-logic-driven-ensembling-part-3/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | artemnazemtsev-neurogolf-logic-driven-ensembling-part-4 | `data/public/artemnazemtsev-neurogolf-logic-driven-ensembling-part-4/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | beicicc-neurogolf-6061-74-public-score-open-solution | `data/public/beicicc-neurogolf-6061-74-public-score-open-solution/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | beicicc-neurogolf-6202-73-public-score-open-solution | `data/public/beicicc-neurogolf-6202-73-public-score-open-solution/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | beicicc-neurogolf-6233-36-public-score-open-solution | `data/public/beicicc-neurogolf-6233-36-public-score-open-solution/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | imaadmahmood-neurogolf-2026-rule-based-onnx-solver | `data/public/imaadmahmood-neurogolf-2026-rule-based-onnx-solver/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | jonathanchan-ngc26-constraint-smart-logic-mix-blending | `data/public/jonathanchan-ngc26-constraint-smart-logic-mix-blending/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | konbu17-neurogolf-2026-blended-401-tasks-lb-6254 | `data/public/konbu17-neurogolf-2026-blended-401-tasks-lb-6254/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | ldausl-the-2026-neurogolf-championship | `data/public/ldausl-the-2026-neurogolf-championship/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | svanikkolli-arc-nano-engine-v2 | `data/public/svanikkolli-arc-nano-engine-v2/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | magmacot-3204-neurogolf-open-datasets | `data/public/magmacot-3204-neurogolf-open-datasets/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | magmacot-5600-neurogolf-logic-driven-ensemble | `data/public/magmacot-5600-neurogolf-logic-driven-ensemble/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | magmacot-5700-neurogolf-logic-driven-ensemble | `data/public/magmacot-5700-neurogolf-logic-driven-ensemble/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | svanikkolli-arc-nano-engine-v2 | `data/public/svanikkolli-arc-nano-engine-v2/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | magmacot-3204-neurogolf-open-datasets | `data/public/magmacot-3204-neurogolf-open-datasets/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | magmacot-5600-neurogolf-logic-driven-ensemble | `data/public/magmacot-5600-neurogolf-logic-driven-ensemble/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | magmacot-5700-neurogolf-logic-driven-ensemble | `data/public/magmacot-5700-neurogolf-logic-driven-ensemble/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | svanikkolli-arc-nano-engine-v2 | `data/public/svanikkolli-arc-nano-engine-v2/submission.zip` |
| task101 | 3.394 | 4134759 | 138871 | magmacot-3204-neurogolf-open-datasets | `data/public/magmacot-3204-neurogolf-open-datasets/submission.zip` |
| task101 | 3.394 | 4134759 | 138871 | svanikkolli-arc-nano-engine-v2 | `data/public/svanikkolli-arc-nano-engine-v2/submission.zip` |
| task133 | 3.224 | 12180672 | 484782 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task133 | 3.224 | 12180672 | 484782 | svanikkolli-arc-nano-engine-v2 | `data/public/svanikkolli-arc-nano-engine-v2/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | beicicc-neurogolf-5433-43-public-score-open-solution | `data/public/beicicc-neurogolf-5433-43-public-score-open-solution/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | beicicc-neurogolf-5509-31-public-score-open-solution | `data/public/beicicc-neurogolf-5509-31-public-score-open-solution/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | beicicc-neurogolf-5518-02-public-score-open-solution | `data/public/beicicc-neurogolf-5518-02-public-score-open-solution/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | needless090-neurogolf-4250 | `data/public/needless090-neurogolf-4250/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | rockerritesh-neurogolf-2026-5353-lb-public-submission | `data/public/rockerritesh-neurogolf-2026-5353-lb-public-submission/submission.zip` |
| task363 | 3.121 | 2844908 | 125501 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task110 | 0.545 | 28516 | 16541 | afr1ste-neurogolf-6225-51-public-score-open-solution | `data/public/afr1ste-neurogolf-6225-51-public-score-open-solution/submission.zip` |
| task110 | 0.545 | 28516 | 16541 | beicicc-neurogolf-6202-73-public-score-open-solution | `data/public/beicicc-neurogolf-6202-73-public-score-open-solution/submission.zip` |
| task110 | 0.545 | 28516 | 16541 | jazivxt-infinitesimals | `data/public/jazivxt-infinitesimals/submission.zip` |

## Drop Unless Generator-Level Breakthrough (59)

These are too expensive for manual scoring gains. Touch them only when a family generator changes the cost class.

| Task | Cost | Points | Needed Cost For +0.5 | Families |
| --- | ---: | ---: | ---: | --- |
| task366 | 20894994 | 8.145 | 12673454 | reduce_or_crop;separator_structure |
| task233 | 12375088 | 8.669 | 7505870 | reduce_or_crop |
| task076 | 12344719 | 8.671 | 7487450 | same_shape_rule |
| task133 | 12180672 | 8.685 | 7387951 | same_shape_rule |
| task209 | 9892462 | 8.893 | 6000081 | reduce_or_crop |
| task101 | 4134759 | 9.765 | 2507858 | same_shape_rule;low_color |
| task363 | 2844908 | 10.139 | 1725523 | same_shape_rule;separator_structure;small_grid;low_color |
| task089 | 788461 | 11.422 | 478225 | same_shape_rule |
| task080 | 786980 | 11.424 | 477327 | same_shape_rule;separator_structure |
| task173 | 778214 | 11.435 | 472010 | same_shape_rule |
| task290 | 504852 | 11.868 | 306208 | reduce_or_crop |
| task115 | 352077 | 12.228 | 213545 | reduce_or_crop;separator_structure |
| task159 | 349304 | 12.236 | 211863 | reduce_or_crop |
| task239 | 340146 | 12.263 | 206308 | expand_or_tile;separator_structure;small_grid;new_output_colors |
| task285 | 326188 | 12.305 | 197843 | same_shape_rule |
| task396 | 316892 | 12.334 | 192204 | reduce_or_crop |
| task382 | 315325 | 12.339 | 191254 | same_shape_rule;low_color |
| task259 | 304046 | 12.375 | 184413 | reduce_or_crop;separator_structure;small_grid;new_output_colors |
| task109 | 287460 | 12.431 | 174353 | reduce_or_crop;separator_structure |
| task245 | 272888 | 12.483 | 165514 | same_shape_rule;small_grid;low_color |
| task324 | 269591 | 12.495 | 163515 | same_shape_rule;separator_structure |
| task255 | 262000 | 12.524 | 158911 | same_shape_rule;new_output_colors |
| task275 | 250999 | 12.567 | 152238 | expand_or_tile;separator_structure |
| task201 | 245129 | 12.590 | 148678 | reduce_or_crop |
| task216 | 239252 | 12.615 | 145113 | crop_exact_variable_offset;low_color |
| task203 | 220932 | 12.694 | 134002 | same_shape_rule;separator_structure |
| task170 | 215245 | 12.720 | 130552 | reduce_or_crop |
| task313 | 208617 | 12.752 | 126532 | same_shape_rule;separator_structure |
| task308 | 200818 | 12.790 | 121802 | reduce_or_crop;separator_structure |
| task377 | 197914 | 12.804 | 120040 | reduce_or_crop;separator_structure |
| task215 | 196524 | 12.811 | 119197 | same_shape_rule;separator_structure |
| task066 | 192569 | 12.832 | 116799 | same_shape_rule |
| task332 | 180900 | 12.894 | 109721 | same_shape_rule;small_grid;low_color;new_output_colors |
| task088 | 172030 | 12.945 | 104341 | reduce_or_crop |
| task195 | 168869 | 12.963 | 102424 | reduce_or_crop;low_color |
| task219 | 167853 | 12.969 | 101807 | same_shape_rule;separator_structure;low_color;new_output_colors |
| task021 | 166815 | 12.975 | 101178 | reduce_or_crop;separator_structure |
| task165 | 165854 | 12.981 | 100595 | same_shape_rule |
| task238 | 161273 | 13.009 | 97817 | reduce_or_crop |
| task199 | 157738 | 13.031 | 95672 | same_shape_rule;new_output_colors |
| task365 | 151656 | 13.071 | 91984 | crop_exact_variable_offset;small_grid |
| task213 | 139746 | 13.152 | 84760 | reduce_or_crop;separator_structure |
| task118 | 138879 | 13.159 | 84234 | same_shape_rule;low_color;new_output_colors |
| task398 | 130877 | 13.218 | 79380 | expand_or_tile;separator_structure |
| task091 | 129824 | 13.226 | 78742 | crop_exact_variable_offset;low_color |
| task057 | 126850 | 13.249 | 76938 | reduce_or_crop;small_grid |
| task107 | 125881 | 13.257 | 76350 | expand_or_tile;separator_structure;new_output_colors |
| task400 | 124700 | 13.266 | 75634 | reduce_or_crop |
| task367 | 121928 | 13.289 | 73953 | same_shape_rule;low_color;new_output_colors |
| task163 | 116948 | 13.331 | 70932 | same_shape_rule;separator_structure |
| task174 | 116750 | 13.332 | 70812 | crop_exact_variable_offset;small_grid |
| task018 | 112605 | 13.368 | 68298 | same_shape_rule |
| task391 | 110028 | 13.392 | 66735 | reduce_or_crop |
| task108 | 108000 | 13.410 | 65505 | expand_or_tile |
| task113 | 108000 | 13.410 | 65505 | same_shape_rule;separator_structure;small_grid |
| task211 | 108000 | 13.410 | 65505 | expand_or_tile;separator_structure;small_grid |
| task379 | 106969 | 13.420 | 64879 | same_shape_rule;separator_structure;low_color |
| task271 | 103752 | 13.450 | 62928 | crop_exact_variable_offset;small_grid;low_color |
| task036 | 101796 | 13.469 | 61742 | crop_exact_variable_offset |

## Next Generator Spec

1. Build one analyzer that reads a task JSON and infers shared predicate axes from all train/test/arc-gen examples: dimensions, color set, row/column counts, local pattern signatures, and phase/orientation.
2. Build one `bank_factorizer` that replaces one predicate per output with a product of shared axes, then emits the smallest exact ONNX graph it can verify locally.
3. Run it across Lane A and Lane B first; submit only batches with verified local gain `> +0.5` and no metric-risk flags.
4. Treat `23+` as a research target for new primitives, not the immediate submission gate. The immediate gate is raising many tasks to `20-22` without exploit behavior.
