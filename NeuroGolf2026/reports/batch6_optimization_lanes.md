# Batch6 Optimization Lanes

Cost formula: `params + memory_bytes + MACs`; task score is `max(1, 25 - ln(cost))`.
To score `20+`, a task needs `cost <= 148`. That makes 20+ realistic only for tiny graph families, not ordinary 30x30 spatial compute.

Current audited zip: `generated/submit_compact_canvas_batch6/submission.zip`
Current local estimate: `6499.404381` over `399` valid tasks.

Lane policy:
- `20+ strict`: keep or improve only tiny graph/color/lookup/transpose tasks. Avoid turning normal spatial tasks into 20+ unless the math is truly tiny.
- `15-17 crop`: use Slice pushdown plus Pad for small-canvas tasks where cost can fall below about 22k.
- `Skip`: do not spend time when the best plausible crop still leaves six-figure cost unless it is a known high-leverage target.

## 20+ Strict Static Current

| Task | Points | Cost | Static | Max shape | Bucket | Families | Ops hint |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task104 | 21.311 | 40 | 1 | 3x3 | B_template | expand_or_tile;small_grid;low_color | Cast;Compress;Concat;Constant;Reshape;Slice |
| task186 | 20.889 | 61 | 1 | 3x3 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Compress;Concat;Constant;Equal;ReduceSum;Slice |
| task056 | 20.873 | 62 | 1 | 3x3 | C_medium | reduce_or_crop;separator_structure;small_grid | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task061 | 20.327 | 107 | 1 | 18x18 | C_medium | same_shape_rule;separator_structure | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |

## 20+ Current But Dynamic Or Compress-Low

| Task | Points | Cost | Static | Max shape | Bucket | Families | Ops hint |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task067 | 22.167 | 17 | 0 | 5x15 | A_exact | crop_exact_same_offset;separator_structure;small_grid | Add;Cast;Compress;Constant;Conv;Div;Greater;Mul |
| task085 | 22.167 | 17 | 0 | 16x30 | D_hard | same_shape_rule | Add;Compress;Constant;Conv;Floor;Greater;Mul;ReduceSum |
| task141 | 22.167 | 17 | 0 | 21x21 | D_hard | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum |
| task161 | 22.167 | 17 | 0 | 20x25 | D_hard | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Less;Mul |
| task220 | 22.167 | 17 | 0 | 16x16 | C_medium | same_shape_rule;new_output_colors | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum |
| task224 | 22.167 | 17 | 0 | 16x16 | D_hard | same_shape_rule | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceMax |
| task225 | 22.167 | 17 | 0 | 6x6 | C_medium | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum |
| task229 | 22.167 | 17 | 0 | 3x3 | C_medium | same_shape_rule;separator_structure;small_grid;new_output_colors | Add;Cast;Compress;Constant;Greater;Mul;ReduceMax;ReduceSum |
| task267 | 22.167 | 17 | 0 | 7x7 | C_medium | same_shape_rule;small_grid | And;Cast;Compress;Constant;Greater;Less;Mul;ReduceSum |
| task341 | 22.167 | 17 | 0 | 10x10 | C_medium | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Constant;Greater;Max;MaxPool;Mul;ReduceMax |
| task004 | 22.056 | 19 | 0 | 16x16 | D_hard | same_shape_rule | Constant;Conv;Mul;ReduceMax;Relu |
| task015 | 22.056 | 19 | 0 | 9x9 | B_template | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task073 | 22.056 | 19 | 0 | 5x5 | B_template | same_shape_rule;separator_structure;small_grid;low_color | Constant;Conv;Mul;ReduceMax |
| task081 | 22.056 | 19 | 0 | 7x7 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task098 | 22.056 | 19 | 0 | 20x22 | D_hard | same_shape_rule | Constant;Conv;Mul;ReduceMax |
| task120 | 22.056 | 19 | 0 | 16x15 | C_medium | same_shape_rule;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task122 | 22.056 | 19 | 0 | 21x21 | C_medium | same_shape_rule;low_color | Constant;Conv;Mul;ReduceMax;Relu |
| task139 | 22.056 | 19 | 0 | 9x9 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task147 | 22.056 | 19 | 0 | 6x6 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task151 | 22.056 | 19 | 0 | 12x12 | D_hard | same_shape_rule;separator_structure;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task180 | 22.056 | 19 | 0 | 8x8 | C_medium | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax |
| task181 | 22.056 | 19 | 0 | 6x9 | B_template | same_shape_rule;small_grid;low_color | Add;Constant;Conv;Mul;ReduceMax;Relu |
| task262 | 22.056 | 19 | 0 | 3x3 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task272 | 22.056 | 19 | 0 | 5x5 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task282 | 22.056 | 19 | 0 | 9x9 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task296 | 22.056 | 19 | 0 | 5x7 | C_medium | reduce_or_crop;small_grid | Constant;Conv;Mul;ReduceMax;Relu |
| task320 | 22.056 | 19 | 0 | 11x9 | C_medium | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task336 | 22.056 | 19 | 0 | 10x10 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Constant;Conv;Mul;ReduceMax;Relu |
| task344 | 22.056 | 19 | 0 | 10x10 | C_medium | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task352 | 22.056 | 19 | 0 | 10x10 | C_medium | same_shape_rule;small_grid;new_output_colors | Constant;Conv;Mul;ReduceMax |
| task024 | 21.704 | 27 | 0 | 15x15 | C_medium | same_shape_rule | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum |
| task063 | 21.704 | 27 | 0 | 14x14 | C_medium | same_shape_rule;separator_structure;low_color;new_output_colors | Add;Clip;Compress;Concat;Constant;Greater;Max;Mul |
| task166 | 21.704 | 27 | 0 | 14x14 | C_medium | same_shape_rule;low_color;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum |
| task254 | 21.704 | 27 | 0 | 9x9 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Compress;Concat;Constant;Greater;Min;Mul;ReduceMax |
| task288 | 21.704 | 27 | 0 | 9x9 | C_medium | same_shape_rule;small_grid | Add;Compress;Constant;Conv;Greater;Mul;ReduceSum;Reshape |
| task299 | 21.704 | 27 | 0 | 6x6 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Compress;Concat;Constant;Greater;Mul;ReduceMax;ReduceSum;Reshape |
| task303 | 21.704 | 27 | 0 | 30x30 | D_hard | same_shape_rule;new_output_colors | Add;Compress;Constant;Conv;Greater;Mul;ReduceMax;ReduceSum |
| task323 | 21.704 | 27 | 0 | 13x13 | B_template | same_shape_rule;low_color;new_output_colors | Compress;Constant;ConvTranspose;Greater;Mul;ReduceSum;Reshape;Slice |
| task032 | 21.668 | 28 | 0 | 6x6 | C_medium | same_shape_rule;small_grid | Add;Cast;Compress;Constant;Conv;Greater;Mul;ReduceSum |
| task278 | 21.503 | 33 | 0 | 18x18 | C_medium | same_shape_rule;low_color;new_output_colors | Cast;Compress;Concat;Constant;Conv;Greater;MaxPool;Mul |
| task334 | 21.416 | 36 | 0 | 5x5 | C_medium | reduce_or_crop;separator_structure;small_grid;new_output_colors | Compress;Concat;Constant;Greater;ReduceSum;Slice |
| task016 | 21.311 | 40 | 0 | 3x3 | A_exact | color_remap;separator_structure;small_grid | Constant;Gather |
| task276 | 21.311 | 40 | 0 | 6x6 | A_exact | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task309 | 21.311 | 40 | 0 | 3x6 | A_exact | color_remap;separator_structure;small_grid;low_color;new_output_colors | Constant;Gather |
| task337 | 21.311 | 40 | 0 | 5x5 | A_exact | color_remap;separator_structure;small_grid | Constant;Gather |
| task190 | 20.957 | 57 | 0 | 10x10 | C_medium | same_shape_rule;small_grid | Clip;Compress;Concat;Constant;Conv;Greater;Identity;Max |
| task092 | 20.857 | 63 | 0 | 30x30 | D_hard | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul |
| task129 | 20.857 | 63 | 0 | 3x3 | B_template | same_shape_rule;separator_structure;small_grid | Compress;Concat;Constant;Equal;Greater;ReduceSum;Reshape;Slice |
| task132 | 20.857 | 63 | 0 | 15x15 | D_hard | same_shape_rule | Add;Cast;Compress;Concat;Constant;CumSum;Greater;Mul |
| task389 | 20.857 | 63 | 0 | 5x5 | C_medium | same_shape_rule;separator_structure;small_grid;new_output_colors | Cast;Compress;Concat;Constant;Greater;Mul;ReduceSum;Reshape |

## Near-20 Candidates To Inspect

| Task | Points | Cost | Static | Max shape | Bucket | Families | Ops hint |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task266 | 19.989 | 150 | 1 | 3x5 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Cast;Compress;Concat;Constant;Reshape;Slice |
| task160 | 19.925 | 160 | 0 | 10x10 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Cast;Constant;Conv;Greater;Mul;Sub;Sum |
| task167 | 19.785 | 184 | 1 | 3x3 | B_template | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | Add;Cast;Compress;Concat;Constant;Equal;Greater;ReduceSum |
| task248 | 19.748 | 191 | 1 | 10x10 | B_template | same_shape_rule;small_grid;low_color | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task357 | 19.748 | 191 | 1 | 10x10 | C_medium | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task399 | 19.424 | 264 | 1 | 7x7 | C_medium | small_grid;low_color;new_output_colors | Compress;Concat;Constant;Equal;ReduceSum;Slice |
| task053 | 19.402 | 270 | 0 | 3x3 | B_template | same_shape_rule;separator_structure;small_grid;low_color | Gather |
| task385 | 19.402 | 270 | 0 | 10x4 | C_medium | same_shape_rule;small_grid | Gather |
| task058 | 19.117 | 359 | 1 | 20x20 | C_medium | same_shape_rule;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task376 | 19.114 | 360 | 1 | 6x17 | D_hard | expand_or_tile | Cast;Compress;Concat;Constant;Reshape;Slice |
| task312 | 18.891 | 450 | 0 | 12x12 | C_medium | same_shape_rule;separator_structure | Add;Constant;Gather;Mul;Sub |
| task176 | 18.828 | 479 | 1 | 3x25 | C_medium | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task292 | 18.781 | 502 | 1 | 3x20 | C_medium | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task294 | 18.766 | 510 | 0 | 10x10 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Add;Constant;Conv;Mul;Relu;Sub |
| task168 | 18.396 | 738 | 0 | 10x10 | C_medium | same_shape_rule;small_grid | Add;Cast;Compress;Concat;Constant;Conv;Greater;Identity |
| task200 | 18.315 | 800 | 1 | 10x10 | C_medium | same_shape_rule;small_grid;new_output_colors | Cast;Compress;Concat;Constant;Reshape;Slice |
| task103 | 18.150 | 944 | 1 | 3x3 | B_template | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Equal;Not;ReduceMin |
| task375 | 18.048 | 1045 | 1 | 15x15 | C_medium | same_shape_rule;separator_structure | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task202 | 18.021 | 1074 | 0 | 42x38 | D_hard | same_shape_rule;separator_structure | Add;And;Cast;Concat;Constant;Equal;Greater;Identity |
| task397 | 17.971 | 1129 | 0 | 10x10 | C_medium | same_shape_rule;small_grid;new_output_colors | Add;And;Cast;Concat;Conv;ConvTranspose;Equal;Greater |
| task038 | 17.828 | 1303 | 1 | 9x9 | C_medium | reduce_or_crop;small_grid;low_color | Cast;Compress;Concat;Constant;Conv;Equal;ReduceSum;Slice |
| task373 | 17.679 | 1512 | 1 | 2x6 | B_template | same_shape_rule;separator_structure;small_grid | And;Cast;Compress;Concat;Constant;Reshape;Slice |
| task047 | 17.654 | 1550 | 0 | 9x9 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Add;Conv;Mul;ReduceMax;ReduceMean |
| task171 | 17.610 | 1619 | 1 | 9x9 | C_medium | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Compress;Concat;Constant;Identity;Not;Reshape |
| task331 | 17.379 | 2040 | 0 | 10x10 | B_template | same_shape_rule;small_grid;low_color;new_output_colors | Add;Constant;Conv;Mul;Relu;Sub |

## 15-17 Clean Crop Candidates

| Task | Points | Cost | Static | Max shape | Bucket | Families | Ops hint |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task076 | 8.671 | 12344719 | 1 | 15x15 | D_hard | same_shape_rule | Cast;Constant;Equal;Gather;Mul;ReduceMin;ReduceSum;Reshape |
| task209 | 8.893 | 9892462 | 1 | 20x20 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Concat;Constant;Conv;ConvTranspose;CumSum |
| task101 | 9.765 | 4134759 | 1 | 17x21 | C_medium | same_shape_rule;low_color | Abs;Add;Cast;Concat;Constant;Conv;ConvTranspose;Greater |
| task363 | 10.139 | 2844908 | 1 | 10x10 | B_template | same_shape_rule;separator_structure;small_grid;low_color | Add;Cast;Concat;Conv;Greater;Less;Mul;Pad |
| task089 | 11.422 | 788461 | 1 | 13x13 | C_medium | same_shape_rule | Add;Cast;Concat;ConvTranspose;Equal;Greater;MatMul;MaxPool |
| task290 | 11.868 | 504852 | 1 | 15x15 | D_hard | reduce_or_crop | Add;ArgMax;Constant;Conv;Gather;Mul;Pad;ReduceMax |
| task324 | 12.239 | 348375 | 1 | 20x20 | C_medium | same_shape_rule;separator_structure | ArgMax;Cast;Flatten;Gather;MatMul;Mod;Mul;ReduceSum |
| task239 | 12.263 | 340146 | 0 | 4x4 | D_hard | expand_or_tile;separator_structure;small_grid;new_output_colors | Add;Cast;Concat;Constant;Greater;Less;MatMul;Mul |
| task396 | 12.334 | 316892 | 1 | 18x18 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Concat;Conv;Gather;Greater;Less |
| task382 | 12.339 | 315325 | 1 | 20x20 | C_medium | same_shape_rule;low_color | Cast;Concat;Gather;Greater;Less;MatMul;Mul;Pad |
| task259 | 12.375 | 304046 | 1 | 7x7 | D_hard | reduce_or_crop;separator_structure;small_grid;new_output_colors | Add;ArgMax;Cast;Constant;Gather;Less;Mul;ReduceMax |
| task109 | 12.431 | 287460 | 1 | 13x13 | D_hard | reduce_or_crop;separator_structure | Add;Cast;Constant;Div;Gather;Less;Max;Mul |
| task245 | 12.483 | 272888 | 1 | 10x10 | C_medium | same_shape_rule;small_grid;low_color | Add;And;ArgMax;Cast;Constant;Gather;Greater;Less |
| task201 | 12.590 | 245129 | 1 | 13x13 | D_hard | reduce_or_crop | Add;And;ArgMax;Cast;Equal;Gather;Greater;Less |
| task216 | 12.615 | 239252 | 1 | 20x20 | B_template | crop_exact_variable_offset;low_color | Add;And;ArgMax;Cast;Concat;CumSum;Gather;Greater |
| task203 | 12.694 | 220932 | 1 | 18x18 | C_medium | same_shape_rule;separator_structure | Add;And;Cast;Concat;Constant;Conv;Div;Equal |
| task313 | 12.752 | 208617 | 1 | 20x20 | D_hard | same_shape_rule;separator_structure | Add;And;Cast;Conv;Gather;Greater;Less;Mul |
| task308 | 12.790 | 200818 | 1 | 20x20 | D_hard | reduce_or_crop;separator_structure | Add;ArgMax;Cast;Equal;Greater;Less;MatMul;Max |
| task215 | 12.811 | 196524 | 1 | 20x20 | D_hard | same_shape_rule;separator_structure | And;Cast;Concat;Constant;Conv;Equal;Gather;Greater |
| task066 | 12.832 | 192569 | 1 | 20x20 | D_hard | same_shape_rule | Add;And;Cast;Flatten;Gather;Less;MatMul;Mul |
| task332 | 12.894 | 180900 | 0 | 3x20 | C_medium | same_shape_rule;small_grid;low_color;new_output_colors | And;Cast;Concat;Constant;Conv;Equal;Greater;Less |
| task195 | 12.963 | 168869 | 1 | 17x19 | C_medium | reduce_or_crop;low_color | Abs;Add;Cast;Clip;Constant;Conv;Less;MatMul |
| task219 | 12.969 | 167853 | 1 | 15x10 | B_template | same_shape_rule;separator_structure;low_color;new_output_colors | Cast;Concat;Gather;Greater;Less;MatMul;Mul;Pad |
| task165 | 12.981 | 165854 | 1 | 20x20 | C_medium | same_shape_rule | Add;Cast;Constant;Conv;Greater;MatMul;MaxPool;Mul |
| task238 | 13.009 | 161273 | 1 | 16x16 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Gather;Greater;Less;Mul;Pad |
| task199 | 13.031 | 157738 | 0 | 15x15 | D_hard | same_shape_rule;new_output_colors | Abs;Add;Cast;Concat;Constant;Div;Floor;Less |
| task365 | 13.071 | 151656 | 1 | 10x10 | B_template | crop_exact_variable_offset;small_grid | Add;And;ArgMax;Cast;Concat;CumSum;Gather;Greater |
| task091 | 13.226 | 129824 | 1 | 15x15 | B_template | crop_exact_variable_offset;low_color | Add;And;ArgMax;Cast;Constant;Gather;Greater;Less |
| task057 | 13.249 | 126850 | 1 | 8x8 | C_medium | reduce_or_crop;small_grid | Add;ArgMax;Constant;Gather;Pad;ReduceMax;Slice;Transpose |
| task367 | 13.289 | 121928 | 0 | 20x20 | C_medium | same_shape_rule;low_color;new_output_colors | Add;Cast;Concat;Flatten;Greater;Less;MatMul;Mul |
| task163 | 13.331 | 116948 | 1 | 11x11 | C_medium | same_shape_rule;separator_structure | Add;Cast;Div;Equal;Greater;Less;MatMul;Mul |
| task174 | 13.332 | 116750 | 1 | 10x10 | B_template | crop_exact_variable_offset;small_grid | Add;ArgMax;Cast;Concat;Equal;Gather;Greater;Less |
| task391 | 13.392 | 110028 | 1 | 15x16 | D_hard | reduce_or_crop | Add;Cast;Concat;OneHot;Pad;ReduceSum;Reshape;Slice |
| task211 | 13.410 | 108000 | 1 | 3x2 | C_medium | expand_or_tile;separator_structure;small_grid | Constant;Gather;Reshape |
| task113 | 13.410 | 108000 | 1 | 10x10 | C_medium | same_shape_rule;separator_structure;small_grid | Constant;Gather;Reshape |
| task379 | 13.420 | 106969 | 0 | 20x20 | C_medium | same_shape_rule;separator_structure;low_color | Add;And;Cast;Concat;Constant;Conv;CumSum;Div |
| task271 | 13.450 | 103752 | 1 | 9x9 | B_template | crop_exact_variable_offset;small_grid;low_color | Add;ArgMax;Cast;Conv;Div;Gather;Greater;Mul |
| task079 | 13.516 | 97151 | 1 | 14x14 | B_template | crop_exact_variable_offset | Add;ArgMax;Cast;Conv;Gather;Greater;Pad;ReduceMax |
| task111 | 13.537 | 95102 | 1 | 10x10 | B_template | crop_exact_variable_offset;small_grid | Mul;Pad;Slice;Sum |
| task178 | 13.548 | 94114 | 1 | 14x13 | D_hard | reduce_or_crop;separator_structure;small_grid | Cast;Conv;Gather;Greater;Less;MatMul;Mul;Pad |
| task293 | 13.592 | 90000 | 0 | 15x15 | C_medium | same_shape_rule;separator_structure | Add;And;Cast;Concat;Constant;Conv;Equal;Gather |
| task096 | 13.655 | 84509 | 0 | 19x19 | D_hard | reduce_or_crop;separator_structure | ArgMax;Cast;Concat;Div;Equal;Floor;Gather;MatMul |
| task143 | 13.677 | 82734 | 1 | 10x10 | C_medium | same_shape_rule;small_grid | Add;Cast;Conv;Equal;Gather;Greater;Mul;Pad |
| task049 | 13.677 | 82725 | 1 | 20x20 | B_template | crop_exact_variable_offset | Add;ArgMax;Cast;Concat;Constant;Gather;Greater;Mul |
| task300 | 13.680 | 82491 | 1 | 9x15 | B_template | crop_exact_variable_offset;small_grid | Add;ArgMax;Concat;Constant;Gather;Pad;ReduceMax;ReduceSum |
| task031 | 13.681 | 82394 | 1 | 12x12 | B_template | crop_exact_variable_offset | Add;ArgMax;Concat;Constant;Pad;ReduceMax;Reshape;Slice |
| task384 | 13.719 | 79334 | 1 | 9x9 | C_medium | reduce_or_crop;small_grid;low_color | Add;Cast;Concat;Constant;Conv;ConvTranspose;Greater;Mul |
| task319 | 13.758 | 76272 | 1 | 19x19 | D_hard | reduce_or_crop;separator_structure | Abs;Cast;Conv;Equal;Flatten;Gather;Less;MatMul |
| task218 | 13.799 | 73169 | 1 | 21x21 | D_hard | reduce_or_crop | ArgMax;Cast;Concat;Conv;CumSum;Equal;GatherND;Greater |
| task387 | 13.826 | 71245 | 0 | 18x18 | D_hard | same_shape_rule;new_output_colors | Add;And;Cast;Concat;Conv;Div;Equal;Greater |

## Skip Unless Breakthrough

| Task | Points | Cost | Static | Max shape | Bucket | Families | Ops hint |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| task366 | 8.145 | 20894994 | 1 | 30x34 | D_hard | reduce_or_crop;separator_structure | Add;Cast;Constant;Equal;Gather;Mul;ReduceMin;ReduceSum |
| task233 | 8.669 | 12375088 | 1 | 30x30 | D_hard | reduce_or_crop | Cast;Constant;Equal;Gather;Mul;ReduceMin;ReduceSum;Reshape |
| task076 | 8.671 | 12344719 | 1 | 15x15 | D_hard | same_shape_rule | Cast;Constant;Equal;Gather;Mul;ReduceMin;ReduceSum;Reshape |
| task133 | 8.685 | 12180672 | 1 | 30x30 | D_hard | same_shape_rule | Add;ArgMax;Cast;Concat;Constant;Conv;ConvTranspose;Equal |
| task209 | 8.893 | 9892462 | 1 | 20x20 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Concat;Constant;Conv;ConvTranspose;CumSum |
| task080 | 11.424 | 786980 | 1 | 31x31 | D_hard | same_shape_rule;separator_structure | Add;ArgMax;Cast;Clip;Concat;Equal;Gather;Greater |
| task173 | 11.435 | 778214 | 1 | 25x25 | D_hard | same_shape_rule | And;Cast;Concat;Conv;Equal;Gather;Greater;Less |
| task290 | 11.868 | 504852 | 1 | 15x15 | D_hard | reduce_or_crop | Add;ArgMax;Constant;Conv;Gather;Mul;Pad;ReduceMax |
| task115 | 12.228 | 352077 | 1 | 30x29 | D_hard | reduce_or_crop;separator_structure | Add;Cast;Constant;Div;Equal;Greater;Mul;ReduceMax |
| task159 | 12.236 | 349304 | 1 | 30x30 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Gather;Less;Mul;Pad;ReduceMax |
| task324 | 12.239 | 348375 | 1 | 20x20 | C_medium | same_shape_rule;separator_structure | ArgMax;Cast;Flatten;Gather;MatMul;Mod;Mul;ReduceSum |
| task239 | 12.263 | 340146 | 0 | 4x4 | D_hard | expand_or_tile;separator_structure;small_grid;new_output_colors | Add;Cast;Concat;Constant;Greater;Less;MatMul;Mul |
| task285 | 12.305 | 326188 | 1 | 30x30 | D_hard | same_shape_rule | ArgMax;Cast;Concat;Flatten;Gather;Less;MatMul;Mul |
| task396 | 12.334 | 316892 | 1 | 18x18 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Concat;Conv;Gather;Greater;Less |
| task382 | 12.339 | 315325 | 1 | 20x20 | C_medium | same_shape_rule;low_color | Cast;Concat;Gather;Greater;Less;MatMul;Mul;Pad |
| task259 | 12.375 | 304046 | 1 | 7x7 | D_hard | reduce_or_crop;separator_structure;small_grid;new_output_colors | Add;ArgMax;Cast;Constant;Gather;Less;Mul;ReduceMax |
| task109 | 12.431 | 287460 | 1 | 13x13 | D_hard | reduce_or_crop;separator_structure | Add;Cast;Constant;Div;Gather;Less;Max;Mul |
| task255 | 12.524 | 262000 | 1 | 30x30 | D_hard | same_shape_rule;new_output_colors | Add;And;Cast;Flatten;Gather;Less;MatMul;Mul |
| task201 | 12.590 | 245129 | 1 | 13x13 | D_hard | reduce_or_crop | Add;And;ArgMax;Cast;Equal;Gather;Greater;Less |
| task216 | 12.615 | 239252 | 1 | 20x20 | B_template | crop_exact_variable_offset;low_color | Add;And;ArgMax;Cast;Concat;CumSum;Gather;Greater |
| task170 | 12.720 | 215245 | 1 | 28x28 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Concat;Conv;Gather;Greater;Less |
| task313 | 12.752 | 208617 | 1 | 20x20 | D_hard | same_shape_rule;separator_structure | Add;And;Cast;Conv;Gather;Greater;Less;Mul |
| task308 | 12.790 | 200818 | 1 | 20x20 | D_hard | reduce_or_crop;separator_structure | Add;ArgMax;Cast;Equal;Greater;Less;MatMul;Max |
| task377 | 12.804 | 197914 | 1 | 30x30 | D_hard | reduce_or_crop;separator_structure | And;ArgMax;Cast;Equal;Gather;Greater;Less;Mul |
| task215 | 12.811 | 196524 | 1 | 20x20 | D_hard | same_shape_rule;separator_structure | And;Cast;Concat;Constant;Conv;Equal;Gather;Greater |
| task066 | 12.832 | 192569 | 1 | 20x20 | D_hard | same_shape_rule | Add;And;Cast;Flatten;Gather;Less;MatMul;Mul |
| task088 | 12.945 | 172030 | 0 | 24x24 | D_hard | reduce_or_crop | Add;Cast;Equal;Greater;Less;MatMul;Mul;ReduceMax |
| task021 | 12.975 | 166815 | 0 | 70x75 | D_hard | reduce_or_crop;separator_structure | Add;And;Cast;Concat;Conv;Equal;Greater;Less |
| task165 | 12.981 | 165854 | 1 | 20x20 | C_medium | same_shape_rule | Add;Cast;Constant;Conv;Greater;MatMul;MaxPool;Mul |
| task238 | 13.009 | 161273 | 1 | 16x16 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Gather;Greater;Less;Mul;Pad |
| task199 | 13.031 | 157738 | 0 | 15x15 | D_hard | same_shape_rule;new_output_colors | Abs;Add;Cast;Concat;Constant;Div;Floor;Less |
| task213 | 13.152 | 139746 | 1 | 24x24 | D_hard | reduce_or_crop;separator_structure | Add;Cast;Equal;Greater;Less;Mul;Not;Pad |
| task118 | 13.159 | 138879 | 1 | 25x28 | D_hard | same_shape_rule;low_color;new_output_colors | Cast;Compress;Equal;Gather;GatherND;Greater;Mul;ReduceMax |
| task398 | 13.218 | 130877 | 1 | 1x5 | D_hard | expand_or_tile;separator_structure | Add;Cast;Compress;Constant;Conv;Equal;Greater;Less |
| task107 | 13.257 | 125881 | 0 | 5x5 | D_hard | expand_or_tile;separator_structure;new_output_colors | Gather;Less;Mul;Pad;ReduceMax;ReduceSum;Squeeze;Sub |
| task400 | 13.266 | 124700 | 1 | 24x24 | D_hard | reduce_or_crop | Add;ArgMax;Cast;Constant;Gather;Greater;Mul;Pad |
| task367 | 13.289 | 121928 | 0 | 20x20 | C_medium | same_shape_rule;low_color;new_output_colors | Add;Cast;Concat;Flatten;Greater;Less;MatMul;Mul |
| task018 | 13.368 | 112605 | 1 | 24x24 | D_hard | same_shape_rule | Abs;Add;ArgMax;Cast;Compress;Concat;ConstantOfShape;Gather |
| task391 | 13.392 | 110028 | 1 | 15x16 | D_hard | reduce_or_crop | Add;Cast;Concat;OneHot;Pad;ReduceSum;Reshape;Slice |
| task379 | 13.420 | 106969 | 0 | 20x20 | C_medium | same_shape_rule;separator_structure;low_color | Add;And;Cast;Concat;Constant;Conv;CumSum;Div |
