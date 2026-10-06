# Sharp NeuroGolf Work Queue

Cost file: `reports/task292_two_stage_compact_task_costs.csv`
Candidate compare: `reports/candidate_compare_after_batch8.csv`

## Rules

- `20+` needs `cost <= exp(5) ~= 148.4`.
- `+0.5` score needs at least a `39.3%` cost cut because `new_cost <= old_cost * exp(-0.5)`.
- Six-figure tasks stay out of the queue unless a verified candidate clears `+0.5` local points.

## Current Clean Static 20+ Tasks (8)

| Task | Bucket | Cost | Points | Static | Families |
| --- | --- | ---: | ---: | ---: | --- |
| task104 | B_template | 28 | 21.668 | 1 | expand_or_tile;small_grid;low_color |
| task056 | C_medium | 44 | 21.216 | 1 | reduce_or_crop;separator_structure;small_grid |
| task186 | B_template | 61 | 20.889 | 1 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task061 | C_medium | 80 | 20.618 | 1 | same_shape_rule;separator_structure |
| task266 | B_template | 105 | 20.346 | 1 | same_shape_rule;small_grid;low_color;new_output_colors |
| task292 | C_medium | 137 | 20.080 | 1 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task248 | B_template | 143 | 20.037 | 1 | same_shape_rule;small_grid;low_color |
| task357 | C_medium | 143 | 20.037 | 1 | same_shape_rule;small_grid;low_color;new_output_colors |

## Public Low-Cost But Non-Static 20+ Tasks (61)

Useful as score evidence, but not part of the strict clean lane unless rebuilt with static IO.

| Task | Bucket | Cost | Points | Static | Families |
| --- | --- | ---: | ---: | ---: | --- |
| task067 | A_exact | 17 | 22.167 | 0 | crop_exact_same_offset;separator_structure;small_grid |
| task085 | D_hard | 17 | 22.167 | 0 | same_shape_rule |
| task141 | D_hard | 17 | 22.167 | 0 | same_shape_rule |
| task161 | D_hard | 17 | 22.167 | 0 | same_shape_rule |
| task220 | C_medium | 17 | 22.167 | 0 | same_shape_rule;new_output_colors |
| task224 | D_hard | 17 | 22.167 | 0 | same_shape_rule |
| task225 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid |
| task229 | C_medium | 17 | 22.167 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors |
| task267 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid |
| task341 | C_medium | 17 | 22.167 | 0 | same_shape_rule;small_grid;new_output_colors |
| task004 | D_hard | 19 | 22.056 | 0 | same_shape_rule |
| task015 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors |
| task073 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color |
| task081 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task098 | D_hard | 19 | 22.056 | 0 | same_shape_rule |
| task120 | C_medium | 19 | 22.056 | 0 | same_shape_rule;new_output_colors |
| task122 | C_medium | 19 | 22.056 | 0 | same_shape_rule;low_color |
| task139 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task147 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task151 | D_hard | 19 | 22.056 | 0 | same_shape_rule;separator_structure;new_output_colors |
| task180 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid |
| task181 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color |
| task262 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task272 | B_template | 19 | 22.056 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task282 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task296 | C_medium | 19 | 22.056 | 0 | reduce_or_crop;small_grid |
| task320 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task336 | B_template | 19 | 22.056 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task344 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors |
| task352 | C_medium | 19 | 22.056 | 0 | same_shape_rule;small_grid;new_output_colors |
| task024 | C_medium | 27 | 21.704 | 0 | same_shape_rule |
| task063 | C_medium | 27 | 21.704 | 0 | same_shape_rule;separator_structure;low_color;new_output_colors |
| task166 | C_medium | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors |
| task254 | B_template | 27 | 21.704 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task288 | C_medium | 27 | 21.704 | 0 | same_shape_rule;small_grid |
| task299 | B_template | 27 | 21.704 | 0 | same_shape_rule;small_grid;low_color;new_output_colors |
| task303 | D_hard | 27 | 21.704 | 0 | same_shape_rule;new_output_colors |
| task323 | B_template | 27 | 21.704 | 0 | same_shape_rule;low_color;new_output_colors |
| task032 | C_medium | 28 | 21.668 | 0 | same_shape_rule;small_grid |
| task278 | C_medium | 33 | 21.503 | 0 | same_shape_rule;low_color;new_output_colors |
| task334 | C_medium | 36 | 21.416 | 0 | reduce_or_crop;separator_structure;small_grid;new_output_colors |
| task016 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | color_map=1->5,2->6,3->4,4->3,5->1,6->2,8->9,9->8 |
| task276 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=6->2,7->7 |
| task309 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=1->1,7->5,8->8 |
| task337 | A_exact | 40 | 21.311 | 0 | color_remap;separator_structure;small_grid | color_map=1->1,2->2,3->3,4->4,5->8,6->6,7->7,8->5,9->9 |
| task190 | C_medium | 57 | 20.957 | 0 | same_shape_rule;small_grid |
| task092 | D_hard | 63 | 20.857 | 0 | same_shape_rule |
| task129 | B_template | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid |
| task132 | D_hard | 63 | 20.857 | 0 | same_shape_rule |
| task389 | C_medium | 63 | 20.857 | 0 | same_shape_rule;separator_structure;small_grid;new_output_colors |
| task162 | C_medium | 67 | 20.795 | 0 | same_shape_rule;separator_structure;new_output_colors |
| task192 | D_hard | 67 | 20.795 | 0 | same_shape_rule |
| task193 | D_hard | 67 | 20.795 | 0 | same_shape_rule |
| task025 | D_hard | 69 | 20.766 | 0 | same_shape_rule;separator_structure |
| task226 | B_template | 77 | 20.656 | 0 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task196 | C_medium | 117 | 20.238 | 0 | same_shape_rule;low_color;new_output_colors |
| task230 | C_medium | 117 | 20.238 | 0 | same_shape_rule;low_color;new_output_colors |
| task298 | C_medium | 122 | 20.196 | 0 | same_shape_rule;separator_structure;small_grid |
| task212 | B_template | 137 | 20.080 | 0 | same_shape_rule;separator_structure;small_grid |
| task050 | C_medium | 147 | 20.010 | 0 | same_shape_rule;low_color;new_output_colors |
| task314 | C_medium | 148 | 20.003 | 0 | same_shape_rule;small_grid |

## 20+ Watchlist (7)

These are close enough that a tiny lookup/remap/gather rewrite can matter.

| Task | Bucket | Cost | Points | Static | Families |
| --- | --- | ---: | ---: | ---: | --- |
| task167 | B_template | 184 | 19.785 | 1 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |
| task376 | D_hard | 252 | 19.471 | 1 | expand_or_tile |
| task399 | C_medium | 264 | 19.424 | 1 | small_grid;low_color;new_output_colors |
| task058 | C_medium | 269 | 19.405 | 1 | same_shape_rule;low_color;new_output_colors |
| task176 | C_medium | 359 | 19.117 | 1 | same_shape_rule;small_grid;low_color;new_output_colors |
| task200 | C_medium | 560 | 18.672 | 1 | same_shape_rule;small_grid;new_output_colors |
| task375 | C_medium | 745 | 18.387 | 1 | same_shape_rule;separator_structure |

## Crop 15-17 Lane (48)

Use these for bounded crop/pad cleanup only; do not chase 20+ unless the graph becomes tiny.

| Task | Bucket | Cost | Points | Static | Families |
| --- | --- | ---: | ---: | ---: | --- |
| task244 | D_hard | 4046 | 16.695 | 0 | reduce_or_crop;separator_structure |
| task048 | B_template | 4500 | 16.588 | 0 | crop_exact_variable_offset;separator_structure;small_grid;low_color |
| task135 | A_exact | 4500 | 16.588 | 0 | crop_exact_same_offset;small_grid |
| task257 | C_medium | 4500 | 16.588 | 0 | reduce_or_crop;separator_structure;small_grid |
| task183 | D_hard | 6482 | 16.223 | 0 | reduce_or_crop;separator_structure;small_grid |
| task100 | C_medium | 7529 | 16.073 | 1 | reduce_or_crop;separator_structure;small_grid |
| task207 | B_template | 8100 | 16.000 | 0 | crop_exact_variable_offset;small_grid |
| task026 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep1_and_0_0_to_8_else_0 |
| task072 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 |
| task144 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_3_else_0 |
| task146 | B_template | 9017 | 15.893 | 1 | crop_exact_variable_offset;separator_structure;small_grid |
| task153 | C_medium | 9017 | 15.893 | 1 | reduce_or_crop;small_grid |
| task236 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 |
| task318 | A_exact | 9017 | 15.893 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_0_else_3 |
| task326 | A_exact | 9017 | 15.893 | 1 | crop_exact_same_offset;separator_structure |
| task372 | D_hard | 9017 | 15.893 | 1 | reduce_or_crop;separator_structure |
| task087 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 |
| task140 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 |
| task179 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=transpose |
| task241 | A_exact | 9053 | 15.889 | 1 | geometric_exact;small_grid | exact_transform=transpose |
| task274 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;low_color |
| task325 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;low_color |
| task360 | C_medium | 9053 | 15.889 | 1 | reduce_or_crop;separator_structure;small_grid |
| task380 | A_exact | 9053 | 15.889 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=rot270 |
| task121 | D_hard | 9248 | 15.868 | 1 | reduce_or_crop |
| task149 | C_medium | 9249 | 15.868 | 1 | reduce_or_crop;separator_structure;low_color;new_output_colors |
| task247 | D_hard | 9253 | 15.867 | 1 | reduce_or_crop;small_grid |
| task184 | D_hard | 9263 | 15.866 | 1 | reduce_or_crop |
| task006 | A_exact | 9387 | 15.853 | 1 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep5_and_1_1_to_2_else_0 |
| task227 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors |
| task347 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors |
| task386 | A_exact | 9387 | 15.853 | 1 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_vertical_sep1_and_0_0_to_3_else_0 |
| task395 | B_template | 9387 | 15.853 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors |
| task235 | B_template | 9498 | 15.841 | 1 | reduce_or_crop;separator_structure;small_grid;low_color;new_output_colors |
| task321 | C_medium | 9560 | 15.835 | 1 | reduce_or_crop;separator_structure;small_grid |
| task130 | C_medium | 9625 | 15.828 | 1 | reduce_or_crop;small_grid |
| task263 | B_template | 10010 | 15.789 | 1 | crop_exact_variable_offset;separator_structure;small_grid |
| task355 | B_template | 10125 | 15.777 | 1 | crop_exact_variable_offset;separator_structure |
| task039 | B_template | 10738 | 15.718 | 1 | crop_exact_variable_offset;small_grid |
| task189 | C_medium | 10980 | 15.696 | 1 | reduce_or_crop;separator_structure;small_grid |
| task029 | B_template | 13098 | 15.520 | 1 | crop_exact_variable_offset;separator_structure |
| task046 | D_hard | 13458 | 15.493 | 1 | reduce_or_crop;separator_structure;small_grid |
| task264 | D_hard | 13729 | 15.473 | 1 | reduce_or_crop |
| task150 | A_exact | 13827 | 15.466 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=flip_h |
| task155 | A_exact | 13827 | 15.466 | 1 | geometric_exact;separator_structure;small_grid | exact_transform=flip_v |
| task065 | B_template | 13832 | 15.465 | 1 | crop_exact_variable_offset;separator_structure |
| task346 | B_template | 14022 | 15.452 | 1 | crop_exact_variable_offset;separator_structure |
| task177 | D_hard | 15130 | 15.376 | 1 | reduce_or_crop |

## Candidate Replacements Over +0.5 (8)

| Task | Delta | Base Cost | Candidate Cost | Source | Zip |
| --- | ---: | ---: | ---: | --- | --- |
| task076 | 4.855 | 12344719 | 96126 | needless090-neurogolf-4250 | `data/public/needless090-neurogolf-4250/submission.zip` |
| task089 | 4.228 | 788461 | 11498 | afr1ste-neurogolf-6225-51-public-score-open-solution | `data/public/afr1ste-neurogolf-6225-51-public-score-open-solution/submission.zip` |
| task233 | 4.219 | 12375088 | 182127 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task209 | 3.967 | 9892462 | 187324 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task101 | 3.394 | 4134759 | 138871 | magmacot-3204-neurogolf-open-datasets | `data/public/magmacot-3204-neurogolf-open-datasets/submission.zip` |
| task133 | 3.224 | 12180672 | 484782 | amanatar-neurogolf-2026-optimal-blending-ensemble | `data/public/amanatar-neurogolf-2026-optimal-blending-ensemble/submission.zip` |
| task363 | 3.121 | 2844908 | 125483 | beicicc-neurogolf-5433-43-public-score-open-solution | `data/public/beicicc-neurogolf-5433-43-public-score-open-solution/submission.zip` |
| task110 | 0.545 | 28516 | 16541 | afr1ste-neurogolf-6225-51-public-score-open-solution | `data/public/afr1ste-neurogolf-6225-51-public-score-open-solution/submission.zip` |

## Drop Unless Big-Cut (59)

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
