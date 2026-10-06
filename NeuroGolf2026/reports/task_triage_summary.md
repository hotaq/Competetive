# NeuroGolf Task Triage Summary

Total tasks: 400

## Buckets

- A_exact: 22
- B_template: 99
- C_medium: 162
- D_hard: 117

## Family Counts

- same_shape_rule: 251
- small_grid: 196
- separator_structure: 157
- new_output_colors: 130
- low_color: 128
- reduce_or_crop: 73
- expand_or_tile: 36
- crop_exact_variable_offset: 23
- geometric_exact: 7
- split_binary_relation: 7
- color_remap: 4
- crop_exact_same_offset: 4

## Top 60 Easiest Heuristic Targets

| Task | Difficulty | Bucket | Shapes | Colors | Families | Notes |
| --- | ---: | --- | --- | --- | --- | --- |
| task016 | 5 | A_exact | 3x3 -> 3x3 | 12345689 -> 12345689 | color_remap;separator_structure;small_grid | color_map=1->5,2->6,3->4,4->3,5->1,6->2,8->9,9->8 |
| task276 | 5 | A_exact | 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 -> 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 | 67 -> 27 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=6->2,7->7 |
| task309 | 5 | A_exact | 3x4;3x5;3x6 -> 3x4;3x5;3x6 | 178 -> 158 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=1->1,7->5,8->8 |
| task337 | 5 | A_exact | 3x3;4x4;5x5 -> 3x3;4x4;5x5 | 123456789 -> 123456789 | color_remap;separator_structure;small_grid | color_map=1->1,2->2,3->3,4->4,5->8,6->6,7->7,8->5,9->9 |
| task087 | 8 | A_exact | 3x3 -> 3x3 | 123456789 -> 123456789 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 |
| task140 | 8 | A_exact | 3x3 -> 3x3 | 0123456789 -> 0123456789 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 |
| task150 | 8 | A_exact | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | 1267 -> 1267 | geometric_exact;separator_structure;small_grid | exact_transform=flip_h |
| task155 | 8 | A_exact | 4x4;5x5;6x6;7x7;8x8 -> 4x4;5x5;6x6;7x7;8x8 | 123478 -> 123478 | geometric_exact;separator_structure;small_grid | exact_transform=flip_v |
| task179 | 8 | A_exact | 3x3 -> 3x3 | 123456789 -> 123456789 | geometric_exact;separator_structure;small_grid | exact_transform=transpose |
| task241 | 8 | A_exact | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | 0123456789 -> 0123456789 | geometric_exact;small_grid | exact_transform=transpose |
| task380 | 8 | A_exact | 3x3 -> 3x3 | 0123456789 -> 0123456789 | geometric_exact;separator_structure;small_grid | exact_transform=rot270 |
| task006 | 10 | A_exact | 3x7 -> 3x3 | 015 -> 02 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep5_and_1_1_to_2_else_0 |
| task026 | 10 | A_exact | 5x7 -> 5x3 | 019 -> 08 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep1_and_0_0_to_8_else_0 |
| task053 | 10 | B_template | 3x3 -> 3x3 | 012 -> 012 | same_shape_rule;separator_structure;small_grid;low_color |  |
| task072 | 10 | A_exact | 13x5 -> 6x5 | 024 -> 03 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 |
| task073 | 10 | B_template | 5x5 -> 5x5 | 015 -> 015 | same_shape_rule;separator_structure;small_grid;low_color |  |
| task144 | 10 | A_exact | 9x4 -> 4x4 | 0247 -> 03 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_3_else_0 |
| task236 | 10 | A_exact | 9x4 -> 4x4 | 0124 -> 03 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 |
| task318 | 10 | A_exact | 9x4 -> 4x4 | 0124 -> 03 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_0_else_3 |
| task386 | 10 | A_exact | 4x7 -> 4x3 | 0157 -> 03 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_vertical_sep1_and_0_0_to_3_else_0 |
| task067 | 12 | A_exact | 2x6;3x9;4x12;5x15 -> 2x2;3x3;4x4;5x5 | 0123456789 -> 0123456789 | crop_exact_same_offset;separator_structure;small_grid |  |
| task078 | 12 | B_template | 10x10 -> 10x10 | 012 -> 012 | same_shape_rule;small_grid;low_color |  |
| task135 | 12 | A_exact | 9x9 -> 3x3 | 0123456789 -> 0123456789 | crop_exact_same_offset;small_grid |  |
| task136 | 12 | B_template | 10x10 -> 10x10 | 012 -> 012 | same_shape_rule;small_grid;low_color |  |
| task167 | 12 | B_template | 3x3 -> 3x3 | 234 -> 05 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |  |
| task181 | 12 | B_template | 6x9 -> 6x9 | 048 -> 048 | same_shape_rule;small_grid;low_color |  |
| task186 | 12 | B_template | 3x3 -> 3x3 | 01 -> 02 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |  |
| task188 | 12 | A_exact | 2x4;2x6;2x8;3x4;3x6;3x8;4x2;4x3;4x4;4x6;4x8;6x2;6x3;6x4;8x2;8x3;8x4 -> 2x2;2x3;2x4;3x2;3x3;3x4;4x2;4x3;4x4 | 123456789 -> 123456789 | crop_exact_same_offset;separator_structure;small_grid |  |
| task250 | 12 | B_template | 10x10 -> 10x10 | 025 -> 025 | same_shape_rule;small_grid;low_color |  |
| task254 | 12 | B_template | 9x9 -> 9x9 | 05 -> 012 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |  |
| task262 | 12 | B_template | 3x3 -> 3x3 | 05 -> 234 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |  |
| task326 | 12 | A_exact | 4x4;4x8;6x6;6x12;8x4;8x8;12x6;12x12 -> 2x2 | 0123456789 -> 0123456789 | crop_exact_same_offset;separator_structure |  |
| task345 | 12 | B_template | 10x10 -> 10x10 | 025 -> 025 | same_shape_rule;small_grid;low_color |  |
| task356 | 12 | B_template | 10x10 -> 10x10 | 08 -> 08 | same_shape_rule;small_grid;low_color |  |
| task010 | 14 | B_template | 9x9 -> 9x9 | 05 -> 01234 | same_shape_rule;separator_structure;small_grid;low_color;new_output_colors |  |
| task027 | 14 | B_template | 10x10 -> 10x10 | 01 -> 012 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task042 | 14 | B_template | 10x10 -> 10x10 | 03 -> 038 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task043 | 14 | B_template | 10x10 -> 10x10 | 05 -> 025 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task081 | 14 | B_template | 7x7 -> 7x7 | 08 -> 018 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task095 | 14 | B_template | 9x9 -> 9x9 | 05 -> 015 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task139 | 14 | B_template | 9x9 -> 9x9 | 04 -> 047 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task160 | 14 | B_template | 10x10 -> 10x10 | 01 -> 012 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task273 | 14 | B_template | 10x10 -> 10x10 | 04 -> 024 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task277 | 14 | B_template | 10x10 -> 10x10 | 08 -> 012 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task282 | 14 | B_template | 9x9 -> 9x9 | 05 -> 015 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task294 | 14 | B_template | 10x10 -> 10x10 | 05 -> 025 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task317 | 14 | B_template | 9x9 -> 9x9 | 05 -> 01 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task330 | 14 | B_template | 10x10 -> 10x10 | 05 -> 012 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task336 | 14 | B_template | 10x10 -> 10x10 | 05 -> 058 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task363 | 14 | B_template | 10x10 -> 10x10 | 025 -> 025 | same_shape_rule;separator_structure;small_grid;low_color |  |
| task381 | 14 | B_template | 10x10 -> 10x10 | 02 -> 029 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task047 | 16 | B_template | 9x9 -> 9x9 | 078 -> 0278 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task094 | 16 | B_template | 15x15 -> 15x15 | 18 -> 168 | same_shape_rule;separator_structure;low_color;new_output_colors |  |
| task154 | 16 | B_template | 15x15 -> 15x15 | 025 -> 025 | same_shape_rule;low_color |  |
| task156 | 16 | B_template | 10x10 -> 10x10 | 04 -> 0124 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task157 | 16 | B_template | 10x15 -> 10x15 | 025 -> 012 | same_shape_rule;separator_structure;low_color;new_output_colors |  |
| task164 | 16 | B_template | 3x3 -> 3x6 | 168 -> 168 | expand_or_tile;separator_structure;small_grid;low_color |  |
| task169 | 16 | B_template | 10x10 -> 10x10 | 05 -> 0123 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task210 | 16 | B_template | 3x3 -> 6x3 | 01 -> 01 | expand_or_tile;separator_structure;small_grid;low_color |  |
| task219 | 16 | B_template | 15x10 -> 15x10 | 08 -> 018 | same_shape_rule;separator_structure;low_color;new_output_colors |  |

## Official Starter Tasks

| Task | Difficulty | Bucket | Shapes | Colors | Families | Notes |
| --- | ---: | --- | --- | --- | --- | --- |
| task006 | 10 | A_exact | 3x7 -> 3x3 | 015 -> 02 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep5_and_1_1_to_2_else_0 |
| task095 | 14 | B_template | 9x9 -> 9x9 | 05 -> 015 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task127 | 20 | B_template | 3x11;7x11 -> 3x11;7x11 | 012345 -> 56789 | same_shape_rule;separator_structure;small_grid;new_output_colors |  |
| task261 | 22 | C_medium | 3x3;4x4;5x5;6x6;7x7 -> 3x3;4x4;5x5;6x6;7x7 | 08 -> 02 | same_shape_rule;small_grid;low_color;new_output_colors |  |
| task331 | 16 | B_template | 10x10 -> 10x10 | 01 -> 012678 | same_shape_rule;small_grid;low_color;new_output_colors |  |

## Recommended First Lane

1. Keep task006 as the first self-built static ONNX proof.
2. Implement exact families first: color_remap, geometric_exact, split_binary_relation, crop_exact_same_offset.
3. Use B_template tasks next only after a reusable ONNX template exists.
4. Treat D_hard tasks as manual rescue backlog.
