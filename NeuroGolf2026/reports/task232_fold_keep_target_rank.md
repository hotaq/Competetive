# Clean Best Target Rank

Base zip: `generated/custom_task232_fold_keep/submission.zip`
Audited tasks: 400
Locally valid tasks: 399
Static 1x10x30x30 valid tasks: 287
Local estimated points: 6510.985445

## Bucket Totals

| Bucket | Tasks | OK | Points |
| --- | ---: | ---: | ---: |
| A_exact | 22 | 22 | 375.627504 |
| B_template | 99 | 98 | 1644.246153 |
| C_medium | 162 | 162 | 2701.325268 |
| D_hard | 117 | 117 | 1789.786520 |

## Highest Priority Clean Targets

| Task | Bucket | Difficulty | Cost | Points | Static | IO Shape | Shapes | Families | Notes | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| task363 | B_template | 14 | 2844908 | 10.138959 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x10 -> 10x10 | same_shape_rule;separator_structure;small_grid;low_color |  | ok |
| task188 | A_exact | 12 | 49690 | 14.186441 | 1 | 1x10x30x30 -> 1x10x30x30 | 2x4;2x6;2x8;3x4;3x6;3x8;4x2;4x3;4x4;4x6;4x8;6x2;6x3;6x4;8x2;8x3;8x4 -> 2x2;2x3;2x4;3x2;3x3;3x4;4x2;4x3;4x4 | crop_exact_same_offset;separator_structure;small_grid |  | ok |
| task150 | A_exact | 8 | 13827 | 15.465622 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | geometric_exact;separator_structure;small_grid | exact_transform=flip_h | ok |
| task155 | A_exact | 8 | 13827 | 15.465622 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x4;5x5;6x6;7x7;8x8 -> 4x4;5x5;6x6;7x7;8x8 | geometric_exact;separator_structure;small_grid | exact_transform=flip_v | ok |
| task087 | A_exact | 8 | 9053 | 15.889149 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 | ok |
| task140 | A_exact | 8 | 9053 | 15.889149 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 | ok |
| task179 | A_exact | 8 | 9053 | 15.889149 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=transpose | ok |
| task241 | A_exact | 8 | 9053 | 15.889149 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | geometric_exact;small_grid | exact_transform=transpose | ok |
| task380 | A_exact | 8 | 9053 | 15.889149 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot270 | ok |
| task006 | A_exact | 10 | 9387 | 15.852919 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x7 -> 3x3 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep5_and_1_1_to_2_else_0 | ok |
| task386 | A_exact | 10 | 9387 | 15.852919 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x7 -> 4x3 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_vertical_sep1_and_0_0_to_3_else_0 | ok |
| task026 | A_exact | 10 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 5x7 -> 5x3 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep1_and_0_0_to_8_else_0 | ok |
| task072 | A_exact | 10 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 13x5 -> 6x5 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 | ok |
| task144 | A_exact | 10 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_3_else_0 | ok |
