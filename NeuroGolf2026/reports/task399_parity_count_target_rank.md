# Clean Best Target Rank

Base zip: `generated/custom_task399_parity_count/submission.zip`
Audited tasks: 400
Locally valid tasks: 399
Static 1x10x30x30 valid tasks: 287
Local estimated points: 6506.111865

## Bucket Totals

| Bucket | Tasks | OK | Points |
| --- | ---: | ---: | ---: |
| A_exact | 22 | 22 | 375.627504 |
| B_template | 99 | 98 | 1642.732589 |
| C_medium | 162 | 162 | 2700.582587 |
| D_hard | 117 | 117 | 1787.169185 |

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
| task236 | A_exact | 10 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 | ok |
| task318 | A_exact | 10 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_0_else_3 | ok |
| task326 | A_exact | 12 | 9017 | 15.893133 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x4;4x8;6x6;6x12;8x4;8x8;12x6;12x12 -> 2x2 | crop_exact_same_offset;separator_structure |  | ok |
| task016 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x3 -> 3x3 | color_remap;separator_structure;small_grid | color_map=1->5,2->6,3->4,4->3,5->1,6->2,8->9,9->8 | ok |
| task276 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 -> 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=6->2,7->7 | ok |
| task309 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x4;3x5;3x6 -> 3x4;3x5;3x6 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=1->1,7->5,8->8 | ok |
| task337 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x3;4x4;5x5 -> 3x3;4x4;5x5 | color_remap;separator_structure;small_grid | color_map=1->1,2->2,3->3,4->4,5->8,6->6,7->7,8->5,9->9 | ok |
| task135 | A_exact | 12 | 4500 | 16.588167 | 0 | 1x10x?x? -> 1x10x?x? | 9x9 -> 3x3 | crop_exact_same_offset;small_grid |  | ok |
| task067 | A_exact | 12 | 17 | 22.166787 | 0 | ?x?x?x? -> ?x?x?x? | 2x6;3x9;4x12;5x15 -> 2x2;3x3;4x4;5x5 | crop_exact_same_offset;separator_structure;small_grid |  | ok |
| task216 | B_template | 18 | 239252 | 12.614727 | 1 | 1x10x30x30 -> 1x10x30x30 | 20x20 -> 3x10;3x11;3x12;3x13;3x14;3x15;3x16;3x17;3x18;4x5;4x6;4x7;4x8;4x9;4x10;4x11;4x12;4x13;4x14;4x16;4x17;4x18;5x6;5x7;5x8;5x9;5x10;5x11;5x12;5x13;5x14;5x15;5x16;5x17;5x18;6x5;6x6;6x7;6x8;6x10;6x11;6x12;6x14;6x15;6x16;6x17;7x5;7x6;7x7;7x8;7x9;7x10;7x11;7x12;7x14;7x15;7x17;7x18;8x4;8x5;8x6;8x7;8x8;8x9;8x10;8x11;8x12;8x14;8x16;8x18;9x4;9x5;9x6;9x7;9x8;9x9;9x10;9x12;9x13;9x15;10x5;10x6;10x7;10x8;10x9;11x4;11x5;11x6;11x8;11x9;11x10;11x11;11x17;12x4;12x5;12x6;12x8;12x10;12x11;13x4;13x5;13x6;13x7;13x12;14x4;14x5;14x6;14x7;14x8;15x4;15x5;15x6;15x7;15x8;16x4;16x5;16x8;16x9;17x4;17x6;17x7;18x5;18x6;18x7 | crop_exact_variable_offset;low_color |  | ok |
