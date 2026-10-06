# Clean Best Target Rank

Base zip: `generated/local_plus_rescan_subset_076_089_101/submission.zip`
Audited tasks: 400
Locally valid tasks: 399
Static 1x10x30x30 valid tasks: 287
Local estimated points: 7549.781184

## Bucket Totals

| Bucket | Tasks | OK | Points |
| --- | ---: | ---: | ---: |
| A_exact | 22 | 22 | 455.115021 |
| B_template | 99 | 98 | 1923.600664 |
| C_medium | 162 | 162 | 3165.893184 |
| D_hard | 117 | 117 | 2005.172315 |

## Highest Priority Clean Targets

| Task | Bucket | Difficulty | Cost | Points | Static | IO Shape | Shapes | Families | Notes | Reason |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| task363 | B_template | 14 | 2844908 | 10.138959 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x10 -> 10x10 | same_shape_rule;separator_structure;small_grid;low_color |  | ok |
| task188 | A_exact | 12 | 49690 | 14.186441 | 1 | 1x10x30x30 -> 1x10x30x30 | 2x4;2x6;2x8;3x4;3x6;3x8;4x2;4x3;4x4;4x6;4x8;6x2;6x3;6x4;8x2;8x3;8x4 -> 2x2;2x3;2x4;3x2;3x3;3x4;4x2;4x3;4x4 | crop_exact_same_offset;separator_structure;small_grid |  | ok |
| task155 | A_exact | 8 | 4551 | 16.576898 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x4;5x5;6x6;7x7;8x8 -> 4x4;5x5;6x6;7x7;8x8 | geometric_exact;separator_structure;small_grid | exact_transform=flip_v | ok |
| task150 | A_exact | 8 | 4546 | 16.577997 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | geometric_exact;separator_structure;small_grid | exact_transform=flip_h | ok |
| task016 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x3 -> 3x3 | color_remap;separator_structure;small_grid | color_map=1->5,2->6,3->4,4->3,5->1,6->2,8->9,9->8 | ok |
| task276 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 -> 3x4;3x6;4x4;4x5;4x6;5x4;5x5;5x6;6x4;6x5;6x6 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=6->2,7->7 | ok |
| task309 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x4;3x5;3x6 -> 3x4;3x5;3x6 | color_remap;separator_structure;small_grid;low_color;new_output_colors | color_map=1->1,7->5,8->8 | ok |
| task337 | A_exact | 5 | 40 | 21.311121 | 0 | 1x10x?x? -> 1x10x?x? | 3x3;4x4;5x5 -> 3x3;4x4;5x5 | color_remap;separator_structure;small_grid | color_map=1->1,2->2,3->3,4->4,5->8,6->6,7->7,8->5,9->9 | ok |
| task135 | A_exact | 12 | 4500 | 16.588167 | 0 | 1x10x?x? -> 1x10x?x? | 9x9 -> 3x3 | crop_exact_same_offset;small_grid |  | ok |
| task380 | A_exact | 8 | 61 | 20.889126 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot270 | ok |
| task087 | A_exact | 8 | 56 | 20.974648 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 | ok |
| task179 | A_exact | 8 | 56 | 20.974648 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=transpose | ok |
| task140 | A_exact | 8 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3 -> 3x3 | geometric_exact;separator_structure;small_grid | exact_transform=rot180 | ok |
| task241 | A_exact | 8 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x3;4x4;5x5;6x6;7x7;8x8;9x9 -> 3x3;4x4;5x5;6x6;7x7;8x8;9x9 | geometric_exact;small_grid | exact_transform=transpose | ok |
| task006 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 3x7 -> 3x3 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep5_and_1_1_to_2_else_0 | ok |
| task026 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 5x7 -> 5x3 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_vertical_sep1_and_0_0_to_8_else_0 | ok |
| task072 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 13x5 -> 6x5 | split_binary_relation;reduce_or_crop;small_grid;low_color;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 | ok |
| task144 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_3_else_0 | ok |
| task236 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_xor_0_0_to_3_else_0 | ok |
| task318 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x4 -> 4x4 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_horizontal_sep4_and_0_0_to_0_else_3 | ok |
| task386 | A_exact | 10 | 16 | 22.227411 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x7 -> 4x3 | split_binary_relation;reduce_or_crop;small_grid;new_output_colors | split_vertical_sep1_and_0_0_to_3_else_0 | ok |
| task326 | A_exact | 12 | 61 | 20.889126 | 1 | 1x10x30x30 -> 1x10x30x30 | 4x4;4x8;6x6;6x12;8x4;8x8;12x6;12x12 -> 2x2 | crop_exact_same_offset;separator_structure |  | ok |
| task067 | A_exact | 12 | 17 | 22.166787 | 0 | ?x?x?x? -> ?x?x?x? | 2x6;3x9;4x12;5x15 -> 2x2;3x3;4x4;5x5 | crop_exact_same_offset;separator_structure;small_grid |  | ok |
| task216 | B_template | 18 | 239252 | 12.614727 | 1 | 1x10x30x30 -> 1x10x30x30 | 20x20 -> 3x10;3x11;3x12;3x13;3x14;3x15;3x16;3x17;3x18;4x5;4x6;4x7;4x8;4x9;4x10;4x11;4x12;4x13;4x14;4x16;4x17;4x18;5x6;5x7;5x8;5x9;5x10;5x11;5x12;5x13;5x14;5x15;5x16;5x17;5x18;6x5;6x6;6x7;6x8;6x10;6x11;6x12;6x14;6x15;6x16;6x17;7x5;7x6;7x7;7x8;7x9;7x10;7x11;7x12;7x14;7x15;7x17;7x18;8x4;8x5;8x6;8x7;8x8;8x9;8x10;8x11;8x12;8x14;8x16;8x18;9x4;9x5;9x6;9x7;9x8;9x9;9x10;9x12;9x13;9x15;10x5;10x6;10x7;10x8;10x9;11x4;11x5;11x6;11x8;11x9;11x10;11x11;11x17;12x4;12x5;12x6;12x8;12x10;12x11;13x4;13x5;13x6;13x7;13x12;14x4;14x5;14x6;14x7;14x8;15x4;15x5;15x6;15x7;15x8;16x4;16x5;16x8;16x9;17x4;17x6;17x7;18x5;18x6;18x7 | crop_exact_variable_offset;low_color |  | ok |
| task219 | B_template | 16 | 167853 | 12.969156 | 1 | 1x10x30x30 -> 1x10x30x30 | 15x10 -> 15x10 | same_shape_rule;separator_structure;low_color;new_output_colors |  | ok |
| task365 | B_template | 18 | 151656 | 13.070630 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x10 -> 3x3;3x4;3x5;3x6;4x3;4x4;4x5;4x6;5x3;5x4;5x5;5x6;6x3;6x4;6x5;6x6 | crop_exact_variable_offset;small_grid |  | ok |
| task091 | B_template | 18 | 129824 | 13.226065 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x9;9x10;9x11;9x12;9x13;9x14;9x15;10x9;10x10;10x11;10x12;10x13;10x14;10x15;11x9;11x10;11x11;11x12;11x13;11x14;11x15;12x9;12x10;12x11;12x12;12x13;12x14;12x15;13x9;13x10;13x11;13x12;13x13;13x14;13x15;14x9;14x10;14x11;14x12;14x13;14x14;14x15;15x9;15x10;15x11;15x12;15x13;15x14;15x15 -> 3x3;3x4;3x5;3x6;3x7;3x8;3x9;3x10;3x11;3x12;3x13;4x3;4x4;4x5;4x6;4x7;4x8;4x9;4x10;4x11;4x12;4x13;5x3;5x4;5x5;5x6;5x7;5x8;5x10;5x11;5x12;5x14;6x3;6x4;6x5;6x6;6x7;6x8;6x9;6x10;6x12;7x3;7x4;7x5;7x6;7x7;7x8;7x9;7x10;7x11;8x3;8x4;8x5;8x6;8x7;8x8;8x9;8x10;8x11;8x12;9x3;9x4;9x5;9x6;9x7;9x8;9x9;9x10;9x12;9x13;10x3;10x4;10x5;10x6;10x7;10x8;10x9;10x11;10x13;11x3;11x5;11x6;11x7;11x10;11x11;11x12;12x5;12x6;12x7;12x8;12x9;12x12;13x5;13x7;13x9;13x13;14x6;14x7;14x8 | crop_exact_variable_offset;low_color |  | ok |
| task174 | B_template | 18 | 116750 | 13.332210 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x10 -> 1x2;1x5;2x1;2x2;2x3;2x4;2x5;3x1;3x2;3x3;3x4;4x1;4x2;4x3;5x1;5x2 | crop_exact_variable_offset;small_grid |  | ok |
| task271 | B_template | 18 | 103752 | 13.450241 | 1 | 1x10x30x30 -> 1x10x30x30 | 9x9 -> 3x3 | crop_exact_variable_offset;small_grid;low_color |  | ok |
| task036 | B_template | 18 | 101796 | 13.469274 | 1 | 1x10x30x30 -> 1x10x30x30 | 30x30 -> 2x3;3x2;3x3;3x4;3x5;4x3;4x4;4x5;5x3;5x4;5x5 | crop_exact_variable_offset |  | ok |
| task079 | B_template | 18 | 97151 | 13.515978 | 1 | 1x10x30x30 -> 1x10x30x30 | 14x14 -> 3x3 | crop_exact_variable_offset |  | ok |
| task014 | B_template | 18 | 82725 | 13.676723 | 1 | 1x10x30x30 -> 1x10x30x30 | 15x15;15x16;15x17;15x19;15x21;15x22;15x23;15x24;15x25;16x15;16x16;16x18;16x19;16x20;16x21;16x22;16x23;16x24;16x25;17x15;17x16;17x17;17x19;17x20;17x21;17x23;17x24;17x25;18x15;18x16;18x17;18x18;18x19;18x20;18x21;18x22;18x23;18x25;19x15;19x16;19x17;19x18;19x19;19x20;19x21;19x22;19x23;19x24;19x25;20x15;20x16;20x17;20x19;20x20;20x21;20x22;20x24;20x25;21x15;21x17;21x18;21x19;21x20;21x21;21x22;21x23;21x25;22x15;22x16;22x17;22x18;22x19;22x20;22x21;22x22;22x23;22x24;23x16;23x17;23x18;23x19;23x20;23x21;23x22;23x23;23x24;24x15;24x16;24x17;24x18;24x19;24x20;24x22;24x23;24x24;24x25;25x15;25x16;25x17;25x19;25x20;25x21;25x23;25x25 -> 5x5;5x6;5x7;5x8;5x9;5x10;5x12;5x14;5x15;6x5;6x6;6x7;6x8;6x9;6x10;6x11;6x12;6x13;6x14;6x17;7x5;7x6;7x7;7x8;7x9;7x10;7x11;7x13;7x14;7x18;8x5;8x6;8x7;8x8;8x9;8x10;8x11;8x12;8x13;8x17;9x5;9x6;9x7;9x8;9x9;9x10;9x13;9x14;9x16;10x5;10x6;10x7;10x8;10x10;10x11;10x12;10x13;11x5;11x6;11x7;11x8;11x9;11x10;11x11;12x5;12x6;12x8;12x9;12x10;12x11;12x12;12x14;13x5;13x7;13x8;13x9;13x12;14x5;14x10;15x5;15x7;15x8;15x9;15x11;16x7;16x10;17x6 | crop_exact_variable_offset |  | ok |
| task049 | B_template | 18 | 82725 | 13.676723 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x10;10x12;10x14;10x15;10x16;10x18;10x19;10x20;11x10;11x11;11x13;11x14;11x15;11x16;11x17;11x18;11x19;11x20;12x10;12x11;12x12;12x14;12x15;12x16;12x18;12x19;12x20;13x10;13x11;13x12;13x13;13x15;13x16;13x17;13x18;13x20;14x10;14x11;14x12;14x13;14x14;14x15;14x16;14x17;14x18;14x19;14x20;15x10;15x11;15x12;15x14;15x15;15x16;15x17;15x19;15x20;16x10;16x12;16x13;16x14;16x15;16x16;16x18;16x19;16x20;17x10;17x11;17x12;17x13;17x14;17x15;17x16;17x17;17x18;17x19;17x20;18x11;18x12;18x13;18x14;18x15;18x16;18x17;18x18;18x19;18x20;19x10;19x11;19x12;19x13;19x14;19x15;19x17;19x18;19x19;19x20;20x10;20x11;20x12;20x14;20x15;20x16;20x18;20x20 -> 2x2;2x3;2x4;2x5;2x6;2x7;2x8;2x9;2x10;2x11;2x14;3x2;3x3;3x4;3x5;3x6;3x7;3x9;4x2;4x3;4x4;4x5;4x6;4x7;4x9;4x10;4x11;4x15;5x2;5x3;5x4;5x5;5x6;5x8;5x9;6x2;6x3;6x4;6x5;6x8;6x9;7x2;7x3;7x4;7x5;7x6;7x10;7x12;8x2;8x3;8x4;9x2;9x3;10x2;10x3;10x5;11x3 | crop_exact_variable_offset |  | ok |
| task310 | B_template | 18 | 82725 | 13.676723 | 1 | 1x10x30x30 -> 1x10x30x30 | 20x20;21x21;22x22;23x23;24x24;25x25;26x26;27x27;28x28;29x29;30x30 -> 5x5;6x6;7x7;8x8 | crop_exact_variable_offset;separator_structure |  | ok |
| task300 | B_template | 18 | 82491 | 13.679556 | 1 | 1x10x30x30 -> 1x10x30x30 | 5x9;5x10;5x11;5x12;5x13;5x14;5x15;6x10;6x11;6x12;6x13;6x14;7x9;7x10;7x11;7x12;7x13;8x9;8x10;8x11;8x12;9x9;9x10;9x11 -> 2x3;3x2;3x3;4x3 | crop_exact_variable_offset;small_grid |  | ok |
| task031 | B_template | 18 | 82394 | 13.680732 | 1 | 1x10x30x30 -> 1x10x30x30 | 10x12;11x12;12x12 -> 3x4;3x5;3x6;3x7;3x8;4x3;4x4;4x5;4x6;4x7;4x8;5x3;5x4;5x5;5x6;6x2;6x3;6x4;6x5;6x6;6x7;7x4;7x5 | crop_exact_variable_offset |  | ok |
