# Factorable Bank Axes

Cost file: `reports/task292_two_stage_compact_task_costs.csv`
Data root: `data/extracted`

## Summary

- `build_first`: `7`
- `needs_richer_features`: `4`

| Task | Lane | Cost | Points | Unique Outputs | Features | Bank Groups | Selected Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| task399 | build_first | 264 | 19.424051 | 5 | 1 | 5 | count_c2 |
| task058 | build_first | 269 | 19.405289 | 16 | 1 | 16 | area |
| task176 | build_first | 359 | 19.116678 | 21 | 1 | 21 | area |
| task373 | build_first | 1080 | 18.015284 | 72 | 2 | 72 | cell_0_0;cell_1_0 |
| task171 | build_first | 1223 | 17.890938 | 49 | 2 | 49 | area;col0_count_c0 |
| task376 | build_first | 252 | 19.470571 | 36 | 3 | 36 | cell_3_0;cell_0_10;cell_5_1 |
| task375 | build_first | 745 | 18.386616 | 55 | 3 | 55 | cell_3_3;area;cell_0_0 |
| task167 | needs_richer_features | 184 | 19.785064 | 3 | 0 | 1 |  |
| task103 | needs_richer_features | 944 | 18.149874 | 2 | 0 | 1 |  |
| task038 | needs_richer_features | 1303 | 17.827575 | 5 | 0 | 1 |  |
| task200 | needs_richer_features | 560 | 18.672063 | 80 | 8 | 65 | cell_9_0;cell_9_1;cell_9_2;cell_9_3;cell_9_4;cell_9_5;cell_9_6;cell_9_7 |

## Interpretation

- `build_first`: the observed output bank is separated by <=4 cheap axes; this is the best target for automatic ONNX bank factorization.
- `build_second`: separated by <=8 axes; still promising, but the emitted graph may need grouped predicates or a small decision tree.
- `many_features`: exact separation exists but likely overfits unless the features map to an obvious ARC rule.
- `needs_richer_features`: add rule-specific axes such as connected components, bounding boxes, symmetry, or periodicity.
