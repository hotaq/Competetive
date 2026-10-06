# Data Explorer — Autonomous Agent Prediction Beta

Source data: `data/raw/data/`, fetched from Kaggle on 26 July 2026.

## Overview

- 16 independent binary-classification tasks (`train_01`–`train_16`)
- 177,752 training rows in total
- 10,000 test rows per task (160,000 total)
- All tasks use `row_id` as the unique identifier and `target` as the binary label
- `target` is absent from test data; submissions require `row_id,target`

## Task profile

| Task | Train rows | Test rows | Features | Target 0 / 1 | Empty cells in train |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_01 | 14,957 | 10,000 | 12 | 7,517 / 7,440 | 13,493 |
| train_02 | 14,929 | 10,000 | 28 | 7,417 / 7,512 | 46,041 |
| train_03 | 3,501 | 10,000 | 18 | 1,734 / 1,767 | 6,139 |
| train_04 | 8,775 | 10,000 | 12 | 4,377 / 4,398 | 13,113 |
| train_05 | 1,060 | 10,000 | 9 | 536 / 524 | 1,523 |
| train_06 | 10,803 | 10,000 | 9 | 5,442 / 5,361 | 12,693 |
| train_07 | 10,417 | 10,000 | 17 | 5,284 / 5,133 | 46,977 |
| train_08 | 8,173 | 10,000 | 12 | 3,997 / 4,176 | 10,159 |
| train_09 | 1,109 | 10,000 | 18 | 556 / 553 | 2,537 |
| train_10 | 11,800 | 10,000 | 27 | 5,868 / 5,932 | 44,781 |
| train_11 | 28,879 | 10,000 | 20 | 14,531 / 14,348 | 66,191 |
| train_12 | 49,432 | 10,000 | 8 | 24,682 / 24,750 | 11,958 |
| train_13 | 500 | 10,000 | 9 | 250 / 250 | 701 |
| train_14 | 11,108 | 10,000 | 23 | 5,530 / 5,578 | 27,145 |
| train_15 | 500 | 10,000 | 30 | 250 / 250 | 2,098 |
| train_16 | 1,809 | 10,000 | 21 | 907 / 902 | 1,591 |

## Checks performed

- Confirmed every `train.csv` has a unique `row_id` for every row.
- Confirmed every task has a 10,000-row `test.csv`.
- Confirmed binary targets use labels `0` and `1`, with near-balanced class counts.
- Counted empty-string cells in each training split.
- Read the supplied `DATA.md` files: features vary by task and are documented as numeric, categorical, ordinal, or count variables.

## Modeling implications

- Treat each `train_XX` directory as a separate task; feature counts and types differ.
- Do not use `row_id` as a predictive feature unless exploratory validation demonstrates a legitimate relationship.
- Build preprocessing that explicitly handles missing values and mixed feature types.
- The small training sets in `train_13` and `train_15` (500 rows each) need conservative validation and strong regularization.
- Class weighting is unlikely to be needed for the supplied target distributions, but use stratified validation.

## Quality assessment

The supplied data is structurally ready for modeling: identifiers are unique and labels are balanced. Empty cells are the main ingestion concern. Their meaning should be assessed per feature type before imputation, because they may represent missing values or a valid category.
