# Synthetic-supervised tree model benchmark

## Validation design

- Normal fit, calibration, and evaluation rows come from disjoint `tcp.stream` groups.
- Each validation run excludes one complete synthetic Attack family from training.
- The excluded family is used only for stress-test recall.
- Thresholds use the 99.5th percentile of unseen Normal calibration probabilities.

These proxy metrics are not leaderboard estimates. Synthetic attacks cannot fully represent hidden competition attacks.

## Model comparison

| model | mean_normal_fpr | max_normal_fpr | mean_held_out_recall | min_held_out_recall | mean_proxy_f1_at_10pct | mean_proxy_f1_at_20pct |
| --- | --- | --- | --- | --- | --- | --- |
| catboost | 0.017 | 0.056 | 0.656 | 0 | 0.625 | 0.657 |
| extra_trees | 0.02 | 0.056 | 0.641 | 0 | 0.601 | 0.635 |

## Leave-one-family-out details

| model | held_out_family | normal_fpr | held_out_family_recall | proxy_f1_at_10pct | proxy_f1_at_20pct |
| --- | --- | --- | --- | --- | --- |
| catboost | syn_flood_row | 0 | 0 | 0 | 0 |
| extra_trees | syn_flood_row | 0 | 0 | 0 | 0 |
| catboost | mqtt_invalid_control | 0.009 | 0.279 | 0.409 | 0.423 |
| extra_trees | mqtt_invalid_control | 0.025 | 0.203 | 0.284 | 0.311 |
| catboost | mqtt_credential | 0 | 1 | 1 | 1 |
| extra_trees | mqtt_credential | 0 | 1 | 0.999 | 1 |
| catboost | mqtt_will_payload | 0.056 | 1 | 0.8 | 0.9 |
| extra_trees | mqtt_will_payload | 0.056 | 1 | 0.8 | 0.9 |
| catboost | mqtt_publish | 0.02 | 1 | 0.918 | 0.962 |
| extra_trees | mqtt_publish | 0.019 | 1 | 0.92 | 0.963 |

Low held-out-family recall means the model memorized known synthetic mutations instead of learning general attack structure. Normal FPR remains the most trustworthy local metric.

## Final calibration

| model | threshold_quantile | threshold | calibration_normal_rows | test_rows_above_threshold |
| --- | --- | --- | --- | --- |
| catboost | 0.995 | 0.01 | 15867 | 902 |
| extra_trees | 0.995 | 0.078 | 15867 | 1160 |

## Candidate variants

| variant | candidate_attack_rows | candidate_attack_pct | exact_normal_behavior_conflicts |
| --- | --- | --- | --- |
| candidate_tree_consensus | 1473 | 14.73 | 84 |
| candidate_catboost_hybrid | 1474 | 14.74 | 84 |
| candidate_tree_union | 1675 | 16.75 | 85 |
| candidate_catboost_stream_context | 4291 | 42.91 | 349 |

- `tree_consensus`: both supervised trees agree, or a high-confidence protocol/SYN rule fires.
- `catboost_hybrid`: CatBoost plus high-confidence protocol/SYN rules.
- `tree_union`: either tree, row Isolation Forest, or high-confidence rules.
- `catboost_stream_context`: propagates suspicious SYN stream context to every packet in that stream.

No submission file is written automatically.
