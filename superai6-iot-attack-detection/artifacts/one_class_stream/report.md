# One-class + synthetic attacks + stream features

## Design

- Isolation Forest is fitted only on known-Normal packets.
- CatBoost is fitted on Normal packets and five protocol-informed synthetic families.
- Both estimators receive packet features and capture-stable `tcp.stream` aggregates.
- Final variants use native estimator predictions; no probability cutoff or protocol-rule postprocessing is used.
- Validation keeps every `tcp.stream` wholly inside one fold and excludes one synthetic family from supervised training.

## Validation summary

| model | mean_normal_fpr | max_normal_fpr | mean_held_out_recall | min_held_out_recall | mean_proxy_f1_at_10pct | mean_proxy_f1_at_20pct |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_stream_native | 0 | 0 | 0.2 | 0 | 0.2 | 0.2 |
| iforest_stream_native | 0.215 | 1 | 0.278 | 0.017 | 0.136 | 0.178 |
| model_consensus | 0 | 0 | 0.004 | 0 | 0.007 | 0.007 |
| model_union | 0.215 | 1 | 0.474 | 0.017 | 0.317 | 0.366 |

## Leave-one-family-out details

| model | held_out_family | normal_fpr | held_out_family_recall | proxy_f1_at_10pct | proxy_f1_at_20pct |
| --- | --- | --- | --- | --- | --- |
| catboost_stream_native | syn_flood_row | 0 | 0 | 0 | 0 |
| iforest_stream_native | syn_flood_row | 1 | 1 | 0.182 | 0.333 |
| model_consensus | syn_flood_row | 0 | 0 | 0 | 0 |
| model_union | syn_flood_row | 1 | 1 | 0.182 | 0.333 |
| catboost_stream_native | mqtt_invalid_control | 0 | 0 | 0 | 0 |
| iforest_stream_native | mqtt_invalid_control | 0.01 | 0.035 | 0.062 | 0.064 |
| model_consensus | mqtt_invalid_control | 0 | 0 | 0 | 0 |
| model_union | mqtt_invalid_control | 0.01 | 0.035 | 0.062 | 0.064 |
| catboost_stream_native | mqtt_credential | 0 | 0 | 0 | 0 |
| iforest_stream_native | mqtt_credential | 0.005 | 0.017 | 0.033 | 0.033 |
| model_consensus | mqtt_credential | 0 | 0 | 0 | 0 |
| model_union | mqtt_credential | 0.005 | 0.017 | 0.033 | 0.033 |
| catboost_stream_native | mqtt_will_payload | 0 | 1 | 1 | 1 |
| iforest_stream_native | mqtt_will_payload | 0.014 | 0.019 | 0.033 | 0.035 |
| model_consensus | mqtt_will_payload | 0 | 0.019 | 0.037 | 0.037 |
| model_union | mqtt_will_payload | 0.014 | 1 | 0.94 | 0.972 |
| catboost_stream_native | mqtt_publish | 0 | 0 | 0 | 0 |
| iforest_stream_native | mqtt_publish | 0.046 | 0.32 | 0.369 | 0.425 |
| model_consensus | mqtt_publish | 0 | 0 | 0 | 0 |
| model_union | mqtt_publish | 0.046 | 0.32 | 0.369 | 0.425 |

## Final training

| normal_training_rows | synthetic_training_rows | synthetic_attack_families | iforest_contamination | manual_probability_threshold | protocol_rule_postprocessing |
| --- | --- | --- | --- | --- | --- |
| 25000 | 3750 | syn_flood_row,mqtt_invalid_control,mqtt_credential,mqtt_will_payload,mqtt_publish | 0.01 | False | False |

## Test candidates

| variant | candidate_attack_rows | candidate_attack_pct | exact_normal_behavior_conflicts | streams_touched |
| --- | --- | --- | --- | --- |
| candidate_catboost_stream_native | 2368 | 23.68 | 174 | 41 |
| candidate_iforest_stream_native | 3612 | 36.12 | 2148 | 432 |
| candidate_model_consensus | 547 | 5.47 | 11 | 39 |
| candidate_model_union | 5433 | 54.33 | 2311 | 434 |

`model_consensus` requires both estimators to flag a row. `model_union` accepts either model. Candidate generation never changes `submission.csv`.
