# Robust anomaly ensemble and protocol rules

## Research-backed attack signals

- Dictionary: abnormal CONNECT credential lengths and repeated authentication behavior.
- Invalid subscription: SUBSCRIBE/SUBACK message types or topic shapes absent from Normal support.
- Will payload: CONNECT packets with unseen Will flag/state or abnormal MQTT length.
- Publish: PUBLISH packets with topic, MQTT payload, or TCP payload outside the Normal envelope.
- TCP SYN flood: streams with more SYN-without-ACK packets than any Normal stream; packet and stream-context variants are kept separate.

Sources:
- mqtt_spec: https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html
- syn_flood_rfc: https://www.rfc-editor.org/rfc/rfc4987.html
- mqtt_authentication: https://www.hivemq.com/blog/mqtt-security-fundamentals-authentication-username-password/

## Validation design

- Fit, threshold calibration, and Normal evaluation use disjoint `tcp.stream` groups.
- Six anomaly scores use only capture-stable protocol/length features; raw RTT, window size, `Id`, and `tcp.stream` are excluded.
- Thresholds use the 0.995 Normal calibration quantile.
- One synthetic family is stress-tested in each fold; these metrics are diagnostics, not leaderboard estimates.

## Validation summary

| model | mean_normal_fpr | max_normal_fpr | mean_held_out_recall | min_held_out_recall | mean_proxy_f1_at_10pct | mean_proxy_f1_at_20pct |
| --- | --- | --- | --- | --- | --- | --- |
| rules_only | 0 | 0 | 1 | 1 | 1 | 1 |
| vote_4_or_rules | 0.005 | 0.02 | 1 | 1 | 0.978 | 0.99 |
| vote_3_or_mqtt_or_syn_stream | 0.02 | 0.025 | 1 | 1 | 0.919 | 0.962 |
| vote_3_or_rules | 0.02 | 0.025 | 1 | 1 | 0.919 | 0.962 |
| vote_2_or_rules | 0.032 | 0.076 | 1 | 1 | 0.882 | 0.942 |
| mahalanobis | 0.009 | 0.025 | 0.8 | 0 | 0.762 | 0.782 |
| vote_3 | 0.02 | 0.025 | 0.702 | 0 | 0.663 | 0.701 |
| vote_4 | 0.005 | 0.02 | 0.663 | 0 | 0.675 | 0.687 |
| vote_2 | 0.032 | 0.076 | 0.702 | 0 | 0.633 | 0.684 |
| knn | 0.021 | 0.077 | 0.681 | 0 | 0.652 | 0.683 |
| lof | 0.014 | 0.057 | 0.651 | 0 | 0.655 | 0.677 |
| iqr | 0.018 | 0.021 | 0.663 | 0 | 0.634 | 0.667 |
| mad | 0.018 | 0.021 | 0.663 | 0 | 0.634 | 0.667 |
| zscore | 0.018 | 0.021 | 0.663 | 0 | 0.634 | 0.667 |

## Per-family validation

| model | held_out_family | normal_fpr | held_out_family_recall | proxy_f1_at_20pct |
| --- | --- | --- | --- | --- |
| zscore | syn_flood_row | 0.007 | 0 | 0 |
| iqr | syn_flood_row | 0.007 | 0 | 0 |
| mad | syn_flood_row | 0.007 | 0 | 0 |
| mahalanobis | syn_flood_row | 0 | 0 | 0 |
| knn | syn_flood_row | 0 | 0 | 0 |
| lof | syn_flood_row | 0 | 0 | 0 |
| vote_2 | syn_flood_row | 0.007 | 0 | 0 |
| vote_3 | syn_flood_row | 0.007 | 0 | 0 |
| vote_4 | syn_flood_row | 0 | 0 | 0 |
| rules_only | syn_flood_row | 0 | 1 | 1 |
| vote_4_or_rules | syn_flood_row | 0 | 1 | 1 |
| vote_3_or_rules | syn_flood_row | 0.007 | 1 | 0.986 |
| vote_2_or_rules | syn_flood_row | 0.007 | 1 | 0.986 |
| vote_3_or_mqtt_or_syn_stream | syn_flood_row | 0.007 | 1 | 0.986 |
| zscore | mqtt_invalid_control | 0.019 | 1 | 0.963 |
| iqr | mqtt_invalid_control | 0.019 | 1 | 0.963 |
| mad | mqtt_invalid_control | 0.019 | 1 | 0.963 |
| mahalanobis | mqtt_invalid_control | 0.011 | 1 | 0.979 |
| knn | mqtt_invalid_control | 0.006 | 1 | 0.989 |
| lof | mqtt_invalid_control | 0.006 | 1 | 0.989 |
| vote_2 | mqtt_invalid_control | 0.024 | 1 | 0.954 |
| vote_3 | mqtt_invalid_control | 0.024 | 1 | 0.954 |
| vote_4 | mqtt_invalid_control | 0.001 | 1 | 0.998 |
| rules_only | mqtt_invalid_control | 0 | 1 | 1 |
| vote_4_or_rules | mqtt_invalid_control | 0.001 | 1 | 0.998 |
| vote_3_or_rules | mqtt_invalid_control | 0.024 | 1 | 0.954 |
| vote_2_or_rules | mqtt_invalid_control | 0.024 | 1 | 0.954 |
| vote_3_or_mqtt_or_syn_stream | mqtt_invalid_control | 0.024 | 1 | 0.954 |
| zscore | mqtt_credential | 0.021 | 1 | 0.96 |
| iqr | mqtt_credential | 0.021 | 1 | 0.96 |
| mad | mqtt_credential | 0.021 | 1 | 0.96 |
| mahalanobis | mqtt_credential | 0.007 | 1 | 0.987 |
| knn | mqtt_credential | 0.006 | 1 | 0.987 |
| lof | mqtt_credential | 0.002 | 0.851 | 0.915 |
| vote_2 | mqtt_credential | 0.025 | 1 | 0.952 |
| vote_3 | mqtt_credential | 0.021 | 1 | 0.96 |
| vote_4 | mqtt_credential | 0.002 | 1 | 0.996 |
| rules_only | mqtt_credential | 0 | 1 | 1 |
| vote_4_or_rules | mqtt_credential | 0.002 | 1 | 0.996 |
| vote_3_or_rules | mqtt_credential | 0.021 | 1 | 0.96 |
| vote_2_or_rules | mqtt_credential | 0.025 | 1 | 0.952 |
| vote_3_or_mqtt_or_syn_stream | mqtt_credential | 0.021 | 1 | 0.96 |
| zscore | mqtt_will_payload | 0.021 | 0.317 | 0.453 |
| iqr | mqtt_will_payload | 0.021 | 0.317 | 0.453 |
| mad | mqtt_will_payload | 0.021 | 0.317 | 0.453 |
| mahalanobis | mqtt_will_payload | 0.004 | 1 | 0.993 |
| knn | mqtt_will_payload | 0.077 | 0.403 | 0.471 |
| lof | mqtt_will_payload | 0.057 | 0.403 | 0.494 |
| vote_2 | mqtt_will_payload | 0.076 | 0.511 | 0.562 |
| vote_3 | mqtt_will_payload | 0.021 | 0.511 | 0.64 |
| vote_4 | mqtt_will_payload | 0.002 | 0.317 | 0.479 |
| rules_only | mqtt_will_payload | 0 | 1 | 1 |
| vote_4_or_rules | mqtt_will_payload | 0.002 | 1 | 0.996 |
| vote_3_or_rules | mqtt_will_payload | 0.021 | 1 | 0.96 |
| vote_2_or_rules | mqtt_will_payload | 0.076 | 1 | 0.867 |
| vote_3_or_mqtt_or_syn_stream | mqtt_will_payload | 0.021 | 1 | 0.96 |
| zscore | mqtt_publish | 0.02 | 1 | 0.961 |
| iqr | mqtt_publish | 0.02 | 1 | 0.961 |
| mad | mqtt_publish | 0.02 | 1 | 0.961 |
| mahalanobis | mqtt_publish | 0.025 | 1 | 0.952 |
| knn | mqtt_publish | 0.017 | 1 | 0.967 |
| lof | mqtt_publish | 0.008 | 1 | 0.985 |
| vote_2 | mqtt_publish | 0.025 | 1 | 0.952 |
| vote_3 | mqtt_publish | 0.025 | 1 | 0.952 |
| vote_4 | mqtt_publish | 0.02 | 1 | 0.961 |
| rules_only | mqtt_publish | 0 | 1 | 1 |
| vote_4_or_rules | mqtt_publish | 0.02 | 1 | 0.961 |
| vote_3_or_rules | mqtt_publish | 0.025 | 1 | 0.952 |
| vote_2_or_rules | mqtt_publish | 0.025 | 1 | 0.952 |
| vote_3_or_mqtt_or_syn_stream | mqtt_publish | 0.025 | 1 | 0.952 |

## Detector thresholds

| detector | quantile | threshold | test_hits |
| --- | --- | --- | --- |
| zscore | 0.995 | 11.406 | 568 |
| iqr | 0.995 | 114.478 | 568 |
| mad | 0.995 | 114.478 | 568 |
| mahalanobis | 0.995 | 27.757 | 682 |
| knn | 0.995 | 4.5 | 435 |
| lof | 0.995 | 23153231121.671 | 531 |

## Protocol rule counts

| rule | test_hits |
| --- | --- |
| rule_dictionary | 54 |
| rule_invalid_subscription | 94 |
| rule_will_payload | 49 |
| rule_publish | 59 |
| rule_invalid_control | 175 |
| rule_syn_packet | 390 |
| rule_syn_stream | 3644 |
| rule_mqtt_any | 328 |
| rule_high_confidence | 718 |

## Candidate variants

| variant | candidate_attack_rows | candidate_attack_pct | exact_normal_behavior_conflicts | streams_touched |
| --- | --- | --- | --- | --- |
| candidate_vote_2 | 642 | 6.42 | 16 | 61 |
| candidate_vote_3 | 571 | 5.71 | 16 | 61 |
| candidate_vote_4 | 415 | 4.15 | 0 | 61 |
| candidate_rules_only | 718 | 7.18 | 84 | 86 |
| candidate_vote_4_or_rules | 847 | 8.47 | 84 | 89 |
| candidate_vote_3_or_rules | 1003 | 10.03 | 100 | 89 |
| candidate_vote_2_or_rules | 1074 | 10.74 | 100 | 89 |
| candidate_vote_3_or_mqtt_or_syn_stream | 3764 | 37.64 | 364 | 89 |

No candidate is copied to `submission.csv` automatically.
