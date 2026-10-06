# IoT Attack Detection EDA

## Dataset overview

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| X_train.csv | 100,000 | 23 |
| X_test.csv | 10,000 | 23 |
| sample_submission.csv | 10,000 | 2 |

- Training labels are absent because every training row is known Normal.
- Feature columns: 22 after excluding `Id`.
- Train duplicate feature rows: 71,212.
- Test duplicate feature rows: 5,687.
- Test rows matching a Normal row including capture-local `tcp.stream`: 2,973.
- Test rows matching Normal packet behavior after excluding `tcp.stream`: 3,115.
- Current `submission.csv` invalid or missing labels: 10,000.

## Input validation

- None

The real submission schema is `Id,label`. The capitalization must match the sample file.

## Missingness shifts

| feature | train_missing_pct | test_missing_pct | missing_shift |
| --- | --- | --- | --- |
| mqtt.qos | 64.319 | 73.74 | 9.421 |
| mqtt.retain | 64.319 | 73.74 | 9.421 |
| mqtt.topic_len | 64.319 | 73.22 | 8.901 |
| tcp.analysis.initial_rtt | 46.822 | 54.39 | 7.568 |
| mqtt.msgtype | 56.825 | 62.9 | 6.075 |
| mqtt.len | 56.825 | 62.9 | 6.075 |
| mqtt.conflag.qos | 97.515 | 96.02 | 1.495 |
| mqtt.conflag.cleansess | 97.515 | 96.02 | 1.495 |
| mqtt.kalive | 97.515 | 96.02 | 1.495 |
| mqtt.username_len | 97.515 | 96.02 | 1.495 |

Missing values are structurally meaningful here: MQTT fields are absent on TCP-only frames. They should not be blindly mean-imputed without missingness indicators.

## Values in test that are absent from Normal training

| feature | train_unique | test_unique | test_unseen_count | test_unseen_pct |
| --- | --- | --- | --- | --- |
| tcp.analysis.initial_rtt | 2485 | 387 | 4561 | 45.61 |
| tcp.window_size | 456 | 169 | 2537 | 25.37 |
| frame.len | 60 | 123 | 829 | 8.29 |
| tcp.len | 57 | 119 | 600 | 6 |
| mqtt.len | 28 | 93 | 596 | 5.96 |
| mqtt.kalive | 1 | 3 | 175 | 1.75 |
| mqtt.topic_len | 9 | 41 | 111 | 1.11 |
| mqtt.msgtype | 6 | 8 | 94 | 0.94 |
| mqtt.username_len | 10 | 7 | 54 | 0.54 |
| mqtt.passwd_len | 6 | 7 | 54 | 0.54 |

There are 6,062 test rows with at least one non-null value never observed in Normal training and 1,409 rows outside at least one Normal numeric range. Continuous features such as RTT naturally create unseen values, so support violations are clues rather than labels.

## Strong rule candidates

| feature | unseen_test_rows | outside_train_range_rows |
| --- | --- | --- |
| tcp.analysis.initial_rtt | 4561 | 528 |
| tcp.window_size | 2537 | 882 |
| frame.len | 829 | 0 |
| tcp.len | 600 | 0 |
| mqtt.len | 596 | 49 |
| mqtt.kalive | 175 | 175 |
| mqtt.topic_len | 111 | 111 |
| mqtt.msgtype | 94 | 0 |
| mqtt.username_len | 54 | 54 |
| mqtt.passwd_len | 54 | 54 |
| mqtt.conflag.willflag | 49 | 49 |

Rows with explicit MQTT state/value violations are especially interesting because attack types include invalid subscription, dictionary, Will-payload, and publish attacks. TCP-only SYN patterns should be analyzed separately for SYN flooding.

## Identifier-aware behavioral novelty

`tcp.stream` is a connection identifier local to each packet capture, so it must not determine whether a packet matches known Normal behavior. Excluding both `Id` and `tcp.stream` gives:

- 3,115 rows matching at least one Normal packet behavior.
- 6,885 rows with behavior absent from Normal training.
- 4,984 rows belonging to a novel pattern repeated in the test set.

### Mutually exclusive behavior cohorts

| behavior_cohort | rows | streams | unique_behavior_patterns | mqtt_rows | syn_without_ack_rows | median_test_pattern_frequency | median_support_violations | median_range_violations | test_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| other_repeated_novel | 3901 | 107 | 434 | 1497 | 0 | 15 | 1 | 0 | 39.01 |
| seen_normal_behavior | 3115 | 225 | 56 | 908 | 223 | 223 | 0 | 0 | 31.15 |
| other_novel_singleton | 1597 | 225 | 1597 | 704 | 0 | 1 | 1 | 0 | 15.97 |
| novel_syn_without_ack | 786 | 564 | 3 | 0 | 786 | 553 | 1 | 0 | 7.86 |
| mqtt_shape_outside_normal | 332 | 44 | 57 | 332 | 0 | 33 | 4 | 1 | 3.32 |
| mqtt_control_outside_normal | 269 | 52 | 267 | 269 | 0 | 1 | 7 | 2 | 2.69 |

These cohorts are descriptive triage groups, not inferred ground-truth labels. `seen_normal_behavior` is the strongest Normal anchor. MQTT control/shape violations are stronger attack candidates than generic TCP novelty because they violate protocol-specific Normal support. `novel_syn_without_ack` isolates previously unseen SYN-without-ACK packets, while the remaining novelty groups may contain both attacks and capture shift.

### Stream-disjoint Normal calibration

| fold | holdout_rows | holdout_streams | behavior_matches | behavior_match_pct | natural_novelty_pct |
| --- | --- | --- | --- | --- | --- |
| 0 | 36529 | 1 | 0 | 0 | 100 |
| 1 | 15868 | 626 | 2479 | 15.623 | 84.377 |
| 2 | 15868 | 619 | 2451 | 15.446 | 84.554 |
| 3 | 15868 | 629 | 2488 | 15.679 | 84.321 |
| 4 | 15867 | 626 | 2502 | 15.769 | 84.231 |

Even known-Normal holdout streams contain substantial behavior absent from the other Normal streams. Therefore, novelty alone cannot be treated as an Attack label. Conversely, test behavior matches Normal more often than any holdout fold (31.150%), reinforcing that exact matches are useful high-confidence Normal anchors despite incomplete Normal coverage.

### Largest contiguous cohort runs by `Id`

| behavior_cohort | start_id | end_id | rows | streams | unique_behavior_patterns |
| --- | --- | --- | --- | --- | --- |
| other_repeated_novel | 2424 | 2432 | 9 | 6 | 9 |
| other_repeated_novel | 5321 | 5329 | 9 | 3 | 9 |
| other_repeated_novel | 9174 | 9182 | 9 | 6 | 9 |
| other_repeated_novel | 3712 | 3719 | 8 | 4 | 8 |
| other_repeated_novel | 895 | 901 | 7 | 6 | 7 |
| other_repeated_novel | 6752 | 6758 | 7 | 5 | 7 |
| other_repeated_novel | 7405 | 7411 | 7 | 6 | 7 |
| other_repeated_novel | 9513 | 9519 | 7 | 3 | 7 |
| seen_normal_behavior | 343 | 348 | 6 | 5 | 4 |
| other_repeated_novel | 1505 | 1510 | 6 | 5 | 6 |
| seen_normal_behavior | 2534 | 2539 | 6 | 2 | 4 |
| seen_normal_behavior | 2952 | 2957 | 6 | 5 | 5 |

Long homogeneous runs would suggest capture ordering leakage; short fragmented runs suggest the test rows were shuffled. This structure should be inspected before using `Id` or row order in a model.

## TCP stream behavior

| Statistic | Normal train streams | Mixed test streams |
| --- | ---: | ---: |
| Stream count | 2,501 | 600 |
| Median packets per stream | 9.0 | 1.0 |
| Maximum packets per stream | 36,529 | 2,550 |
| Median TCP-only ratio | 0.778 | 1.000 |
| Median SYN-without-ACK ratio | 0.111 | 1.000 |

Test streams outside at least one Normal aggregate range: 596. Test streams containing at least one unseen low-cardinality Normal state: 600.

### Largest stream-level distribution shifts

| feature | train_median | test_median | median_shift_iqr | test_outside_range_streams |
| --- | --- | --- | --- | --- |
| tcp_window_max | 64620 | 512 | 64107999999999.992 | 98 |
| tcp_window_mean | 31886.556 | 512 | 31374555555555.555 | 90 |
| tcp_window_min | 5755 | 512 | 5243000000000 | 75 |
| tcp_len_sum | 101 | 0 | 101000000000 | 0 |
| frame_len_max | 151 | 54 | 97000000000 | 371 |
| tcp_len_max | 97 | 0 | 97000000000 | 0 |
| frame_len_mean | 67 | 54 | 13000000000 | 371 |
| tcp_len_mean | 11.222 | 0 | 11222222222.222 | 0 |
| packet_count | 9 | 1 | 8000000000 | 0 |
| unique_packet_patterns | 9 | 1 | 8000000000 | 1 |
| ack_count | 8 | 0 | 8000000000 | 0 |
| row_span | 9 | 1 | 8000000000 | 0 |

### Stream features exceeding the Normal envelope

| feature | train_min | train_max | test_outside_range_streams | test_unseen_state_streams |
| --- | --- | --- | --- | --- |
| frame_len_mean | 62 | 150.599 | 371 | 0 |
| frame_len_max | 62 | 1514 | 371 | 0 |
| packet_density | 0.105 | 1 | 224 | 0 |
| syn_count | 0 | 2 | 222 | 222 |
| tcp_window_max | 256 | 64620 | 98 | 445 |
| tcp_window_mean | 253.862 | 35190 | 90 | 0 |
| syn_without_ack_count | 0 | 2 | 84 | 84 |
| tcp_window_min | 250 | 5760 | 75 | 593 |
| syn_with_ack_count | 0 | 1 | 69 | 69 |
| keepalive_max | 15 | 15 | 51 | 51 |
| mqtt_type_1_0_count | 0 | 1 | 50 | 50 |
| mqtt_type_2_0_count | 0 | 1 | 49 | 49 |

These connection aggregates are more aligned with the named attacks than isolated packet rows. In particular, SYN floods should concentrate in `syn_without_ack_*`, while dictionary and MQTT attacks should alter CONNECT/CONACK, message-type, credential, topic, payload, and Will behavior within a stream.

## Modeling implications

1. Treat this as one-class anomaly detection, not ordinary supervised classification.
2. Exclude `Id` from row-level models. Treat `tcp.stream` cautiously because it identifies connections and may encode capture grouping.
3. Keep missingness patterns and protocol-aware features.
4. Use exact Normal behavior matches that exclude `tcp.stream` as a high-confidence Normal anchor set.
5. Combine rule signals with multiple anomaly models; do not select the final threshold from a single score distribution.
6. Optimize and report F1 only when pseudo-label or trusted validation labels exist; Normal-only cross-validation can measure false-positive stability but not competition F1.

## Generated artifacts

- `feature_summary.csv`: missingness, cardinality, support, ranges, and medians.
- `rule_summary.csv`: per-feature support and range violations.
- `test_anomaly_signals.csv`: row-level EDA indicators, not final predictions.
- `train_stream_features.csv` and `test_stream_features.csv`: connection-level aggregates.
- `stream_feature_summary.csv`: train/test stream-envelope comparison.
- `test_stream_signals.csv`: connection-level novelty indicators, not final predictions.
- `test_behavior_profiles.csv`: identifier-aware packet novelty and descriptive cohorts.
- `behavior_cohort_summary.csv`: sizes and characteristics of mutually exclusive cohorts.
- `behavior_signature_runs.csv`: contiguous cohort runs in test `Id` order.
- `normal_behavior_baseline.csv`: natural novelty measured on stream-disjoint Normal folds.
- `summary.json`: machine-readable dataset checks and headline counts.
