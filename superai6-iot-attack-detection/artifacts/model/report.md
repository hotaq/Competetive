# Two-level IoT anomaly model

## Model

- Row layer: capture-stable Isolation Forest excluding raw RTT/window shift.
- Packet rules: MQTT control states and shapes outside the Normal envelope.
- Stream layer: Isolation Forest on protocol/ratio/count aggregates.
- SYN rule: stream `syn_without_ack_count` must exceed every Normal stream.

No competition submission is written by this script.

## Stream validation

| mean_normal_score_fpr | max_normal_score_fpr | mean_normal_syn_rule_fpr | mean_synthetic_syn_score_recall | mean_synthetic_syn_rule_recall | mean_synthetic_syn_hybrid_recall |
| --- | --- | --- | --- | --- | --- |
| 0.205 | 1 | 0 | 0.205 | 1 | 1 |

The SYN envelope rule is the primary flood detector. The stream Isolation Forest remains supporting evidence because train/test stream geometry differs substantially.

## Candidate variants

| variant | candidate_attack_rows | candidate_attack_pct | exact_normal_behavior_conflicts | streams_touched |
| --- | --- | --- | --- | --- |
| candidate_high_confidence | 718 | 7.18 | 84 | 86 |
| candidate_packet_ensemble | 945 | 9.45 | 85 | 89 |
| candidate_stream_context | 3725 | 37.25 | 349 | 89 |
| candidate_broad_stream | 7204 | 72.04 | 2565 | 464 |

- `high_confidence`: MQTT protocol rules plus SYN packets inside envelope-breaking streams.
- `packet_ensemble`: high-confidence signals plus row anomaly score.
- `stream_context`: labels every packet in a SYN-envelope-breaking stream.
- `broad_stream`: also labels every packet in a stream-level Isolation Forest outlier.

The difference between packet and stream-context variants reflects unknown competition label semantics. Review exact-Normal conflicts before selecting one.

## Component thresholds

| component | quantile | threshold |
| --- | --- | --- |
| row_protocol_stable | 0.995 | 0.14 |
| stream_isolation_forest | 0.995 | 0.239 |

## Test stream flags

| test_streams | stream_score_hits | syn_envelope_hits |
| --- | --- | --- |
| 600 | 384 | 84 |
