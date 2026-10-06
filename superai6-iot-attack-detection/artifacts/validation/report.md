# Proxy validation report

## What this validates

- Model fitting, threshold calibration, and Normal evaluation use disjoint `tcp.stream` groups.
- Thresholds are calibrated at several Normal-score quantiles.
- Synthetic stress tests cover row-level SYN flood, invalid MQTT control state, credential abuse, Will payload, and publish anomalies.
- Proxy F1 is recomputed for several assumed Attack prevalences.

This is not a leaderboard F1 estimate. There are no trusted Attack labels locally, and synthetic attacks are intentionally simplified.

## Cross-validation summary

| model | threshold_quantile | mean_normal_fpr | max_normal_fpr | mean_normal_rule_hit_rate | mean_synthetic_macro_recall | min_synthetic_macro_recall | mean_recall__syn_flood_row | mean_recall__mqtt_invalid_control | mean_recall__mqtt_credential | mean_recall__mqtt_will_payload | mean_recall__mqtt_publish |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_features | 0.95 | 0.245 | 1 | 0 | 0.627 | 0.318 | 0.8 | 0.249 | 0.774 | 0.31 | 1 |
| all_features | 0.975 | 0.059 | 0.115 | 0 | 0.5 | 0.318 | 0.4 | 0.164 | 0.766 | 0.169 | 1 |
| all_features | 0.99 | 0.014 | 0.022 | 0 | 0.266 | 0.22 | 0 | 0.022 | 0.297 | 0.027 | 0.984 |
| all_features | 0.995 | 0.011 | 0.022 | 0 | 0.232 | 0.202 | 0 | 0.012 | 0.17 | 0.018 | 0.958 |
| protocol_hybrid | 0.95 | 0.084 | 0.251 | 0 | 0.92 | 0.8 | 0.6 | 1 | 1 | 1 | 1 |
| protocol_hybrid | 0.975 | 0.082 | 0.251 | 0 | 0.84 | 0.8 | 0.2 | 1 | 1 | 1 | 1 |
| protocol_hybrid | 0.99 | 0.041 | 0.084 | 0 | 0.84 | 0.8 | 0.2 | 1 | 1 | 1 | 1 |
| protocol_hybrid | 0.995 | 0.016 | 0.066 | 0 | 0.84 | 0.8 | 0.2 | 1 | 1 | 1 | 1 |
| protocol_stable | 0.95 | 0.084 | 0.251 | 0 | 0.589 | 0.405 | 0.6 | 0.121 | 1 | 0.226 | 1 |
| protocol_stable | 0.975 | 0.082 | 0.251 | 0 | 0.506 | 0.402 | 0.2 | 0.12 | 1 | 0.21 | 1 |
| protocol_stable | 0.99 | 0.041 | 0.084 | 0 | 0.413 | 0.207 | 0.2 | 0.04 | 0.774 | 0.049 | 1 |
| protocol_stable | 0.995 | 0.016 | 0.066 | 0 | 0.298 | 0.202 | 0.2 | 0.034 | 0.21 | 0.046 | 1 |

`normal_fpr` is the most trustworthy local metric. Synthetic recall measures whether a model notices deliberately corrupted protocol behavior. A good candidate needs both a stable Normal FPR and broad family recall.

## Proxy F1 sensitivity at 20% assumed Attack prevalence

| model | threshold_quantile | assumed_attack_prevalence | mean_proxy_f1 | min_proxy_f1 | max_proxy_f1 |
| --- | --- | --- | --- | --- | --- |
| all_features | 0.95 | 0.2 | 0.545 | 0.333 | 0.666 |
| all_features | 0.975 | 0.2 | 0.568 | 0.452 | 0.638 |
| all_features | 0.99 | 0.2 | 0.401 | 0.346 | 0.452 |
| all_features | 0.995 | 0.2 | 0.361 | 0.329 | 0.452 |
| protocol_hybrid | 0.95 | 0.2 | 0.829 | 0.666 | 0.905 |
| protocol_hybrid | 0.975 | 0.2 | 0.788 | 0.666 | 0.888 |
| protocol_hybrid | 0.99 | 0.2 | 0.84 | 0.796 | 0.888 |
| protocol_hybrid | 0.995 | 0.2 | 0.882 | 0.877 | 0.888 |
| protocol_stable | 0.95 | 0.2 | 0.608 | 0.505 | 0.67 |
| protocol_stable | 0.975 | 0.2 | 0.542 | 0.504 | 0.619 |
| protocol_stable | 0.99 | 0.2 | 0.508 | 0.343 | 0.671 |
| protocol_stable | 0.995 | 0.2 | 0.407 | 0.331 | 0.696 |

If the true prevalence differs, precision and F1 will change even when model recall/FPR stay fixed. See `prevalence_sensitivity.csv` for 5% through 50% assumptions.

## Test candidate rates

| model | threshold_quantile | threshold | protocol_rule_rows | candidate_attack_rows | candidate_attack_pct |
| --- | --- | --- | --- | --- | --- |
| all_features | 0.95 | 0.052 | 0 | 2092 | 20.92 |
| all_features | 0.975 | 0.069 | 0 | 1316 | 13.16 |
| all_features | 0.99 | 0.119 | 0 | 613 | 6.13 |
| all_features | 0.995 | 0.142 | 0 | 511 | 5.11 |
| protocol_stable | 0.95 | 0.044 | 0 | 1920 | 19.2 |
| protocol_stable | 0.975 | 0.078 | 0 | 950 | 9.5 |
| protocol_stable | 0.99 | 0.094 | 0 | 442 | 4.42 |
| protocol_stable | 0.995 | 0.14 | 0 | 402 | 4.02 |
| protocol_hybrid | 0.95 | 0.044 | 328 | 2021 | 20.21 |
| protocol_hybrid | 0.975 | 0.078 | 328 | 1093 | 10.93 |
| protocol_hybrid | 0.99 | 0.094 | 328 | 595 | 5.95 |
| protocol_hybrid | 0.995 | 0.14 | 328 | 555 | 5.55 |

Large disagreement between calibrated Normal quantiles and test candidate rates indicates capture shift or a high Attack prevalence. These rates are diagnostics only; this script does not write `submission.csv`.

## Decision rule before submission

1. Reject configurations with unstable or excessive stream-disjoint Normal FPR.
2. Do not trust a high synthetic F1 if SYN or another family has near-zero recall.
3. Prefer the capture-stable feature set when raw RTT/window features cause test candidate inflation.
4. Add stream-level scoring before submission if row-level SYN stress recall is weak.
5. Create a submission only after reviewing the selected score distribution and threshold sensitivity.
