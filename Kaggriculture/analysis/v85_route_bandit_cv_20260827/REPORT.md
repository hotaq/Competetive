# V85 Observable Route Contextual-Bandit CV — 2026-08-27

## Verdict

Status: **fail_stop_no_checkpoint**.

Nested held-family observable selection missed at least one locked mechanical/economic gate. Stop without checkpoint, runtime, or submission.

This is development-only Rust-prefix evidence. No sealed/old/fresh/novel trace, checkpoint, runtime candidate, submission, or upload was opened or created.

## Locked observable protocol

- 32 candidate-relative contexts: four proxy families x four consumed seeds x two candidate seats.
- Live Rust prefix only; observation captured before actions at steps 24/48/72/96/120 and execution stopped before the step-120 action.
- `encode_selector_features` contributes 81 values per snapshot; model input is exactly 405 finite values and contains no explicit seat/family/seed/ID.
- Selectable actions are exactly KEEP V67, FORCE B120, FORCE C120. All five V84 margins are retained separately for audit.
- CPU-only nested leave-one-family-out CV; every normalization and ridge fit uses only its current training fold.
- Uncertainty is delete-one-family jackknife prediction variance plus grouped-OOF residual RMSE. LCB must clear both absolute-positive and delta-threshold gates.

## Outer held-family results

| Held family | Wins | V67 wins | Worst | Worst Δ | Mean Δ | Interventions | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| v16_route_family | 5/8 | 5/8 | -3585 | 0 | 0.00 | 0 | FAIL |
| ryo_selector_family | 5/8 | 5/8 | -21048 | 0 | 0.00 | 0 | FAIL |
| route_99612231_family | 8/8 | 8/8 | 3741 | 0 | 0.00 | 0 | PASS |
| robust_cluster_family | 8/8 | 8/8 | 1237 | 0 | 0.00 | 0 | PASS |

## Aggregate

- Wins: 26/32 (V67 26/32).
- Worst margin: -21048.
- Paired worst / mean delta: 0 / 0.00.
- CVaR10 margin: -12197.75.
- Interventions: 0/32.

## Mechanical and guardrail gates

- contexts_32: **PASS**
- feature_shape_32x405: **PASS**
- exact_v84_prefix_parity_32_of_32: **PASS**
- identifiers_separate_from_model_input: **PASS**
- all_five_v84_margins_retained: **PASS**
- candidate_sanitizer_zero_32_of_32: **PASS**
- all_12960_feature_values_finite: **PASS**
- no_forbidden_model_input_fields: **PASS**
- Independent reproduction: **PASS**
- Overall locked gate: **FAIL**

## Exact hashes

- Frozen V67: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- Frozen V84 results: `e7f1e1c701fce8951bb57b02ab8d408a49ef916170a3065dd782fa4c7286cd55`
- Feature collector: `a5d69bd1c447f7b0f5a5eebf0020a71163778e66dc5bedd9b999771e3d1749ef`
- CV runner: `1e3b69b4a92eb8e6760ba2726eee377592ecea722284e27ec8e64195aa902ae9`
- Dataset file: `9524ddc12f2bf5f5b575a10e43511ea26b6c51ed6092c099936ac14f52af01b8`
- Dataset core: `f44e78d8b8075d61404cf88aa8067e15df35b9f1723a82df5f926da1a92e3f47`
- CV core: `34aeaf21cc932e726d635d87e8fccee7e563acf5ce682b129fcd6a43f7b0242d`
- Results file: `482ad1addad3de681f02340d3c9445de52701e1b9045395804cd903bf549af5e`

A failed gate is a hard stop: do not serialize weights, build a runtime, or submit.
