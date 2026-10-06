# V84 Route-Option On-Policy Oracle — 2026-08-27

## Verdict

Status: **pass_action_support_only**.

The cheap action-support falsifier passed. This permits only the separately gated observable-selection test, not promotion or submission.

This is development-only Rust-engine falsification evidence. It did not train a model, open a replay/holdout, build a submission, or upload anything.

## Frozen protocol

- Options fixed before results: `ABSTAIN_KEEP_V67`, `FORCE_BASE_SUFFIX`, `FORCE_A120_SUFFIX`, `FORCE_B120_SUFFIX`, `FORCE_C120_SUFFIX`.
- Forced choices are injected only into frozen V67's captured embedded V59 choice state at step 120; V67's full overlay remains callable.
- Four live executable proxy families, seeds 27100100..27100103, both candidate seats: 32 keys and 160 unique option games.
- Opponents are called live from changed state; no opponent replay tape is used.

## Mechanical gates

- Complete 719-transition games: 160 / 160.
- Candidate sanitizer / fallback / exception / partial events: 0 / 0 / 0 / 0.
- V67 proposed-action prefix parity through step 119: 128 checks, all=True.
- ABSTAIN full action+score parity against frozen catalog V67: 32 checks, all=True.
- Non-abstain differences never begin before step 120: 128 checks, all=True.
- Independent full rerun core hash match: True (`b65bfbfda6c0e81a1360a78c4ced8394c98a97d5b1a60a19fdc4761c9f885f2c`).

## Economic action-support gate

- Hindsight oracle positive: 32 / 32 keys (required 32/32).
- Positive oracle worst margin in every family: True.
- V67 losses rescued by non-abstain options: 6 / 6 (required 6/6).
- Smallest all-key positive fixed-action cover: 3.

| Family | Keys | V67 wins | Oracle wins | V67 worst | Oracle worst | Oracle mean |
|---|---:|---:|---:|---:|---:|---:|
| v16_route_family | 8 | 5 | 8 | -3585 | 396 | 6666.00 |
| ryo_selector_family | 8 | 5 | 8 | -21048 | 775 | 12224.38 |
| route_99612231_family | 8 | 8 | 8 | 3741 | 3741 | 9358.25 |
| robust_cluster_family | 8 | 8 | 8 | 1237 | 1237 | 10938.00 |

## Guardrails and hashes

- Sealed inputs opened: none.
- Old/fresh/novel replay actions or rewards opened: none.
- Training/checkpoint/candidate/package/submission/upload: none.
- Frozen V67 SHA-256: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`.
- Results SHA-256: `e7f1e1c701fce8951bb57b02ab8d408a49ef916170a3065dd782fa4c7286cd55`.
- Deterministic 32-key outcome core SHA-256: `b65bfbfda6c0e81a1360a78c4ced8394c98a97d5b1a60a19fdc4761c9f885f2c`.

The route catalog is frozen for this decision. No option may be added after seeing failed keys.
