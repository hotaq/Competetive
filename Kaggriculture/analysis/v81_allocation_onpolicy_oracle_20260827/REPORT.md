# V81 Allocation On-Policy Oracle — 2026-08-27

## Result

Status: **FAIL_STOP**.

This is a development-only paired Rust-engine oracle. It does not train, create, package, or upload a candidate and is not promotion evidence.

Fail-fast stop reason: `executor_partial_or_pending:v16_route_family:seed=27100100:seat=0:FAIL_PARTIAL_INTERVENTION`.

## Availability versus benefit

Availability means the exact V67 step-248 signature was present and the strict option started. Benefit separately means positive paired terminal margin delta.

| Family | Games | Available / complete | Benefit / neutral / regress | Wins base→forced (Δ) | Mean base→forced (Δ) | Worst base→forced (Δ) | Paired Δ mean / worst | Any regression |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v16_route_family | 1 | 1 / 0 | 0 / 0 / 1 | 0→0 (+0) | -3585.0→-11737.0 (-8152.0) | -3585.0→-11737.0 (-8152.0) | -8152.0 / -8152.0 | YES |

Only the first family has a result: fail-fast left the other three families unmeasured, so no cross-family benefit claim is possible.

## Executor and action-contract evidence

- Completed paired games: 1 / 32.
- Forced option starts/completions/partials: 1 / 0 / 1.
- Candidate sanitizer-changed steps, ABSTAIN/forced: 0 / 0.
- Exact linked proposed/sanitized command differences: 1 / 1.
- Out-of-contract candidate action differences: 368 (proposed and sanitized counted separately).

## First exact failure

The option rewrote step 248, then aborted at step 249 with `linked_signature_mismatch`. The on-policy V67 base emitted `hands[4]=PASS` after the SHEEP purchase instead of the frozen `PICKUP COW 2` token, so the current executor cannot complete the chain.

This diagnostic game regressed by -8152 terminal margin (baseline -3585, forced -11737).

## Exact action window, steps 248-261

Every movement step is included. Values show candidate hand 4 and the complete market list for baseline versus forced, before and after Rust sanitization.

| Step | Base proposed H4 | Forced proposed H4 | Base sanitized H4 | Forced sanitized H4 | Base proposed market | Forced proposed market |
|---:|---|---|---|---|---|---|
| 248 | `["DROP"]` | `["DROP"]` | `["DROP"]` | `["DROP"]` | `[["SELL","MELON",6],["BUY_ANIMAL","COW",2]]` | `[["SELL","MELON",6],["BUY_ANIMAL","SHEEP",2]]` |
| 249 | `["PICKUP","COW",2]` | `["PASS"]` | `["PICKUP","COW",2]` | `["PASS"]` | `[["BUY_SEED","MELON",4]]` | `[["BUY_SEED","MELON",4]]` |
| 250 | `["PICKUP","WHEAT",2]` | `["PICKUP","WHEAT",2]` | `["PICKUP","WHEAT",2]` | `["PICKUP","WHEAT",2]` | `[["BUY_PRODUCT","WHEAT",1]]` | `[["BUY_PRODUCT","WHEAT",1]]` |
| 251 | `["WEST"]` | `["WEST"]` | `["WEST"]` | `["WEST"]` | `[]` | `[]` |
| 252 | `["NORTH"]` | `["NORTH"]` | `["NORTH"]` | `["NORTH"]` | `[["SELL","MELON",6],["BUY_LAND"],["SELL","MELON",6]]` | `[["SELL","MELON",6],["BUY_LAND"],["SELL","MELON",6]]` |
| 253 | `["SOUTH"]` | `["SOUTH"]` | `["SOUTH"]` | `["SOUTH"]` | `[["BUY_PRODUCT","WHEAT",2]]` | `[["BUY_PRODUCT","WHEAT",2]]` |
| 254 | `["SOUTH"]` | `["SOUTH"]` | `["SOUTH"]` | `["SOUTH"]` | `[]` | `[]` |
| 255 | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `[]` | `[]` |
| 256 | `["PLACE","COW",1]` | `["PASS"]` | `["PLACE","COW",1]` | `["PASS"]` | `[]` | `[]` |
| 257 | `["FEED"]` | `["PASS"]` | `["FEED"]` | `["PASS"]` | `[["SELL","MELON",6]]` | `[["SELL","MELON",6]]` |
| 258 | `["CARE"]` | `["PASS"]` | `["CARE"]` | `["PASS"]` | `[["SELL","MELON",6],["BUY_PRODUCT","WHEAT",2]]` | `[["SELL","MELON",6],["BUY_PRODUCT","WHEAT",2]]` |
| 259 | `["EAST"]` | `["EAST"]` | `["EAST"]` | `["EAST"]` | `[]` | `[]` |
| 260 | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `["BUILD_PASTURE"]` | `[]` | `[]` |
| 261 | `["PLACE","COW",1]` | `["PASS"]` | `["PLACE","COW",1]` | `["PASS"]` | `[["SELL","MELON",6]]` | `[["SELL","MELON",6]]` |

Proxy actions may react on-policy to the changed state. The four-path constraint applies to the candidate's proposed and sanitized streams.

## Frozen scope

- Representatives: `v22`, `v12_ryo`, `v44_route`, `robust_cluster`.
- Already-consumed seeds only: `27100100..27100103`; both seats.
- No replay, holdout, or live Kaggle action/reward trace was opened.
- No training, candidate, package, submission, or upload was created.

## Reproduction

```bash
.venv/bin/python analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py
.venv/bin/python analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py --enrich-existing
.venv/bin/ruff check analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py
.venv/bin/ruff format --check analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py
```

Input hashes and exact action-trace hashes are in `results.json`.
