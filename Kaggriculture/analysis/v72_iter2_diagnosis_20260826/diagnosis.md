# V72 Iteration 2 diagnosis — old80 only

## Decision

**Stop promotion and do not tune quotas yet.** The first blocker is an execution-state semantic bug: V72 counts animals in the shed or a unit inventory as if they had already become live production capacity. It then releases or starves that multi-turn target, spends through its operating cash, and retries blocked land/work without invoking the existing inventory-recovery primitive.

The frozen V72 controller and wrapper were not modified. The trace covers action steps 120–718 and the final state at 719 for two exact old80 cases only. Both traces reproduce the frozen league result exactly.

## Representative cases

| Case | Episode | V72 money | Tape money | Margin | Paired V59 money |
|---|---:|---:|---:|---:|---:|
| Median representative (29 below the overall median margin) | 99614514, seat 0 | 1,682 | 135,589 | -133,907 | 106,073 |
| Worst | 99589578, seat 0 | 1,258 | 198,124 | -196,866 | 107,066 |

Margins are not directly comparable between V72 and V59 because changing the candidate changes shared-market interactions with the recorded tape. Candidate money is still a useful development oracle, not live-policy proof.

## Earliest failure chain

Worst episode 99589578:

1. Step 120 starts with 359 money. V72 sells 5 fertilizer, hires 2 hands, and picks up wheat; money becomes 826.
2. Step 121 emits another hire plus `BUY_ANIMAL COW 2`; money collapses to 24. The two cows remain in the shed.
3. At step 122 the controller already switches to strawberry because `FarmSnapshot.cows` includes live + shed + unit inventory. The purchase therefore looks like completion before either cow is placed.
4. Step 123 spends again on wheat seed, reaches 4 money, and `LAND_EXPANSION:NE` first returns `KG-EXEC-INVENTORY` while a unit holds wheat.
5. Live COW/SHEEP capacity reaches zero at step 167. Land remains one quadrant through step 719 despite 98 inventory-blocked land reports.
6. Step 696 liquidates 24 wheat for +1,118 but cannot monetize the stranded COW 2 / SHEEP 1; final money is 1,258.

Median episode 99614514 is not rescued by eventual progress:

- Land is first blocked at step 124 and expands only at step 173.
- It buys 19 cows, hires 130 hands, and buys 83 wheat products. Fixed-price listed purchases alone are bounded above by 11,804; realized negative step cash deltas total 14,652.
- It reaches COW 7 / SHEEP 4, sells some milk and wool, but all live animals are gone by step 575. It finishes with only WHEAT 8 and 1,682 money.
- Strawberry is physically below quota for 575 traced turns and is never selected while physically deficient; wheat's longest such starvation run is 497 turns.

## Quantified scheduler failure

| Signal | Median | Worst |
|---|---:|---:|
| Production reports marked `PLANNED` | 550 | 560 |
| `PLANNED` with no same-step composition progress | 294 (53.5%) | 246 (43.9%) |
| Target runs | 181 | 113 |
| One-step target runs | 109 | 54 |
| Land `KG-EXEC-INVENTORY` reports | 97 | 98 |
| HIRE quantity | 130 | 213 |
| Sum negative cash deltas | -14,652 | -8,342 |
| Sum positive cash deltas | +15,975 | +9,241 |

The no-same-step-progress count is supporting evidence only: deterministic movement may be useful even when composition is unchanged. The causal issue is that the Python controller has no persistent active target and interprets total ownership as physical animal progress. In source, animal snapshot construction combines tile, shed, and unit inventory before `_production_target` reads `snapshot.cows/sheep`; urgency contributes 4 while bounded deficit plus age contributes at most 3, so an urgent target can starve another indefinitely. `PLANNED` clears backoff, but does not mean the macro target is complete.

## Comparison with prior controllers

- V18 Gate 0 already showed that legal, deterministic state-aware execution is insufficient: 0/8 wins, mean money 17,107.875, mean margin -119,663.25. That is a different eight-game corpus, so it is descriptive only.
- V59 on the exact old80 corpus scored 66/80 wins, mean money 90,301.55, and worst money 44,099. V72 scored 0/80, mean money 1,578.875, and worst money 297.
- On the two traced seat-matched episodes, V59 earns 106,073 and 107,066 versus V72's 1,682 and 1,258. This rules out quota-only tuning as the next rational experiment.

## Ranked causes

1. **Owned/live conflation (direct, high confidence).** A shed purchase closes the animal deficit before deployment.
2. **No sticky completion plus starvation (direct, high confidence).** `PLANNED` is reselected next turn; urgency can dominate deficit and age forever.
3. **No cash-reserve admission (direct, high confidence).** Worst spends 826→24→4 in two turns while opening production is still incomplete.
4. **Backoff without recovery (direct, high confidence).** Inventory blocks merely defer retries; the existing Rust `RECOVERY:INVENTORY` primitive is used only in terminal liquidation by V72.
5. **Stranded terminal capital (symptom, high confidence).** Cows and sheep remain in the worst shed because liquidation can sell products, not undeployed animals.

## Minimal next iteration

No Rust schema change is needed.

1. Split `live_animals` from `owned_animals` in the Python snapshot. Keep irreversible quota floors on owned count; judge production completion and starvation on live count.
2. Make scheduling lexicographic: inventory recovery and urgent maintenance first, then completion of stranded unit work, then capital expansion, then new growth. Land expansion is inadmissible while any urgent/stranded state remains.
3. Store one `active_production_key` per player. Give it a bounded dwell until physical progress or explicit block; `PLANNED` alone cannot release it. After the dwell/progress boundary, round-robin among physical deficits and force the oldest deficit by 24 turns.
4. Preserve a 400-money floor when admitting fixed-price workforce/animal/seed/land targets. Cap animal growth to the currently affordable quantity, and require reserve plus land cost before expansion.
5. On `KG-EXEC-INVENTORY`, prepend the existing `RECOVERY:INVENTORY` target and suppress competing unit-heavy work until inventories clear.

Required synthetic tests: shed animals do not satisfy live completion; target persists across buy→pickup→place; urgent crop cannot starve a live-animal deficit beyond 24 turns; inventory block triggers recovery; cash never crosses the floor after a fixed-price purchase; player contexts remain isolated and deterministic; opening 0–119 and liquidation ≥696 remain unchanged.

## Oracle stop gate

Run only these same two old80 traces first. Do **not** open full old80 unless both pass:

- 719 transitions, final step 719, and zero post-handoff normalize/sanitize/fallback;
- cash ≥400 after every fixed-price purchase;
- any bought animal is live or under recovery within 24 turns;
- urgent maintenance and stranded unit inventory always preempt land expansion;
- no physical quota remains unselected for more than 48 turns;
- worst case reaches land ≥2 by step 192 and ends with no COW/SHEEP in shed;
- candidate money reaches at least half of paired V59: 53,036.5 (median representative) and 53,533 (worst).

If either replay fails one hard condition, stop and diagnose that condition. Do not run full old80, fresh, third/future, or upload.

## Reproduction and limitations

- Trace: `analysis/v72_iter2_diagnosis_20260826/trace_old80.py`
- Machine-readable summary: `analysis/v72_iter2_diagnosis_20260826/summary.json`
- The trace is deterministic, passes Ruff, and asserts exact frozen candidate/tape money before producing a summary.
- Action-tape opponents become strategically stale after state divergence. These are development screens, not proof of live win rate.
- Listed fixed purchase cost is an upper bound because an accepted listed quantity may still be only partially fillable.
