# Independent critic: next falsifiable experiments after V78

Date: 2026-08-27  
Scope: read-only synthesis of existing V59/V67/V71–V78 evidence and the active mission rubric. No replay was run, no sealed/future result was opened, no agent or submission was changed, and nothing was uploaded.

## Decision

Keep **V67 frozen as the incumbent**. None of the tested successors has earned promotion, and the next work should not be another whole-route splice or a broad closed-loop rewrite.

The best next hypothesis is a **small, state-conditioned allocation change at a real expansion event**. A cheap market-only upper-bound probe is second. If both lack enough causal headroom, the honest long-term path is a **V67-anchored residual RL policy trained on-policy in the official engine**, not replay imitation and not a route selector.

The mission remains failed. Its rubric requires one standalone archive with official-engine legality, at least 95% aggregate wins and positive worst-case against every available strong proxy family, 24 consecutive live wins against identified current Top1–20 opponents, no V16/V22 regression, and reproducible hashes (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/rubric.md:1`). Current evidence does not meet any decisive-promotion interpretation.

## Authoritative baseline facts

### Evidence

- V59 reached 66/80 on old80 and 79/95 on the then-fresh set, but its late-route oracle still stayed below the mission gate (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:6`).
- V67 preserved legality and exact standalone parity, but reached only 83/95 on the then-fresh set and 106/160 on the later disjoint current-Top20 development gate (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:7-8`). These sets are now consumed development evidence, not reusable holdouts.
- Uploaded V67 is exactly reproduced by the local standalone over 20,132/20,132 public action steps. Its observed public record is 24/28, worst margin -21,061, and opponent rank is absent, so it does not prove a Top1–20 streak (`analysis/v67_live_loss_audit_20260826/REPORT.md:6-23`).
- All four observed public losses and twenty base-route wins share the same opening/route, land unlocks, terminal land 3, hands 9, COW 11, SHEEP 4, and zero terminal inventory. That rules out opening failure, land timing, no-action, and terminal inventory stranding as the common explanation (`analysis/v67_live_loss_audit_20260826/REPORT.md:25-34`).
- The cohort-level gap becomes material late: loss mean margin is +476 at step 360, +438 at 480, -2,833 at 576, and -8,967 at 719. The loss cohort is milk-glutted and wool-scarce while requested sale quantities remain almost fixed (`analysis/v67_live_loss_audit_20260826/REPORT.md:36-63`).
- Fixed routes contain real hindsight headroom, but observable selection has not extracted it. Four routes reach 134/160, expanded experts 148/160, while linear leave-one-family-out selection reaches only 107–108/160 (`analysis/v59_v67_route_oracle_development160_20260826.md:11-19`, `:76-105`, `:135-153`). A later whole-route selector had a 80/80 hindsight oracle but only 64/80 or 63/80 actual selector performance (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:11`).

### Inference

V67's remaining failure is more consistent with **state-dependent economic allocation under a late shared-market regime** than with an executor, opening, land, workforce-count, or inventory-cleanup failure. This is supported but not proven causal: there are only four public losses, fixed-tape counterfactuals become strategically stale after divergence, and cow-heavy controls also win.

## Ranked next hypotheses

| Rank | Falsifiable hypothesis | Causal support | Generalization risk | Cost | Why it ranks here |
|---:|---|---|---|---|---|
| 1 | Reallocate a true new pasture expansion from COW to SHEEP only when an identity-free value condition is persistent | Medium-high structural support | High | Medium | It targets a real available decision that V78 proved exists, while preserving the V67 opening and route outside a narrow asset-allocation event |
| 2 | A market-only value oracle can show whether late sell timing/quantity has enough headroom without changing production | Medium-low | Medium-high | Low | It is the cheapest discriminating probe; failure should kill the remaining market-only branch before more policy work |
| 3 | A V67-anchored residual RL policy can learn a few local macro deltas on-policy and abstain elsewhere | Medium | High | High | Route/action headroom exists, but hand rules and whole-route selectors fail; on-policy residual learning is the remaining mechanism with plausible adaptability |

### 1. Expansion-allocation switch, not late replacement

**Hypothesis.** When V67 creates genuinely new pasture capacity, selecting COW versus SHEEP from persistent normalized MILK/WOOL value, current inventory, feed capacity, and observed shop demand can rescue the cow-heavy loss regime without altering the frozen opening or choosing a different whole route.

**Evidence.** V78 found that the step-248 COW x2 sequence in every target loss is new capacity expansion into two newly built pastures, not replacement. COW and SHEEP share the same pasture type, and no animals are removed in the window (`analysis/v78_late_replacement_feasibility_20260827/REPORT.md:32-68`). V78 explicitly identifies expansion allocation as outside the rejected late-replacement hypothesis (`analysis/v78_late_replacement_feasibility_20260827/REPORT.md:117-119`). The live loss audit independently observed a wool-scarce/milk-glutted regime and proposed preserving livestock optionality rather than rewriting the route (`analysis/v67_live_loss_audit_20260826/REPORT.md:49-63`, `:152-161`).

**Risk.** The same cow-heavy expansion wins controls, and the predeclared 24-step relative-price condition at step 248 held in only one of four losses (`analysis/v78_late_replacement_feasibility_20260827/REPORT.md:55-68`). Therefore a raw `WOOL/MILK` threshold is not supported. Changing animal type also changes pickup/place/harvest semantics; parity must be proven outside the selected expansion block.

**Cheapest falsifier.** Before implementing a policy, compute a generous value upper bound for substituting exactly the new expansion units, including forgone milk, attainable wool events, feed/labor cost, and shared-market price impact. Use only the consumed gate8. Stop if the optimistic bound cannot make all four target margins positive. If the bound passes, freeze one identity-free rule and require on gate8:

- target losses 4/4 become positive;
- controls remain 4/4 positive and no control margin decreases;
- zero sanitizer/fallback changes;
- V67 actions remain byte-identical outside the declared type-dependent expansion/production block.

Passing is only a causal screen, never promotion evidence.

### 2. Late market-only value upper bound

**Hypothesis.** After physical capital allocation is observably complete, changing only sale timing and quantities could recover enough value from the late price inversion to rescue at least the small public deficits while leaving movement and production unchanged.

**Evidence.** V67's loss cohort diverges mainly after step 480/576, market prices invert sharply, and sale requests barely react (`analysis/v67_live_loss_audit_20260826/REPORT.md:36-63`).

**Counter-evidence.** The existing evidence is unfavorable. V71's repeated one-event early-WOOL rule rescued 6 but regressed 9 and worsened mean/CVaR because early cash and prices cascaded into later behavior (`artifacts/v71_market_governor_20260826/VERDICT.md:6-19`, `:21-47`). V77's observable MILK acceleration rescued 0/4 target losses (`analysis/v77_causal_macro_optionality_20260827/DIAGNOSTIC.md:25-61`). V67 finishes with zero shed/carry inventory, so there is no hidden terminal stockpile to liquidate (`analysis/v67_live_loss_audit_20260826/REPORT.md:25-34`).

**Cheapest falsifier.** Run a market-only hindsight oracle on the consumed gate8 while forcing every farmer, hands, land, animal, seed, and product-buy action to remain byte-identical. The oracle must model fills and endogenous shared-market price changes, not scheduled quantities alone. Stop the entire market-only path if even the optimistic oracle cannot make all four target margins positive while preserving all four controls. Do not threshold-tune another early-WOOL or MILK-only rule.

### 3. Safe residual RL anchored to V67

**Hypothesis.** A policy that defaults exactly to V67 and learns only a small residual over enumerated macro decisions can capture local economic headroom while avoiding the catastrophic behavior of a replacement controller. The residual should abstain to V67 outside supported states.

**Evidence.** Expanded hindsight experts reach 148/160, so alternative actions/routes sometimes have substantial value, but selectors fail to generalize (`analysis/v59_v67_route_oracle_development160_20260826.md:76-105`, `:135-153`). V71 shows that a hand-coded observable rule can have true causal rescues yet still regress through repeated composition (`artifacts/v71_market_governor_20260826/VERDICT.md:6-19`, `:40-47`). This supports learning an action-value and uncertainty boundary rather than adding more fixed thresholds.

**Design boundary.** This must not be behavior cloning from opponent tapes or whole-route classification. Train against executable strong proxy agents in the official engine, both seats and new seeds. Limit the residual to explicit decisions such as expansion allocation and late market order sizing. Optimize win rate with worst-case/CVaR constraints, and require calibrated abstention back to V67 when lower-confidence advantage is non-positive.

**Falsifier before a new holdout.** On development-only on-policy simulations with grouped leave-opponent-family-out validation, every fold must be no worse than V67 on wins, worst margin, and CVaR; the combined candidate must improve at least three independent opponent families with zero regressed families. If it fails, stop before candidate freeze and before any new holdout is generated.

## Explicitly rejected paths

“Rejected” here applies to the tested intervention and evidence boundary, not to every future state-aware or RL architecture.

| Path | Verdict and direct reason |
|---|---|
| Whole-route selector / add more tapes | Reject. Hindsight headroom is large, but observable leave-family-out selectors collapse to 64/80 or 63/80 even with a 80/80 oracle. More experts worsen the identification problem (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:11`). |
| V72/V74/V75 closed-loop rewrite | Reject as successor architecture. V72 scored 0/80 and destroyed the production economy; V74 reached only 2,925/4,167 and V75 only 6,799/16,127 on the two authorized traces, far below roughly 53k gates (`analysis/v72_iter2_diagnosis_20260826/diagnosis.md:51-63`; `analysis/v74_iter2_screen_20260826/VERDICT.md:3-26`; `analysis/v75_maintenance_screen_20260826/VERDICT.md:3-30`). |
| Generic workforce fallback | Reject as an unconditioned V67 patch. V59's sell→hire→production loop is necessary in the failed replacement controllers (`analysis/v59_v74_pair_trace_20260826/SUMMARY.md:107-117`), but V67's public losses already finish with hands 9, land 3, full livestock composition, and no inventory (`analysis/v67_live_loss_audit_20260826/REPORT.md:25-34`). No completed V76 result artifact exists; the repository contains only its screen script. Thus there is no discriminating causal support for “hire more” on V67. |
| MILK acceleration | Reject. The predeclared V77 intervention rescued 0/4; two targets had no eligible event and the other two remained losses (`analysis/v77_causal_macro_optionality_20260827/DIAGNOSTIC.md:38-75`). |
| V78 late COW→SHEEP replacement | Reject. There were zero true replacements, so its optimistic upper bound was zero and the gate passed 0/4 (`analysis/v78_late_replacement_feasibility_20260827/REPORT.md:3-15`). |
| Naive V61 livestock/option append | Reject. Existing variants reached only 47–57/80 despite zero sanitizer steps, materially below V59/V67 (`analysis/v77_causal_macro_optionality_20260827/DIAGNOSTIC.md:63-75`; `.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:6-7`). |
| Repeat V71 early-WOOL threshold tuning | Reject. Full paired evaluation produced 6 rescues and 9 regressions with worse tail risk; the registered stop rule already forbade threshold search (`artifacts/v71_market_governor_20260826/VERDICT.md:21-47`). |

## Promotion ladder that preserves V67

### Gate 0 — incumbent lock

- Keep V67 standalone SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0` immutable.
- A candidate remains an overlay/branch; it never replaces the V67 artifact or its evaluation records.
- old80, fresh95, third160, development160, and public gate8 are all **consumed development evidence**. None may be called fresh, sealed, or holdout again.

### Gate 1 — pre-registered causal screen

- Write one mechanism, exact observable features, intervention scope, pass/fail threshold, and stop rule before evaluation.
- Use consumed development data only.
- Require exact V67 parity outside the declared intervention, zero legality/sanitizer/fallback changes, target rescues, and non-regression controls.
- A failed screen kills that hypothesis; do not search thresholds on the same cases.

### Gate 2 — development generalization

- Evaluate candidate and V67 paired on official-engine **on-policy** games against executable proxy agents, not fixed opponent tapes.
- Split by opponent-policy family and seed; use grouped/nested validation so the same episode, seed, or family cannot cross train/test folds.
- Require every fold to be no worse than V67 on wins, worst margin, and CVaR, plus improvement across at least three independent families.
- This gate selects one candidate only. It is not promotion evidence.

### Gate 3 — freeze before new data

Freeze and hash all of the following before generating the holdout:

- standalone candidate archive;
- V67, V16, and V22 baselines;
- official engine/runtime lock;
- evaluator and legality checker;
- executable opponent-family artifacts;
- metric definitions and exact pass thresholds.

After freeze, an independent evaluator creates the seed list from fresh randomness or a future public beacon and stores its hash. The candidate must not be changed after seeing any result. If it fails, the entire holdout becomes consumed and the next candidate needs a newly generated holdout after its own freeze.

### Gate 4 — truly disjoint official-engine holdout

- No seed, episode, replay, or derived state used in design/training may appear.
- Run every available strong executable proxy family in both seats with at least five new seeds per family and at least 100 games total.
- Apply the mission rubric literally: at least 95% aggregate wins **and positive worst margin against every proxy family**. Because positive worst forbids any non-positive game, do not reinterpret this as “95% is enough despite losses.”
- On the identical holdout, candidate must not regress below V67, V16, or V22 on win rate, mean, CVaR, or worst margin.
- Fixed replay-tape counterfactuals may be reported only as diagnostics; they cannot satisfy this gate.

### Gate 5 — archive and runtime acceptance

- Raw standalone smoke in both seats: all 719 executable transitions complete, non-idle behavior, valid action keys, zero sanitizer/fallback changes.
- Re-run exact archive/local parity, latency, reproducible hashes, and packaging checks.
- Any artifact change after Gate 4 invalidates the holdout and returns to Gate 3.

### Gate 6 — explicit upload approval and live evidence

- Upload only after explicit user approval.
- Identify the opponent cohort as current Top1–20 using evidence captured at match time; current V67 public artifacts cannot do this because opponent rank is absent (`analysis/v67_live_loss_audit_20260826/REPORT.md:17-23`).
- Require 24 consecutive positive-margin live games against that identified cohort. A loss resets the streak and rejects decisive promotion; 24 wins mixed with losses is not equivalent.

### Gate 7 — promotion

Only after Gates 0–6 pass may the candidate become the new incumbent. Until then, V67 remains the recoverable baseline and no score claim should exceed the evidence.

## Unknowns and limits

- The four public losses are too few to estimate a stable meta rule (`analysis/v67_live_loss_audit_20260826/REPORT.md:165-170`).
- Recorded opponent actions become strategically stale after candidate state divergence. Replay screens establish causal plausibility, not live win rate.
- The repository does not yet prove that the expansion-allocation upper bound is large enough, that a market-only oracle can rescue the worst -21,061 deficit, or that residual RL can maintain calibrated abstention.
- TokenSave SQLite currently indexes only 55 files and contains none of the current V59/V67/V71–V78 or mission artifacts, so this critic used direct reads of the named reports after the required SQLite-first check.

## Final recommendation

Run the two cheapest **oracles**, not two more agents:

1. expansion-allocation value upper bound;
2. strict market-only value upper bound.

Whichever lacks 4/4 target-rescue headroom is killed immediately. If both fail, stop hand patches and invest in the V67-anchored on-policy residual RL track. In every branch, preserve V67 and generate the first genuinely disjoint holdout only after a single candidate archive is frozen.
