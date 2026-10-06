# V115 — Strong-panel tape screen: REPORT (no-winner, clone-reject)

## Outcome

**No winner. No bundle built. Nothing submitted.** The screen winner passes G1,
G2, and E0 on merit but fails the lineage gate: its opening 72 actions are
94.4% identical to the live omar route — it is omar lineage wearing an mtmr
jersey. Per the frozen prereg (no-build-on-fail, clone-reject), the run stops
here. This is the acceptable honest result the prereg allowed for.

## Winner (rejected)

- **W = `mtmr_s1__ep111947463__seat0`** (manifest final_money 140350; the
  losing side of ep111947463, verbatim replay, no overlays).
- Screen (seeds 20000101-20000200, 10-opponent G4 strong panel, both seats,
  self-games excluded): **point 0.9172** (1800 games), rank 1/12; paired money
  vs YANNIK **+4997, 95% CI [1757, 8119]** — beats yannik head-to-head.
- Holdout (fresh seeds 20001101-20001400, 300 seeds): **point 0.9402**
  (5400 games); paired money vs YANNIK **+3317, CI [1667, 4986]**.
- Reference arms reproduce the informal G4 check almost exactly on fresh seeds:
  screen YANNIK 0.8560 / OMAR 0.8920 (informal was 0.8565/0.8945); holdout
  YANNIK 0.8537 / OMAR 0.8983. Panel and harness are comparable.
- W2 (diagnostic, never built): `thunder_thunder__ep111943074__seat1` —
  holdout point 0.6598 but paired money vs YANNIK −20130 (CI excludes 0
  downward). Lineage-clean (prefix-72 vs omar 0.014, vs yannik 0.0) but fails
  on merit. No other candidate passes: the next-best screen points
  (m___m___p___q 0.655, majkel 0.539, kaggledew 0.533) all lose to yannik by
  12k-24k paired money.

## Gates verdict

| gate | result |
|---|---|
| G1 (screen: top point ≥ 0.60 + beats yannik, CI excludes 0) | PASS |
| G2 (holdout: point ≥ 0.60 + beats yannik, CI excludes 0) | PASS |
| E0 (180-game re-run, 0 money mismatches) | PASS (0 mismatches) |
| LIN (prefix-72 identity < 0.90 vs yannik AND vs omar) | **FAIL — 0.9444 vs omar → clone-reject** |
| G3 (bundle + both-seat smokes) | not attempted (no-build-on-fail) |

Lineage detail: W `actions[1:73]` match omar route steps 1-72 in 68/72 steps
(0.9444); vs yannik 0/72 (0.0). Full-tape identity vs omar is only 0.2879 — the
clone is the opening through the first branch step (72), exactly the
route-determining window. Shipping it would mean uploading a near-copy of the
decaying Slot-A lineage, not a challenger. The candidate-set filter excluded
omar-*named* tapes but could not see action-level lineage; the LIN gate caught
what the filter missed — the gate worked as designed.

## What was run (timed)

| stage | games | wall |
|---|---|---|
| wiring probe (seeds 20000001-20000002, never reused) | 32 | <1s |
| screen: 12 tapes + YANNIK + OMAR vs 10 strong, 100 seeds | 26,000 | 9s |
| holdout: W, W2, YANNIK, OMAR vs 10 strong, 300 seeds | 23,400 | 9s |
| E0 determinism re-run | 180 | <1s |

Engine `{"episodeSteps": 720, "weedSpawnChance": 0.005}`, `play_batch`, both
seats. Files: `PREREGISTRATION.md` (written before any eval),
`screen_rows.jsonl`, `holdout_rows.jsonl`, `e0_rerun.jsonl`,
`screen_gates.json`, `holdout_gates.json`, `gates.json`. New scripts only:
`scripts/evaluate_v115_strong_screen.py`,
`scripts/analyze_v115_strong_gates.py`. No changes to src/tests/rust/bindings/
or existing scripts; no Kaggle calls; nothing committed.

## Submit command

None — there is no bundle, so there is nothing for a human to submit. Slot A
(omar fallback) is unchanged. A follow-up worth considering: re-screen the pool
with an up-front action-level lineage filter (prefix-72 identity vs omar/yannik
< 0.90 as an *entry* criterion), so screen winners are eligible by construction.
