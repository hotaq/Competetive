# V95 — Does routing on the revealed shop *pair* beat the best single tape, on a 1,390-tape pool?

Pre-registered 2026-08-30. **FROZEN on sign-off, 2026-08-30.** The gates, the seed ranges and the stopping rule below are not to be edited. Any change is an Amendment appended below with its own date and reason, in the V94 style — never an edit in place.

Signed off before any stage-1 result was inspected; see Disclosure.

## Origin

`analysis/v94_shop_router_20260830/REPORT.md` is an OVERALL FAIL, and its failure mode is
specific and actionable: the fitted first-shop table picked `yusuke/ep102574290` in **all
eight** cells, so the routed arm and the single-tape arm were the identical policy and G2/G3
passed vacuously. V94's own diagnosis:

> What the pool lacks is a **specialist that beats the generalist**. Routing is only worth
> something when different tapes win different cells; here one tape wins all eight.

V94 also established that the branch variable is real — `kenjo/ep102545173` scores 0.664
pooled but **0.096** when the first shop is YARN_STORE, a 0.57 point-rate swing on 17,280
held-out games — and that the published gain lives in the *second* shop, not the first.
Yusuke's own `FIRST_ROUTES` branches in only 1 of 8 first-shop cells; his reported
0.678 -> 0.814 comes from the 64-cell ordered-pair table.

V94 closed with three ordered next steps. This experiment is steps 1-3.

## What changed since V94

| | V94 | V95 |
|---|---:|---:|
| harvested files | 51 | 1,390 |
| **unique action sequences** | **43** | **1,390** |
| source episodes | Top 5 | 982 episodes, Top 20 |
| comparison: Yusuke's pool | 1,446 | 1,446 |

Pool `artifacts/tape_pool_20260830.tar.gz`, manifest
`artifacts/tape_pool_20260830.manifest.json`, built by `scripts/harvest_tape_pool.py`
(both seats kept, deduplicated by action hash). The binding constraint V94 named is
resolved: the pool is now the same order of magnitude as the published method's.

## Hypothesis

**H1.** With 1,390 unique sequences, the 64 ordered (first, second) shop cells contain at
least one specialist that beats the pooled-best tape in its own cell, and a table fitted on
training seeds beats the best single tape on unseen seeds.

**H0.** One tape remains best in every populated cell — the V94 outcome — and routing buys
nothing. A second failure here is a statement about tape pools generally, not about pool size.

## Protocol

Three stages. Each stage uses a **disjoint** seed range, and every range is disjoint from
V94's consumed ranges (13100001-13100040, 13200001-13200240, 13300001-13300240).

**Stage 1 — screen (running).** All 1,385 candidates x 1 opponent x 6 seeds x 2 seats =
16,620 games. Seeds **14100001-14100006**. Purpose: cut the pool to a shortlist. This is a
screen and carries **no gate**. Output `stage1.json` / `stage1.partial.jsonl`.

**Stage 2 — fit.** Top 40 candidates by stage-1 point rate, x **1 opponent** (the same
`OPPONENT_HASHES[0]` used in stage 1) x seeds **14200001-14202000** x 2 seats =
**160,000 games**, ~2.4 h. The pair table is fitted here: for each of the 64 ordered cells,
pick the candidate with the highest point rate in that cell, requiring **>= 30 games per
cell** for a cell to be fitted; cells below that threshold fall back to the pooled-best tape
and are reported as unfitted.

Budget decided 2026-08-30. A 3-opponent fit (480,000 games, ~7.1 h) was specified first and
then revised down to this on cost grounds; a 15-candidate variant (180,000 games) was the
third option and was not taken. V94 sized this step at roughly 18,000 games, so stage 2 is
still ~9x that estimate.

**Consequence to report honestly in the REPORT:** fitting against a single opponent means
the table can overfit to that opponent's play. Stage 3 keeps all 3 opponents, so the holdout
is a test of generalisation across opponents as well as across seeds, and G2 (per-opponent
no-regression) is the gate that would expose such overfitting. If G2 fails only on the two
opponents absent from the fit, that is a diagnosis, not a surprise, and must be reported as
a limitation of this budget rather than a property of shop routing.

**Engine path.** At this size both transports fit the 12 h notebook limit — ~2.4 h on
`step_typed` (160 ms/game) against ~3.5 h on `Engine.step` (236 ms/game). Typed is still
preferred. Measured 2026-08-30 on pool tapes: the two paths produce identical final money
across 3 seeds, so this is a transport change only and does not touch semantics. Switch
`run_shop_router.py` to `step_typed` **after stage 1 completes**, so that no stage carries
results from two different code paths.

**Stage 3 — holdout.** Seeds **14300001-14302000**, same 3 opponents, both seats, two arms
(routed table vs best single tape) = **24,000 games**, ~21 min. Read **once**. The fitted
table is not touched after stage 3 is read.

Opponents are the 3 tapes identified by sha256 in the notebook (`OPPONENT_HASHES`), excluded
from the candidate set.

## Gates

All four must pass. Failure of any one is an OVERALL FAIL and a hard stop — no checkpoint,
no runtime, no submission.

| gate | criterion |
|---|---|
| **G1** | Routed point rate exceeds best-single point rate on holdout by **>= 0.02**, and the 95% CI on the paired per-(seed, seat, opponent) difference **excludes 0**. |
| **G2** | No regression against any of the 3 opponents taken separately: routed point rate >= single - 0.01 for each. |
| **G3** | Routed mean margin >= best-single mean margin on holdout. |
| **G4** | The fitted table selects **>= 2 distinct tapes** across the 64 cells, and at least one cell that branches away from the pooled-best tape holds **>= 30 games**. |

**G4 exists because V94 passed G2 and G3 vacuously.** If the table degenerates to one tape,
the two arms are the same policy, and G1-G3 compare a policy against itself. G4 makes that
outcome an explicit FAIL rather than a silent pass.

## Disclosure — stage-1 data predates this document

Stage 1 was launched on Kaggle before this file was written. Its first run
(`kaggriculture-shop-router-screen` v3, 2026-08-30 14:02) crashed at candidate 702/1385 on
`KG-SCHEMA-INVALID at quantity` and produced **8,412 of the 16,620 rows**, preserved as
`stage1.kaggle_run1.partial.jsonl`. Version 4 resumes the remaining 684 candidates.

At the time of writing, the contents of those 8,412 rows have **not been inspected** — only
row count, key structure, and per-candidate completeness, which were needed to build the
resume list. No margin, point, or shop field has been read. Stage 1 carries no gate, and the
stage-2 and stage-3 seed ranges are untouched, so the gates above are unconditioned on any
outcome.

`stage1.partial.jsonl` in this directory is a **different, earlier local run** (262 rows,
opponents keyed `rank003_yusuke_hayashi/...`). It is not part of this experiment.

## Known defect fixed before stage 1 completed

`clean()` admitted market orders with quantity <= 0. Two pool tapes emit `SELL <product> 0`
(`wei_chance__ep103172055__seat0`, 57 orders at step 713; `norbert_szolnoki__ep103183194__seat0`,
10). The Kaggle environment treats these as silent no-ops; the fail-closed executor raises.
Fixed in `scripts/play_against_tape.py`, `scripts/falsify_q3_identifiability.py`, and the
notebook's embedded runner. Dropping a zero-quantity order is semantically identical to the
competition's no-op, so stage-1 rows recorded before and after the fix are comparable.

## Stopping rule

If G4 fails, report the per-cell winner table as V94 did and stop. Do not widen the shortlist,
add seeds, or re-fit to manufacture a branch. Do not re-read the stage-3 holdout under any
amendment.

---

## Amendment 1 — stage 1b, a tie-break the original document did not specify (2026-08-31)

**This rule was written after stage-1 outcomes were read.** It is recorded here, dated and
appended rather than edited into the text above, so that a reader can weigh it accordingly.

### What went wrong

Stage 2 was specified as "Top 40 candidates by stage-1 point rate". Stage 1 gives each
candidate 12 games (6 seeds x 2 seats), so point rate can only take the values k/12, and the
1,385 candidates collapse onto a handful of them:

| point rate | candidates | cumulative |
|---:|---:|---:|
| 1.0000 | 17 | 17 |
| 0.8333 | 211 | 228 |
| 0.7500 | 1 | 229 |
| 0.6667 | 453 | 682 |

Only 17 candidates are determined. The remaining 23 places must come out of a 211-way exact
tie, and the original document names no tie-break. The screen therefore does not produce the
shortlist stage 2 consumes.

The choice is not cosmetic. Ranking by point rate and ranking by mean margin agree on **3 of
40** candidates. Within the 0.8333 tier mean margin runs from **+12,073**
(`ant__ep103143136__seat1`, the largest in the pool) down to **-2,016**
(`milan_leonard__ep102038335__seat1`), while all 17 candidates at 1.0000 sit between +3,861
and +4,834 — they win every game narrowly. **G3 is a margin gate**, so the tie-break decides
what stage 2 has to work with.

### Rule adopted

Rather than pick a ranking criterion with the outcomes already visible, the tie is resolved
by measurement on seeds not yet used. The cause is that n = 12 is too coarse, and that is
what stage 1b fixes.

**Stage 1b — tie-break.** The **228 candidates with stage-1 point rate >= 0.8333** x the same
`OPPONENT_HASHES[0]` x seeds **14150001-14150030** x 2 seats = **13,680 games**, ~12 min.
The candidate set is frozen in `stage1b_shortlist.json`
(sha256 `70813bd4a4038f7507ae39595f98415bc70ed6bf941846cb2019ac9b6616733b`, 228 entries).
The seed range is disjoint from stage 1 (14100001-14100006), stage 2 (14200001-14202000),
stage 3 (14300001-14302000) and every V94 range.

**Selection for stage 2.** Rank the 228 by **stage-1b point rate alone**, over the 60 fresh
games. Stage-1 point rate is *not* pooled in: the 228 were selected because they scored high
on those 12 games, so their stage-1 scores carry a winner's-curse bias, and stage 1b is an
unbiased re-measurement of an already-selected set. Ties are broken by **stage-1b mean
margin**, then by candidate name ascending, so the rule is deterministic. Take the top 40.

Stage 1b carries **no gate**, exactly as stage 1 does not. It is a screen.

### Engine path

Stage 1b runs on `step_typed`, the switch the document above schedules for "after stage 1
completes". All 228 candidates are measured on that one path within stage 1b, so the ranking
this stage produces is internally consistent; no candidate is compared against another
across transports. Stages 2 and 3 use `step_typed` as well.

### What is unchanged

G1-G4, the stage-2 and stage-3 designs, their seed ranges, and the stopping rule are
untouched. Stage 3 remains read exactly once.
