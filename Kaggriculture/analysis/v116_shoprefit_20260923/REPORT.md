# V116 — Fresh shop-router refit on Sept-22 winners: REPORT (no-winner, empty donor set)

## Outcome

**No winner. No bundle built. Nothing submitted.** The Yusuke-method refit
collapses at the construction step: not one of the 10 named Sept-22 winner
tapes shares the yannik base's canonical opening through the first branch
step (identity 0.0000 for all 10), so no splice is legal and the route table
is the base route by the frozen rule. The screen then measures new ≡ yannik
exactly, which cannot pass a +0.02 superiority gate. Per the frozen prereg
(no-build-on-fail), the run stops here. This is the acceptable honest result
the prereg allowed for.

## Why the donor set is empty (method-level finding, not a bug)

- Full 92-tape pool canonical scan (`normalize_legacy_action`, the v97 rule)
  vs the yannik base: identity 1.0 through step 72 **only** in yannik's own
  7 tapes. Every non-yannik tape — all 10 named donors included — scores
  **0.0000**: yannik's sheep-rush opening shares no full action with any of
  them, not even step 0 (BUY_SEED wheat 3+4 + sheep vs the pool's
  BUY_PRODUCT wheat 20 / SELL 15 omar-lineage opening).
- Cross-check: mtmr_s1__ep111947463__seat0 matches the *omar* route 0.9444
  through step 72 (v115's lineage finding reproduced), while yannik matches
  omar 0.0000. The Sept-22 winner pool is omar-lineage (or otherwise
  divergent) openings throughout; yannik is a lineage of one.
- Splicing any winner suffix onto the yannik prefix would be exactly the
  cross-divergent-farm chimera v106 proved dies — the frozen rule forbids
  it, so the artifact is the base route with identity tables (all 8
  first-routes → 0, all 64 second-routes → 0), asserted mechanically by
  `scripts/build_v116_shoprefit_routes.py` (`fit.json`: `eligible_donors: []`).

A Yusuke-method router *needs* donors from the base's own lineage (his 10
routes were his own episodes). Refitting onto winners of a foreign lineage
is unimplementable without chimeras. Follow-up worth considering: a yannik-
lineage donor pool (yannik's own 7 pool tapes) — a different prereg, not
this one.

## Gates verdict (300 fresh gate seeds 20101101–20101400, 16 opponents, both seats)

| gate | result |
|---|---|
| G1 (point(new)−point(yannik) ≥ +0.02, CI excludes 0) | **FAIL** — diff 0.0000, CI [0, 0] |
| G2 (point(new)−point(omar) CI lower > −0.01) | **FAIL** — diff −0.0011, CI [−0.0136, +0.0114] |
| G3 (no per-opponent regression vs yannik AND omar) | **FAIL** — 11/16 pass; 5 fail vs omar (table below) |
| G4 (played route == table lookup, 9600/9600 games) | PASS — 0 mismatches |
| G5 (unseen GATE-pair draws: point(new) ≥ point(yannik)) | PASS — 0.8773 vs 0.8773 (14,396 gate / 14,404 fit games) |
| E0 (320-game re-run, 0 money mismatches) | PASS — 0 mismatches |

Pooled points: new 0.8850 / yannik 0.8850 / omar 0.8861 (28,800 games).

G3 detail (FAIL rows are all new≡yannik losing to omar lineage, never to yannik):

| opponent | new | yannik | omar | verdict |
|---|---|---|---|---|
| decem__ep111955463__seat1 | 0.7467 | 0.7467 | 1.0000 | FAIL vs omar |
| dsm__ep111948590__seat1 | 0.7867 | 0.7867 | 1.0000 | FAIL vs omar |
| majkel1337__ep111958811__seat1 | 0.6617 | 0.6617 | 0.7950 | FAIL vs omar |
| vadim_vasilenko__ep111940707__seat1 | 0.8183 | 0.8183 | 1.0000 | FAIL vs omar |
| ymg_aq__ep110781289__seat0 | 0.9533 | 0.9533 | 1.0000 | FAIL vs omar |
| all other 11 opponents | == yannik | — | within 0.02 | PASS |

Note the panel cuts against yannik twice: the strong 10 are omar-lineage
tapes, so a yannik-base router concedes decem/dsm/vadim/ymg_aq/majkel to omar
while beating omar on mtmr/otter/kaggledew/thunder/sida. Neither lineage
dominates the 16-tape panel (omar 0.8861 vs yannik 0.8850, CI overlapping).

## What was run (timed)

| stage | games | wall |
|---|---|---|
| parity subset (Rust play_batch vs Python Engine, yannik-calibrated) | 36 + 36 | ~2 min |
| gate screen: new/yannik/omar vs 16 opponents, 300 seeds, both seats | 28,800 | 13 s |
| E0 determinism re-run | 320 | <1 s |

Engine `{"episodeSteps":720,"weedSpawnChance":0.005}`, `play_batch`, both
seats. Parity: 36/36 exact money matches on wiring-only seeds
20100001–20100003 (never reused). Files: `PREREGISTRATION.md` (written before
any eval seed), `fit.json`, `parity.json`, `screen_rows.json`, `gates.json`.
New files only: `artifacts/shop_router_v116/routes.json`,
`scripts/build_v116_shoprefit_routes.py`,
`scripts/verify_v116_router_parity.py`, `scripts/analyze_v116_gates.py`
(ruff clean). `scripts/evaluate_shop_router_fast.py` used UNMODIFIED. No
changes to src/tests/rust/bindings/existing scripts/routes; `submissions/`
untouched; no Kaggle calls; nothing committed.

## Submit command

None — there is no bundle, so there is nothing for a human to submit. Slot A
(omar fallback) is unchanged.
