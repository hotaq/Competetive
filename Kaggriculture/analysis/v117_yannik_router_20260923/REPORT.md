# V117 — Same-lineage shop-router on yannik's own tapes: REPORT (FAIL, no bundle)

## Outcome

**OVERALL FAIL. No bundle built. Nothing submitted.** The v116 residual
follow-up constructs fine — 7/7 yannik donors share the canonical prefix
(identity 1.0 through step 72), the fit screen selects 14 legal cells across
4 donor suffixes — but the gate screen measures new 0.8846 vs yannik 0.8721
(diff +0.0125, bootstrap CI [+0.0084, +0.0169]): a real but sub-bar gain that
clears 0 yet misses the frozen +0.02 superiority bar (G1 FAIL), and the panel
still concedes 6 omar-lineage tapes beyond tolerance (G3 FAIL). Per the frozen
prereg (no-build-on-fail, no tuning on these seeds ever again), the run stops
here. Slot A (omar fallback) is unchanged.

## Why construction succeeds where v116 collapsed

- Canonical scan (`normalize_legacy_action`, v97 rule) of all 7
  yannik_schiffner pool tapes vs the yannik base: identity **1.0000 through
  step 72 for all 7** (each 720 actions, verified computationally). Through
  step 144: 1.0000 for 6 of 6 non-base donors except ep111957345 (0.6667,
  first-level cells only). No cross-lineage splice exists anywhere in the
  build (asserted mechanically by `scripts/build_v117_yannik_routes.py`).
- Fit (44,800 games: 7 donor-tape arms vs the frozen 16-opponent panel, fit
  seeds 20200101–20200300, FIT-pair games only): base tape is the best pooled
  donor (0.8812), yet 14 cells clear min_gain 0.01 with no opponent regression
  beyond tolerance 0.05 (min_cell_games 30, min_opponent_games 8): 1
  first-level (YARN_STORE → ep111958489) + 13 second-level cells across donors
  ep111936440 / ep111955057 / ep111956211 / ep111958489 (`fit.json`).
- Same-lineage donors are NOT near-identical programs: pooled fit points span
  0.3911 (ep111957345) to 0.8812 (base). The shared opening ends at step 72;
  the suffixes play very different games. The prereg's "tie-break scale"
  expectation was wrong in the interesting direction — the fit found real
  cells — but winner's curse applies: in-sample cell gains up to +0.26 shrink
  to +0.0125 pooled on gate seeds.

## Gates verdict (300 fresh gate seeds 20201101–20201400, 16 opponents, both seats)

| gate | result |
|---|---|
| G1 (point(new)−point(yannik) ≥ +0.02, CI excludes 0) | **FAIL** — diff +0.0125, CI [+0.0084, +0.0169] (excludes 0, misses bar) |
| G2 (point(new)−point(omar) CI lower > −0.01) | PASS — diff +0.0062, CI [−0.0076, +0.0208] |
| G3 (no per-opponent regression vs yannik AND omar) | **FAIL** — 10/16 pass; 6 fail vs omar (table below) |
| G4 (played route == table lookup, 9600/9600 games) | PASS — 0 mismatches |
| G5 (unseen GATE-pair draws: point(new) ≥ point(yannik)) | PASS — 0.8574 vs 0.8574 (14,330 gate / 14,470 fit games) |
| E0 (320-game re-run, 0 money mismatches) | PASS — 0 mismatches |

Pooled points: new 0.8846 / yannik 0.8721 / omar 0.8783 (28,800 games).
Seat symmetry of new−yannik: seat0 +0.0127, seat1 +0.0123 (no seat skew).

G3 detail (FAIL rows are all new≥yannik but short of omar — never a
regression vs yannik itself, worst otter 0.9967 vs 1.0000):

| opponent | new | yannik | omar | verdict |
|---|---|---|---|---|
| decem__ep111955463__seat1 | 0.7367 | 0.7083 | 1.0000 | FAIL vs omar |
| dsm__ep111948590__seat1 | 0.7867 | 0.7733 | 1.0000 | FAIL vs omar |
| majkel1337__ep111958811__seat1 | 0.6867 | 0.6383 | 0.7717 | FAIL vs omar |
| third_farm_club__ep111957345__seat0 | 0.8350 | 0.8300 | 0.8683 | FAIL vs omar |
| vadim_vasilenko__ep111940707__seat1 | 0.8117 | 0.8050 | 1.0000 | FAIL vs omar |
| ymg_aq__ep110781289__seat0 | 0.9633 | 0.9600 | 1.0000 | FAIL vs omar |
| all other 10 opponents | ≥ yannik − 0.02 and ≥ omar − 0.02 | — | — | PASS |

The panel cuts against yannik-lineage twice, as in v116: the strong 10 are
omar-lineage tapes. New beats yannik on 15/16 opponents yet still concedes
the same omar stronghold. Neither lineage dominates (new 0.8846 vs omar
0.8783, CI overlapping).

## Two audit notes (both resolved inside the frozen gates, no relaxation)

1. G4 first ran 2456/9600 mismatches from a `shops[:1]` approximation of the
   second-step lookup in the analysis script. Wrong: shop2 is recorded from
   live state AT step 144 (zero new-arm rows have empty shop2), so the
   prereg's literal "(shop1, shop2) lookup" uses the full pair at step 144.
   Corrected audit: 0 mismatches. The gate criterion never changed.
2. Rust-vs-Python fallback semantics (installed `kaggriculture_rs` returns 0
   on a missing second key where `route_index_for` falls back to the first
   route) is unreachable with this table: all 64 pair keys are present, and
   Rust/Python `route_index` agree on every probed cell. Parity 36/36 exact
   money matches; E0 320/320.
3. G5 bitwise equality (0.8573839662447258 both arms) is consistent, not a
   bug: all 4740 GATE games are money-identical new vs yannik (draws agree
   100% across arms — shop draws are seed-determined). All 13 selected
   second cells are FIT pairs, and the first-level YARN suffix differences
   are engine-inert through step 144.

## What was run (timed)

| stage | games | wall |
|---|---|---|
| fit screen: 7 donor tapes vs 16 opponents, 200 seeds, both seats | 44,800 | ~24 s |
| parity subset (Rust play_batch vs Python Engine, yannik-calibrated) | 36 + 36 | ~2 min |
| gate screen: new/yannik/omar vs 16 opponents, 300 seeds, both seats | 28,800 | ~13 s |
| E0 determinism re-run | 320 | <1 s |

Engine `{"episodeSteps":720,"weedSpawnChance":0.005}`, `play_batch`, both
seats. Parity: 36/36 exact money matches on wiring-only seeds
20200001–20200003 (never reused). Files: `PREREGISTRATION.md` (written before
any eval seed), `fit_rows.json`, `fit.json`, `parity.json`,
`screen_rows.json`, `gates.json`. New files only:
`artifacts/shop_router_v117/routes.json`,
`scripts/build_v117_yannik_routes.py`,
`scripts/verify_v117_router_parity.py`, `scripts/analyze_v117_gates.py`
(ruff clean). `scripts/evaluate_shop_router_fast.py` used UNMODIFIED. No
changes to src/tests/rust/bindings/existing scripts/routes; `submissions/`
untouched (the one untracked entry predates this run, Sep 23); no Kaggle
calls; nothing committed.

## Submit command

None — G1 and G3 fail, so there is no bundle and nothing for a human to
submit. Slot A (omar fallback) is unchanged. The 202xxxxx seed grids are now
burnt and must never be tuned on.
