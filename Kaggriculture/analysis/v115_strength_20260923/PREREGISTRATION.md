# V115 — Strong-panel tape screen (2026-09-23/24): which pool tape beats 2500-track opposition

Pre-registered BEFORE running any seed below. No game has been played on seeds
20000101-20000200 or 20001101-20001400 anywhere in analysis/ or docs (verified:
no 200/201/202xxxxx seed appears in analysis/, docs/, scripts/, src/).
Live before start: yannik 56495539 (~1846, climbing), omar 56361130 (~1442).
Target 2500 needs consistent wins vs 2500-rated teams; top-10 cutoff ~2950,
deadline 2026-09-30. Hedge rule: Slot A = omar fallback always; Slot B = gated
challenger only. No `kaggle competitions submit` in this run (human fires).
Nothing is committed. No changes to src/, tests/, rust/, bindings/, or existing
scripts (new scripts only). No overlay/modification of any tape: verbatim replay
only. No tuning on burnt seeds.

## Frozen inputs (verified, not re-harvested)

- Pool: `artifacts/tape_pool_20260922_top10/` — 92/92 tapes verified
  `length == 720` with action keys exactly `{farmer, hands, market}`.
- Candidates (12 = top-12 distinct-team tapes by manifest `final_money`,
  excluding yannik_schiffner / omar lineage / zero waste):
  1. `third_farm_club__ep111957345__seat0` (160916)
  2. `unknown_mother_goose__ep111950709__seat0` (155143)
  3. `majkel1337__ep111958811__seat1` (153588)
  4. `m___m___p___q__ep111946154__seat1` (146916)
  5. `otter_vibe__ep111947463__seat1` (142784)
  6. `mtmr_s1__ep111947463__seat0` (140350)
  7. `dsm__ep111948590__seat1` (138720)
  8. `kaggledew_valley__ep111928231__seat0` (136745)
  9. `vadim_vasilenko__ep111940707__seat1` (126672)
  10. `吃白饭的大肥鱼__ep111928231__seat1` (120751)
  11. `thunder_thunder__ep111943074__seat1` (111266)
  12. `decem__ep111955463__seat1` (110078)
- Opponent panel = exact G4 strong set (10 tapes, from
  `/tmp/strong_test/strong.json` context): `majkel1337__ep111958811__seat1`,
  `unknown_mother_goose__ep111950709__seat0`,
  `third_farm_club__ep111957345__seat0`, `m___m___p___q__ep111946154__seat1`,
  `otter_vibe__ep111947463__seat1`, `mtmr_s1__ep111947463__seat0`,
  `dsm__ep111948590__seat1`, `kaggledew_valley__ep111928231__seat0`,
  `vadim_vasilenko__ep111940707__seat1`, `decem__ep111955463__seat1`.
- Reference arms (native policy, verbatim, no overlays):
  YANNIK = Rust `ShopRouter` on
  `analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json`
  (single route, branch 72/144; loads OK);
  OMAR = Rust `ShopRouter` on `artifacts/shop_router_v4/base_tape_routes.json`
  (single route `omar_althobaiti__ep110807162__seat1`, 719 actions, branch 72/144).
- Engine config unchanged: `{"episodeSteps": 720, "weedSpawnChance": 0.005}`,
  `play_batch`, both seats always. Candidates played as Rust `Tape`
  (`actions[1:720]`), verbatim open-loop.
- Context only (NOT gates): prior G4-style informal check on burnt seeds
  19800101-19800200 gave omar 0.8945 / yannik 0.8565 vs this panel.
  Burnt and avoided: 18400101-18400400, 18500101-18500300, 18600101-18600300,
  18700101-18700300, 18800101-18800300, 19100001-19101000, 19200011-19201010,
  19400001-19401000, 195xxxxx (refresh screen/round-robin/holdout),
  19600101-19600300, 19700101-19700300, 19800101-19800300, 19900101-19900300.

## Fresh seeds (never used as eval grids)

- Screen: 20000101-20000200 (100 seeds).
- Holdout: 20001101-20001400 (300 seeds).
- E0 probe: first 10 holdout seeds (20001101-20001110), all opponents x both seats.

## Stages (new scripts only: `scripts/evaluate_v115_strong_screen.py`,
`scripts/analyze_v115_strong_gates.py`; each stage timed)

1. **Screen**: 12 candidate tapes + YANNIK + OMAR vs 10-opponent panel,
   seeds 20000101-20000200, both seats. Self-pairings (candidate == opponent
   tape) SKIPPED and excluded from pooled points (10 of 12 candidates overlap
   the panel; mirror self-games would drag them toward 0.5). Output
   `screen_rows.jsonl` (row schema: arm, opponent, seed, seat, money,
   opponent_money, margin, point).
2. **G1 select**: W = highest pooled screen point among the 12 (self-games
   excluded); W2 = runner-up (diagnostic only, never built).
3. **Holdout**: arms W, W2, YANNIK, OMAR vs same panel, seeds 20001101-20001400,
   both seats. Output `holdout_rows.jsonl`.
4. **Checks**: E0 determinism (re-run W tape on E0 probe jobs; 0 money
   mismatches required); LIN lineage (W `actions[1:73]` per-step identity vs
   yannik route actions and vs omar route actions; both < 0.90 or clone-reject).
5. **Build** (only if G1 ∧ G2 ∧ E0 ∧ LIN pass): single-route file from W
   (schema of `base_tape_routes.json`: branch 72/144, all branch tables map to
   route 0) + UNMODIFIED `scripts/build_shop_router_v1_submission.py` into ONE
   new `submissions/<name>_standalone/` dir. G3: `candidate.json`
   `standalone_validated` (960 action-equivalence checks) + official-env
   `smoke_kaggle_raw_submission.py` both seats DONE/DONE with money matching
   `play_fast`.

## Gates (frozen; never relaxed after seeing outcomes)

Paired unit = same (opponent, seed, seat); CI = 95% seed-cluster bootstrap
(2000 replicates; resample seeds, keep both seats + all opponents per seed).

| gate | criterion |
|---|---|
| G1 | W = highest pooled screen point AND point(W) >= 0.60 on screen AND mean paired money(W − YANNIK) > 0 with bootstrap CI excluding 0 |
| G2 | point(W) >= 0.60 on holdout AND mean paired money(W − YANNIK) > 0 with bootstrap CI excluding 0 on holdout |
| E0 | determinism probe 0 mismatches |
| LIN | W prefix-72 identity vs yannik route AND vs omar route both < 0.90 |
| G3 | bundle equivalence + both-seat official-env smokes pass with matching money |

Decision: build at most ONE bundle, for W only, iff G1 ∧ G2 ∧ E0 ∧ LIN ∧ G3.
If W fails any gate: NO bundle, report and stop (no-winner is acceptable).
