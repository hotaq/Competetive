# Refresh 2026-09-23 — tape screen on the frozen Sept-22 pool, no re-harvest

Pre-registered 2026-09-23 BEFORE running any seed below. No screen/holdout game
has been played on these seeds. Live before start: omar_base_tape_standalone
(submission 56361130, public 1431.8 COMPLETE, decaying); top-1 DSM 3166.0.
Deadline 2026-09-30 23:59 UTC; team "zero waste". Hedge rule: Slot A = omar
fallback always; Slot B = gated challenger only. No `kaggle competitions submit`
in this run (human fires). Nothing is committed.

## Frozen inputs (verified, not re-harvested)

- Pool: `artifacts/tape_pool_20260922_top10/` — 92 tapes, all `length == 720`
  with action keys exactly `{farmer, hands, market}` (verified 92/92).
- Opponents (V100 set, same 10 v101 used — points comparable to v101):
  `analysis/v100_cembas_router_20260905/opponents/` (a_poor_vul, ai是我的豆包,
  andrey_tikhomirov, baran_akyıldız, giulio_ravasio, keiz,
  lagrangianlocomotive, peikopon, yuk, yukino).
- ROUTER reference (live omar): `artifacts/shop_router_v4/base_tape_routes.json`
  (single route `omar_althobaiti__ep110807162__seat1`, 719 actions).
- Live tape (Slot-A lineage): `artifacts/tape_pool_20260919/tapes/omar_althobaiti__ep110807162__seat1.json`.
- Engine config unchanged: `{"episodeSteps": 720, "weedSpawnChance": 0.005}`,
  `play_batch`/`play_fast`, both seats always.

## Fresh seeds (never used as eval grids anywhere in analysis/ or docs)

- Screen: 19500101–19500200 (100 seeds).
- Round-robin: 19500201–19500400 (200 seeds).
- Holdout: 19300001–19301000 (1000 seeds; v103-planned, never run — REPORT confirms no holdout ran).
- Burnt and avoided: 181/182/183xxxxx (v101), 18400101–18400400 (dry-run+),
  18500101–18500300, 18600101–18600300, 18700101–18700300, 18800101–18800300
  (v109 screen), 19100001–19101000, 19200011–19201010, 19400001–19401000.

## Stages

1. **Screen**: all 92 pool tapes + ROUTER vs 10 V100 opponents on 19500101–19500200
   (`evaluate_tape_league.py --pool artifacts/tape_pool_20260922_top10
   --opponents <10 V100> --router base_tape_routes.json`). Keep best pooled point
   per team; top 12 teams forward.
2. **Round-robin**: 12 shortlisted tapes + live omar tape, all pairs, no `--router`,
   seeds 19500201–19500400. W = highest pooled point among the 12 (games vs the
   live tape excluded); W2 = runner-up from a DIFFERENT team with prefix-72
   canonical-action identity vs W < 0.90. Output `round_robin.json`.
3. **Holdout**: arms W, W2, live-omar-tape (`--tape` each) + `--routes` base arms,
   vs 10 holdout opponents (shortlist minus W, W2), seeds 19300001–19301000
   (`evaluate_shop_router_fast.py`). Output `holdout.json` (row schema:
   arm, opponent, seed, seat, money, margin, point, shop1/shop2).
4. **Checks**: E0 determinism (re-run W on holdout seeds 19300001–19300010 for all
   opponents × both seats via the same script; 0 money mismatches required);
   lineage (W actions[1:73] identity vs live omar route and vs live tape < 0.90).
5. **Build** (only if gates pass): single-route file from W (schema of
   `base_tape_routes.json`, all branch tables → route 0) + unmodified
   `scripts/build_shop_router_v1_submission.py` into ONE new
   `submissions/<name>_standalone/` dir. G3: `candidate.json`
   `standalone_validated` (960 action-equivalence checks) + official-env
   `smoke_kaggle_raw_submission.py` both seats DONE/DONE, zero invalid-key steps.

## Gates (frozen; never relaxed after seeing outcomes)

Primary comparator = `live` tape arm (the exact tape Slot A was built from).
Paired unit = same (opponent, seed, seat); CI = 95% seed-cluster bootstrap
(resample seeds, keep both seats + all opponents per seed).

| gate | criterion |
|---|---|
| G1 | mean paired money(W − live) > 0 with bootstrap CI excluding 0 (lower bound > 0) |
| G2 | pooled point(W) − point(live) >= −0.01 on holdout |
| E0 | determinism probe 0 mismatches |
| LIN | W prefix-72 identity vs live omar < 0.90 (not a clone) |
| G3 | bundle equivalence + both-seat official-env smokes pass |

Decision: build at most ONE bundle, for W only, iff G1 ∧ G2 ∧ E0 ∧ LIN ∧ G3.
W2 is hedge diagnostic only — never built in this run. If W fails any gate:
no bundle, report and stop (no-winner is an acceptable honest result).
