# Refresh 2026-09-24 — fresh top-10 harvest, challenger for Slot B

Pre-registered 2026-09-24 BEFORE any game on the seeds below. Live at start:
yannik refresh (submission 56495539, public 1849.3, team best) and omar
(56361130, public 1438.2, decaying, expendable). Top-1 DSM 3098.8. Deadline
2026-09-30 23:59 UTC. No `kaggle competitions submit` in this run (human fires).
Nothing is committed.

## Change vs refresh_20260923 (decided before any game)

The V100 screen panel is stale: on 09-23 it ranked DSM/DECEM (now #1/#2 live)
at 0.17/0.27. The screen panel is therefore the Sept-22 best-per-team set
(12 tapes incl. live yannik), and the holdout is the current top-10 teams'
tapes. Comparator = live yannik tape (Slot A = 56495539 stays; Slot B = omar
is replaced only by a gated winner).

## Frozen inputs

- Pool: `artifacts/tape_pool_20260924_top10/` — 98 tapes / 49 episodes,
  10/10 top-10 teams, harvested 2026-09-24 05:27 UTC, 0 dropped.
- Screen opponents (12): `analysis/refresh_20260923/roundrobin_pool/tapes/*`
  minus `omar_althobaiti__ep110807162__seat1`.
- ROUTER reference: `analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json`.
- Live tape: `artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json`.
- Engine unchanged (`play_batch`/`play_fast`), both seats always.

## Fresh seeds (grep of analysis/ scripts/ docs/ found none)

- Screen: 23000101–23000200 (100).
- Round-robin: 23000201–23000400 (200).
- Holdout: 23100001–23101000 (1000).
- E0 probe: 23100001–23100010 (subset of holdout).

## Stages

1. Screen: 98 pool tapes + ROUTER vs 12 screen opponents. Best pooled point per
   team; top 12 teams forward.
2. Round-robin: 12 shortlisted + live yannik tape, all pairs. W = highest pooled
   point among the 12 excluding games vs live; W2 = runner-up from a different
   team with prefix-72 identity vs W < 0.90.
3. Holdout: arms W, W2, live (`--tape`) vs the 10 remaining shortlisted tapes
   (all current top-10 teams), `evaluate_shop_router_fast.py`.
4. Checks: E0 determinism, LIN.
5. Build only on pass (single-route file + unmodified
   `scripts/build_shop_router_v1_submission.py`), then G3 smokes both seats.

## Gates (frozen)

| gate | criterion |
|---|---|
| G1 | mean paired money(W − live) CI95 lower bound > 0 (seed-cluster bootstrap) |
| G2 | pooled point(W) − point(live) >= −0.01 on holdout |
| G4 | point(W) >= 0.50 on the holdout (every holdout opponent is a current top-10 team tape) |
| E0 | 0 mismatches on determinism probe |
| LIN | W prefix-72 identity vs live yannik < 0.90 |
| G3 | bundle equivalence + both-seat official-env smokes DONE/DONE, 0 invalid keys |

Build at most ONE bundle (W) iff all pass. W2 is diagnostic only. No winner is
an acceptable result: then nothing is built and 56495539 + 56361130 ride.
