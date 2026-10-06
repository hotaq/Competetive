# Live-episode counterfactual screen — 56495539 (yannik), 2026-09-24

Pre-registered BEFORE the full-panel run. Context: yannik 56495539 has
plateaued (1849.3 → 1845.9, 79/131 wins, ~56-61% in every rating band).
The Sept-24 tape league cannot find a better tape because top-10 teams are
reactive (`analysis/refresh_20260924/REPORT.md`). This screen uses the real
live games instead: the seed and the opponent's actual actions from every
episode 56495539 played.

## What has been seen already (exploratory, disclosed)

- E0: Rust `play_batch` replay of our + opponent's recorded actions reproduces
  live final money for both seats exactly on the first 13 episodes.
- Bias probe on the first 17 episodes: swapping our tape to omar / umg / dsm
  changes the opponent's final money by a mean 23-33% (market interaction).
  Their recorded actions answered yannik, so a counterfactual candidate faces an
  opponent that cannot react. This favours every non-live tape. Points seen:
  omar 13, umg 9, dsm 9 vs live 15 (of 17). Nothing else has been run.

## Frozen panel and candidates

- Panel: every episode of 56495539 harvested into `replays/` (target 131; the
  count is recorded in REPORT). Opponent = recorded actions `[1:720]`, seed =
  replay seed, our seat as played. Engine `{"episodeSteps": 720, "weedSpawnChance": 0.005}`.
- Split by episode id: SELECT = even id, HOLDOUT = odd id.
- Candidates: all tapes in `artifacts/tape_pool_20260919/` (380),
  `artifacts/tape_pool_20260922_top10/` (92), `artifacts/tape_pool_20260924_top10/` (98).
- LIVE = recorded result (exact). CONTROL = the Sept-22 pool yannik_schiffner
  tapes other than ep111952809 (same team, not live, so they carry the same
  can't-react bias as any candidate).

## Metric

point per episode (win 1, tie 0.5, loss 0). Secondary: point on episodes whose
opponent's current score >= 1846 (our rating; these are the games that move Elo).

## Selection

W = highest point on SELECT (ties: higher mean margin). No other tuning.

## Holdout gates (frozen)

| gate | criterion |
|---|---|
| H1 | point(W) − point(LIVE) >= +0.10 on HOLDOUT, paired episode bootstrap CI95 lower bound > 0 |
| H2 | point(W) >= point(LIVE) on HOLDOUT episodes with opponent score >= 1846 |
| H3 | point(W) − mean point(CONTROL) >= +0.05 on HOLDOUT (beats the bias baseline, not just live) |
| G3 | bundle equivalence + both-seat official-env smokes, as in refresh_20260923 |

Build one bundle (W, via the refresh_20260923 single-route path) iff H1-H3 pass,
then G3. It would replace Slot B (omar 56361130) only; the human submits.
All fail = honest negative; nothing built.
