# Graft top-10 routes into N3's reactive chassis (pre-registered 2026-09-26, before any game)

Why: the current top-10 share only 1–5% of N3's opening (a different strategy family, private
code). Their recorded tapes lose when played open-loop. Hypothesis: a top-10 route inside N3's
reactive chassis (weed/hand repair, sell lead, race, terminal layers) beats N3.

## Variants (overlay appended to N3 main.py md5 e58cd76f; nothing else changes)

Donor tape per team = the best game in `artifacts/tape_pool_20260924_top10` for each of the 10
leaderboard teams (win first, then own money): DSM, DECEM, Vadim, M&M&P&Q, THIRD FARM, Majkel,
吃白饭的大肥鱼, Kaggledew, Unknown Mother-Goose, Fourth Quadrant. The chassis tape is
actions[1:720] (tape index = observation step).
- G<team>-full: every N3 route id → donor tape
- G<team>-648: every route id except 2 → donor tape. Route 2 (N3's day-27+ endgame) is kept.
20 variants.

## Stage A — select on the N3-live panel (136, `analysis/n3_live_20260925/replays`)

N3 0.5551, SB682 0.6397. W = highest point. It must be > 0.6397, or the line stops.
Also reported, not gated: the recorded top-10 panel T24 (79).

## Stage B — confirm W

| gate | criterion |
|---|---|
| C1 | SB682 live panel (173, `analysis/n3_clone_detect_20260926/sb682_panel/replays`): point(W) >= SB682 recorded 0.7514 and >= N3 replay 0.6994 + 0.03 with paired bootstrap CI95 lower > 0 |
| C2 | arena, fresh seeds 26500001–26500020 both seats vs the 17-agent field: pooled point(W) >= N3's pooled on the same seeds − 0.02 |
| C3 | h2h vs N3, seeds 26600001–26600050 both seats: point >= 0.50 |
| C4 | 0 exceptions; G3 official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose W. Fail → report; nothing changes.
