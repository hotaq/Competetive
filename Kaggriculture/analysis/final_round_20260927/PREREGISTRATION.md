# Final round: strawberry timing for the current meta on MS (pre-registered 2026-09-27)

Context: team rank 782 (2221.3); the silver cutoff is rank 504 (2346.7). The active pair is MS
(56578534, SB682 + learned day-12/18 route switch) and SB682 (56541172). Deadline is
2026-09-30 23:59 UTC. The human authorized auto-submission if every gate passes.
A diagnosis of MS's 40 latest live losses (descriptive, disclosed; those games sit mostly
in the confirm half below) showed opponents now dumping strawberries from d25h1. MS dumps
at d29h10.
T0 (done): MS replayed locally reproduces 40/40 of its live games exactly.

## Variants (MS main.py; only the SB overlay's constants change)

`_SB_STEP` = S ∈ {551 (d23h23), 575 (d24h23), 599 (d25h23), 623 (d26h23), 647 (d27h23)}
× mode {once (as in SB682), cont (re-fires every turn with >= 5 held)}. 10 variants.
Baseline = MS (S = 681, once).

## Panel

328 public episodes of 56578534 and 56541172 absent from every earlier panel
(2026-09-26 07:02 → 2026-09-27 16:34), sorted by end_time. SELECT = the older 164,
CONFIRM = the newer 164.

## Selection

W = the variant with the highest SELECT point (ties: mean margin). It must beat MS on SELECT,
or the line stops.

## Gates

| gate | criterion |
|---|---|
| F1 | CONFIRM: point(W) − point(MS) >= +0.03, paired bootstrap CI95 lower > 0 |
| F2 | arena vs the 17-agent field, fresh seeds 27300001–27300020 both seats: pooled(W) >= pooled(MS) − 0.02 |
| F3 | 0 exceptions, no game ending < 20k; G3 official env from the built tarball, both seats vs MS, money equals play.py |

Pass all → submit (authorized); this retires SB682, leaving W + MS as the final pair.
Any fail → no submission; MS + SB682 stay.
