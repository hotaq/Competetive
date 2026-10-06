# Mirror breaker round 2 (pre-registered after round 1 FAIL, before any round-2 game)

New hypothesis: the mirror edge grows with the race horizon beyond the old cap.
Round-1 seeds are burnt; every round-2 seed is fresh.

## Variants (edits of C's last bindings; everything else identical)

- H50, H54, H60, H72: V9_RACE_DEFAULT = V9_RACE_MAX = h (lines 6662-6663)
- H50c, H54c, H60c, H72c: the same plus _CA_MARGIN = -7.0
- R48c9, R48c11: DEFAULT 48 (MAX 48) plus _CA_MARGIN = -9.0 / -11.0
- Reference arm (not selectable twice): round-1 V = race48 + ca−7

## Stage 1: each vs C, seeds 25300001–25300020, both seats (40 games)

V2 = best stage-1 point among the 10 new variants (ties: mean margin). No combos.

## Stage 2: identical gates to round 1, fresh K1 seeds

| gate | criterion |
|---|---|
| K1 | V2 vs C, seeds 25400001–25400100 both seats: point >= 0.70 and CI95 lower > 0.50 |
| K2 | point(V2) >= point(C) − 0.02 on C live panel (68, C 0.7353), yannik (0.9237), omar (0.9688) |
| K3 | arena seeds 25000001–25000050 vs A,D,E,F,G,X7,X5,Y: pooled >= C pooled 0.77875 − 0.03 |
| K4 | 0 exceptions |
| G3 | official env from the built tarball, both seats vs C, money equals play.py |

Pass → propose to the human. Fail → stop the knob line; report.
