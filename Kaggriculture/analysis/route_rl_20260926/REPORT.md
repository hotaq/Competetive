# Learned midgame route choice REPORT — FAIL (R1)

Train (700 live episodes × 42 forced routes, all with the SB682 endgame): SB682/native 0.8414
is the best fixed arm (next R107 0.769). Oracle 0.970, mostly coin-flip noise. Candidates
within 0.10: R105, R106, R107, R120, R125, native. CV: depth 1 0.8521, depth 2 0.8307,
depth 3 0.8479. Depth 1 selected: every tree splits only on "is a YARN_STORE among the
first two shops".

Holdout (258): policy 0.7132 vs SB682 0.7016, +0.012, CI95 [−0.023, +0.047]. R1 FAIL.
It picked R120 in 109 games and native in 149. 0 exceptions. Nothing submitted.
Reading: the day-6 route choice holds about +0.01 of real signal; the rest of the oracle is luck.
