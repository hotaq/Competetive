# Newest public agents vs H54c (pre-registered 2026-09-24, before any game)

Goal (user): close the gap to the top-10. Our best live agent is H54c (56519374,
2357 after 27 games). Cheapest lever: a stronger public agent published since
this morning. Code scanned statically: no network/subprocess; `open(path)` and
`os.environ.get("V92_SELL_LIB")` only read an optional, unset library path.

## Candidates (artifacts/public_notebooks_20260924b/outputs, md5[:8])

N1 harvest-ledger new version 7f612276 · N2 population-robust-economy 647b3906 ·
N3 multi-route-farming-agent e58cd76f (cha22 identical) · N4 autonomous-ai-farming-agent 919466e4 ·
N5 herd-safe-v3 5cbf90be · N6 master-engine-v3 592cd7cb · N7 herd-safe race ca25 (tar main.py) ·
N8 shepherds-ledger 715cbd1d · N9 v40-challenger fe95fba0.
(demand-preserving = 099c956e = B, already known to play like C: skipped.)

## Stage 1 — screen

Each N: h2h vs H54c seeds 25500001–25500020 both seats (40), and the live panels
yannik (131; H54c 0.9237), omar (192; H54c 0.9688), C-live (68; H54c 0.7574).
Eligible: live pooled point (391 games) >= H54c's − 0.02. W3 = eligible
candidate with the highest h2h vs H54c (ties: mean margin); must be >= 0.55.

## Stage 2 — confirm W3

| gate | criterion |
|---|---|
| M1 | h2h vs H54c, seeds 25600001–25600100 both seats: point >= 0.60, CI95 lower > 0.50 |
| M2 | each live panel point >= H54c's − 0.02 |
| M3 | arena seeds 25000001–25000050 vs A,D,E,F,G,X7,X5,Y and C: pooled >= H54c's (0.795 on the 8 + 0.765 vs C) − 0.03 |
| M4 | 0 exceptions |
| G3 | official env from W3's own tarball, both seats vs H54c, money equals play.py |

Pass → propose W3 to the human. Fail → keep H54c; report.
