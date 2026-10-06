# Newest public agents vs H54c REPORT — PASS: W3 = N3 (multi-route-farming-agent, e58cd76f)

Stage 1 (h2h seeds 25500001–25500020; live panels 391 games; H54c live pooled 0.9169):

| id | h2h vs H54c | yannik | omar | C-live | live pooled |
|---|---:|---:|---:|---:|---:|
| **N3 multi-route (= cha22)** | **0.575** (+559) | 0.977 | 0.971 | 0.809 | **0.9450** |
| N6 master-engine-v3 | 0.075 | 0.962 | 0.964 | 0.838 | 0.9412 |
| N5 herd-safe-v3 / N8 shepherds (identical) | 0.375 | 0.969 | 0.974 | 0.662 | 0.9182 |
| N1 harvest-ledger (new) | 0.250 | 0.924 | 0.958 | 0.691 | 0.9003 |
| N7 ca25 | 0.100 | 0.969 | 0.969 | 0.515 | 0.8900 |
| N4 autonomous | 0.075 | 0.977 | 0.958 | 0.500 | 0.8849 |
| N9 v40-challenger | 0.225 | 0.924 | 0.969 | 0.529 | 0.8772 |
| N2 population-robust | 0.000 | 0.794 | 0.729 | 0.235 | 0.6650 |

Stage 2 (`gates.json`):

| gate | N3 | H54c | verdict |
|---|---|---|---|
| M1 h2h vs H54c, seeds 25600001–25600100 (200) | 0.835, CI95 [0.76, 0.90], +1,034, 0 ties | — | PASS |
| M2 yannik / omar / C-live | 0.977 / 0.971 / 0.809 | 0.924 / 0.969 / 0.757 | PASS |
| M3 arena pooled vs A,D,E,F,G,X7,X5,Y,C | 0.904 (X7 0.98, C 0.86) | 0.792 (X7 0.00) | PASS |
| M4 exceptions | 0 | — | PASS |
| G3 official env, own tarball, seed 25600001, both seats vs H54c | 134,958 / 134,239 equal to play.py | — | PASS |

N3 beats every agent we hold, including X7, which C and H54c lose to 100/100.
Tarball `artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/submission_cha22.tar.gz`
(sha256 63e1554c93305577…, main.py only; Apache notices are inside main.py). Not submitted by this run.

Submitted 2026-09-24 on the human's order as **56521684** (tar sha256 63e1554c93305577…). At submit time: H54c 2402.4, C 2291.5, yannik 1843.8, X7 1670.1.
