# Mirror breaker round 1 REPORT — FAIL (K1)

Stage 1 (seeds 25100001–25100020, 40 games vs C): 22 of 30 single-constant
variants are inert or near-inert (>= 34/40 identical games). The lever is the
market-race horizon: V9_RACE_DEFAULT 38 → 0.375, 42 → 0.575, 43 → 0.600,
44 → 0.650, 46 → 0.600, 48 → 0.675. ca_margin −7.0 → 0.675. Pre-declared combo
race48 + ca−7 → 0.800 (+130, 4 ties) = V.

Stage 2 (`confirm/`, `gates.json`):

| gate | V | C | verdict |
|---|---|---|---|
| K1 vs C, seeds 25200001–25200100 (200 games) | 0.655, CI95 [0.57, 0.74], +79, 24 ties | — | FAIL (< 0.70) |
| K2 C live panel (68) / yannik (131) / omar (192) | 0.757 / 0.924 / 0.969 | 0.735 / 0.924 / 0.969 | PASS |
| K3 arena pooled vs A,D,E,F,G,X7,X5,Y | 0.793 | 0.779 | PASS |
| K4 exceptions | 0 | — | PASS |

Verdict: FAIL, nothing proposed. The mirror edge is real (CI excludes 0.5) but
below the pre-set 0.70 bar; stage 1 overstated it (0.80 → 0.655). No cost
anywhere else. The horizon is pinned at its cap (V9_RACE_MAX = 48;
`horizon = min(MAX, max(DEFAULT, lead + MARGIN))`), which motivates round 2.

# Round 2 REPORT — PASS: V2 = H54c (horizon 54/54 + _CA_MARGIN −7.0)

Stage 1 (seeds 25300001–25300020 vs C): H54c 0.750 (+221), R48c11 0.725,
R48c9 0.725, round-1 V 0.700, H60c/H50c/H72c 0.700, H54 0.675, H60/H50/H72 0.625.

| gate | H54c | C | verdict |
|---|---|---|---|
| K1 vs C, seeds 25400001–25400100 (200) | 0.765, CI95 [0.69, 0.835], +146, 18 ties | — | PASS |
| K2 C live / yannik / omar | 0.757 / 0.924 / 0.969 | 0.735 / 0.924 / 0.969 | PASS |
| K3 arena pooled (A,D,E,F,G,X7,X5,Y) | 0.795 | 0.779 | PASS |
| K4 exceptions | 0 | — | PASS |
| G3 official env from tarball, seed 25400001, both seats vs C | 95,859 / 95,642 equal to play.py | — | PASS |

Bundle: `submissions/mirror_h54c_standalone/` built by `build_bundle.py`
(tar sha256 9466cd3456027010…; main.py + C's LICENSE.txt, NOTICE.txt; header
line records the modification). Not submitted by this run. Still loses to X7 0/100
like C does, and the edge is only against C-lineage clones (mean +146 coins).

Submitted 2026-09-24 on the human's order as **56519374** (tar sha256 9466cd3456027010…). At submit time: C 2283.8, yannik 1843.8, X7 1689.0.
