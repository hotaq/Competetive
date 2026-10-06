# Closed-loop arena REPORT (per ARENA_PREREGISTRATION.md)

Seeds 25000001–25000050, both seats, 100 games per pair, 0 exceptions.
Fidelity: C vs X7 on seed 25000001 equals the official env exactly (both seats).

| row vs col | C | A | D | E | F | G | X7 | X5 | Y | pooled |
|---|---|---|---|---|---|---|---|---|---|---|
| C | – | .90 | 1.00 | 1.00 | .82 | .95 | **.00** | 1.00 | .56 | .779 |
| X7 | **1.00** | .99 | .94 | .94 | .98 | .06 | – | .89 | .46 | **.782** |
| E | .00 | .85 | .75 | – | .76 | .98 | .06 | 1.00 | .62 | .627 |
| D | .00 | .85 | – | .25 | .76 | .98 | .06 | 1.00 | .62 | .565 |
| F | .18 | .83 | .24 | .24 | – | 1.00 | .02 | 1.00 | .58 | .511 |
| Y | .44 | .42 | .38 | .38 | .42 | .46 | .54 | .54 | – | .448 |
| A | .10 | – | .15 | .15 | .17 | .99 | .01 | 1.00 | .58 | .394 |
| G | .05 | .01 | .02 | .02 | .00 | – | .94 | 1.00 | .54 | .323 |
| X5 | .00 | .00 | .00 | .00 | .00 | .00 | .11 | – | .46 | .071 |

Strongly non-transitive: C beats every public agent except X7 (0/100);
X7 beats all but G; G beats only X7 and X5.

Challenger rule: X7 h2h vs C 1.00 and pooled 0.782 > 0.779, so X7 qualifies
and was checked on the live panels:

| panel | X7 | C |
|---|---:|---:|
| CONFIRM 131 (LIVE yannik 0.603) | 0.947 | 0.924 |
| strong subset (LIVE 0.574) | 0.941 | 0.941 |
| SELECT 192 omar | 0.865 | 0.969 |
| recorded top-10 T24 / T22 | 0.785 / 0.743 | 0.835 / 0.797 |

X7: P1 +0.3435 CI [+0.252, +0.435] PASS, P2 PASS, P3 PASS, G3 PASS
(`g3_X7.json`, tar sha256 11fd4428d5f3a4fe…, archive = main.py only).
X7 − C on CONFIRM: +0.023 CI [−0.023, +0.076], a tie against the live population.

Reading: C and X7 are equals against the live field and hard counters of each
other. As the two final slots (team score = the better one) they hedge against
opposite metas; the yannik tape is dominated by both. The 80%-vs-top-10 goal
cannot be certified locally: recorded top-10 panels inflate everyone (yannik
0.772), and the only near-top-10 reactive code we have (X7) beats C 100/100.

## Submitted

2026-09-24: X7 submitted on the human's order as **56515322** (tar sha256 11fd4428d5f3a4fe…). At submit time C (56513871) was at 1748.0, yannik 1840.8.
