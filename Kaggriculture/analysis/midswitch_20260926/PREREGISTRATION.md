# Mid-game route switch at day 12 / day 18 on SB682 (pre-registered 2026-09-26)

Idea (user): let the agent re-decide mid-game instead of locking the route at day 6.
Unlike 144 and 648, steps 288 and 432 are new splice points, so connectivity is checked first.

## Variants (overlay on SB682 = N3 + d29h10 strawberry block sale)

Native router until obs step S, then forced route R until 648 (N3's endgame route 2 afterwards,
plus the SB682 sale). S ∈ {288, 432}; R ∈ {R105, R106, R107, R110, R120, R125} (the day-6
candidates from `analysis/route_rl_20260926`). 12 variants. Step-S features are captured:
money of both sides, their difference, tile similarity, hands, shops, native route.

## Stage 0 — connectivity (N3-live 136)

A splice (S, R) is usable only if it has 0 exceptions, no game ending < 20k, and mean money
>= 0.9 × SB682's mean money on the same games. Unusable splices are dropped.

## Stage 1 — oracle and learning (train = 700 live episodes, usable splices only)

Report the oracle over {SB682} ∪ usable splices. Per-option regression trees at depth 1–3
(min leaf 30, 5-fold CV), with SB682 as the default when an option is not predicted +0.01 better.
Stop if CV <= SB682 (0.8414).

## Gates (holdout 258, `analysis/rl_endgame_20260926/holdout`)

| gate | criterion |
|---|---|
| M1 | point(policy) − point(SB682) >= +0.03, paired bootstrap CI95 lower > 0 |
| M2 | arena vs the 17-agent field, fresh seeds 27100001–27100020 both seats: pooled >= SB682 − 0.02 |
| M3 | 0 exceptions; G3 official env from the built tarball, both seats vs N3, money equals play.py |

All runs detached (`nohup caffeinate`). Pass → propose. Fail → report.
