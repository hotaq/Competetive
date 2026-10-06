# Learned midgame route choice on top of SB682 (pre-registered 2026-09-26)

Exploration (`analysis/route_oracle_20260926`, N3-live 136, disclosed): forcing each of N3's
41 routes for obs steps 144–647 connects cleanly. The native route reproduces N3 exactly on
136/136, with 0 errors and no crashed games. The per-game oracle is 0.893 vs N3 0.555, but
no fixed route beats N3 (best R110 0.533). The oracle is inflated by near-coin-flip games.

## Agent (overlay on N3 main.py md5 e58cd76f)

Phase 1: N3's opening (unchanged). Phase 2: route chosen at obs step 144 by the learned policy
(fallback = N3's native router). Phase 3: N3's endgame route plus the SB682 block sale at obs
step 681. Baseline = SB682 (native route + phase 3).

## Data

Train: the same 700 recorded live episodes as `analysis/rl_endgame_20260926` (N3-live,
SB682 panel, C-live, yannik, omar). Every forced route R plus the SB682 endgame is replayed on
each; features come from the observation at step 144 (the first two unlocked shops, our and
the opponent's money and their difference, public-tile similarity, identical-state count,
hands on both sides).
Holdout: the 258 fresh episodes in `analysis/rl_endgame_20260926/holdout`. That set was used
only as the untouched test of the endgame policy and never to fit or select anything about
routes.

## Learning

Only routes with train point within 0.10 of the best fixed arm are candidates (limits noise
from 41 draws). One regression tree per candidate route on point (depth in {1,2,3}, min leaf
30, 5-fold CV). Policy = argmax predicted point, with native as the default when native is
within 0.01. If CV point <= SB682's train point, stop (no holdout run).

## Gates

| gate | criterion |
|---|---|
| R1 | holdout: point(policy) − point(SB682) >= +0.03, paired bootstrap CI95 lower > 0 |
| R2 | arena vs the 17-agent field, fresh seeds 26900001–26900020 both seats: pooled >= SB682 − 0.02 |
| R3 | 0 exceptions, no game ending < 20k; G3 official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose. It must be submitted before N3 is re-submitted last, since each new
submission retires the older active one. Fail → report; the active pair stays.
