# Reactive dump-follow on SB682 (pre-registered 2026-09-26, before any game)

Idea (user): let the agent correct the game turn by turn instead of on fixed steps. In the
V39 clone war, games are lost when a rival dumps first and we drip into the crashed price.

## Overlay (on SB682 = N3 + d29h10 strawberry block sale, `submissions/n3_sb682_standalone/main.py`)

From obs step 624 (d26): if an item's market price falls by >= D (fraction) from the previous
observation, we did not SELL that item on the previous turn, and we hold >= 5, then block-sell
everything we hold of it now (the parent's sells of that item are dropped that turn). Each item
fires at most once.
Variants: RD30s (D=0.30, STRAWBERRY) · RD50s (D=0.50, STRAWBERRY) ·
RD30a (D=0.30; STRAWBERRY, MILK, WOOL, EGG, CARROT) · RD50a (D=0.50; same items).

## Select (train = the same 700 recorded live episodes)

Comparator: SB682 (`analysis/route_rl_20260926/train/native_*`, 0.8414). W = the variant with
the highest train point. It must be > 0.8414, or the line stops.

## Gates

| gate | criterion |
|---|---|
| D1 | holdout 258 (`analysis/rl_endgame_20260926/holdout`): point(W) − point(SB682) >= +0.03, paired bootstrap CI95 lower > 0 |
| D2 | arena vs the 17-agent field, fresh seeds 27000001–27000020 both seats: pooled >= SB682 − 0.02 |
| D3 | 0 exceptions, no game ending < 20k; G3 official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose. Note: a new submission retires the older active one (N3 56521684), leaving
W + SB682. Fail → report; nothing changes.
