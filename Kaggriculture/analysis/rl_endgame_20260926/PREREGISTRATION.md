# Learned endgame sell policy on N3: contextual bandit (pre-registered 2026-09-26)

Goal: close games in the V39 clone war are decided by endgame sell timing (SB682, N3M,
`analysis/strong_losses_20260924`). No fixed timing wins everywhere: d29h10 helps in
2100–2300 and against exact mirrors, while d27 dumps cost ~1,100 coins against most agents.
Learn when to dump from context.

## Policy (overlay on N3 main.py md5 e58cd76f)

At obs step 630 (d27h6) the overlay computes features from its own observation and picks
one action for the rest of the game:
- a0 N3 default (drip) · a1–a4 strawberry block sale at the first obs step >= 640 / 660 / 672 / 681
  with >= 5 held · a5 block sale of STRAWBERRY, MILK, WOOL, EGG at 672
Features: our/opponent money and their difference; public-tile similarity; count of steps
with identical tiles and money; our shed (STRAWBERRY, MILK, WOOL, EGG, CARROT, WHEAT);
strawberry and carrot plants on both farms; market prices of those six goods; hands on both
sides; how many unlocked town shops buy strawberries.

## Data

Train = 704 recorded live episodes (N3-live 136, SB682 panel 173, C-live 68, yannik 131,
omar 192). Every action a0–a5 is replayed on every episode (full information, no
exploration), with the opponent's recorded actions.
Holdout = 258 public episodes of 56521684 / 56541172 / 56519374 absent from every earlier
panel, harvested 2026-09-26. Untouched until the policy is frozen.

## Learning

Per-action regression tree on point (depth in {1,2,3}, min leaf 20), depth chosen by 5-fold
CV on train. Policy = argmax predicted point; ties go to a0. The fitted trees are written out
as plain if/else, so there is no library at runtime. BEST_FIXED = the single action with the
highest train point.

## Gates (holdout and fresh seeds)

| gate | criterion |
|---|---|
| H1 | holdout: point(policy) − point(N3) >= +0.03, paired bootstrap CI95 lower > 0 |
| H2 | holdout: point(policy) >= point(BEST_FIXED) |
| H3 | arena vs the 17-agent field, seeds 26700001–26700020 both seats: pooled >= N3 − 0.01 |
| H4 | mirror vs N3, seeds 26800001–26800050 both seats: point >= 0.60 |
| H5 | 0 exceptions; G3 official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose. Fail → report; nothing changes.
