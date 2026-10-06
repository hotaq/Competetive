# N3 strawberry pre-emptive block sale (pre-registered 2026-09-24, before any game)

Diagnosis (`analysis/generalization_20260924/live_56521684/ep112896893.json`): N3's only
live loss (−676 vs sAIzeria Harness 2494) came from the day-29 strawberry race. The rival
dumped 20 at d29h11 (price 59), and N3 dripped 3 per turn from h13 into a crashed price.
Counterfactual on the recorded game: dumping N3's 20 one to seven turns earlier wins by
+276 to +598. Raising N3's race horizon (48/54/60/72) does not change this game (−672):
the race layer reacts only after a rival sale. Across 395 harvested live games, 94–100% of
opponents dump at least 10 strawberries on days 28–30.

## Variants (overlay appended to N3 main.py, md5 e58cd76f; N3 code untouched)

The overlay wraps the callable Kaggle would pick for N3 (the last callable in the module).
At the first observation step >= S where the shed holds >= 5 strawberries, it drops the
parent's STRAWBERRY sells and prepends one SELL STRAWBERRY <all held> (market list capped
at 10). It fires once per game and does nothing else.

| id | block-sale time (recorded action index / day-hour) | obs step S |
|---|---|---|
| SB669 | 669 / d28h21 | 668 |
| SB673 | 673 / d29h01 | 672 |
| SB682 | 682 / d29h10 | 681 |
| SB691 | 691 / d29h19 | 690 |

## Stage S — select on 391 recorded live games

Panels: yannik (131), omar (192), C-live (68). Paired against N3's existing results on the
same episodes (`analysis/public_agents_20260924b/screen/N3_*`). W = the variant with the
largest mean paired margin gain over N3. It must be > 0, or the line stops.

## Stage C — confirm W

| gate | criterion |
|---|---|
| Q1 | N3-live panel (24 episodes of 56521684, frozen now): mean paired margin(W − N3 replay) >= 0 and point(W) >= point(N3 replay) |
| Q2 | closed-loop arena, fresh seeds 25800001–25800020 both seats, vs N1,N2,N4,N5,N6,N7,N9,A,D,E,F,G,X7,X5,C,H54c,Y: pooled point(W) >= pooled point(N3) − 0.02 (N3 is re-run on the same seeds) |
| Q3 | mirror: W vs N3, fresh seeds 25900001–25900050 both seats: point >= 0.60, seed-cluster CI95 lower > 0.50 |
| Q4 | 0 exceptions |
| G3 | official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose to the human. Fail → N3 stays; report.
