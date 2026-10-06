# The winning mix is melon-heavy, and V67's schedule cannot reach it

Date 2026-08-30. Crop rules read from `rust/kaggriculture-engine/src/rules.rs:45`.
Prices and impact slopes from `analysis/market_process_20260830/`. Plant-to-harvest
timing instrumented on seed 27083001 against `meta_v16_rc6_adaptive_market_standalone`.

## Labour is the scarce input, so rank crops per harvest action

V67 issues 415 `HARVEST` commands a game and that number is fixed. What varies is what
each one collects.

| crop | seed | first yield | repeats | units/harvest | price@360 | gross/harvest | units to halve price |
|---|---:|---:|---:|---:|---:|---:|---:|
| MELON | 80 | day 10 | once | 6 | 224 | **1,344** | 78 |
| STRAWBERRY | 100 | day 10 | 2d | 4 | 204 | **816** | 75 |
| TOMATO | 50 | day 8 | 1d | 4 | 66 | 264 | 28 |
| WHEAT | 10 | day 2 | once | 6 | 41 | 246 | 663 |
| CARROT | 20 | day 2 | once | 4 | 42 | 168 | 78 |

V67 plants WHEAT 133, STRAWBERRY 40, CARROT 20, MELON 11. Two thirds of its plantings go
to the second-worst crop per harvest, and the best gets 5%.

## The mix it should be selling

Equalising marginal price across the five crops, over the same 261 crop units V67 sells:

| | current | optimal |
|---|---|---|
| STRAWBERRY | 154 units, ending at a price of **1** | 118, ending at 45 |
| MELON | **8** | **125** |
| TOMATO | 0 | 18 |
| WHEAT + CARROT | 99 | 0 as goods; wheat stays as feed |
| revenue | 21,020 | **32,449** |
| harvest actions | 58 | **55** |

**+11,429 a game, 1.54x, for three fewer harvest actions.** It is not a labour problem.
For scale, the 19 live losses run from -298 to -29,677.

V67 already plants 11 melons, which should yield 66 units. It sells 8. Almost the whole
melon crop is planted and then not collected.

## Why an in-place crop swap cannot deliver it

The cheapest possible intervention would rewrite only the crop argument of `PLANT`,
leaving movement, watering and harvest timing untouched. Measuring the gap between each
planting and the first harvest on that tile shows there is no room for it:

| crop | plants | median days to first harvest | p90 | gaps >= 8 days |
|---|---:|---:|---:|---:|
| WHEAT | 116 | 4.0 | 4.1 | **0** |
| CARROT | 19 | 3.0 | 3.1 | **0** |
| STRAWBERRY | 40 | 10.0 | 10.3 | 40 |
| MELON | 11 | 10.0 | 10.4 | 11 |

Every planting is harvested within a day of maturing. The schedule is fitted to the crops
it plants, so no wheat or carrot tile is idle long enough to carry a melon, which needs
10 to 12 days. The only long gaps belong to strawberry, and strawberry is ongoing while
melon is one-shot, so substituting there would forfeit every later harvest on that tile.

Twelve tiles are planted and never harvested at all.

## Conclusion

The strategy is identified and priced: melon-heavy, strawberry reduced to the level its
market can absorb, wheat kept as feed rather than sold. The gain is 1.54x on crop revenue
using less labour than V67 already spends.

It is not reachable by patching V67. Every patch route is now closed by measurement —
route selection (V85), land timing (V88), seed supply, and the sell side (V79, V89, V90) —
and this report closes the last cheap one, the in-place crop swap. Reaching this mix
means building a harvest schedule around it, which is a new agent, not an overlay.

---

## Correction: the per-harvest ranking above is wrong

The table at the top ranks crops by gross per `HARVEST` action, which counts only the
collecting. Planting and watering are labour too, and crops die after two consecutive
unwatered days, so a plant must be watered on at least every other day of its life.
Costing the full cycle reverses the order:

| crop | life (days) | waters | harvests | total actions | units | net | **net per action** |
|---|---:|---:|---:|---:|---:|---:|---:|
| STRAWBERRY | 20 | 15 | 10 | 26 | 40 | 8,060 | **310** |
| MELON | 12 | 6 | 1 | 8 | 6 | 1,264 | **158** |
| TOMATO | 22 | 15 | 22 | 38 | 88 | 5,758 | **152** |
| WHEAT | 4 | 2 | 1 | 4 | 6 | 236 | **59** |
| CARROT | 3 | 2 | 1 | 4 | 4 | 148 | **37** |

**Strawberry is the most labour-efficient crop in the game**, because one planting yields
ten harvests. V67's choice of it is correct. What is wrong is the quantity: at the 154
units it actually sells, the realised price is about 100 rather than 204, which halves
strawberry's true net per action to roughly 155 and brings it level with melon and
tomato. The market, not the crop, is what caps it.

So the conclusion stands and the reasoning changes. The three good crops are worth
roughly the same per action once price impact is counted, so the mix should spread across
all three thin markets rather than concentrate in one — which is what the marginal-price
allocation already said.

## What is actually being wasted

The numbers above use the step-360 price and ignore price impact, so they overstate any
crop we would scale up. Ranking by net per action is the labour question; market depth is
a separate cap.

V67 spends **135 of its 186 crop plantings on wheat and carrot**, the two crops worth 37
to 59 per labour action, against 118 to 310 for the other three. That is roughly 540
labour actions in the cheapest crops in the game.

Those actions cannot all move into crops, because every good crop market is thin: melon
halves in 78 units, strawberry in 75, tomato in 28. The one deep market among the goods
V67 can produce is **EGG, at 217 units to halve and a price of 52**, and eggs cost no
planting or watering at all — geese need `FEED` and `CARE`. Wheat is the feed, which is
the one honest reason to keep growing it.

That is the shape of the agent worth building: strawberry and melon sized to their market
depth, tomato as the third outlet, wheat grown only in the quantity the animals eat, and
the labour this frees moved into the animal economy, where the deep market is.
