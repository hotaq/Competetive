# Analysing the rank-1 agent versus V67 — 2026-08-29

Milan Leonard, rank 1, score 3025.9. Twelve public replays harvested to compact tapes
(`artifacts/opponent_tapes_20260829/rank001_milan_leonard/`). Compared against V67's own
28 preserved live tapes from submission 55797143
(`artifacts/preserved_tapes_20260829/v67_live_55797143/`).

## The economies are nearly identical

| | Milan (rank 1) | V67 live |
|---|---|---|
| HIRE actions per game | 280 | 279 |
| BUY_LAND steps | 151, 266 | 161, 253 |
| Seeds bought | WHEAT 188, STRAWBERRY 46, MELON 12, CARROT 10 | WHEAT 136, STRAWBERRY 39, CARROT 19, MELON 13 |
| Animals | SHEEP 7, COW 6 | COW 10, SHEEP 5 |

Milan's step-2 market order is also byte-identical to the `boatlee` tape's. The top of the
leaderboard has converged on a common opening, and V67 is already playing essentially the
same macro-economic game. Two differences stand out: Milan sells 386 WHEAT against V67's
700, keeping more of its revenue in FERTILIZER (311), STRAWBERRY (206), MILK (197) and
WOOL (165); and Milan passes 952 times against V67's 443, moving far less.

## Money looks better for Milan, but the comparison is confounded

| | n | Win rate | Money mean | Money min | Money sd | Margin mean | Margin worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| Milan | 12 | 75% | 106,057 | 75,381 | 19,187 | +2,276 | -35,552 |
| V67 | 28 | 86% | 89,190 | 46,926 | 32,369 | +23,051 | -21,061 |

V67 has the *better* win rate and a mean margin ten times larger. An initial reading was
that Milan has a higher floor, but that does not survive checking: V67 **won** its lowest
money games (46,926 against an opponent on 43,791), because absolute money is set by the
episode's economy rather than by agent strength. V67's four losses sit mid-range at 72,129
to 91,247.

## The actual explanation: opponent strength

Mapping every opponent in both samples onto the leaderboard:

| | Opponent score, mean | Opponent rank, median |
|---|---:|---:|
| Milan | **2,992.5** | **2** |
| V67 | **1,373.9** | **1,792** |

**V67's 86% win rate and +23,051 mean margin were earned against opponents rated roughly
1,600 points below the ones Milan beats.** Those statistics carry no evidence of strength.
Milan's thinner +2,276 margin, taken against the very top of the field, is worth far more.

This is consistent with V67's live trajectory: 162 episodes, 71% wins overall but 48% over
the last 25 as matchmaking raised the difficulty.

## Consequence

Every comparison this project has used to promote a candidate — local head-to-head on
twelve seeds, and live aggregate win rate and margin — has been measured against weak
opposition. Win rate and margin must be conditioned on opponent rating before they mean
anything.
