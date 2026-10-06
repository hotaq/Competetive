# New-meta baseline: repo agents versus the rank-3 leaderboard agent — 2026-08-29

## Why this reference is valid

Tapes were harvested for the top five leaderboard teams
(`artifacts/opponent_tapes_20260829/`, via `scripts/harvest_opponent_tapes.py`). Action
identity across three replays each, spanning different seeds, seats, and opponents:

| Rank | Team | Score | Cross-episode action identity | Final money |
|---:|---|---:|---:|---|
| 1 | Milan Leonard | 3025.9 | 14.3-16.8% | 86,669-129,490 |
| 2 | Crop Dusta | 3023.3 | 19.0-20.1% | 122,221-137,034 |
| 3 | Yusuke Hayashi | 2913.3 | **100.0%** | 79,699-108,893 |
| 4 | Kenjo1209 | 2902.7 | 90.4-93.8% | 72,126-105,154 |
| 5 | Subramanya N | 2894.4 | 11.4-13.6% | 79,494-113,657 |

Ranks 1, 2, and 5 are strongly state-reactive, so replaying them as tapes would produce a
sanitized opponent once state diverges, which is weak evidence. **Yusuke Hayashi is
different: 100.0% identical with zero divergence across all three pairs.** That agent does
not adapt in real play either, so replaying its tape is an exact reproduction rather than a
degraded proxy. It is the first high-quality reference in this project that is not a
descendant of its own agents.

Agent: `submissions/opponent_rank3_yusuke_tape/` (from episode 102587748).

## Result: 12 seeds, both seats, 24 games each, seed base 27082901

| Candidate | Wins / 24 | Our money | Yusuke money | Mean margin | Worst margin |
|---|---:|---:|---:|---:|---:|
| v16_rc6_adaptive_market_standalone | 24 | 104,892 | 82,090 | 22,802 | 9,663 |
| v67_quantity_conserving_front_run_standalone | 24 | 107,247 | 80,630 | 26,617 | 6,210 |
| tape_live_55861364 | 16 | 76,573 | 72,294 | 4,279 | -4,902 |

**V16 and V67 both win 24/24 against a genuine 2913.3-rated agent, with positive worst-case
margin.** The competitor `boatlee` tape, which rates 2495.5 live against our 2135.8,
manages only 16/24 here and has a negative worst case.

## What this does and does not show

It does not show that V16 or V67 would rate above 2913. Rating comes from the whole field,
this is one opponent on twelve seeds, and ranks 1, 2, and 5 cannot be tested this way
because they are reactive.

It also does **not** show that V67 is underrated. That was the initial reading, and it is
wrong. V67's live submission `55797143` has played 162 episodes, winning 71% overall but
only 48% of its last 25, so 2135.8 is a converged rating and its real ceiling.

The correct conclusion is the uncomfortable one: **local head-to-head does not predict
leaderboard rating.** V16 and V67 beat, by 24/24 with positive worst case, an agent rated
778 points above them. The same pattern holds against the `boatlee` tape, rated 360 points
above V67 while losing 22/24 locally. Any promotion rule gated on 12-seed head-to-head
wins - which is what the whole V16 through V86 series used - is not measuring the thing
that scores.
