# OrbitWars New Strategy Board

Status: active team research board. Do not submit to Kaggle from this board.

## Current baseline
- main.py and submission.py are currently the previous best after rollback from a harmful big-first prototype.
- Smoke after rollback passed: VALIDATION_SMOKE DONE,DONE rewards 1,1 steps 80.
- Latest lost ladder replay: .omx/research/ladder_replays/episode_75526435.json.

## Key replay clue: episode 75526435
- 4 seats all named hotaq; rewards -1,1,-1,-1.
- Seat 1 wins; final observed around step 224: seat 1 total 2275, prod 48, 32 planets; others eliminated.
- Seat 1 pattern: waits until enough ships for a high-confidence early capture, then launches larger follow-up from newly captured planet.
- Failed prototype: blindly merging early same-source/same-target moves into one large launch hurt deterministic 1v1 and was reverted.

## Shared hypotheses to challenge
1. Opening needs an assured capture + hold rule, not just send big.
2. 4P needs neighbor-collapse / anti-leader reaction earlier than current recovery trigger.
3. Low-production/high-neutral-ship maps need different opening priorities than rich maps.
4. Captured frontier planets should launch only when hold risk is acceptable or when the follow-up wins tempo.

## Worker ownership
Workers should write detailed notes in their scratch files and only append/update their own section below.
- Worker 1 scratch: .omx/research/team_strategy/worker-1.md
- Worker 2 scratch: .omx/research/team_strategy/worker-2.md
- Worker 3 scratch: .omx/research/team_strategy/worker-3.md

## Worker 1: lost replay forensic lane
Evidence pass complete in `.omx/research/team_strategy/worker-1.md`.

High-confidence replay findings from `episode_75526435`:
- Seat 1 did **not** win by blind big-first. It waited until step 15, sent `38` from home p5 to capture prod-2 p9 at step 20 with `1` left, then immediately reinforced/staged from that foothold.
- The first decisive swing was a frontier steal: seat 0 captured p20 at step 24 with `9` left, then seat 1 launched `35` from p9 at step 31 and flipped p20 at step 34 with `6` left. Symmetric seat 2 did the same to seat 3 p23.
- Seats 0/3 repeatedly made low-garrison frontier captures that were flipped: p20/p23 at step 34 and p11/p8 at step 69. Copy the recapture/denial window; avoid unprotected frontier grabs.
- Seat 1 broke symmetry with seat 2 after step 120 by taking high-production anchors: p6 at step 120 (`42` left), p23/p24 at 121-122, p7/p10 at step 131, then p22 at step 146 and p4 at step 161.
- Production snowball evidence: seat 1 moved from `10 planets / prod19` at step 131 to `19 planets / prod33` at step 161 and `30 planets / prod46` at step 176; seat 2 collapsed to `3 planets / prod4` at step 161 and zero planets by step 173.

Rules to copy: assured staged capture, immediate reinforcement of low-leftover new captures, urgent recapture of fresh enemy prod-2/prod-3 planets with low garrison, and high-production anchor priority after a core exists.
Rules to avoid: broad big-first, expensive prod-1 neutral hunting in opening, unprotected frontier captures, and duplicate same-source/same-target spam unless a deliberate timed overkill rule justifies it.

Audit addendum for Worker 3 validation outputs:
- Full notes: `.omx/research/team_strategy/worker-1-audit.md`.
- `worker3_frontier_survival_candidate.py` is code-material (4P leader-pressure + high-prod frontier reinforcement multipliers), but the available paired controls show no behavior delta versus current: old100/101 results are identical, and public902 is identical at `267 total / prod8 / 2 planets` with first capture still delayed `21` vs `9` for `submissionB`.
- Do **not** integrate the frontier-survival candidate yet; it remains an unproven W3-A/W3-B scaffold that does not fix the public902 opening collapse.
- Keep `worker3_static_race_rejected_candidate.py` rejected: public902 worsened from current/candidate `267 / prod8 / 2 planets` to `0 / prod0 / 0 planets`.

## Worker 2: replay-folder / sim-regression lane
Evidence pass complete in `.omx/research/team_strategy/worker-2.md`.
Challenger/validator pass added in `.omx/research/team_strategy/worker-2-challenge.md`.

High-confidence rejects / danger flags for leader integration:
- Reject broad `big-first` opening variants: `patch_big_first*` lost 1-3 on seeds 100-103 and produced zero-prod eliminations on seeds 102/103.
- Reject global close/low-value capture loosening: `patch_low_close*` was only 4-4 and had multiple zero-prod eliminations across unswapped and swapped seeds 100-107.
- Reject rank/top2-only 4P validation: public-hard seeds 900-901 show top2=7/8 but eliminated_or_zero=5/8.
- Reject old-base-only acceptance: current still collapses on old-base hard seeds 556/557/10001/10002/1213/1217 and public-hard FFA.
- Use only post-module-RNG-seeding reports, or rerun older GUI labels before trusting them.

Challenger verdicts:
- W1-A staged opening is promising only if it stays close/prod-2+/reinforced; reject immediately if it recreates `big-first` zero-prod failures on 100-107 or delays public-hard first capture.
- W1-B frontier recapture is the strongest replay-derived rule, but must be prod-2+, recent, short-ETA, and hold-positive; reject if capture count rises while final production/planets fall.
- Worker 3 scratch was not present at read time, but `worker3_frontier_survival_candidate.py` and reports were present. The candidate is small/4P-scoped, yet public seed 902 still loses to `submissionB` with latest `267 total / prod8 / 2 planets` vs `1352 / prod27 / 10 planets`, and first capture is delayed `21` vs `9`; not integration-ready without control-normalized public survival gains.

Compact validation matrix:
| candidate | challenge focus | hard accept | hard reject |
|---|---|---|---|
| W1-A staged opening | Avoid broad big-first; require close prod-2+ target, home guard, immediate hold/reinforce. | No new zero-prod/zero-total on 100-107; public 900-901 eliminated_or_zero <= 2/8; GUI 902-905 no repeated near-zero current seats. | Any new 100-107 zero-prod, long/low-prod opening launch, or public first-capture delay. |
| W1-B frontier recapture | Deny recent weak prod-2+ enemy captures without low-value churn. | Net production/planet survival improves; extra captures also improve final prod/planets. | More captures with lower prod/planets, prod-1 chase, long ETA recapture spam, or new zero-prod seats. |
| Worker3 4P leader-pressure candidate | Step-55 anti-leader nudge must improve public-hard survival, not just stay neutral in 1v1. | Match/beat current-control first-place while reducing zero/near-zero seats on public 900-905. | Public 902 remains prod < 10 or planets < 3, or first capture stays far later than public agents. |

Recommended gates for any next candidate: seat-swapped 1v1 seeds 100-107, known hard old-base seeds 555-560 / 1212-1217 / 9999-10002, plus public-hard FFA seeds 900-901 and GUI 902-905. Do not accept old-base 100-101 alone for a 4P candidate.

## Worker 3: implementable patch / benchmark lane
Evidence pass complete in `.omx/research/team_strategy/worker-3.md`.
Candidate scaffold: `.omx/research/team_strategy/worker3_frontier_survival_candidate.py`.
Rejected candidate copy: `.omx/research/team_strategy/worker3_static_race_rejected_candidate.py`.

High-confidence implementation notes:
- Current `main.py` and `submission.py` were byte-identical at inspection (`dc3e1d0aedfa` short SHA); Worker 3 did not mutate either file.
- Public B mirrors current on many core 4P constants (`FOUR_PLAYER_ROTATING_SEND_RATIO=0.55`, `FOUR_PLAYER_TARGET_MARGIN=2`, `PROACTIVE_DEFENSE_RATIO=0.28`, `MULTI_ENEMY_PROACTIVE_RATIO=0.35`), while public D/E spend more aggressively (`0.62` send, margin `3`) with weaker proactive defense. Do not expect a simple public-constant copy to solve this.
- Public-hard seed 902 control: current lost to `submissionB` with `267 total / prod8 / captures29 / first_capture21` versus `1352 / prod27 / captures52 / first_capture9`; the opening report shows current fired at static p14 on step 4, then p2 captured p14 at step 9. This confirms opening race/target choice is a real failure mode.
- Static-neutral race filtering was tested and rejected: the exact candidate worsened seed 902 from `267` to `0` total (`submissionB` rose to `3238`). Do not integrate that filter as-is.
- Recommended conservative path is 4P-only production-leader pressure plus mild high-prod frontier reinforcement scoring. It is a scaffold, not proven improvement yet; validate on public-hard 900-905 before touching `main.py`.

Patch designs:
1. **W3-A production-leader pressure**: add `_dominant_enemy`, gate by 4P + step >= 55 + enemy prod gap >= 6, and mildly boost target value/score against the production leader. Risk medium; avoids over-focusing weakest enemy while `submissionB/D` snowball.
2. **W3-B high-prod frontier reinforcement scoring**: in `reinforce_value`, add 4P-only `1.12-1.18` value bump for prod>=3 own frontier planets within ~28 distance of enemies. Risk low/medium; value-only until sims prove it.
3. **W3-C staged opening/recapture design only**: implement later as close prod-2+ assured capture + hold/reinforce + follow-up. Do not use broad big-first or static race filtering.

Verification evidence recorded in Worker 3 scratch:
- `python3 -m py_compile .omx/research/team_strategy/worker3_frontier_survival_candidate.py .omx/research/team_strategy/worker3_static_race_rejected_candidate.py` passed.
- Current public 902 control and static-race rejected sim commands/results recorded with exact `rtk run ... uv run ... gui_evolve.py` invocations.
- Current and candidate 2P control seeds 100-101 both returned `latest_wins=1 old_wins=1` with totals `[[0,4528],[7011,6148]]`, confirming the 4P-scoped scaffold did not explain the known 2P hard loss.

## Candidate strategy backlog
| id | idea | evidence | risk | validation |
|---|---|---|---|---|
| W1-A | 4P assured staged opening for close/edge prod-2 neutrals, with home guard and immediate reinforcement/hold check. | Replay 75526435: seat 1 waited to step 15, sent 38 from p5, captured p9 at step 20, reinforced, then used p9 to flip p20 at step 34. | Medium/high: broad big-first already regressed; must be narrow and map/seat gated. | old-base hard seeds 100-107 / 555-560 / 1212-1217 / 9999-10002 plus public FFA 900-905; reject zero-prod eliminations. |
| W1-B | Fresh frontier recapture/denial boost for enemy prod-2/prod-3 planets captured recently with low garrison. | Replay 75526435: p20/p23 stolen at step 34; p11/p8 stolen at step 69; seat 1 later won by taking p6/p7/p22/p4 high-prod anchors. | Medium: can over-chase if applied to low-prod bait or long travel. | Same guardrails as W1-A, plus inspect events CSV for increased low-value churn. |
| W3-A | 4P production-leader pressure: track `_dominant_enemy` and mildly boost attacks on the production leader when step>=55 and enemy prod gap>=6. | Public-hard 900-901 eliminated_or_zero=5/8; seed902 current lost `267/prod8/captures29/first_capture21` vs `submissionB 1352/prod27/captures52/first_capture9`; replay 75526435 shows production leader snowball to prod48/32 planets. | Medium: can distract from weakest-enemy cleanup; keep 4P-only and gap-gated. | Public-hard FFA 900-905 seat-rotated; old-base hard 555-560 / 1212-1217 / 9999-10002; 2P 100-107 should be unchanged. |
| W3-B | 4P high-production frontier reinforcement scoring for own prod>=3 frontier planets near enemies. | Replay 75526435 p20/p23 and p11/p8 frontier flips; public-hard top2 hides zero/near-zero eliminations. | Low/medium: too much reinforcement can turtle; use value-only multiplier first. | Events CSV should show fewer high-prod flips and better final prod/planets without lower first-place. |
| W3-C | Staged opening/recapture design: close prod-2+ assured capture with hold/reinforce and follow-up only when hold-positive. | Seat1 in episode 75526435 used p5->p9 then p9->p20; current public902 first capture lags 21 vs 9. | High: broad big-first and static-race filtering already failed. | Must pass old-base 100-107 and public 900-905; reject any zero-prod opening collapse. |


## Rejected / dangerous approaches
| idea | why rejected / evidence |
|---|---|
| Blindly merge early same-source same-target launches into one larger launch | Local sims regressed hard versus previous best; it loses useful timing/route behavior in 1v1 and some 4P seats. Keep rejected unless narrowed by map/seat predicate and revalidated. |
| Broad `big-first` opening variants | `patch_big_first_vs_before_1v1`, `patch_big_first_limited_vs_before_1v1`, and `patch_big_first_4p_only_vs_before_1v1` all lost 1-3 on seeds 100-103; seeds 102/103 produced zero-prod eliminations against the previous best. |
| Global close/low-value capture loosening | `patch_low_close_vs_before_1v1` and swapped run were only 4-4 and had repeated zero-prod eliminations: unswapped seeds 104/105/106 and swapped seeds 102/105/106. Use only as a narrow recovery/recapture idea if it survives seat-swapped validation. |
| Rank/top2-only 4P validation | Public-hard seeds 900-901 gave top2=7/8 but eliminated_or_zero=5/8. A candidate can look rank-safe while dying to `submissionB`/`submissionD` snowballs. |
| Old-base-only or 220-turn-only acceptance | Deterministic old-base hard seeds still show zero/near-zero collapses (556, 557, 10001, 10002, 1213, 1217), and public-hard FFA first_place was only 2/8. Require public-hard and long/seat-rotated guardrails. |
| Trusting pre-seeding-fix GUI evidence | Older GUI labels used displayed seeds without seeding Orbit Wars module RNG. Use post-fix artifacts or rerun before accepting/rejecting a strategy. |
| Static-neutral opening race filter as tested by Worker 3 | It directly targeted public902 p14 race but worsened the control from current total `267` to `0` while `submissionB` rose to `3238`; keep only as evidence that opening fixes need hold-positive staged capture, not broad static filtering. |

## Leader integration notes
- Worker 3 recommends testing W3-A + mild W3-B first as a 4P-scoped candidate; do not integrate the static-race rejected candidate.
- Any integrated candidate should be accepted only if public-hard survival improves, not merely top2 rank.

## Leader implementation note: W1-B v2 promoted (2026-04-27)
Implemented narrow 4P fresh enemy-capture recapture/denial in `main.py` and mirrored to `submission.py` from `.omx/research/team_strategy/w1b_fresh_recapture_candidate_v2.py`.

Evidence before promotion:
- Public FFA 900-905 vs `submissionB/D/E`: wins unchanged `2/6`, eliminated/zero-prod unchanged `3/6`; seed902 improved from `1552 total / +180 vs best opponent / prod48 / 15 planets` to `3057 / +1389 / prod52 / 18 planets`; avg total `797.5 -> 1048.3`.
- Public FFA 906-911: wins unchanged `5/6`, eliminated unchanged `0/6`; seed909 rank improved `4 -> 3`, total `2838 -> 3372`, prod `18 -> 24`; avg rank `1.50 -> 1.33`, avg total `3365.5 -> 3488.0`.
- Combined public 900-911: wins unchanged `7/12`, but avg rank improved `2.17 -> 2.08`; zero-total/zero-prod unchanged.
- 2P hard old-base 555-560 unchanged vs current log: wins `4/6`, zero-total `2/6`, exact totals `[[3017,0],[0,2722],[0,3673],[6774,6341],[3425,1189],[8057,5117]]`.

Verification after promotion:
- `python3 -m py_compile main.py submission.py test_agent.py` passed.
- `python3 test_agent.py` passed 7 tests.
- `cmp -s main.py submission.py` passed.
- `./scripts/smoke_submission.sh` passed: `VALIDATION_SMOKE [DONE, DONE] [1, 1] 80`.

Caution: rating does not use win margin, so this is a conservative rank/survival improvement, not a proven ladder-win jump. Keep watching public-hard eliminate cases 900/901/904.

## External strategy research synthesis (2026-04-27)

Research artifact: `.omx/research/external_strategy_research.md`.

Borrowed from Planet Wars / Galcon / Risk / Diplomacy / game-AI papers:
- EXT-A hold-positive assured opening: close prod>=2 target, capture with small surplus, keep/reinforce until enemy ETA window passes.
- EXT-B cluster-anchor scoring: value static prod>=3 planets that control a local production cluster, then stage rear ships there.
- EXT-C line-commitment penalty: punish long/cross-map launches unless endgame or high-value fresh recapture.
- EXT-D 4P balance-of-power: pressure production leader after step ~60 if leader is snowballing; avoid weakest-chase too early.
- EXT-E per-planet bid allocator: owned planets bid for defense/reinforce/capture/reserve so frontier needs are not starved.
- EXT-F tacking/decoy pressure: only later, because it is high-risk unless opponents overreact.

Priority: implement EXT-A + EXT-C first as narrow candidates; validate public FFA 900-905 and 1v1 hard 555-560 before touching submit file.

## Orbit Wars Lab integration (2026-04-27)

Research artifact: `.omx/research/orbit_wars_lab_notes.md`.

External lab used: <https://github.com/automatylicza/orbit-wars-lab>.

Important engine note from Kaggle discussion #694310: old mirror-symmetry 4P maps were timing-asymmetric for orbiting planets; the lab pins Kaggle env commit `27660ecca553b96b1e3d6b282ca855fd1f274847`, which includes the 4P rotational-symmetry fix. Use this lab as the stronger official-engine validation layer.

Local agent id in lab: `mine/current-w1b-counter`.

New commands:

```bash
rtk run './scripts/setup_orbit_wars_lab.sh'
rtk run './scripts/run_lab_gauntlet.sh 1 777'
rtk run 'FORMAT=4p ./scripts/run_lab_gauntlet.sh 1 888'
```

Observed:

- 2P one-game public gauntlet (`runs/2026-04-27-010`): current 5W / 2L; losses to `tamrazov-starwars` and `yuriygreben-architect`.
- 2P two-game public gauntlet (`runs/2026-04-27-008`): current 10W / 4L; losses both samples to `pilkwang-structured` and `tamrazov-starwars`.
- 4P top-public sample (`runs/2026-04-27-011`): Pilkwang won; scores `[11855, 12413, 10951, 6291]`, current close second by score.

Strategy implication: next evolution should optimize against strong public agents, not only `old_base_v2`. Highest-value targets are anti-Tamrazov/Yuriy 2P stability and anti-Pilkwang 4P production-cluster hold.

## Promoted: hostile-only cluster target boost (2026-04-27)

Candidate path: `.omx/research/candidates/cluster_hostile_candidate.py`.
Promoted to: `main.py` and `submit_current_w1b_counter.py`.

Reasoning:
- Full cluster-hold reserve/reinforce over-turtled: local evolve `cluster_hold_v2_vs_current_s555` only 2-2 and seed100 1-3; lab 4P collapsed to `[0,3572,0,0]`.
- Soft cluster target for neutrals+hostiles improved one focused sample but hurt aggregate score slightly.
- Hostile-only cluster boost kept neutral expansion intact while slightly improving pressure against enemy prod>=3 frontier/cluster anchors.

Deterministic Orbit Wars Lab evidence after seeding fix:
- selected public core, 2 games each (`runs/2026-04-27-030`) vs current baseline (`runs/2026-04-27-029`): both 4W/2L, candidate score sum `17144` vs current `16920`.
- full public gauntlet, 1 game each (`runs/2026-04-27-031`) vs current baseline (`runs/2026-04-27-021`): both 5W/2L, candidate score sum `21665` vs current `20930`.
- 4P sample (`runs/2026-04-27-032`) unchanged vs current (`runs/2026-04-27-027`): both eliminated, Pilkwang wins.

Validation:
- `python3 -m py_compile main.py submit_current_w1b_counter.py` passed.
- `./scripts/smoke_submission.sh 4242 80` passed.

Next strategy board:
- Need a separate 4P anti-Pilkwang layer; hostile-only cluster boost does not solve 4P collapse.
- Keep broad cluster reserve/reinforce rejected unless redesigned with much stricter trigger.

## 2026-04-27 Pattern scout → counter candidate v1

Source: latest local Kaggle replay scout for submission `52093209` using `scripts/scout_kaggle_patterns.py`.
Report: `.omx/research/opponent_profiles/52093209.pattern_scout.md`.

Key opponent patterns found:
- `hostile_rusher`: early fleets into owned planets / home-raids.
- `long_crossmap`: long ETA cross-map attacks that punish empty fronts.
- `nearby_expander`: wins nearby neutral race first, then attacks exposed ring anchors.
- `heavy_neutral_chaser`: spends into high-ship neutrals; counter is to avoid matching the spend and snipe/recapture after capture.
- `scarce_4p`: low planet count 4P maps punish far wars; compact high-production ring is safer.

Implemented candidate: `submit_pattern_counter_v1.py` plus working copy `.omx/research/candidates/pattern_counter_v1.py`.
Runtime note: Kaggle agent cannot see opponent name/submission id in match, so implementation uses only visible behavior detectors from fleet targets/ETAs.

Validation:
- `python3 -m py_compile .omx/research/candidates/pattern_counter_v1.py scripts/scout_kaggle_patterns.py`
- `SUBMIT_FILE=submit_pattern_counter_v1.py ./scripts/smoke_submission.sh 4242 80` → `VALIDATION_SMOKE [DONE, DONE] [1, 1] 80`
- 4P Lab seed 888, 4 games: pattern v1 `1W/3L`, current `1W/3L`; pattern v1 improved some survival/score but same wins.
- 4P Lab seed 902, 4 games: pattern v1 `1W/3L`, current `0W/4L`.
- 2P selected seed 777: pattern v1 `2W/1L`, current `2W/1L`; matchup flips (better vs Pilkwang, worse vs Tamrazov), so not a universal promotion yet.

Submit command if choosing this anti-pattern candidate:

```bash
rtk run kaggle competitions submit -c orbit-wars -f submit_pattern_counter_v1.py -m pattern scout counters v1
```

## 2026-04-27 1v1 loss scout for submission 52093209

Refreshed scout with:

```bash
rtk run ./scripts/scout_kaggle_patterns.py 52093209 --fetch-count 20 --limit 80
```

Main output: `.omx/research/opponent_profiles/52093209.pattern_scout.md` now includes a **1v1 losses to study first** table.
Detailed 1v1 forensic reports were generated under `.omx/research/opponent_profiles/1v1_losses/`.

Current important 1v1 losses:
- `75539203` vs `EMR_DATA`: us `0`, opp `3284`; prod90 `16/23`; high-prod losses `14/7`; notes `high_prod_anchor_bleed`, `opponent_recaptures_more`, `opponent_higher_opening_tempo`.
- `75538740` vs `bgsszsa`: us `0`, opp `2661`; prod90 `3/3`; high-prod losses `24/18`; notes `high_prod_anchor_bleed`, `opponent_recaptures_more`, `long_crossmap_pressure`.
- `75537684` vs `datnt114`: us `0`, opp `4155`; prod90 `44/36`; high-prod losses `25/16`; notes `high_prod_anchor_bleed`, `opponent_recaptures_more`, `long_crossmap_pressure`.

Key 1v1 diagnosis:
- The common loss is not just opening expansion; it is midgame anchor bleeding. We capture/contest prod-3+ planets, then opponent recaptures more often and keeps the high-prod chain.
- Winners often launch far more total ships after midgame (`EMR_DATA` 7110 vs our 3087; `datnt114` 13360 vs our 6522 in the replay forensics), with many own/recapture/defense moves.
- Directly hard-blocking far low-prod opening lanes was tested and rejected: it made the Lab selected 1v1 smoke go `0W/3L` (bad). Do not promote that filter.

Candidate created:
- `.omx/research/candidates/onevone_fresh_counter_v1.py`
- `submit_1v1_fresh_counter.py`

Candidate idea: enable the W1-B fresh recapture system in 1v1 for prod-3+ enemy anchors with larger allowed garrisons. This is safe syntactically and validation-smoke passes, but Lab 2p win-rate was the same as current (`11W/3L` on 14-game gauntlet seed 777), so it is not clearly better yet.

Validation:
- `python3 -m py_compile .omx/research/candidates/onevone_fresh_counter_v1.py`
- `SUBMIT_FILE=submit_1v1_fresh_counter.py ./scripts/smoke_submission.sh 4242 80` → `VALIDATION_SMOKE [DONE, DONE] [1, 1] 80`

Submit command if intentionally testing this 1v1-focused candidate:

```bash
rtk run kaggle competitions submit -c orbit-wars -f submit_1v1_fresh_counter.py -m 1v1 fresh anchor recapture v1
```

## 2026-04-27 1v1 lab check — tournament 2026-04-27-061/062

- UI run `2026-04-27-061` showed `mine/current-w1b-counter` 0W/3L, but that run has no replay files and the lab does not snapshot the exact agent hash, so it is not safe to diagnose from that run alone.
- Re-ran same selected 1v1 seed_base=777 with current hash `3ac7a642...` and `SAVE_REPLAYS=1` as run `2026-04-27-062`: current is 2W/1L.
  - Beat `external/pilkwang-structured`: 2734-0, turns 215.
  - Lost to `external/tamrazov-starwars`: 0-4040, turns 287.
  - Beat `external/yuriygreben-architect`: 2427-0, turns 180.
- Loss pattern vs tamrazov: symmetric opening until about step 110, then enemy wins high-prod recapture tempo. Critical example: our p0 captures p6(prod3) at step 116 with 1 ship, enemy steals it at step 117; enemy later steals p8(prod5) at step 140 with a 149-ship near counter. Launch tempo by final: us 479 launches / 13,173 ships, tamrazov 863 launches / 20,835 ships.
- Tried `submit_1v1_anchor_hold_v1.py` candidate (future high-prod anchor-hold support). Validation passed, but selected run `2026-04-27-064` still 2W/1L and tamrazov loss worsened/unchanged enough (0-4220), so do not promote.
- Current lab agent was reset to `submit_current_w1b_counter.py` after testing.

Next 1v1 idea: not just defensive anchor hold; need a tamrazov-style midgame tempo release / chain-attack mode once both sides are symmetric and high-prod anchors are contested. Candidate should increase useful launch frequency from multiple front planets without over-reserving, while preserving current wins vs pilkwang/yuriy.

## 2026-04-27 v6 strategy experiment: Production Denial + Tempo Flood

Base: `submit_mirror_tempo_4p_focus_v5.py`.

Implemented candidates:
- `submit_denial_flood_v6.py`: Production Denial + Tempo Flood.
- `submit_tempo_flood_v6a.py`: Tempo Flood only, denial disabled.
- `submit_denial_only_v6b.py`: Denial only branch for isolation.

Results:
- v5 baseline selected 1v1 run `2026-04-27-072`: 6W/0L.
- v5 4P seed902 run `2026-04-27-073`: 2W/2L.
- v5 4P seed888 run `2026-04-27-074`: 2W/2L.
- v6 combined selected 1v1 run `2026-04-27-075`: 4W/2L. Reject: denial caused tempo loss versus tamrazov/yuriy.
- v6a tempo-only selected 1v1 run `2026-04-27-076`: 6W/0L, no regression but no measurable improvement in this guard.
- v6 combined 4P seed902 run `2026-04-27-077`: 2W/2L, no win-rate gain and lower scores in wins than v5.

Decision:
- Do not submit v6 combined.
- Production Denial as raw partial-pressure mission is harmful: it spends ships damaging prod anchors when a normal capture/hold mission would preserve tempo better.
- Safe current submit remains `submit_mirror_tempo_4p_focus_v5.py`.
- Tempo Flood implementation is safe in selected 1v1 but likely too conservative/rare; keep as candidate only until it shows 4P/opening improvement.

## 2026-04-27 Episode 75539733 hotaq mirror study

Source replay: `https://www.kaggleusercontent.com/episodes/75539733.json` (downloaded to `.omx/research/episodes/75539733.json`).
Generated forensic report: `.omx/research/analysis/episodes/75539733_detail.md` plus launch/change CSVs.

What happened:
- 2-player mirror match, team names both `hotaq`, rewards `[1, -1]`, final observed at step 269: p0 `2361 / prod52 / 36 planets`, p1 `0 / prod0 / 0 planets`.
- Game is almost perfectly symmetric until about step 150: both are `939 / prod20 / 12 planets`.
- First meaningful edge appears by step 176: p0 `1379 / prod22 / 14 planets`, p1 `1335 / prod20 / 12 planets`.
- Decisive break is step 176-201: p0 jumps to `1163 / prod34 / 20 planets`, p1 falls to `768 / prod9 / 7 planets`.
- After that it is cleanup: step 226 p0 `prod41 / 26 planets`, p1 `prod3 / 2 planets`.

Launch tempo evidence:
- Full game launch volume: p0 `576` actions / `10377` ships, p1 `235` actions / `4401` ships.
- The important window is not opening, it is midgame tempo release:
  - steps 175-199: p0 `91` actions / `2437` ships, p1 `83` / `2112`.
  - steps 200-224: p0 `139` actions / `2162` ships, p1 `41` / `771`.
  - steps 225-249: p0 `152` actions / `2717` ships, p1 `2` / `68`.
- p0 sends many actions to own planets too (`steps 200-224`: 77 own-target launches), so winner is doing reinforcement/chain-forwarding/recapture swarm, not just one-shot attack.

Key tactical pattern:
- Both sides repeatedly trade prod3 anchors p5/p6. p0 wins after it can keep recapturing while also expanding/forwarding.
- High-production losses are common on both sides: p0 lost prod>=3 anchors 8 times, p1 lost them 10 times. So this replay does **not** support conservative “send huge margin and turtle” as the main fix.
- The winning pattern is: symmetric expansion -> small prod/planet edge -> immediate multi-source fleet spam -> reinforce own fresh captures while attacking enemy/neutral nearby -> opponent launch volume collapses.

Implication for our bot:
- v7 high-prod hold/opening experiments were the wrong direction when they slow tempo. Empirical runs confirmed that:
  - `submit_4p_hold_opening_v7.py` seed902: `0W/4L`.
  - `submit_4p_opening_v7a.py` seed902: `1W/3L`.
  - `submit_4p_hold_lite_v7c.py` seed902: `1W/3L`.
  - visible-counter hold v7d/e/f seed902: `2W/2L`, but flips one v5 loss into a win while regressing one v5 win, so not promotion-worthy yet.
- Next promising direction: build a **midgame tempo release / chain-forward mode** after we own ~10+ planets or prod>=18, especially for 1v1 and 4P frontier clusters:
  1. when a capture succeeds, immediately schedule follow-up from nearby own planets within ETA<=8 to either reinforce the fresh capture or retake the closest prod2/3 enemy anchor;
  2. allow multiple small/medium actions from the same front per turn if source remains above guard;
  3. score own fresh captures as valid targets for reinforcement if visible enemy fleets can arrive soon;
  4. avoid global margin/turtle changes unless there is visible incoming enemy mass.

Decision now:
- Keep current safe submission recommendation as `submit_mirror_tempo_4p_focus_v5.py`.
- Do not promote v7f: seed902 result `2W/2L` (`[10623,8838,2367,5453]` win, `[0,0,2267,0]` loss, `[4508,7857,1924,3581]` loss, `[5194,90,1970,567]` win). It helps seed293 but still loses seed694 and regresses seed681 compared with v5.
