# Harvest report — tape_pool_20260922_top10

Date: 2026-09-22. Refresh cadence: every 2-3 days (deadline 2026-09-30).
Method: `scripts/harvest_top10_pool.py` (same pattern as
`scripts/harvest_tape_pool.py`: one replay download per episode, both seats kept
as compact tapes, dedup by action sha256; tape JSON schema identical to
`artifacts/tape_pool_20260919`).

## Leaderboard snapshot (`kaggle competitions leaderboard kaggriculture -s`, 2026-09-22)

| rank | team | score | submission date (UTC) | tapes |
| --- | --- | --- | --- | --- |
| 1 | DSM | 3113.7 | 2026-09-22 05:27:49 | 6 |
| 2 | Vadim Vasilenko | 3102.7 | 2026-09-21 21:16:54 | 9 |
| 3 | Majkel1337 | 3070.6 | 2026-09-21 09:39:46 | 10 |
| 4 | THIRD FARM CLUB | 3052.7 | 2026-09-22 02:00:05 | 7 |
| 5 | Unknown Mother-Goose | 3052.5 | 2026-09-22 00:34:39 | 7 |
| 6 | M & M & P & Q | 3046.3 | 2026-09-21 18:20:57 | 6 |
| 7 | Kaggledew Valley 🏆 | 3038.8 | 2026-09-22 02:21:08 | 7 |
| 8 | KawattaTaido | 2991.4 | 2026-09-21 05:13:15 | 6 |
| 9 | Yannik Schiffner | 2983.0 | 2026-09-22 04:54:21 | 7 |
| 10 | Otter Vibe | 2978.4 | 2026-09-19 08:48:39 | 9 |

Leaderboard shifts daily; re-check before the next refresh.

## Pool contents

- 92 unique tapes from 46 distinct episodes, 6 most-recent public episodes per
  top-10 team (deduped across teams that played each other, e.g. DSM vs Vadim
  Vasilenko ep111940707).
- Every top-10 team has >= 6 tapes. Success criterion (>= 1 tape for most of
  top-10) met 10/10.
- Bonus opponent tapes (both seats harvested): mtmr_s1 x5, 吃白饭的大肥鱼 x3,
  DECEM x3, QQ x2, Azat Akhtyamov / SpaTaro / THUNDER THUNDER / Densike /
  Orbital Terraformer x1 each. Per memory note, these opponent tapes are the
  valuable ones against RL/closed-loop top teams; `wanted_by` in manifest.json
  records which top-10 list each episode came from, and `own_tape_for` marks
  tapes that are the listed team's own seat (empty = opponent tape).

## Teams with no retrievable replays

None. All 10 teams yielded public completed episodes.

## Sanitize check

Every tape: `length == 720` and every action's keys exactly
`{farmer, hands, market}` (fail-closed executor requirement). 0 tapes rejected,
0 dropped downloads, 0 "zero waste" (own-team) seats encountered.
`manifest.json` (`dropped`, `teams_with_no_replays`, `own_team_seats_skipped`)
logs all three categories explicitly.

## Provenance

`manifest.json`: per-tape `team_name`, `episode`, `seat`, `date` (episode
`endTime`), `seed`, `final_money`/`opponent_money`, `rank`/`score`/`submission`
via top-level `leaderboard` + per-tape `wanted_by`; `episodes` lists all 46
episode ids. Validation: every listed file exists, parses as JSON, actions
length 720, tape sha256 matches manifest entry.

## Cost note

46 replay downloads (~32 MB each, discarded after reduction); no overlap with
the 20260919 pool (all 46 episodes are new — leaderboard moved on), so nothing
was copied from the old pool.
