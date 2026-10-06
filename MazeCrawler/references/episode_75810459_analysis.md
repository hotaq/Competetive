# Episode 75810459 Analysis

Source: `episodes/75810459.json`

Original URL was provided by the user from Kaggleusercontent.

## Metadata

- Episode ID: `75810459`
- Environment: `crawl`
- Version: `1.1.0`
- Module version: `1.29.0`
- Seed: `1846759712`
- Teams: `hotaq` vs `hotaq`
- Rewards: `[0.5, 0.5]`
- Statuses: `DONE`, `DONE`
- Recorded steps: `13`

## What Happened

The replay shows a symmetrical no-progress game:

- Both agents have empty action dicts (`{}`) throughout the recorded episode.
- Player 0 factory starts at `(5, 2)` with energy `1000`.
- Player 1 factory starts at `(14, 2)` with energy `1000`.
- Both factories stay on their starting cells.
- Energy drops by 1 each step while they idle.
- The southern boundary advances from `0` to `1` at step 4, to `2` at step 8, and to `3` at step 12.
- At step 12, both factories are below the southern boundary and are removed.
- Because both factories are eliminated simultaneously, the episode ends as a draw with reward `0.5` for both.

## Starting Position Detail

For player 0, the starting factory cell `(5, 2)` has wall bitfield `3`, meaning:

- North wall: blocked
- East wall: blocked
- South wall: open
- West wall: open

This is useful because a safe fallback movement should be able to choose `WEST` rather than idling forever.

## Implications

1. A Kaggle validation episode can complete even if a bot makes no strategic progress.
2. Local validation must check progress, not just whether the environment returns `DONE`.
3. Useful harness checks:
   - Did the factory move above its start row?
   - Did the agent return non-empty actions?
   - Did either factory die to the first scroll boundary?
   - Did a self-play game draw only because both bots idled?
4. For `v1`, the same public starting observation makes the current local `main.py` return `WEST` for player 0, so this replay should not be treated as proof that current `v1` always idles. It may be an idle/empty submission, an action-redacted replay, or a replay not matched to the latest local code.

## Next Harness Requirement

When we build `scripts/evaluate.py`, include a `progress_score` in addition to win/loss:

- max factory row reached
- steps survived
- number of non-empty action turns
- number of build actions
- number of movement actions
- simultaneous early-idle draw flag
