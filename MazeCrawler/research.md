# Maze Crawler Research

## Competition Snapshot

- Kaggle URL: https://www.kaggle.com/competitions/maze-crawler/overview
- Kaggle slug for CLI: `maze-crawler`
- Local environment name from starter docs: `crawl`
- Category: Playground
- Reward: Knowledge
- Deadline from Kaggle CLI: `2026-06-16 23:59:00`
- Joined status from Kaggle CLI: `userHasEntered=True`
- Team count at setup time: `39`
- Initial leaderboard top visible score at setup time: `1238.4`

## Local Workspace

- Root: `competetive/MazeCrawler/`
- Starter submission: `main.py`
- Downloaded Kaggle bundle: `data/maze-crawler.zip`
- Extracted reference files: `data/AGENTS.md`, `data/README.md`, `data/main.py`
- Working folders: `artifacts/`, `data/`, `docs/`, `notebooks/`, `public_notebooks/`, `references/`, `scripts/`, `src/`

## Game Summary

Maze Crawler is a two-player fog-of-war strategy game. Each side starts with a factory in a 20-wide maze that scrolls north over time. The factory builds robots, survives the scrolling boundary, and wins by outlasting the opponent or by tiebreaker at step 500.

Key mechanics:

- Map is east/west symmetric with occasional doors across the center.
- Fog remembers walls/layout and mines, but not crystals, enemy robots, or mining nodes.
- Factory can build every 10 turns and jump every 20 turns.
- Scouts cost 50, move every turn, and have vision 5.
- Workers cost 200, move every 2 turns, and can build/remove walls for 100 energy.
- Miners cost 300, move every 2 turns, and can transform on mining nodes into mines.
- Mines generate 50 energy per turn up to 1000.
- Every robot consumes 1 energy per turn.
- Combat has friendly fire. Crush order is Factory > Miner > Worker > Scout. Same-type collisions destroy all units of that type.
- Southern boundary starts advancing once every 4 turns and reaches every turn by step 400.
- Tiebreakers are total energy, then unit count, then draw.

## First Strategy Hypotheses

1. A starter worker-rush north policy is valid but likely weak because it ignores scouting, mining, and collision avoidance.
2. Early scouts should map walls/crystals faster than workers and may improve path choice before committing factory jumps.
3. Factory movement/jump safety is critical because off-board landing destroys the factory and scrolling loss is terminal.
4. Workers are useful for opening north walls but wasting 100 energy on fixed/no-op walls is expensive.
5. Mining has high upside if nodes are found early, but miners are slower and costlier than scouts; scout-first node discovery may be needed.
6. Because map halves are mirrored, enemy factory/route inference should be possible from our own explored layout.
7. Collision avoidance is mandatory because friendly fire can erase scouts/workers and newly spawned units can collide on the factory's north cell.

## Kaggle Discussion Findings

Source notes: `references/kaggle_discussion.md`

Captured on 2026-05-03 from `https://www.kaggle.com/competitions/maze-crawler/discussion?sort=votes`.

- Only three visible topics were present: a pinned official getting-started/Discord post, an opponent-allocation concern, and a medals/points discussion.
- No public strategy thread or code technique was visible yet.
- The opponent-allocation thread reports repeated pairing against the same opponent around 5-6 times. Treat public match outcomes as noisy until we can inspect episode/opponent IDs.
- Kaggle staff context confirms this is a for-fun Playground simulation competition, so public writeups may be sparse.
- Main operational takeaway: build a stronger local evaluation harness early, with fixed seeds, swapped sides, and opponent tracking after submissions.

## Kaggle Overview Findings

Source notes: `references/kaggle_overview.md`

Captured on 2026-05-03 from `https://www.kaggle.com/competitions/maze-crawler/overview`.

- Goal is to create/train Maze Crawler bots that survive a 1v1 infinite-scrolling maze and outlast the opponent's factory.
- Teams can submit up to 5 agents per day.
- Kaggle validates new submissions with a self-play episode before adding them to the ladder.
- Valid submissions start with rating `mu0 = 600`; skill is modeled as `N(mu, sigma^2)`.
- Rating updates depend on win/loss/draw, expected result, and uncertainty. The margin or score by which an episode is won does not affect skill updates directly.
- Newer bots get a higher episode rate for faster feedback.
- Kaggle tracks the latest 2 submissions for final submissions, while the leaderboard shows the best scoring bot.
- Final submission deadline is June 16, 2026 at 11:59 PM UTC. Games continue after that until roughly July 17-30, 2026 or leaderboard convergence.

Strategy impact:

- Optimize for win rate first, not reward margin.
- Self-play safety is mandatory because failed validation marks the submission Error.
- Local harness should include candidate-vs-candidate self-play and swapped player order, not just candidate-vs-random.
- Keep two recent stable submissions near the end; avoid leaving the latest 2 as broken experiments.
- Public score changes may lag true quality because the rating system needs many episodes to reduce uncertainty.

## Immediate Next Steps

1. Install or use a Kaggle environment that contains `kaggle_environments` so local games can run.
2. Build a deterministic simulator harness around `make("crawl", configuration={"randomSeed": seed})`.
3. Evaluate the starter `main.py` against `random` over many seeds to get a baseline.
4. Add safe wall navigation helpers: wall lookup, passability, unseen-cell preference, and center-axis fixed-wall handling.
5. Create `v1` strategy: scout-first exploration, factory north movement unless blocked, worker only when north-wall bottleneck exists.
6. Compare candidate agents with fixed seeds before submitting.
7. After first submission, download episodes/logs and record opponent IDs to detect repeated-pair bias.
8. Before every Kaggle upload, run self-play validation locally where possible to catch mirror-collision and spawn-collision failures.

## Submission History

| Local time | File | Message | Status | Public score | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-05-03 20:41 +07 | `main.py` | `v1 safe starter collision-aware` | `SubmissionStatus.COMPLETE` | `600.0` | First v1 safe-starter upload. Passed Kaggle validation; started at default rating. |

## Episode Replay Notes

Source notes: `references/episode_75810459_analysis.md`

- Episode `75810459` is `hotaq` vs `hotaq`, seed `1846759712`, with rewards `[0.5, 0.5]` and statuses `DONE`.
- The replay contains 13 recorded steps and shows both agents returning empty action dicts while factories remain at their starting cells.
- Both factories are removed when the southern boundary reaches row `3`, causing a simultaneous-elimination draw.
- Player 0's starting factory cell has north/east blocked but west/south open, so a progress-capable policy should be able to move laterally instead of idling.
- Harness implication: local evaluation must track progress metrics such as non-empty action turns, max factory row, steps survived, and early-idle draw flags. A completed validation episode alone is not enough evidence of a useful bot.
- Follow-up patch: local `v1b` now returns `JUMP_NORTH` for both player views on the same starting replay states instead of idling/erroring, and it tolerates replay/runtime observations that omit `southBound`/`northBound`.

## Public Notebook Findings

Source notes: `public_notebooks/lightningv08_worker_bfs_baseline/summary.md`

- Notebook: `Worker + BFS Baseline | LB 1100+ | Maze Crawler`
- URL: `https://www.kaggle.com/code/lightningv08/worker-bfs-baseline-lb-1100-maze-crawler`
- Pulled locally under `public_notebooks/lightningv08_worker_bfs_baseline/`
- The live page title claims `LB 1100+`, but the sidebar at capture showed public/best score `726.9` for version 2.
- Transferable strategy: one conservative worker, BFS pathfinding toward the next row, action reservations, worker wall removal, and crystal detours when units are below max energy.
- Important implementation signal: readiness checks use cooldown `<= 1` rather than `== 0`, matching the rules where cooldown tick happens before action validation.
- Important safety signal: `JUMP_NORTH` should check that the landing cell is in bounds, unoccupied/unreserved, and has at least one exit.

Source notes: `public_notebooks/lakhindarpal_1100_lb/summary.md`

- Notebook: `1100+ LB | Maze Crawler`
- URL: `https://www.kaggle.com/code/lakhindarpal/1100-lb-maze-crawler`
- Pulled locally under `public_notebooks/lakhindarpal_1100_lb/`
- Live page sidebar at capture showed public/best score `1121.9`, version 4, runtime 34s.
- This is a stronger reference than the lightningv08 page by live score, and it adds jump-aware factory BFS plus worker energy transfer.
- Transferable strategy: factory-first convoy, normal BFS to far-north targets, jump-aware BFS fallback, emergency jump when the lava gap is low, exactly one worker, worker wall removal, worker convoy mode near danger, and factory-to-worker energy transfer.
- Caveat: the code assumes object-style observations and needs our dict-safe `ObsView` adaptation. Its jump safety is weaker than the lightningv08 landing-exit guard, so we should combine both instead of copying blindly.

Recommended next implementation path:

1. `v2-bfs-worker`: take cooldown `<= 1`, normal BFS movement, one-worker policy, and safe jump landing.
2. `v3-jump-bfs-transfer`: add lakhindarpal's jump-aware BFS and adjacent worker energy transfer after v2 is stable.

## Useful Commands

```bash
python3 -m kaggle.cli competitions files -c maze-crawler
python3 -m kaggle.cli competitions download -c maze-crawler -p competetive/MazeCrawler/data
python3 -m kaggle.cli competitions leaderboard -c maze-crawler --show
python3 -m kaggle.cli competitions submissions -c maze-crawler
python3 -m kaggle.cli competitions submit -c maze-crawler -f competetive/MazeCrawler/main.py -m "starter worker rush"
```

Local smoke test once `kaggle_environments` is available:

```bash
python3 - <<'PY'
from kaggle_environments import make

env = make("crawl", configuration={"randomSeed": 42}, debug=True)
env.run(["main.py", "random"])
print([(i, state.reward, state.status) for i, state in enumerate(env.steps[-1])])
PY
```

## Verification Notes

- Kaggle CLI can list and download files for `maze-crawler`.
- Downloaded files are small starter assets only: `AGENTS.md`, `README.md`, `main.py`.
- Local Python currently does not have `kaggle_environments`, so no local game smoke test was run during setup.
