# 1100+ LB | Maze Crawler

Source: https://www.kaggle.com/code/lakhindarpal/1100-lb-maze-crawler

Pulled with:

```bash
python3 -m kaggle.cli kernels pull lakhindarpal/1100-lb-maze-crawler -p competetive/MazeCrawler/public_notebooks/lakhindarpal_1100_lb -m
```

Live page metadata checked on 2026-05-03:

- Public score: `1121.9`
- Best score: `1121.9`, version 4
- Runtime: 34s
- Votes: 5
- Copies: 2
- Comments: 0
- License: Apache 2.0

The notebook writes a single `submission.py` agent and includes a local `make("crawl", debug=True)` smoke run against `random`. The local notebook was read but not executed.

## Strategy Shape

- Factory-centric convoy policy.
- Strict action order: factory first, then worker, then other units.
- Collision avoidance through a `reserved` destination set.
- Factory tries normal BFS north up to depth 20.
- If normal BFS cannot find a route, factory uses a jump-aware BFS that models jump cooldown.
- Factory emergency jumps north when the southern bound is close.
- Factory transfers energy to an adjacent low-energy worker.
- Worker either clears a north wall, follows/helps factory escape, or targets nearby crystals.

## Important Techniques

1. `bfs_jump_step` is the main new idea compared with the lightningv08 baseline. It searches normal moves and jump moves in one state space, with jump cooldown included in the BFS state.
2. The factory target is not merely next row; it targets up to `row + 20`, across any column. This encourages long-horizon escape instead of local greedy movement.
3. Factory only builds a worker if no movement or jump path was found, no worker exists, and the north spawn cell is free.
4. Worker gets an explicit rescue/convoy role: when `gap <= 4`, target ahead of the factory instead of crystals.
5. Factory can transfer to adjacent worker when worker energy is `<= 150`.

## Risks / Caveats

- It assumes object-style observations (`obs.southBound`, `obs.robots`, etc.), so it needs adaptation for our replay/dict-safe wrapper.
- `can_jump` checks bounds and full-wall landing, but does not require the landing cell to have an open exit. The lightningv08 safety guard is stronger here.
- `can_jump` does not inspect walls between start and landing, which may be intended by game rules, but should be verified in the official env.
- Worker wall remove uses fixed `energy >= 100` rather than `config.wallRemoveCost`.
- Factory build reserve is weaker than lightningv08: `energy >= config.workerCost` without extra reserve.
- The notebook installs `kaggle-environments` in the first cell; the pulled code was not run locally.

## Transfer To Our Bot

Use this as the v3 layer after the v2 BFS-worker patch:

- Keep our `ObsView` / dict-safe compatibility.
- Import lightningv08's landing-exit guard for jumps.
- Add this notebook's `bfs_jump_step` for factory escape after normal BFS fails.
- Add adjacent worker transfer.
- Add worker convoy mode when lava gap is small.
- Keep worker count low; one worker is enough for the first strong baseline.

Recommended implementation order:

1. Patch v2 with cooldown `<= 1`, normal BFS north, one-worker policy, and safer jump landing.
2. Patch v3 with jump-aware BFS and worker energy transfer.
3. Submit v2 first if it passes replay smoke, then submit v3 only after v2's Kaggle score starts moving.
