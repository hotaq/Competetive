# Worker + BFS Baseline Summary

Source: https://www.kaggle.com/code/lightningv08/worker-bfs-baseline-lb-1100-maze-crawler

Pulled with Kaggle CLI on 2026-05-03.

## Metadata

- Author: `lightningv08` / Vladislav Lavrentev
- Kaggle title: `Worker + BFS Baseline | LB 1100+ | Maze Crawler`
- Browser-visible public/best score at capture: `726.9` on version 2
- Notebook votes at capture: `16`
- Runtime shown: `20s`
- License shown: Apache 2.0
- Kaggle CLI metadata: public Python notebook for competition `maze-crawler`

Note: the notebook title claims `LB 1100+`, but the live page sidebar showed `726.9` when captured. Treat `726.9` as the verified live page score for this read.

## Strategy

This is a compact rule-based baseline centered on one worker plus BFS pathfinding.

Core ideas:

- Keep the factory alive and moving north.
- Build at most one worker, and only after movement choices fail or are less urgent.
- Use BFS over discovered wall cells to choose the first step toward any cell in the next row.
- Prefer normal movement before jumping.
- Jump north only if the landing cell is inside bounds, unoccupied/unreserved, and has at least one exit.
- Worker removes north walls when blocked and has enough energy.
- Worker/other units can detour toward visible crystals when not at max energy.
- Use action reservations to reduce friendly collisions.

## Implementation Details Worth Reusing

1. Cooldown checks use `<= 1`, not exactly `0`.

   The game processing order ticks cooldowns before action validation. If observations expose cooldown `1`, an action may become legal after the tick. Our local bot should use `<= 1` for move/build/jump readiness unless testing proves otherwise.

2. Factory route choice uses BFS to the next row.

   It builds target cells as every column on `row + 1`, then runs BFS with a small max depth to find the first legal step around walls. This is better than only checking immediate north/lateral moves.

3. Jump safety checks the landing cell's exits.

   It does not jump into a cell just because it is in-bounds; it checks that the target cell has at least one legal exit. This reduces sealed-cell traps.

4. Worker build timing is conservative.

   The notebook builds one worker only if:

   - factory build cooldown is nearly ready,
   - gap from south boundary is at least 2,
   - north spawn is passable and free,
   - there is no worker yet,
   - factory has worker cost plus extra reserve.

5. Collision reservation is simple but useful.

   Each selected action reserves the destination, and `safe_dirs` rejects targets reserved by earlier units. It also allows following only if the occupant has already selected a non-idle action.

## Differences From Our Current v1b

- Our `v1b` jumps north aggressively when north is blocked or near the boundary; this notebook is more conservative and checks for landing exits.
- Our current movement is score-based local movement; the notebook adds BFS to route around discovered walls.
- Our cooldown checks are mostly `== 0`; the notebook uses `<= 1`, likely important because cooldown ticks before validation.
- Our worker/scout build policy is more mixed; the notebook keeps one worker and avoids scout spam.

## Recommended Next Patch

Make `v2-bfs-worker` by adapting these ideas into our `main.py`:

1. Change move/build/jump readiness to `<= 1`.
2. Add BFS first-step toward the next row.
3. Replace aggressive jump with `can_jump_to_open_cell`.
4. Build exactly one worker with energy reserve.
5. Keep our fallback observation wrapper from `v1b` because replay states can omit boundary fields.
