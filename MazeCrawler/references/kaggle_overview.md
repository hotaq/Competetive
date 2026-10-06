# Kaggle Overview Notes

Source page: https://www.kaggle.com/competitions/maze-crawler/overview

Captured: 2026-05-03

## Competition Goal

Create and/or train AI bots for Maze Crawler, a novel 1v1 infinite-scrolling maze strategy game. The objective is to survive in a mirrored maze, keep the factory ahead of the advancing southern boundary, manage energy, and outmaneuver the opponent.

The match ends when:

- one factory is destroyed by the scrolling boundary,
- one factory is destroyed by direct collision with the enemy factory,
- or both survive until the time limit and the tiebreaker cascade applies.

## Evaluation System

- Teams can submit up to 5 agents per day.
- Each submission plays episodes against other ladder bots with similar skill rating.
- Ratings go up with wins, down with losses, and move closer together on ties.
- Kaggle tracks only the latest 2 submissions for final submissions, while the leaderboard displays the best scoring bot.
- Every bot keeps playing episodes until competition end; newer bots get more frequent episodes for faster feedback.
- Each submission has a Gaussian skill estimate `N(mu, sigma^2)`, where `mu` is estimated skill and `sigma` is uncertainty.
- New valid submissions start at `mu0 = 600` after passing validation.
- Validation is a self-play episode against copies of the submitted bot. If it fails, the submission is marked Error and logs can be downloaded.
- The margin or score by which a bot wins/loses an episode does not affect skill rating updates. Only win/loss/draw matters, adjusted by expected result and uncertainty.

## Timeline

- May 1, 2026: Start Date
- June 9, 2026: Entry Deadline
- June 9, 2026: Team Merger Deadline
- June 16, 2026: Final Submission Deadline
- July 17, 2026 to approximately July 30, 2026: games continue until leaderboard convergence, then leaderboard finalizes
- All deadlines are 11:59 PM UTC unless otherwise noted.

## Gameplay Details From Overview

- Maze window is 20x20 and scrolls north over time.
- Maze has east/west symmetry; players start on opposite halves.
- Factories are mobile, indestructible against non-factory robots, and can build or jump.
- Factory jumps always happen, ignore walls, consume cooldown, and can destroy the factory if the landing cell is off-board.
- Spawned robots appear north of the factory before movement/combat, so spawning into an occupied cell can trigger crush combat.
- Robots consume 1 energy per turn. A robot with 0 energy is forced idle.
- Workers spend 100 energy to build/remove walls, even if the action hits a fixed wall or no-op state.
- Fixed walls are the east/west outer perimeter and central mirror axis.
- Crystals have 6% density and are worth 10-50 energy, but are visible only while in vision.
- Mining nodes have 3% density and are not remembered after leaving vision.
- Mines generate 50 energy per turn up to 1000 and are remembered once discovered.
- Fog permanently remembers walls/layout and mines, but not crystals, enemy robots, or mining nodes.
- Scrolling begins once every 4 turns and ramps to every turn by step 400.
- Time limit is step 500.
- Tiebreaker cascade is total energy, then unit count, then true draw.

## Strategy Implications

1. Optimize for binary outcomes first. The rating system does not reward win margin directly.
2. Validation must be robust against self-play. Friendly collisions and deterministic mirrored movement can cause a bot to destroy itself in validation.
3. Keep at least two recent candidate submissions healthy because Kaggle tracks the latest 2 for final submissions.
4. Public leaderboard movement will be slow/noisy because the skill estimate depends on many episodes and uncertainty decay.
5. Local candidate comparison should report win rate, not just reward magnitude.
6. Because new submissions get more episodes, public feedback may initially overrepresent the newest bot.
7. Avoid risky factory jumps unless landing safety is known; one bad jump can immediately lose.
8. Avoid spawning into occupied friendly cells and same-type collisions, especially in self-play validation.
9. Energy economy matters for tiebreakers, but survival comes first.
