# Latest strategy result snapshot

Current bot: `new_current_counter` in `main.py` / `submit_current_w1b_counter.py`.

## Latest patch intent

Targeted recovery patch against `old_base_v2`:

- lower the map-control recovery trigger from late/severe only to turn >= 90 with enemy production gap >= 5 or enemy planet gap >= 2;
- allow recovery captures against production-1 planets when behind, instead of ignoring them and letting old strategy snowball map control;
- reduce reserve/turtle behavior under pressure (`RECOVERY_RESERVE_MULT = 0.35`) so planets release ships sooner;
- add a severe-behind branch that scales down exact keep (`RECOVERY_SEVERE_EXACT_KEEP_MULT = 0.72`) when enemy production gap >= 16 or enemy planet gap >= 5;
- raise recovery scoring (`RECOVERY_SCORE_MULT = 1.9`) so short capture/counter-capture missions outrank passive holding.
- add an opening guard: in 1v1 openings, skip underpowered partial launches into long neutral targets until a source can actually cover the projected capture need.

## Simulator fix

The local Kaggle Orbit Wars environment uses Python module-level random state for map generation, so `configuration={"seed": ...}` alone was not making GUI comparisons reproducible.

Patched deterministic seeding in:

- `.omx/research/sim/gui_evolve.py`
- `.omx/research/sim/gui_simulate.py`
- `.omx/research/sim/compare_old_new.py`

New reports named `evolve_patch_seeded_*` are reproducible seed batches. Older `evolve_live` / `evolve_patch_check` reports are useful for history, but their displayed seeds were not true deterministic map seeds.

## Verification snapshot

### Deterministic GUI compare, seeds 555-560, 220 turns

Artifact: `.omx/research/sim/gui/replays/evolve_patch_opening_guard_555_560.report.md`

```text
latest_wins=4 old_wins=2
totals=[[3017, 0], [0, 2722], [0, 3673], [6774, 6341], [3425, 1189], [8057, 5117]]
```

### Deterministic GUI compare, seeds 100-103, 220 turns

Artifact: `.omx/research/sim/gui/replays/evolve_patch_opening_guard_100_103.report.md`

```text
latest_wins=4 old_wins=0
totals=[[2565, 2033], [11182, 8495], [7575, 5106], [3923, 2932]]
```

### Deterministic GUI compare, seeds 1212-1217, 220 turns

Artifact: `.omx/research/sim/gui/replays/evolve_patch_opening_guard_1212_1217.report.md`

```text
latest_wins=4 old_wins=2
totals=[[4614, 0], [0, 3350], [5000, 3286], [3178, 0], [6147, 2513], [0, 2808]]
```

### Deterministic GUI compare, seeds 9999-10002, 220 turns

Artifact: `.omx/research/sim/gui/replays/evolve_patch_opening_guard_9999_10002.report.md`

```text
latest_wins=2 old_wins=2
totals=[[2334, 893], [7328, 6813], [6, 4989], [138, 1645]]
```

### Combined deterministic GUI evidence

Across the four deterministic GUI batches above:

```text
20 games: latest_wins=14, old_wins=6
```

This is an improvement over the stalled reference run `evolve_live` (`latest_wins=3 old_wins=4`) and over the recovery-only deterministic set (`16 games: latest_wins=10 old_wins=6`). It is still not stable enough to call solved: seeds 556, 557, 1213, 1217, 10001, and 10002 still show losses, including catastrophic near-eliminations.

### Submission smoke

Command:

```bash
rtk run './scripts/smoke_submission.sh'
```

Observed:

```text
VALIDATION_SMOKE ['DONE', 'DONE'] [1, 1] 80
```

### Syntax / mirror check

Commands:

```bash
rtk run 'python3 -m py_compile main.py submit_current_w1b_counter.py .omx/research/sim/gui_evolve.py .omx/research/sim/gui_simulate.py .omx/research/sim/compare_old_new.py'
rtk run 'cmp -s main.py submit_current_w1b_counter.py'
```

Both passed.

## Current risk / next target

Loss reports show the latest bot is stronger after avoiding some bad opening trickles, but still loses when the opponent gets a fast production cluster and our bot cannot stabilize frontier planets. Next targeted strategy should improve midgame defense/recapture priority on high-production frontier planets after turn 60.

## 4P FFA quick checks

Added a reusable 4-player FFA runner:

```bash
rtk run './scripts/run_ffa4.sh old3 700 2 220'
rtk run './scripts/run_ffa4.sh mixed 800 2 220'
rtk run './scripts/run_ffa4_gui.sh mixed 800 4 220'
```

### Latest vs three `old_base_v2` copies

Artifact: `.omx/research/sim/runs/20260427-032005_ffa4_old3_s700_n2.md`

```text
8 games, seat-rotated over seeds 700-701
current first_place=5/8, top2=6/8, eliminated=2/8
avg_rank=1.62, avg_total=2139.8, avg_production=18.5
```

### Latest vs mixed old/public strategies

Opponents: `old_base_v2`, `old_geometry_v4`, `distance_1100`.

Artifact: `.omx/research/sim/runs/20260427-032358_ffa4_mixed_s800_n2.md`

```text
8 games, seat-rotated over seeds 800-801
current first_place=2/8, top2=6/8, eliminated=2/8
avg_rank=2.00, avg_total=1879.2, avg_production=14.9
```

4P read: latest is competitive against three `old_base_v2` copies, but mixed FFA still favors `old_base_v2` on production snowball. Next 4P-specific target should improve early/midgame production race without changing the 1v1 opening guard.

## Pilkwang public benchmark study

Studied Kaggle notebook: <https://www.kaggle.com/code/pilkwang/benchmark-how-strong-is-your-orbit-wars-agent>

Local study note: `docs/BENCHMARK_STUDY.md`

Main takeaways applied locally:

- benchmark should rotate seats, not trust one replay;
- compare both 2P and 4P;
- read normalized rank/top2/elimination, not just first-place wins;
- seed both Python RNG and Orbit Wars module RNG for reproducibility;
- use the public benchmark pack as stronger opponents than only `old_base_v2`.

Downloaded public agent pack to `.omx/research/kaggle_bench/public_agents/` and added aliases `submissionA` through `submissionF`.

### Public-hard 4P quick signal

Public-hard trio used locally: `submissionB`, `submissionD`, `submissionE`.

Artifacts:

- `.omx/research/sim/runs/20260427-115222_ffa4_public_s900_n1.md`
- `.omx/research/sim/runs/20260427-115330_ffa4_public_s901_n1.md`

Combined read:

```text
8 seat-rotated games over seeds 900-901
current first_place=2/8
current top2=7/8
current eliminated_or_zero=5/8
```

Interpretation: latest bot is still submit-worthy vs old baselines, but public-hard 4P exposes survival weakness. Next improvement target should be 4P anti-collapse / production stabilization against `submissionB` and `submissionD` style openings.

## Orbit Wars Lab public-agent benchmark

Integrated <https://github.com/automatylicza/orbit-wars-lab> under:

```text
.omx/research/upstream/orbit-wars-lab/
```

The lab uses the official Kaggle environment pinned to commit `27660ecca553b96b1e3d6b282ca855fd1f274847`, the rotational-symmetry fix referenced by Kaggle discussion #694310.

Helper commands:

```bash
rtk run './scripts/setup_orbit_wars_lab.sh'
rtk run './scripts/run_lab_gauntlet.sh 1 777'
rtk run 'FORMAT=4p ./scripts/run_lab_gauntlet.sh 1 888'
```

### 2P gauntlet, public agents, one game each

Artifact: `.omx/research/upstream/orbit-wars-lab/runs/2026-04-27-010/results.json`

```text
7 matches: current wins 5, losses 2
wins: nearest-sniper, starter, pilkwang-structured, sigmaborov-reinforce, sigmaborov-starter
losses: tamrazov-starwars, yuriygreben-architect
```

### 2P gauntlet, public agents, two games each

Artifact: `.omx/research/upstream/orbit-wars-lab/runs/2026-04-27-008/results.json`

```text
14 matches: current wins 10, losses 4
lost both to pilkwang-structured and tamrazov-starwars in that sample
```

### 4P top-public sample

Artifact: `.omx/research/upstream/orbit-wars-lab/runs/2026-04-27-011/results.json`

```text
winner=external/pilkwang-structured
scores=[11855, 12413, 10951, 6291]
current was close second by score, but Pilkwang held the stronger production race.
```

Lab read: our current bot is no longer just an old-base counter; it can beat several bundled public rule bots. Next target should be anti-Tamrazov/Yuriy 2P stability plus anti-Pilkwang 4P production-cluster holding.

## Orbit Wars Lab deterministic seeding + hostile-cluster candidate (2026-04-27)

Lab harness fix: patched `.omx/research/upstream/orbit-wars-lab/orbit_wars_app/match.py` so `seed_base` actually seeds Python + Orbit Wars module RNG before each match. Added `SAVE_REPLAYS` to `scripts/run_lab_gauntlet.sh` and added `scripts/run_lab_selected.sh` for focused public-agent tests.

Promoted candidate: `cluster_hostile_candidate` -> `main.py` and `submit_current_w1b_counter.py`.

What changed in agent behavior:

- adds local high-production cluster scoring;
- only applies the cluster boost to **hostile** prod>=3 targets, not neutrals;
- adds a small hostile capture margin/value/score boost for enemy frontier/cluster anchors;
- rejected broader cluster-hold reserve/reinforce candidate because it over-turtled and collapsed in 4P.

Evidence, deterministic lab:

- Current baseline selected 2-game public core (`runs/2026-04-27-029`): 4W/2L; score sum for our bot on listed matches = 16920.
- Hostile-cluster selected 2-game public core (`runs/2026-04-27-030`): 4W/2L; score sum = 17144. Same win-rate, better aggregate score.
- Current full 2P gauntlet (`runs/2026-04-27-021`): 5W/2L; score sum = 20930.
- Hostile-cluster full 2P gauntlet (`runs/2026-04-27-031`): 5W/2L; score sum = 21665. Same win-rate, better aggregate score, especially vs `sigmaborov-reinforce`.
- 4P top-public sample unchanged (`runs/2026-04-27-027` current vs `runs/2026-04-27-032` hostile): both eliminated; Pilkwang wins with `[0,4109,0,0]`.

Validation:

```text
python3 -m py_compile main.py submit_current_w1b_counter.py
./scripts/smoke_submission.sh 4242 80 -> VALIDATION_SMOKE [DONE, DONE] [-1, 1] 80
```

Caution: this is a narrow lab-score improvement, not a ladder-guaranteed jump. It improves hostile target pressure without changing 4P survival; next real target remains anti-Pilkwang 4P collapse.
