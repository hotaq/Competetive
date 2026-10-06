# Pilkwang Orbit Wars Benchmark Study

Source notebook: <https://www.kaggle.com/code/pilkwang/benchmark-how-strong-is-your-orbit-wars-agent>

Pulled locally with:

```bash
uv run --no-project --python 3.11 --with kaggle kaggle kernels pull pilkwang/benchmark-how-strong-is-your-orbit-wars-agent -p .omx/research/kaggle_bench/pilkwang_benchmark -m
uv run --no-project --python 3.11 --with kaggle kaggle datasets download -d pilkwang/orbit-wars-public-dataset-for-benchmarks -p .omx/research/kaggle_bench/public_agents --unzip
```

## What the benchmark does

- Registers candidate agents and opponent agents.
- Runs official `kaggle_environments.make("orbit_wars")` episodes.
- Tests both direct 2-player duels and 4-player FFA.
- Mirrors/rotates seats to reduce starting-position bias.
- Uses suites:
  - `smoke`: tiny sanity sample.
  - `screening`: medium comparison sample.
  - `full`: slower stronger evidence.
- Summarizes by win rate, normalized rank, 2p/4p split, rank distribution, seat spread, map weakness, outcome labels, and runtime health.

## Important implementation detail copied into local tools

The notebook seeds both Python `random` and the Orbit Wars module RNG. Our previous local tools only seeded Python `random`. Local tools now also try to seed `kaggle_environments.envs.orbit_wars.orbit_wars.random` when present.

Updated tools:

- `.omx/research/sim/gui_evolve.py`
- `.omx/research/sim/gui_simulate.py`
- `.omx/research/sim/compare_old_new.py`
- `.omx/research/sim/compare_ffa4.py`

## Public benchmark agent pack

Downloaded dataset: <https://www.kaggle.com/datasets/pilkwang/orbit-wars-public-dataset-for-benchmarks>

Local files:

- `.omx/research/kaggle_bench/public_agents/submissionA.py`
- `.omx/research/kaggle_bench/public_agents/submissionB.py`
- `.omx/research/kaggle_bench/public_agents/submissionC.py`
- `.omx/research/kaggle_bench/public_agents/submissionD.py`
- `.omx/research/kaggle_bench/public_agents/submissionE.py`
- `.omx/research/kaggle_bench/public_agents/submissionF.py`

Local aliases were added so GUI/FFA scripts can call `submissionA` through `submissionF` directly.

## New local commands

Public-hard 4P smoke/screen:

```bash
rtk run ./scripts/run_ffa4.sh public 900 1 220
rtk run ./scripts/run_ffa4.sh public 901 1 220
```

GUI version:

```bash
rtk run ./scripts/run_ffa4_gui.sh public 902 1 30
```

## Current public-hard 4P signal

Public-hard trio: `submissionB`, `submissionD`, `submissionE`.

Seed 900, all current seats:

```text
4 games: current first_place=1/4, avg_rank=1.75-ish by rows, many eliminations/zero totals in non-winning seats.
```

Seed 901, all current seats:

```text
4 games: current first_place=1/4, mostly rank 2, one rank 3 elimination.
```

Combined quick read over these 8 seat-rotated games:

```text
current first_place=2/8
current top2=7/8
current eliminated_or_zero=5/8
```

Interpretation: the bot can win some public-hard 4P seats, but often collapses to zero while still being counted rank 2 because other agents are also eliminated. This is a stronger warning than the old-base-only benchmark.

## Strategy implication

Do not optimize only against `old_base_v2`. Use public-hard FFA as a guardrail before submission. The next improvement should target 4P survival and production stabilization against strong public openings, especially when `submissionB` or `submissionD` snowballs early.
