# Orbit Wars project map

Use this file as the navigation page. Keep Kaggle submission files at repo root.

## Files you edit / submit

- `main.py` — working bot source.
- `submit_current_w1b_counter.py` — Kaggle upload file. This should be copied from `main.py` before submit.

## Scripts you run

- `scripts/run_evolve.sh` — 4-panel live GUI comparison, latest vs old.
- `scripts/run_compare.sh` — batch quick/full evaluation, writes reports under `.omx/research/sim/runs/`.
- `scripts/smoke_submission.sh` — cheap self-play validation for `submit_current_w1b_counter.py`.
- `scripts/setup_orbit_wars_lab.sh` — clone/update Orbit Wars Lab and install its uv/Python 3.12 env.
- `scripts/sync_lab_agent.sh` — copy `submit_current_w1b_counter.py` into Orbit Wars Lab as `mine/current-w1b-counter`.
- `scripts/run_lab_gauntlet.sh` — official-engine public-agent gauntlet via Orbit Wars Lab.
- `scripts/run_lab_selected.sh` — focused Orbit Wars Lab 2P gauntlet against selected public agents (`OPPONENTS=...`).
- `scripts/open_latest.sh` — reopen latest GUI/report/events.
- `scripts/index_runs.py` — refresh `docs/RUNS_INDEX.md`.

## Main research / simulation files

- `.omx/research/sim/gui_evolve.py` — engine for 4-map GUI runs.
- `.omx/research/sim/gui_simulate.py` — single replay GUI runner.
- `.omx/research/sim/evolve_report.py` — turns replay JSON into report/events/summary.
- `.omx/research/sim/compare_old_new.py` — quick/full head-to-head runner.
- `.omx/research/main_public_planner_v2.py` — old baseline used as `old_base_v2`.
- `.omx/research/upstream/orbit-wars-lab/` — cloned public benchmark lab with stronger bundled agents.
- `.omx/research/orbit_wars_lab_notes.md` — integration notes and benchmark reads from the lab.
- `.omx/research/kaggle_discussion_694310.md` — extracted notes from Kaggle symmetry-fix discussion.

## Outputs

- `.omx/research/sim/gui/replays/` — replay JSON, `.report.md`, `.events.csv`, `.summary.json`, `.insights.md`.
- `.omx/research/sim/runs/` — new batch compare outputs from `scripts/run_compare.sh`.
- `docs/RUNS_INDEX.md` — generated index of important outputs.

## Do not move casually

- `main.py` and `submit_current_w1b_counter.py`: Kaggle/simple import paths expect root files.
- `.omx/research/sim/gui/`: GUI paths assume this layout.
- `.omx/research/main_public_planner_v2.py`: compare scripts use this path.
