# Run Orbit Wars locally

Start here when you want to run it yourself.

```bash
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/OrbitWars"
```

## 1) Open 4-panel live GUI compare

Default: latest `new_current_counter` vs `old_base_v2`, seeds 100-103, 220 turns.

```bash
rtk run './scripts/run_evolve.sh'
```

Custom seed/number of games/steps:

```bash
rtk run './scripts/run_evolve.sh 9999 4 220'
```

Shortcut if you only want seed + steps, with 4 games:

```bash
rtk run './scripts/run_evolve.sh 1212 220'
```

Write to a custom replay file:

```bash
rtk run './scripts/run_evolve.sh 100 4 220 .omx/research/sim/gui/replays/my_run.json'
```

Do not auto-open browser:

```bash
NO_OPEN=1 rtk run './scripts/run_evolve.sh 100 4 220'
```

GUI URL after a default run:

```text
http://127.0.0.1:8765/evolve.html?replay=replays/evolve_live.json&watch=1
```

## 2) Reopen latest GUI/report

```bash
rtk run './scripts/open_latest.sh'
```

## 3) Run batch compare

Fast signal:

```bash
rtk run './scripts/run_compare.sh quick'
```

Full 500-turn smoke:

```bash
rtk run './scripts/run_compare.sh full'
```

Outputs go under:

```text
.omx/research/sim/runs/
```

Then open the index:

```bash
rtk run 'open docs/RUNS_INDEX.md'
```

## 4) Validate submit_current_w1b_counter.py before upload

```bash
rtk run './scripts/smoke_submission.sh'
```

## 5) Submit to Kaggle when quota is available

Only run this when you intentionally want to spend quota:

```bash
rtk run 'kaggle competitions submit -c orbit-wars -f submit_current_w1b_counter.py -m "hostile cluster target boost v1"'
```

## 6) Run Orbit Wars Lab public-agent benchmark

Setup/update the external lab and sync our current submit file into it:

```bash
rtk run './scripts/setup_orbit_wars_lab.sh'
```

2P gauntlet against bundled public agents:

```bash
rtk run './scripts/run_lab_gauntlet.sh 1 777'
```

4P top-public sample:

```bash
rtk run 'FORMAT=4p ./scripts/run_lab_gauntlet.sh 1 888'
```

Outputs go under:

```text
.omx/research/upstream/orbit-wars-lab/runs/
```
