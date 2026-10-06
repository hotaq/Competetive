# Orbit Wars Agent

Single-file Kaggle submission agent for the Orbit Wars Python kit.

## Start here

- Runbook: [`RUN.md`](RUN.md)
- File map: [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md)
- Latest result snapshot: [`docs/LATEST_RESULTS.md`](docs/LATEST_RESULTS.md)
- Generated run index: [`docs/RUNS_INDEX.md`](docs/RUNS_INDEX.md)

## Files

- `main.py` - self-contained agent with no third-party dependencies.
- `submit_current_w1b_counter.py` - Kaggle upload copy of `main.py` with a clear non-generic name.
- `test_agent.py` - offline sanity checks for action format, sun avoidance, defense, and comet observations.
- `scripts/` - wrappers for GUI sim, compare runs, and submission smoke tests.

## Strategy Notes

- The ladder updates skill from wins/losses, not score margin, so the bot favors robust expansion and ship preservation over all-in attacks.
- Kaggle validates new submissions against copies of themselves first; `main.py` keeps fallback turn state per player in case multiple copies share one Python process.
- Contested neutral planets are penalized or outbid so the bot wastes fewer ships into 4-player pileups.
- Current patch adds anti-stall recovery pressure: when behind on production/planet count after turn 90, the bot releases reserve and forces short profitable captures.

## Local Checks

```bash
rtk run './scripts/smoke_submission.sh'
rtk run './scripts/run_evolve.sh 100 4 220'
rtk run './scripts/run_compare.sh quick'
```

## Submit

Submit `submit_current_w1b_counter.py` when the Kaggle CLI is configured, quota is available, and the competition rules are accepted:

```bash
rtk run 'kaggle competitions submit -c orbit-wars -f submit_current_w1b_counter.py -m "anti-stall recovery pressure v1"'
```

If you package multiple files later, keep `submit_current_w1b_counter.py`/`main.py` at the archive root.
