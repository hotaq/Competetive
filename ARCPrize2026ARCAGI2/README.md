# ARC Prize 2026 - ARC-AGI-2

Workspace for the Kaggle competition:

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

## Current Snapshot

Checked: 2026-04-29 Asia/Bangkok

- Kaggle title: ARC Prize 2026 - ARC-AGI-2
- Kaggle slug: `arc-prize-2026-arc-agi-2`
- Goal: build an AI system that solves ARC-AGI-2 static reasoning tasks.
- Target: reach 85% accuracy on the private ARC-AGI-2 evaluation set within Kaggle efficiency limits.
- Submission type: Kaggle notebook only.
- Evaluation: custom Abstraction and Reasoning Challenge metric.
- Scoring shape: predict exactly 2 outputs for each test input; a task output scores 1 if either prediction exactly matches the hidden solution, otherwise 0. Final score is the average.
- Prize pool: USD 700,000 for this ARC-AGI-2 track.
- Deadline: 2026-11-02 23:59 UTC.
- Daily submissions: 1.
- Max team size: 5.
- Current local Kaggle status: joined/accepted; data download works.

## File Map

- `research.md` - initial competition notes and strategy framing.
- `data/raw/` - Kaggle/raw files after rules acceptance.
- `data/processed/` - derived local data.
- `scripts/` - local utilities and validators.
- `notebooks/` - local exploration notebooks.
- `public_notebooks/` - reviewed public references, if downloaded later.
- `artifacts/` - generated submissions and experiment outputs.
- `docs/` - deeper notes, diagrams, and runbooks.

## Notebook-First Starter

- `notebooks/arc2026_baseline.ipynb` - Kaggle-ready starter notebook.
  - Loads `/kaggle/input/arc-prize-2026-arc-agi-2`.
  - Includes public-split evaluator and visual task viewer.
  - Debug-only by default; writes `debug_submission.json`, not a leaderboard file.
  - Uses only lightweight heuristic candidates for the first pipeline check.
  - Intended to run on Kaggle editor for inspection, not as the daily submission.
- `notebooks/arc2026_l4x4_original_candidate.ipynb` - current run candidate.
  - Written from scratch for ARC Prize 2026.
  - Uses only 2025 ideas: test-time fitting, augmentation voting, candidate rescoring, strict schema validation.
  - Requires only the official competition input.
  - Emits `submission.json` and `run_meta.json`.
- `notebooks/arc2026_gpu_ttt_architects_candidate.ipynb` - deprecated reference only.
  - Do not run/submit this as-is; it was an adapted 2025 notebook and is kept only as a research reference.

## Kaggle Data Snapshot

Downloaded to `data/raw/` on 2026-04-29:

- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json`
- `sample_submission.json`

Observed local counts:

- Training challenges: 1,000 tasks.
- Training solutions: 1,000 tasks.
- Evaluation challenges: 120 tasks.
- Evaluation solutions: 120 tasks.
- Visible test challenges: 240 tasks.
- Sample submission: 240 task IDs, aligned with the visible test challenge inputs.
- Grid values: integers 0-9.
- Max observed grid shape: 30x30.
- Visible test inputs per task: mostly 1, with a few tasks having 2 or 3.

## First Commands

Download files:

```bash
python3 -m kaggle.cli competitions download -c arc-prize-2026-arc-agi-2 -p data/raw
```

List files without downloading:

```bash
python3 -m kaggle.cli competitions files -c arc-prize-2026-arc-agi-2
```

## Sources

- https://arcprize.org/competitions/2026/arc-agi-2
- https://arcprize.org/competitions/2026
- https://github.com/arcprize/ARC-AGI-2
- https://arcprize.org/guide/1
