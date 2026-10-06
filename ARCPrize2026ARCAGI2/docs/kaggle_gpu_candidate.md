# Kaggle Candidate Runbook

This is the historical Kaggle candidate runbook. After the 2024 Omni-ARC writeup review, the active direction is now the hybrid Omni-ARC 2026 lane in `docs/omni_arc_2026_plan.md`.

Do not spend the one daily leaderboard submission on `arc2026_baseline.ipynb`; that file is only for schema/viewer debugging.

Also do not run `arc2026_gpu_ttt_architects_candidate.ipynb` as-is. It was an adapted 2025 notebook and is kept only as a reference.

The previous `arc2026_l4x4_original_candidate` lane is now a CPU/symbolic candidate source, not the main submission strategy.

## Candidate Notebook

Reference only:

- `notebooks/arc2026_l4x4_original_candidate.ipynb`

Source script:

- `notebooks/arc2026_l4x4_original_candidate.py`

It is an original CPU/symbolic implementation using these 2025-derived ideas:

- test-time fitting per task
- geometric/color augmentation
- candidate voting
- output shape priors
- strict `submission.json` validation

## Kaggle Settings

- Competition input: `arc-prize-2026-arc-agi-2`
- Accelerator: GPU L4 x4 is okay, but this first original candidate does not require external model inputs
- Internet: off
- Run mode: competition rerun/submission notebook
- Submit only after a full run completes and writes `submission.json`

## Required Attached Inputs

Only the official competition input is required:

- `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json`

## Why Keep This Lane

This lane is useful because it is not a brittle 2025 code transplant. It uses the competition-safe parts of the 2025 top-10 pattern:

- fit transformations per task
- augment views and invert predictions
- vote/rerank candidates
- validate schema before output

It is not the final strategy. It is now the CPU/symbolic branch for the Omni-ARC hybrid plan:

- generate relation/symbolic candidates cheaply,
- provide source-diverse `attempt_2` candidates,
- serve as a local sanity check for visualizer and schema,
- avoid wasting GPU runtime on simple exact symbolic relations.

## Do Not Submit

Do not submit:

- `notebooks/arc2026_baseline.ipynb`
- `notebooks/arc2026_gpu_ttt_architects_candidate.ipynb`
- `notebooks/arc2026_l4x4_original_candidate.ipynb` as the only submission lane
- `/Users/chinnaphat/Downloads/submission.json` if it is the identity/solid-grid baseline

Those are debugging/reference artifacts only.
