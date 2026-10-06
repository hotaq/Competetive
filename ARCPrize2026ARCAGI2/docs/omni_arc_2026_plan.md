# Omni-ARC 2026 Plan

Adopt the 2024 Guillermo Barbadillo Omni-ARC direction, but adapt it to ARC Prize 2026 and the current L4 x4 Kaggle setup.

## Positioning

This is the main direction now:

1. Keep a neural TTT generator as the primary solver.
2. Add AIRV-style inference: augment, infer, reverse augmentation, vote.
3. Add a selector/verifier layer to pick the final two attempts.
4. Run CPU symbolic/relation candidates beside the GPU model.
5. Submit only when the full pipeline writes a valid `submission.json`.

Do not replace the neural solver with the symbolic-only candidate. The symbolic lane is a candidate source and second-opinion attempt, not the main model.

## Why This Direction

The 2024 2nd-place writeup says prompting alone was not enough. The score came from:

- a small ARC-specialized Qwen model,
- per-task test-time fine-tuning,
- AIRV voting,
- candidate selection,
- and a CPU solver ensemble.

This matches what we observed:

- raw Qwen outputs often get plausible shapes but weak relations,
- symbolic relation candidates can solve some clean pattern tasks,
- `attempt_2` should not be wasted on all-solid fallback when a second candidate exists,
- GPU utilization is poor if generation is serialized or candidate search is too thin.

## Target Kaggle Pipeline

### Stage 1: Load Data

- Use `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json`.
- During local/eval debugging only, use `arc-agi_evaluation_challenges.json` and `arc-agi_evaluation_solutions.json`.

### Stage 2: Per-Task TTT

For each task:

- Build pseudo-train examples from the task's own train pairs.
- Train with leave-one-out style examples:
  - use `n-1` train pairs as prompt context,
  - predict the held-out output.
- Keep TTT small enough to finish all 240 tasks:
  - start with 4-8 steps for the 4B model,
  - only increase steps for short/simple tasks if runtime allows.

The pure 2024 setting was around 300 steps per task on a 0.5B model. Do not copy that number to a 4B model on L4 x4.

### Stage 3: AIRV Inference

Generate candidates under reversible transforms:

- identity
- transpose
- rotations
- flips
- fixed color permutations from task colors
- train example shuffle

For every augmented prediction:

1. Validate JSON/grid shape.
2. Reverse geometry and color mapping.
3. Drop invalid grids.
4. De-duplicate by exact grid.
5. Keep frequency/vote count and source metadata.

### Stage 4: CPU Symbolic/Relation Branch

Run cheap symbolic candidates in parallel or before GPU result selection:

- relation matrix candidates,
- tiling/repetition,
- mirror/upscale/downscale,
- crop/object movement,
- identity and constant as last-resort fallbacks.

Symbolic candidates get high weight only when they exactly explain all train pairs.

### Stage 5: Selector / Verifier

Rank all candidates, not just the first generated outputs.

Initial non-learned selector score:

- high score for train-consistent symbolic relation candidates,
- neural vote frequency,
- shape prior from train outputs,
- color-set compatibility,
- penalty for all-solid fallback,
- penalty for identity unless train examples support identity,
- source diversity bonus so `attempt_1` and `attempt_2` are not near-duplicates.

Later learned selector:

- train a small selector/verifier offline on public ARC/evaluation-like candidates,
- input: task prompt + candidate output,
- output: whether candidate is correct or choose best among options.

## Attempt Policy

Use the two attempts deliberately:

- `attempt_1`: highest-ranked neural/AIRV/TTT candidate.
- `attempt_2`: best candidate from a different source family, usually symbolic/relation if confident.

If both top candidates are from the same family and nearly duplicate, prefer source diversity over tiny score differences.

## Runtime Policy

For L4 x4:

- shard tasks across 4 workers,
- one model process per GPU is safer than many threads per GPU,
- avoid `torch.cuda.empty_cache()` after CUDA illegal memory access; restart the worker/GPU process instead,
- write shard checkpoint JSON frequently,
- merge shards at the end into `/kaggle/working/submission.json`.

## Immediate Implementation Order

1. Patch current Qwen notebook so it stores more than two raw candidates per task.
2. Add AIRV candidate metadata: transform, temperature, vote count, source.
3. Add relation/symbolic candidate generation before final top-2 selection.
4. Add selector scoring and source-diverse top-2 attempt selection.
5. Run on evaluation split only to inspect exact/shape/viz.
6. Run on test only when eval/viz looks sane and runtime fits.

## Do Not Do

- Do not submit symbolic-only output as the main daily submission.
- Do not spend a Kaggle submission on smoke output.
- Do not hardcode known evaluation/test task answers.
- Do not rely on internet during notebook evaluation.
- Do not copy 2024 hyperparameters blindly to a 4B model.

## Current Useful Local Files

- `research.md` has the 2024 and 2025 writeup notes.
- `notebooks/arc2026_l4x4_original_candidate.py` contains the current CPU/symbolic relation branch.
- `notebooks/arc2026_gpu_ttt_architects_candidate.ipynb` is a deprecated reference for ARChitects-style GPU TTT, not a direct submission target.
- `tools/score_submission.py` scores evaluation submissions.
- `tools/arc_visualizer.html` visualizes train/test/prediction grids.
