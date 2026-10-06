# ARC Prize 2026 - ARC-AGI-2 Research Notes

Research date: 2026-04-29 Asia/Bangkok

Workspace folder: `competetive/ARCPrize2026ARCAGI2/`

## What This Competition Is

ARC-AGI-2 is a static visual reasoning benchmark. Each task gives a few demonstration input/output grid pairs and one or more test input grids. The solver must infer the hidden transformation rule and output the correct grid for each test input.

The grids are rectangular arrays of integers from 0 to 9. These integers represent colors. Grid dimensions can vary from 1x1 to 30x30, and output shape is part of the answer.

This is not a normal supervised tabular/image competition. It is closer to program synthesis, search, rule induction, test-time adaptation, and solver ensembling over abstract grid transformations.

## Official Goal

Reach 85% accuracy on the private ARC-AGI-2 evaluation dataset within Kaggle efficiency limits.

Important constraints:

- Submission must be a Kaggle notebook.
- Evaluation has no internet access.
- Prize eligibility requires open-sourcing code and methods.
- The competition uses an exact-match metric.

## Scoring

For every test input, submit exactly 2 predicted outputs. If either predicted output exactly matches the hidden ground truth, that output scores 1. Otherwise it scores 0. The final score is the average across all task test outputs.

Exact match means shape, colors, and positions must all match.

## Public Data Shape

The official ARC-AGI-2 public repository contains:

- 1,000 public training tasks.
- 120 public evaluation tasks.

Kaggle's competition data download now works after joining/accepting rules. The zip contains:

- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json`
- `sample_submission.json`

Local file counts after extraction:

- Training challenges: 1,000 tasks.
- Training solutions: 1,000 tasks.
- Evaluation challenges: 120 tasks.
- Evaluation solutions: 120 tasks.
- Visible test challenges: 240 tasks.
- Sample submission: 240 task IDs.

The visible challenge schema is:

```json
{
  "task_id": {
    "train": [{"input": [[...]], "output": [[...]]}],
    "test": [{"input": [[...]]}]
  }
}
```

The solution files map each task ID to a list of expected output grids, one per test input.

The sample submission schema is:

```json
{
  "task_id": [
    {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    }
  ]
}
```

For tasks with multiple test inputs, the list contains one attempt object per test input. The downloaded `sample_submission.json` aligns with the visible `arc-agi_test_challenges.json` test-pair counts.

Observed distributions in the visible test file:

- Test tasks: 240.
- Test input count per task: 223 tasks have 1 input, 15 tasks have 2 inputs, 2 tasks have 3 inputs.
- Train pair count per test task: ranges from 2 to 8.
- Max observed grid shape: 30x30.
- Color values: 0 through 9.

Local note: before the user joined the competition, direct download returned HTTP 403. After joining, the same Kaggle CLI download succeeded.

## Initial Strategy Direction

Start with tooling, not model training:

1. Download and inspect data after rule acceptance.
2. Build a local evaluator that reads challenge and solution JSON and checks two-attempt exact match.
3. Build a visual task viewer for fast human/solver debugging.
4. Implement simple reusable grid primitives:
   - identity and constant fallbacks
   - crop, pad, recolor, connected components
   - rotate, mirror, translate
   - repeat/tile, object extraction, bounding boxes
   - color masks and shape matching
5. Create a baseline that emits two guesses per test input.
6. Mine public 2025/2026 notebooks and discussions only after the local evaluator is trustworthy.

Likely useful branches:

- DSL/program synthesis over grid primitives.
- Candidate generation plus ranking.
- Ensemble of public/open solvers, carefully license-checked.
- Test-time adaptation or synthetic-data training, if it can fit the notebook/no-internet/evaluation constraints.

## Important Watchouts

- Notebook-only submission and one daily submission make local validation critical.
- Public evaluation feedback can cause overfitting; separate training/evaluation discipline matters.
- LLM API approaches will not work directly during evaluation because internet is disabled.
- Prize eligibility is tied to reproducible open-source methods.

## Verified Local Commands

List competition files:

```bash
python3 -m kaggle.cli competitions files -c arc-prize-2026-arc-agi-2
```

List competition metadata:

```bash
python3 -m kaggle.cli competitions list -s arc-prize-2026
```

Download full data:

```bash
python3 -m kaggle.cli competitions download -c arc-prize-2026-arc-agi-2 -p competetive/ARCPrize2026ARCAGI2/data/raw
```

Result: succeeded after rules acceptance; extracted six JSON files to `data/raw/`.

## ARC Prize 2024 Writeup Recon

### 2nd - Guillermo Barbadillo, Omni-ARC

Source read: 2026-04-30.

Headline: Omni-ARC trains one small Qwen model across several ARC-related tasks, then uses per-problem test-time fine-tuning, inference augmentation, voting, and a CPU ensemble.

Main idea:

- Base model was `Qwen2.5-0.5B-Instruct`, not a large model.
- It was LoRA fine-tuned on public ARC-style datasets and trained on more than one objective:
  - `examples + input -> output`
  - `inputs -> input`, i.e. generate new inputs to learn the input/grid distribution.
  - The writeup also discusses verifier/selector/code tasks as natural extensions.
- Test-time fine-tuning was the major lift:
  - per private task, train on `n-1` train pairs and hold one pair out as pseudo-test, sampled on the fly.
  - batch size 1.
  - around 300 TTT steps per problem.
  - LR around `8e-5`, higher than pretraining LR.
- Inference used AIRV:
  - data augmentation, inference, reverse augmentation, vote.
  - 96 predictions per problem.
  - temperature 0.
  - VLLM for generation.
- Simple ensemble with the 2020 public CPU solver: first attempt from each solution.

Augmentation details:

- Standard data augmentation: rotations, flips, color changes, and swapping train/test examples.
- Problem augmentation: apply reversible transformations only to inputs or only to outputs, creating composed ARC problems.
- Reversible transformations mentioned: rotations/flips, padding, upscale, mirror.
- Grid text representation was simple but explicit: Markdown code block, first line has grid shape, rows are numbered.

Results/lessons:

- The approach scored 40 on the ARC 2024 leaderboard.
- Without TTT, one model solved 32% of evaluation tasks; with 32 predictions, top-2 accuracy was 22%.
- He says prompting alone is not enough; TTT is needed for novel ARC tasks.
- A selector model beat voting on evaluation with 32 predictions (`top1 68.7%`, `top2 81.0%`), but did not improve LB, likely because frozen-model candidate distribution differed from TTT-model candidates.
- Code/DSL training did not work well for him in 2024: 285 hand-coded tasks only gave around 5% eval and one lucky private solve.
- Dataset quality matters: huge synthetic BARC data did not clearly improve results, likely due low novelty.
- `Qwen2.5-0.5B` was the right speed/compute point for his submission setup; bigger Qwen models were more data-efficient but slower and more VRAM constrained.

Important transfer for ARC 2026:

- For our L4 x4 setup, do not blindly scale TTT on `Qwen3-4B`; throughput can be worse than a smaller specialized ARC model.
- Add an explicit candidate selector/verifier lane instead of relying only on raw generation order or vote count.
- Keep the CPU symbolic/relation solver running beside GPU generation; one attempt from neural and one attempt from CPU/symbolic is a proven ensemble pattern.
- Make AIRV stronger: fixed augmentation set, reverse transforms, de-duplicate candidates, vote/rank by frequency plus shape confidence.
- Improve grid prompts with shape and row numbers; this is cheap and likely helps.
- Treat input-distribution learning as an offline fine-tune objective if we prepare a model/dataset artifact, not as something Kaggle can download during no-internet evaluation.

Links:

- Kaggle writeup: <https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar>
- Better-formatted paper: <https://ironbar.github.io/arc24/05_Solution_Summary/>
- GitHub: <https://github.com/ironbar/arc24>
- Submission notebook: <https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24?scriptVersionId=199282752>

## ARC Prize 2025 Top-10 Writeup Recon

Research date: 2026-04-29 Asia/Bangkok.

Primary sources:

- Kaggle private leaderboard: <https://www.kaggle.com/competitions/arc-prize-2025/leaderboard>
- NVARC writeup: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc>
- The ARChitects writeup: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025>
- MindsAI & Tufa Labs writeup: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution>
- Lonnie writeup/notebook: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place>
- Guillermo Barbadillo notebook: <https://www.kaggle.com/code/ironbar/the-architects-single-task-ttt>
- Guillermo Barbadillo technical report: <https://ironbar.github.io/arc25/05_Solution_Summary/>
- ippeiogawa writeup: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects>
- rxe / Richard Emslie notebook: <https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked>

### Final Private Leaderboard Top 10

Kaggle says the final private leaderboard uses approximately 50% of the test data and reflects final standings. Treat this table as the authority for placement. The downloaded Kaggle CSV is a public leaderboard snapshot, so public ranks/scores can disagree with the final private standings.

| Private rank | Team | Private score | Public/open source status |
| --- | --- | ---: | --- |
| 1 | NVARC | 24.03 | Full Kaggle writeup, GitHub, public notebooks/datasets |
| 2 | the ARChitects | 16.53 | Full Kaggle writeup, technical report, notebook/model links |
| 3 | MindsAI & Tufa Labs | 12.64 | Full Kaggle writeup, PDF, GitHub, HF datasets/models, notebook |
| 4 | Lonnie | 6.67 | Kaggle writeup plus notebook fork of 2024 first-place baseline |
| 5 | Guillermo Barbadillo @ Veridas | 6.53 | Kaggle notebook plus detailed external technical report |
| 6 | ippeiogawa | 6.25 | Kaggle writeup plus two notebooks |
| 7 | Ggmam1 | 6.25 | No public writeup/notebook found in focused Kaggle/web search yet |
| 8 | rxe | 6.11 | Public notebook by albartrose/Richard Emslie; no separate Kaggle writeup found |
| 9 | Abdelmoneim Desouki | 5.83 | No public writeup/notebook found in focused Kaggle/web search yet |
| 10 | Boiled-Sweet-Potato | 5.69 | No public writeup/notebook found in focused Kaggle/web search yet |

Local public notebook copies downloaded for inspection:

- `public_notebooks/arc2025_ironbar_the_architects_single_task_ttt/the-architects-single-task-ttt.ipynb`
- `public_notebooks/arc2025_ippeiogawa_lb10_tweak_2024/lb10-00-with-tweak-to-2024-1st-solution.ipynb`
- `public_notebooks/arc2025_ippeiogawa_lb903_tweak_2024/lb9-03-with-tweak-to-2024-1st-solution.ipynb`
- `public_notebooks/arc2025_rxe_albartrose_arcitects_tweaked/10-4-notebook-arcitects-tweaked.ipynb`

### 1st - NVARC

Headline: synthetic data generation with LLMs and code generation, improved ARChitects, and improved TRM.

Main idea:

- Scale high-quality synthetic ARC-style puzzles, then train/adapt strong grid-as-text models.
- Use puzzle descriptions from H-ARC and BARC, mix descriptions, generate Python programs for input/output grids, and filter by consistency.
- Reported pipeline scale: 716 ARC-AGI-2 training puzzle descriptions after filtering, 266,593 new puzzle summaries, 126,901 input-grid programs, and 103,253 puzzles with one input program plus multiple consistent output programs.
- Final ARChitects-style training corpus: about 104k puzzles and 3.25M augmented samples.
- Used Qwen3-style chat/grid representation, full fine-tuning for the 4B model, then LoRA test-time fine-tuning per puzzle.
- Used batch DFS decoding and candidate rescoring with augmentation log probabilities plus candidate frequency.
- Also explored TRM late in the competition; useful but final gains were mostly from synthetic-data and ARChitects-style scaling.

Important transfer:

- Synthetic puzzle quality and consistency filtering mattered more than just quantity.
- Candidate ranking must combine multiple signals: augmentation probabilities, how often a candidate appears, and exact validity checks.
- For 2026, we should build the data/evaluator/viewer pipeline before training so synthetic generation can be measured locally.

Useful links from their writeup:

- GitHub: <https://github.com/1ytic/NVARC>
- TRM notebook: <https://www.kaggle.com/code/cpmpml/arc2-trm-v31?scriptVersionId=278223801>
- ARChitects-style notebook: <https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue>
- Viewer: <https://www.kaggle.com/code/sorokin/nvarc-viewer>
- Synthetic datasets:
  - <https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles>
  - <https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles>
  - <https://www.kaggle.com/datasets/sorokin/nvarc-augmented-puzzles>
  - <https://www.kaggle.com/datasets/cpmpml/arc-prize-trm-training-data/settings>

### 2nd - the ARChitects

Headline: masked diffusion LLM plus recursive refinement.

Main idea:

- Final approach used LLaDA-8B as a masked diffusion model for ARC instead of pure autoregressive generation.
- Train to reconstruct masked output-grid tokens, then recursively feed model outputs back in for refinement.
- Added 2D positional encoding based on Golden Gate RoPE.
- Used a separate LLaDA shape predictor because output dimensions are a major source of failure.
- Pretraining data included ReARC, ARC-GEN-100K, official ARC1/ARC2, ARC-Heavy, and ConceptARC.
- Test-time fine-tuning was 128 steps per task on L4.
- Earlier autoregressive branch used Mistral-NeMo-Minitron-8B with task-specific TTT, speculative decoding, prefix caching, lower DFS probability cutoffs, and more augmentation scoring.

Important transfer:

- Whole-output refinement is attractive for ARC-AGI-2 because many tasks require global structure, not left-to-right token decisions.
- Shape prediction deserves a separate module or at least separate scoring.
- Public/private divergence was real: they mention a higher-public submission was selected even though another one had better private stability.

Links:

- Technical report: <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>
- Notebook: <https://www.kaggle.com/code/dfranzen/arc-prize-2025-by-the-architects>
- Video: <https://www.youtube.com/watch?v=CcoGi47qD-w>

### 3rd - MindsAI & Tufa Labs

Headline: adapt pretrained models at test time.

Main idea:

- Used a pruned 660M encoder-decoder model from Salesforce CodeT5-Large.
- Trained for months on more than 100M reasoning examples, roughly 70M ARC-style.
- Heavy test-time adaptation drove the score:
  - TTT with permutation-based labeling, around 45k steps.
  - AIRV: augment, inference, reverse, vote; around 10k augmented inferences per task.
  - Self-ensemble of multiple checkpoints.
- Augmentations included rotations/flips, color permutations, input/output swap, prompt/answer reversals, mixup/combine variants, and tokenizer dropout.

Important transfer:

- TTT and AIRV were nearly additive in their experiments.
- Different seeds/checkpoints can beat simply sampling more from one model.
- Encoder depth mattered more than decoder depth for their setup.
- They warn that ARC-AGI-2 may be partially adversarial to the old TTT/AIRV paradigm, so this is a strong baseline family but not enough by itself.

Links:

- GitHub: <https://github.com/jcole75/arc_2025_mindsai>
- Notebook: <https://www.kaggle.com/code/jcole75/mindsai-tufa-2025-v4>
- HF dataset: <https://huggingface.co/datasets/mindware/arc-agi-mega>
- HF models: <https://huggingface.co/mindware>
- arXiv: <https://arxiv.org/abs/2506.14276>
- Extra Kaggle dataset: <https://www.kaggle.com/datasets/jcole75/mindsai-tufa-extra-datasets>

### 4th - Lonnie

Headline: strong 2024 baseline fork, seed/hyperparameter exploration, and private leaderboard variance.

Main idea:

- Forked the 2024 first-place ARChitects/Sanyul/Boris-style baseline.
- Used Nemo Mini, LoRA, task-specific priming, turbo DFS, augmented scoring, caching, and parallel GPU execution.
- Kept the approach close to the known strong baseline and tuned around it.
- The writeup emphasizes random seed as a real hyperparameter. On a small test set, one or two exact solves can move many ranks.
- Reported score variance across seeds was large enough to explain public/private surprises.

Important transfer:

- Keep seed sweeps and selection sweeps as first-class experiments, not afterthoughts.
- A modest tweak to a strong baseline can beat a more ambitious branch if the ambitious branch is not validated well.
- The writeup title/metadata says 5th place, while the final private leaderboard now shows Lonnie at rank 4. Keep this inconsistency in mind when comparing notes.

Notebook:

- <https://www.kaggle.com/code/lonnieqin/lb-5-83-baseline-from-1st-place-of-2024>

### 5th - Guillermo Barbadillo @ Veridas

Headline: single-task TTT on the 2024 ARChitects baseline, plus a broader search-and-learn research program.

Leaderboard/source facts:

- Kaggle notebook `The architects single task TTT` shows best score 6.53 and runtime 13s metadata for version 185, using GPU L4 x4.
- The notebook says it copied the 2024 ARChitects solution and changed:
  - test-time training per task instead of grouped tasks.
  - efficient 4-GPU execution.
  - hyperparameter tuning.
- Notebook config highlights from local copy:
  - `train_epochs = 10`
  - `n_gpus = 4`
  - `max_seq_length_train = 8192`
  - `max_seq_length_infer = 8192`
  - `min_prob = 0.13`
  - `lora_rank = 4`
  - augmentation scoring with transpose, rotation, random color permutation, example shuffle, and `score_full_probmul_3`.

Broader technical report:

- Main research direction was deep-learning-guided program synthesis: search program space, execute code, learn from failed attempts with hindsight relabeling, then search again.
- He reports this search-and-learn branch did not beat the transduction/TTT baseline on the private test, but it validated several ideas:
  - BARC induction model can solve some ARC tasks with DSL/code generation.
  - Hindsight relabeling improved ARC-AGI-1 evaluation pass rate in one experiment from 23.3% to 28.3% under the same 512-prediction budget.
  - BARC induction solved around 22% on ARC-AGI-1 eval at 512 predictions but only around 0.8% on ARC-AGI-2 eval.
  - RL improved a BARC model in one experiment but training collapse was a blocker.
  - Prediction refinement with a smaller BARC-style model was weak compared with frontier-model code-refinement approaches.

Important transfer:

- For 2026, a program-synthesis branch is worth tracking, but it needs strong induction models, refinement, and a way to learn from failed attempts.
- Do not assume infinite synthetic tasks help if their diversity is narrow.
- The practical competition submission may still need a robust transduction/TTT baseline while we experiment with search-and-learn.

Links:

- Notebook: <https://www.kaggle.com/code/ironbar/the-architects-single-task-ttt>
- Technical report: <https://ironbar.github.io/arc25/05_Solution_Summary/>
- GitHub: <https://github.com/ironbar/arc25>

### 6th - ippeiogawa

Headline: small tweaks to the 2024 ARChitects baseline.

Source facts:

- Kaggle writeup metadata says `Solution Writeup - 6th place - Nov 9, 2025`.
- The writeup says the final solution was a lightly modified version of the ARChitects 2024 approach.
- Two linked notebooks:
  - `LB10.00 with tweak to 2024 1st solution`
  - `LB9.03 with tweak to 2024 1st solution`

Main tweaks:

- TTT per task.
- `num_epoch = 24`.
- Reduced training examples to 3 for faster training.
- Batch size 4 and gradient accumulation 2.
- Removed background-only lines for faster training in one of the two submissions.
- Inference used one augmentation/inference pass, reduced examples to 2, two subprocesses for speed, and `top_k = 4` for DFS.
- The writeup says not all tasks finish within the 12-hour limit; reducing epochs hurt performance.

Important transfer:

- This is a strong reminder that notebook budget management matters as much as model quality.
- Background-line compression is a cheap preprocessing idea to test locally, but do not apply it blindly to test grids.
- If we use DFS, `top_k` and probability cutoffs need direct runtime/score sweeps.

Links:

- Writeup: <https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects>
- Notebook 1: <https://www.kaggle.com/code/ippeiogawa/lb10-00-with-tweak-to-2024-1st-solution>
- Notebook 2: <https://www.kaggle.com/code/ippeiogawa/lb9-03-with-tweak-to-2024-1st-solution>

### 7th - Ggmam1

No public writeup or notebook found yet.

Focused checks done:

- Kaggle private leaderboard confirms final rank 7 with score 6.25.
- Kaggle CLI search by `Ggmam1` did not return public competition notebooks.
- Web search for exact team/name did not surface a credible writeup.

Action if we revisit:

- Inspect Kaggle profile manually if it becomes accessible from the leaderboard row.
- Search discussion comments after login if Kaggle exposes more author links.

### 8th - rxe

Headline: ARChitects baseline tweaks with task-level worker scheduling, collator changes, and DFS tweaks.

Source facts:

- The final private leaderboard shows `rxe` at rank 8 with score 6.11.
- Public leaderboard snapshot maps this team to `albartrose`.
- Public notebook: `10.4 notebook (ARCitects tweaked)` by Richard Emslie.

Notebook summary:

- Started from the previous year's ARChitects notebook.
- Introduced a master/worker client-server pattern so ARC tasks were farmed to workers one by one across L4 GPUs.
- Did TTT and inference per task.
- Inference was run on LoRA adapters, slower than the original train-and-merge approach.
- Hyperparameters noted:
  - 10 train epochs.
  - 1 inference epoch.
  - LoRA rank 32, alpha 32.
  - `min_prob = 0.17`.
  - `max_seq_length_train = 6144`.
- Added a collator that skipped examples during training.
- Tweaked DFS to skip 1-2 bad probabilities and extend probability budget for longer grids.
- Author notes that collator + DFS jump was from around 8 to 10.4 public, but later rewrites dropped to 6-7%.

Important transfer:

- Master/worker scheduling is useful for Kaggle L4x4 task-level parallelism.
- DFS changes can create large public gains but are fragile.
- Refactoring a working baseline can silently break performance; keep exact reproducibility checkpoints.

Link:

- <https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked>

### 9th - Abdelmoneim Desouki

No public writeup or notebook found yet.

Focused checks done:

- Kaggle private leaderboard confirms final rank 9 with score 5.83.
- Public leaderboard mirrors map the likely Kaggle username as `amadesouki`, but Kaggle CLI did not return public ARC 2025 notebooks for that user.
- Web search for the exact name plus ARC Prize 2025 did not surface a credible solution writeup.

Action if we revisit:

- Search Kaggle discussion/comments by user handle if exposed in-browser.
- Check if a team member posted code under a different handle.

### 10th - Boiled-Sweet-Potato

No public writeup or notebook found yet.

Focused checks done:

- Kaggle private leaderboard confirms final rank 10 with score 5.69.
- Public leaderboard mirrors map the likely Kaggle username as `boiledsweetpotato`, but Kaggle CLI did not return public ARC 2025 notebooks for that user.
- Web search found leaderboard/profile mirrors but no solution writeup.

Action if we revisit:

- Inspect Kaggle discussion/comments by handle.
- Search after prize winners publish delayed code or a post-competition repository.

### Transferable 2026 Plan From 2025 Recon

Practical plan for our ARC Prize 2026 workspace:

1. Build local evaluator and viewer first.
   - Exact-match two-attempt scoring must be trustworthy before leaderboard submissions.
   - Viewer is essential because many ARC failures are visual and hard to debug from JSON.
2. Reproduce one strong public baseline family.
   - Start from ARChitects-style grid-as-text, LoRA/TTT, DFS/beam, augmentation scoring.
   - Track notebook runtime against the 12-hour Kaggle limit from day one.
3. Add robust candidate selection.
   - Unique candidates, augmentation probability, candidate frequency, and shape confidence should all be logged.
   - Do not submit raw generation order if rescoring is available.
4. Treat output shape as a separate problem.
   - A wrong shape is always zero, so shape prediction or shape-conditioned generation needs its own validation.
5. Run seed/augmentation/runtime sweeps.
   - 2025 writeups repeatedly show seed and small hyperparameters can move exact solves.
   - Public/private divergence means one lucky public score is not enough evidence.
6. Build a synthetic data branch only with strict filters.
   - Generate tasks through programs, not just free-form grids.
   - Filter by consistency and diversity.
   - Track which synthetic source improves which local validation split.
7. Keep a search-and-learn/program-synthesis branch alive.
   - DSL/code approaches are verifiable and can learn from failed attempts.
   - Current open models still need better refinement, but this path looks strategically important for ARC-AGI-2/2026.
8. Preserve exact reproducibility checkpoints.
   - Save notebook/script versions, configs, seeds, public score, local eval score, and runtime.
   - Refactors must prove score parity before replacing a working baseline.
