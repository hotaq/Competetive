# NVIDIA Nemotron Model Reasoning Challenge - Research Notes

Last researched: 2026-04-30, Asia/Bangkok.

Competition URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

## Current Kaggle Metadata

- Title: NVIDIA Nemotron Model Reasoning Challenge
- Organizer: NVIDIA
- Category: Featured
- Goal: advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark.
- Reward: 106,388 USD
- Metric name: NVIDIA Nemotron Metric
- Enabled: 2026-03-16 22:51:41 UTC
- Final deadline: 2026-06-15 23:59 UTC
- New entrant deadline: 2026-06-08 23:59 UTC
- Team merger deadline: 2026-06-08 23:59 UTC
- Max daily submissions: 5
- Max team size: 5
- Current team count at research time: 2514
- Top public leaderboard score seen via Kaggle CLI: 0.87
- Our account submissions: none found yet.

Kaggle API notes:

```bash
python3 -m kaggle.cli competitions list -s nvidia-nemotron-model-reasoning-challenge
python3 -m kaggle.cli competitions files -c nvidia-nemotron-model-reasoning-challenge
python3 -m kaggle.cli competitions leaderboard nvidia-nemotron-model-reasoning-challenge --show --csv
python3 -m kaggle.cli competitions submissions -c nvidia-nemotron-model-reasoning-challenge
```

## Local Workspace

Created folder:

```text
competetive/NvidiaNemotronModelReasoning/
|-- README.md
|-- research.md
|-- artifacts/
|-- data/raw/
|-- notebooks/
`-- src/
```

Downloaded files:

```text
data/raw/nvidia-nemotron-model-reasoning-challenge.zip  628K
data/raw/train.csv                                      2.9M
data/raw/test.csv                                       1.4K
```

Download command used:

```bash
python3 -m kaggle.cli competitions download \
  -c nvidia-nemotron-model-reasoning-challenge \
  -p competetive/NvidiaNemotronModelReasoning/data/raw \
  --force
```

## Data Profile

`train.csv`

- Shape: 9500 rows, 3 columns
- Columns: `id`, `prompt`, `answer`
- Missing values: none
- Unique IDs: 9500
- Prompt length: min 177, max 510, avg about 301 chars
- Answer length: min 1, max 39, avg about 8 chars

`test.csv`

- Shape: 3 rows, 2 columns
- Columns: `id`, `prompt`
- Missing values: none
- The 3 public test IDs overlap with `train.csv`; treat this as a tiny sample/sanity file, not the real hidden evaluation set.

Category counts inferred from prompt templates:

| Category | Train rows | Answer style |
| --- | ---: | --- |
| bit manipulation | 1602 | 8-bit binary strings |
| gravity | 1597 | decimal distance, usually 2 decimals |
| unit conversion | 1594 | decimal converted value |
| cipher | 1576 | short decrypted English phrase |
| numeral | 1576 | Roman-numeral-like strings |
| symbol transformation | 1555 | short symbol strings |

Example task families:

- Bit manipulation: infer an 8-bit transformation from input/output examples using shifts, rotations, XOR, AND, OR, NOT, majority, or choice-like functions.
- Cipher: infer a substitution/encryption rule from several encrypted-to-plain examples, then decrypt a new phrase.
- Gravity: infer the changed gravitational constant from observations, then compute `d = 0.5*g*t^2`.
- Numeral: convert an integer into the Wonderland numeral system; first look suggests Roman numerals.
- Unit conversion: infer a hidden conversion factor from example measurements.
- Symbol transformation: infer a character-level equation/transformation mapping.

## Submission And Evaluation Notes

Official Kaggle metadata only exposes the metric name through the CLI. Secondary and public challenge notes indicate the final artifact is a LoRA adapter for the Nemotron-3-Nano-30B family, likely packaged as `submission.zip`, with generated answers extracted from a final-answer format such as `\boxed{...}`.

Before submitting, re-check the Kaggle Overview, Evaluation, and Rules pages directly because the hard constraints matter:

- exact base model name
- LoRA rank cap
- required zip layout
- allowed external data
- required public notebook/writeup for prizes
- exact final-answer extraction and numeric tolerance behavior

## Initial Strategy

High-confidence direction: this is not ordinary CSV prediction. The visible CSV is a supervised puzzle corpus, while the hidden evaluation appears to score a submitted reasoning-capable adapter.

Good first branch:

1. Build deterministic parsers and local solvers for each of the six prompt families.
2. Use train rows to create trusted reasoning traces: prompt, parsed rule, solved answer, final answer.
3. Make a local verifier that mimics answer extraction: exact string for symbolic/text/binary answers, numeric tolerance for decimal answers.
4. Fine-tune a LoRA on solver-generated traces and require a stable final answer wrapper.
5. Track validation by category, especially bit manipulation and symbol transformation, because public auxiliary notes say these are high-value/hard areas.

Fast baseline branch:

1. Split train by ID into local train/valid.
2. Implement easy solvers first: numeral, gravity, unit conversion, simple cipher.
3. Use those solved families to generate clean SFT examples.
4. Keep bit manipulation and symbol transformation as a separate hard branch with brute-force rule search.

Public artifacts worth reading next:

- Progress Prize repo: https://github.com/tonghuikang/nemotron
- Public SFT trace dataset: https://huggingface.co/datasets/andy279/nemotron-reasoning-challenge
- NVIDIA Nemotron docs: https://docs.nvidia.com/nemotron/latest/nemotron/super3/sft.html

Do not import external traces into training until rules are checked. They are useful immediately for technique mining: category taxonomy, trace format, solver-guided data generation, answer cleaning, and validation design.

## Kaggle Code / Models / Discussion Mining - 2026-04-30

Sources checked:

- Code tab, sorted by votes: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code
- Models tab: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models
- Discussion tab, sorted by votes: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion?sort=votes

Local copies of the most useful public artifacts were saved under:

```text
references/kaggle_code/
references/kaggle_discussion/
references/kaggle_models/
```

### Best Starting Lane

Start from the DGXChen / Tong / Huikang SFT reproduction lane, then add solver-generated traces. Do not start with GRPO or a generic baseline classifier.

Recommended first solution path:

1. Use Ryan Holbrook's submission demo only for the required adapter zip layout and base packaging pattern.
2. Use DGXChen's Unsloth notebook as the most practical copy-and-run training baseline for a public 0.84 to 0.85 style solution.
3. Use Tong/Huikang's 0.85+ notebooks and progress-prize discussion for the real winning idea: deterministic solver traces, SFT only, rank-32 LoRA, long context, and a stable final-answer format.
4. Use Konbu, Mohan, and Kalyan notebooks to build or verify the puzzle-family solvers that generate clean CoT/SFT examples.
5. Validate every adapter with Huikang's adapter-validation notebook before any Kaggle submission.

Why this lane:

- The visible data is a six-family puzzle corpus, while the hidden scoring flow evaluates a Nemotron LoRA adapter.
- Top public evidence says the hard score gains come from better reasoning traces, especially bit manipulation, not from changing the base model.
- Public GRPO discussion indicates GRPO is much slower than SFT for this setup. It can be a later branch, not the first branch.
- The final answer wrapper, usually `\boxed{...}`, is not cosmetic. It is part of reliable metric extraction.

### High-Signal Public Code

| Notebook | Role for us | Key takeaways |
| --- | --- | --- |
| https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo | Submission packaging baseline | Uses official base model path, rank-32 LoRA, adapter zip generation. Use this to avoid zip/layout mistakes. |
| https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb | Best practical starter training run | Reproduces Tong-style SFT around 0.84 to 0.85 LB with Unsloth; rank 32, alpha 32, max length 8192, SFTTrainer, boxed answer prompt. |
| https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85 | Strong solution reference | Shows the 0.85+ lane: generated/tokenized trace data, masked loss, Mamba fast path patch, manual `lm_head` LoRA handling, fp32 LoRA params, bf16 base. |
| https://www.kaggle.com/code/huikang/tinker-submission-notebook | Tinker adapter conversion | Converts a Tinker-trained adapter into Kaggle-ready format and handles tensor/key conversion details. Useful after training elsewhere. |
| https://www.kaggle.com/code/huikang/adapter-validation-notebook | Pre-submit validator | Loads `submission.zip`, base model, and LoRA, extracts boxed answers, and reports per-category accuracy. This should be our gate before submission. |
| https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-copy-run-instantly | Simple starter training | Easier training template with offline packages and LoRA zip checks; useful for smoke tests, weaker than DGXChen/Tong lane. |
| https://www.kaggle.com/code/konbu17/bit-manipulation-solver-cot-generator | Bit solver and CoT generator | Important for the hardest category. Uses bit/byte transforms, per-bit candidates, brute-force scoring, and CoT generation. |
| https://www.kaggle.com/code/mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers | Rule solver overview | Good taxonomy and easy-family solvers. Reports near-perfect visible performance for gravity, unit, and numeral; bit and symbol/cipher remain differentiators. |
| https://www.kaggle.com/code/kalyankkr/all-6-puzzle-types-decoded-sft-training-data | Training data preparation | Good for category decoding and balanced SFT JSONL construction; more of a data-prep reference than a top scorer. |

Core baseline settings to try first:

```text
base model: NVIDIA/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 or official Kaggle mirror
LoRA rank: 32
LoRA alpha: 32
LoRA dropout: 0.0 for Tong/DGXChen lane
max sequence length: 8192
learning rate: 2e-4
objective: SFT on solver-style reasoning traces
answer format: final answer inside \boxed{...}
```

DGXChen-style practical settings:

```text
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
num_train_epochs: 1
bf16: true
gradient_checkpointing: true
packing: false
```

### Useful Public Models

| Model | Use | Notes |
| --- | --- | --- |
| https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained | Inspectable public adapter family | Metadata says rank fixed to eval constraints, alpha aligned with rank, target modules include `in_proj`, `x_proj`, `dt_proj`, `out_proj`, final answer inside `\boxed{}`. Good for config mining or a quick packaging baseline. |
| https://www.kaggle.com/models/atahalam/nvidia-nemotron-model-reasoning-30b-a3b-lora-0-80 | Sparse public LoRA | Exists as a 0.80-ish adapter reference, but metadata is thin. Do not anchor the main branch on it. |
| Official Kaggle/base mirrors | Required base | The adapter must match the expected Nemotron 3 Nano 30B A3B base and rank constraints. Re-check rules before packaging. |

### Top Discussion Takeaways

High-signal threads saved locally:

```text
689915 - Open Progress Prize Publication: SFT to maximize minimum logprob
688461 - Answers To Everything Data: Read Me! 100% Solve Rate
690307 - Strategy to solve 85% of bit manipulation
690161 - Why GRPO is Painfully Slow on Nemotron
687961 - rank-32 LoRA memory discussion
683853 - metric bug for binary string
684212 - base model visualization and completions
683172 - Kaggle CLI and RTX Pro 6000 kernel workflow
```

Main conclusions:

- The leading public solution family is SFT, not RL-first. Tong/Huikang describe a deterministic CoT design that tries to maximize the weakest token logprob.
- Bit manipulation is the biggest practical differentiator. The better public approaches solve bits as boolean relations or transformed bit streams instead of brute-forcing full natural-language expressions.
- Some public comments claim near-complete reverse-engineered solvers, but they are not all independently verified. Treat them as technique leads, not ground truth.
- The binary-string metric has a reported parsing issue in discussion. Our local validator should still compare binary strings exactly so we do not train toward a public-metric accident.
- External traces or LLM-generated training data may be useful, but rules and allowed-data boundaries must be rechecked before training on them.
- The old PyPI Kaggle CLI is enough for research/downloads here. For GPU kernel push/runs, the discussion recommends the newer GitHub Kaggle CLI on Python 3.11 because accelerator flags matter.

### First Sprint Plan

1. Create a local category parser and answer verifier.
   - exact match for binary, cipher, numeral, and symbol answers
   - numeric tolerance only for gravity and unit conversion
   - boxed-answer extraction matching the public validation notebook
2. Build easy-family deterministic solvers first.
   - gravity
   - unit conversion
   - numeral
   - simple cipher and symbol cases
3. Build the bit-manipulation solver as the main high-value branch.
   - start from Konbu/Tong-style bit transforms
   - score candidate transforms on examples
   - generate deterministic bit-level traces for SFT
4. Produce `artifacts/train_sft_solver.jsonl` with prompt, reasoning, and boxed answer.
5. Train the first rank-32 LoRA using the DGXChen/Tong SFT settings.
6. Validate the adapter with Huikang's validation notebook before any submission.

Decision for now:

```text
Default baseline = solver-trace SFT adapter.
Starter notebook = DGXChen 0.85 Unsloth reproduction.
Packaging reference = Ryan submission demo.
Validation gate = Huikang adapter-validation notebook.
Main improvement target = bit manipulation trace quality.
Avoid first = GRPO, external API distillation, generic classifiers.
```
