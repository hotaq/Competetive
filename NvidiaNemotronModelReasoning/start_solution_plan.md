# NVIDIA Nemotron Reasoning - Start Solution Plan

Date: 2026-04-30

## Recommended Start

Build a solver-trace SFT adapter. The best public starting point is the DGXChen/Tong/Huikang SFT lane, with Ryan Holbrook's notebook only as the submission-packaging reference.

Do not start with:

- generic CSV prediction
- GRPO
- blind LoRA training without solver traces
- external LLM distillation before rules are checked

## Starting Stack

```text
base model: Nemotron-3-Nano-30B-A3B-BF16
adapter: LoRA
rank: 32
alpha: 32
dropout: 0.0
max length: 8192
learning rate: 2e-4
objective: SFT
answer wrapper: \boxed{...}
```

## Why This Is The Best First Branch

Public code and discussion evidence points to one consistent pattern: high scores come from clean deterministic reasoning traces, especially for bit manipulation, then SFT into the required Nemotron adapter format.

The first useful submission should therefore prove this pipeline:

```text
parse prompt -> solve or infer rule -> generate CoT trace -> train LoRA -> validate boxed answer extraction -> package submission.zip
```

## Implementation Order

1. Verifier
   - classify prompt category
   - extract final `\boxed{...}` answers
   - exact-match binary/text/symbol answers
   - numeric tolerance only for decimal answers

2. Easy solvers
   - gravity
   - unit conversion
   - numeral
   - simple cipher/symbol cases

3. Bit solver
   - transformed bit streams
   - per-bit boolean functions
   - candidate scoring against examples
   - deterministic trace generator

4. SFT data
   - write JSONL with prompt, reasoning, and boxed final answer
   - keep per-category counts balanced
   - hold out local validation rows

5. First LoRA
   - use DGXChen/Unsloth settings as the first runnable kernel
   - rank 32, alpha 32, max length 8192
   - validate with Huikang adapter-validation notebook

## Kaggle RTX Pro 6000 Kernel

Starter kernel package:

```text
kaggle_rtx6000_sft/
|-- kernel-metadata.json
`-- train_rtx6000_sft.py
```

Push command:

```bash
scripts/push_kaggle_rtx6000.sh
```

The helper uses:

```bash
kaggle kernels push -p kaggle_rtx6000_sft --accelerator NvidiaRtxPro6000
```

Target Kaggle kernel:

```text
chinnaphatkhun/notebooke4b8ccf598
https://www.kaggle.com/code/chinnaphatkhun/notebooke4b8ccf598/edit
```

## Reference Roles

| Reference | Use |
| --- | --- |
| Ryan Holbrook submission demo | zip layout and adapter packaging |
| DGXChen 0.85 Unsloth notebook | first runnable training baseline |
| Huikang end-to-end fine-tuning | strong solution design and adapter details |
| Huikang validation notebook | pre-submit validation gate |
| Konbu bit solver | bit manipulation trace generation |
| Mohan/Kalyan decoded tasks | puzzle taxonomy and easy-family data prep |

## First Success Criteria

- local verifier reports per-category accuracy
- solver-generated SFT JSONL exists
- rank-32 LoRA trains without adapter-layout errors
- validation notebook can load `submission.zip`
- generated final answers consistently use `\boxed{...}`
