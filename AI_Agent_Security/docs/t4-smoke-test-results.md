# Kaggle T4 smoke-test results

Private notebook:
<https://www.kaggle.com/code/chinnaphatkhun/ai-agent-security-local-validation>

Run date: 2026-07-26

## Outcome

Both competition target models successfully loaded and completed a one-step
Gym evaluation on a Kaggle T4 GPU using the embedded project attack package.

| Model | Kernel version | Model load | Evaluation | Findings | Result |
|---|---:|---:|---:|---:|---|
| GPT-OSS 20B | 4 | 94.1s | 16.51s | 0 | Complete |
| Gemma 4 26B-A4B-it | 5 | 117.8s | 28.26s | 0 | Complete |

Both runs used:

- `OptimalGuardrail` public version 1
- Gym environment
- Seed 123
- One search step
- One tool hop
- No runtime, transcript, or framework errors

The zero score is expected. These were model-access and evaluator-compatibility
smoke tests, not scoring experiments.

## Proven T4 profile

```text
llama-cpp-python: 0.3.34
n_ctx: 2048
n_gpu_layers: 20
max_new_tokens: 256
n_batch: 128
n_ubatch: 64
flash_attn: false
offload_kqv: false
GGML_CUDA_FORCE_MMQ: 1
```

## Failed configurations and lessons

### Version 1

Failed before model loading because the notebook imported an SDK CLI helper
that is not publicly exported in SDK 3.1.2.

Resolution: validate syntax, imports, and `AttackAlgorithmBase` inheritance
directly.

### Version 2

Loaded GPT-OSS with the SDK defaults (`n_ctx=8192`, full GPU offload), then
failed inside llama.cpp CUDA at the 60-second evaluation deadline.

### Version 3

The conservative T4 profile loaded successfully, but the 30-second evaluator
budget expired while the first inference was still running.

Resolution: use a 300-second smoke allowance even for a one-step run. The
actual successful evaluation took only 16.51 seconds for GPT-OSS and 28.26
seconds for Gemma after model loading, but shorter evaluator deadlines can
interrupt the first inference.

## Next safe expansion

Increase one dimension at a time:

1. Raise tool hops from 1 to 2 with one search step.
2. Confirm both models still complete.
3. Raise search steps to a small fixed number.
4. Inspect traces and candidate counts.
5. Only then run long discovery or replay validation.

This avoids confusing model-loading, inference-time, memory, search, and replay
failures.

