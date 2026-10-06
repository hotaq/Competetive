# Local validation workflow

Source discussion:
[Validate Your Attack Locally to Save Submission Time](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/708186),
published 2026-06-14.

Linked notebook:
[AAS Local Validation](https://www.kaggle.com/code/llkh0a/aas-local-validation).

The notebook source and metadata are downloaded under:

```text
notebooks/aas-local-validation/
```

## Main idea

Run attack discovery and replay validation before making a competition
submission. Use the two competition target models and the public guardrail to
inspect how each prompt chain behaves. Once strong prompt chains are known,
make the final `AttackAlgorithm` return static `AttackCandidate` objects
immediately instead of spending submission time calling `env.interact()`.

This separates the workflow into:

1. **Discovery attack:** explores the environment and records candidates.
2. **Local replay validation:** evaluates candidates against GPT-OSS, Gemma,
   and `OptimalGuardrail`.
3. **Static submission:** returns only the selected replayable message chains.

## Competition-matched settings

The notebook reads these values from the bundled gateway:

| Setting | Value |
|---|---|
| Budget per model | 9,000 seconds |
| Maximum tool hops | 8 |
| Attack seed | 123 |
| Environment | Gym |
| Models | `gpt_oss`, `gemma` |
| Public guardrail | `OptimalGuardrail` |

This resolves the overview-page runtime inconsistency in favor of the SDK
gateway's 9,000-second value.

## Notebook execution

The notebook:

- Attaches GGUF versions of GPT-OSS 20B and Gemma 4 26B-A4B-it
- Installs `llama-cpp-python` when missing
- Loads one model at a time to control GPU memory
- Calls `evaluate_redteam` using the competition fixtures and Gym environment
- Saves transcripts, framework events, and agent-debug JSONL
- Reports raw score, normalized score, findings, unique cells, canonical
  cells, runtime, guardrail version, and environment
- Provides `inspect_prompt_chain(messages)` for detailed manual trace review

The Gemma model requires accepting Google's model-access agreement in Kaggle.

## Original correlation result

The discussion author reported:

```json
{
  "gpt_oss_public": 0.27,
  "gemma_public": 0.24,
  "local_public_mean": 0.255
}
```

The corresponding public leaderboard score was `0.24`.

This was evidence that local validation was directionally useful at the time,
not a permanent guarantee.

## Important caveats

- The post predates multiple evaluator updates.
- In July 2026, participants reported local results corresponding to public
  scores around 54–59 rather than exact matches.
- The notebook author stated that they had not revalidated the notebook after
  the competition update.
- Submission replay uses a remote GPU pool, so local Kaggle T4 runtime does
  not predict submission wall-clock time exactly.
- Local validation only exposes the public guardrail and public fixtures.
- Hardcoded public secret values or fixture IDs are not private-leaderboard
  robust.

Treat local scores as a ranking and debugging signal. Always keep a safety
margin for runtime and validate transfer across both models.

## Recommended project workflow

### Discovery

Run the content-guided search implementation and save:

- Candidate user-message chains
- Public predicate findings
- Official and semantic cell hashes
- Model name
- Replay success
- Tool sequence
- Runtime cost

### Candidate filtering

Retain candidates that:

- Replay successfully from clean state
- Work on both target models when possible
- Add a new canonical cell or high-severity finding
- Stay comfortably below message and runtime limits
- Do not depend only on public secret values

### Static submission

Generate a self-contained `attack.py` whose `run` method does not explore:

```python
class AttackAlgorithm(AttackAlgorithmBase):
    def run(self, env, config):
        del env, config
        return [
            AttackCandidate.from_messages(chain)
            for chain in VALIDATED_CHAINS
        ]
```

The final file should embed or otherwise package every required chain and
dependency. It must not assume that local `src/` modules are available unless
the submission notebook writes those modules into `/kaggle/working/`.

## Next adaptation steps

1. The current `attack.py` and `src/` package are embedded in our private
   validation notebook.
2. Both GGUF models and Gemma access are confirmed.
3. One-step T4 smoke tests pass for GPT-OSS and Gemma.
4. Expand tool hops and search steps gradually while inspecting traces.
5. Run the full 9,000-second budget only after bounded scoring tests pass.
6. Export successful candidates into a compact versioned candidate manifest.
7. Generate and validate a self-contained static `attack.py`.

Measured smoke-test results and the proven T4 configuration are documented in
`docs/t4-smoke-test-results.md`.
