# First Submission: Reliable Baseline

Archive: `submission.zip`

SHA-256:

```text
a7745604021526377d4dbeb21dcd854111c3482857454f9f5a8998d87b364a50
```

## Contents

- Root `agent.yaml` using `gemini-3.1-flash-lite`
- Reliability-first system prompt
- `reliable-baseline` bundled skill
- Constant-prior fallback submission script
- Cross-validated CatBoost training script with a scikit-learn fallback

The archive contains no datasets, credentials, API keys, notebook outputs,
model binaries, or unrelated project files.

## Validation evidence

- Offline package-contract test suite: 5 passed
- Python syntax validation: passed
- Fallback submission schema check: passed
- Modeled baseline executed successfully on all 16 supplied training tasks
- Every generated prediction file had 10,000 rows
- ZIP integrity test: passed
- `agent.yaml` is at the ZIP root
- No symlinks, bytecode, cache directories, data, or credentials are included

Official ADK compilation and end-to-end Docker evaluation were not run because
the dependency installation was interrupted and the local Docker daemon was not
running.

## Intended upload command

```bash
kaggle competitions submit autonomous-agent-prediction-beta \
  -f submissions/01_baseline/submission.zip \
  -m "First reliable baseline: fallback plus deterministic CatBoost/sklearn agent"
```

## Kaggle result

- Submission reference: `54993344`
- Uploaded: 26 July 2026 at 06:18:48 UTC
- Status: `COMPLETE`
- Public score: **0.815**
- Private score: not yet available
