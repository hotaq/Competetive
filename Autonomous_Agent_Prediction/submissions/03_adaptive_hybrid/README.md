# Third Submission: Adaptive Hybrid

Target: move from `0.821` toward the leaderboard-leading `0.829–0.830` range.

## Why this approach

The deterministic portfolio improved the first score from `0.815` to `0.821`,
but public deterministic replications cluster around `0.822`.

The only public payload plausibly associated with the current `0.830` leader is
byte-identical to Kaggle's official Demo Agent. That agent uses
`gemini-3.5-flash` to generate task-specific modeling code during evaluation;
the generated winning code is not preserved publicly.

This package therefore uses the official adaptive architecture while adding:

- A guaranteed prior-probability submission before EDA or model fitting.
- Explicit post-fit status checks and a six-minute selection reserve.
- Native-categorical CatBoost and LightGBM guidance.
- OOF, stability, and prediction-diversity gates.
- A compact leakage-safe feature-engineering skill.
- A requirement to select a valid submission before terminating.

## Package

- Root model: `gemini-3.5-flash`
- EDA sub-agent: `gemini-3-flash-preview`
- Skills: `feature-engineer`, `reliable-fallback`
- Tools: command/file editing, prediction submission, status, final selection,
  and the EDA sub-agent

## Validation

- Package/schema/unit tests: `15 passed`
- Root and nested YAML includes resolve inside the agent directory
- Root and sub-agent models exist in the official `models.yaml`
- Skill names and manifests satisfy package naming rules
- All bundled Python files compile
- Fallback output preserves the official sample-submission schema
- Feature engineering preserves train/test feature alignment and target rows

This is intentionally stochastic. Public code establishes a credible high
ceiling, but no run can guarantee `0.83`.

## Archive

- Path: `submissions/03_adaptive_hybrid/submission.zip`
- SHA-256:
  `dae7f9279d711b224bac2c970df355b2be38e2976227559bb34a1010f5c4158d`
- ZIP integrity: passed
- Credentials or secrets detected: none
- Data and model binaries included: none

Intended upload command:

```bash
kaggle competitions submit autonomous-agent-prediction-beta \
  -f submissions/03_adaptive_hybrid/submission.zip \
  -m "Adaptive hybrid: reliable fallback plus task-specific AutoML"
```

## Research

See [`reports/public_kaggle_code_review_round2.md`](../../reports/public_kaggle_code_review_round2.md)
for the source-cited evidence and limitations.
