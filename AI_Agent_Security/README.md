# AI Agent Security: Multi-Step Tool Attacks

Competition workspace for the Kaggle challenge:
[AI Agent Security: Multi-Step Tool Attacks](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks).

The goal is to build an `AttackAlgorithm` that searches a deterministic,
sandboxed tool-using agent for replayable multi-step security failures.

## Repository layout

```text
.
├── attack.py                 # Submission entry point
├── src/                      # Search strategies and reusable helpers
├── tests/                    # Fast local tests
├── experiments/              # Experiment configs and results
├── notebooks/                # Kaggle submission and analysis notebooks
├── docs/                     # Competition notes and research log
├── scripts/                  # Local evaluation and packaging commands
├── data/
│   ├── raw/kaggle/           # Official Kaggle bundle (not committed)
│   └── processed/            # Derived local artifacts (not committed)
└── submissions/              # Packaged submissions and score records
```

See [docs/competition-overview.md](docs/competition-overview.md) for the
official task summary, constraints, scoring, and timeline.

See [docs/local-validation.md](docs/local-validation.md) for the two-model
Kaggle validation workflow and static-submission strategy.

## Quick start

1. Download the official competition bundle:

   ```bash
   kaggle competitions download \
     -c ai-agent-security-multi-step-tool-attacks \
     -p data/raw/kaggle
   ```

2. Unpack the archive:

   ```bash
   unzip data/raw/kaggle/ai-agent-security-multi-step-tool-attacks.zip \
     -d data/raw/kaggle/sdk
   ```

3. Use the bundled SDK directly for validation and smoke testing:

   ```bash
   PYTHONPATH=data/raw/kaggle/sdk \
     python -m aicomp_sdk.cli.main validate redteam attack.py
   PYTHONPATH=data/raw/kaggle/sdk \
     python -m aicomp_sdk.cli.main test redteam attack.py \
       --budget-s 10 --agent deterministic
   ```

4. Implement stronger search strategies in `src/strategies/` and keep
   `attack.py` as the Kaggle-compatible entry point.

## Competition guardrails

- Keep all attack development inside the provided offline sandbox.
- Do not target live systems or unrelated services.
- Do not commit competition data, secrets, generated traces, or large model
  artifacts.
- Notebook submissions must run with internet disabled and within Kaggle's
  GPU runtime limit.
