# Autonomous Agent Prediction

Workspace for the [Kaggle Autonomous Agent Prediction Beta](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta) competition.

## Quick start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Configure Kaggle credentials using the Kaggle CLI instructions.
4. Download the competition files:

   ```bash
   bash scripts/download_data.sh
   ```

5. Inspect the files in `data/raw/` and begin exploration in `notebooks/`.

The competition schema and evaluation metric are intentionally not hard-coded until the official files are downloaded.

## Layout

```text
data/raw/          Downloaded, immutable competition files
data/interim/      Intermediate artifacts
data/processed/    Model-ready datasets
configs/           Experiment configuration
notebooks/         Exploration and experiments
src/               Reusable project code
scripts/           Data and training utilities
tests/             Automated tests
reports/           Results, plots, and submission notes
```

Keep generated data, credentials, model artifacts, and notebook checkpoints out of version control.
