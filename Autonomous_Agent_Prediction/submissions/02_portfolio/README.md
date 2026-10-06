# Second Submission: Robust Portfolio

Target: improve the first public score from `0.815` toward `0.83`.

## Strategy

- Submit a constant-prior fallback before modeling.
- Train two CatBoost configurations when CatBoost is available.
- Train LightGBM when LightGBM is available.
- Always train ExtraTrees, histogram gradient boosting, and logistic regression.
- Add an RBF SVM on datasets with at most 2,500 training rows.
- Build OOF-ranked pair, top-three, and top-four rank blends.
- Export at most six modeled candidates.
- Submit candidates in OOF order and select the best observed public score.

## Portable backtest

The portable backtest excludes CatBoost and LightGBM because those optional
packages are not installed locally. It compares the best sklearn portfolio
candidate against the first agent's sklearn fallback on every supplied task.

| Task | First agent OOF AUC | Portfolio OOF AUC | Gain |
| --- | ---: | ---: | ---: |
| train_01 | 0.70744 | 0.71454 | +0.00710 |
| train_02 | 0.96063 | 0.96035 | -0.00029 |
| train_03 | 0.79233 | 0.80837 | +0.01604 |
| train_04 | 0.82345 | 0.82730 | +0.00384 |
| train_05 | 0.65307 | 0.68914 | +0.03608 |
| train_06 | 0.76498 | 0.81060 | +0.04563 |
| train_07 | 0.81409 | 0.81669 | +0.00260 |
| train_08 | 0.84088 | 0.85430 | +0.01343 |
| train_09 | 0.60123 | 0.64203 | +0.04080 |
| train_10 | 0.83007 | 0.83311 | +0.00304 |
| train_11 | 0.80844 | 0.81716 | +0.00872 |
| train_12 | 0.78704 | 0.78711 | +0.00007 |
| train_13 | 0.62289 | 0.66202 | +0.03913 |
| train_14 | 0.80051 | 0.80230 | +0.00179 |
| train_15 | 0.79757 | 0.84490 | +0.04733 |
| train_16 | 0.87494 | 0.88619 | +0.01126 |

- Mean first-agent OOF AUC: `0.77997`
- Mean portfolio OOF AUC: `0.79726`
- Mean gain: `+0.01728`
- Tasks improved: `15 / 16`

These are local OOF results, not a guarantee of a Kaggle public score. The
evaluation sandbox's CatBoost and LightGBM candidates and per-episode public
candidate selection are expected to add diversity beyond this portable test.

## Validation

- Package contract and schema tests: `10 passed`
- Portfolio script executed successfully on all 16 supplied tasks
- Candidate outputs preserve submission columns, IDs, row counts, and finite
  probability values

## Archive

- Path: `submissions/02_portfolio/submission.zip`
- SHA-256:
  `a1de688dffe257a319b55012e501534807201f41b1a147481ee5c8e1e125543b`
- Contains only the agent YAML, prompt, skill manifest, and two Python scripts
- Contains no data, credentials, model binaries, or unrelated project files

Intended upload command:

```bash
kaggle competitions submit autonomous-agent-prediction-beta \
  -f submissions/02_portfolio/submission.zip \
  -m "Robust OOF portfolio with guarded candidate selection"
```

## Kaggle result

- Submission reference: `55019885`
- Uploaded: 27 July 2026 at 06:00:33 UTC
- Status: `COMPLETE`
- Public score: **0.821**
- Change from first submission: `+0.006`
- Private score: not yet available
- Outcome: improved, but below the approximately `0.83` target
