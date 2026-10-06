# Leakage Checklist

- Never use `target` or a target-derived statistic as a feature for the same row.
- Fit imputers, encoders, scalers, bins, and feature selection inside each CV
  training fold.
- Apply each fold's fitted transformation unchanged to its validation rows.
- Fit final inference transformations on training data only.
- Do not derive features from `sample_submission.csv` predictions.
- Exclude identifier columns unless validation proves a stable relationship.
- Treat public scores as noisy tie-breakers, not training labels.
