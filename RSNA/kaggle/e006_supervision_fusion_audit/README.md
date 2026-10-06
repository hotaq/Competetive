# E006 — supervision fusion audit

CPU-only Kaggle experiment that audits three independent public report-label sources
against the 58 official image-adjudicated studies. Steven v4 is retained as a frozen
external comparator because it already blends Steven v2 with Lixin.

The notebook pins all CSVs by SHA-256, uses repeated out-of-fold target-specific
reliability weighting, emits paired-bootstrap diagnostics, and never submits to the
leaderboard.

Build locally:

```bash
python build_notebook.py e006_audit.py e006-supervision-fusion-audit.ipynb
```

Promotion requires the predeclared gate in the notebook introduction. A passing label
audit only authorizes an EfficientNet-B3 scout; it is not an image-model score.
