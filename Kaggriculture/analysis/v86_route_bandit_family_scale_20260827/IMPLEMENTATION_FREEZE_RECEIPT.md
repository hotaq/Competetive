# V86-A Implementation Freeze Receipt

Frozen before any V86 game or economic outcome was generated.

- Collection runner `run_experiment.py`: `7214a9166596cdc00bfef184ccb3856db22c33af894607a524c8a36510899e5c`
- Fixed evaluator `evaluate_cv.py`: `4a99d63638e862572f4afb5ba1296ba5a218d4b09e85b663b2f7522ce6a5ccf4`
- Machine protocol `protocol.json`: `007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30`
- Pre-registration `PRE_REGISTRATION.md`: `43b336348d5c292f9e7f3b294c3dc4abff7b30f6fec37ed8607ef935d981a8bf`
- Family selector `select_families.py`: `32bec44e1c24fbb7c86ef73fb0dcd378c6c91bc21ade07fc155779fda9c64e77`
- Family selection `family_selection.json`: `80e055b41291562053c30ed96e2f4dcd89363b5080c669c3c24335e639683b63`
- Family freeze supplement: `3afbb444990843b426274936e9c281ff73d2e757507cab9b09ec494b4bfbed18`

Pre-outcome static validation passed: both CLIs expose only the frozen collection/evaluation interfaces, Ruff check and format pass, and the evaluator's exact V85 numerical-parity self-test passes. No V86 game, reward, margin, checkpoint, runtime agent, submission, Kaggle upload, or sealed-holdout read occurred before this receipt.
