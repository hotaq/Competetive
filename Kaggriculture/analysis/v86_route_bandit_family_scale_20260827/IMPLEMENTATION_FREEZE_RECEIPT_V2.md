# V86-A Implementation Freeze Receipt V2

This receipt supersedes `IMPLEMENTATION_FREEZE_RECEIPT.md`. It was frozen before any V86 game or V67/B/C candidate economic outcome was generated.

- Collection runner `run_experiment.py`: `30faa31202d938d781413843bf572fa77ade04e4bf45081bf0b884230ca09412`
- Fixed evaluator `evaluate_cv.py`: `4a99d63638e862572f4afb5ba1296ba5a218d4b09e85b663b2f7522ce6a5ccf4`
- Contract receipt `CONTRACT_FREEZE_RECEIPT_V2.md`: `4a5b62e82d0ed671d27095db417c49ae53b0ab917653206a0f2893057080d11c`
- Machine protocol `protocol.json`: `007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30`
- Corrected pre-registration `PRE_REGISTRATION.md`: `875791c2811ffcb1f464b8daeb8959c0626d1341cbed5a8763ccd16cbe4aea6c`
- Family selector `select_families.py`: `79cb474b7a266207d557c69dd2d30a74bf3ac9199fd248dc72d26acf7c0c226f`
- Family selection `family_selection.json`: `9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38`
- Family-selection supplement `FREEZE_SUPPLEMENT_FAMILY_SELECTION.md`: `16e1826923234a049cff74631daba878723050f4878174926a45c39502724060`

Pre-outcome static validation passed: the runner verifies 128 contexts, 384 games, and 17 pinned source/provenance artifacts; both CLIs expose only the frozen collection/evaluation interfaces; Ruff check and format pass; and the evaluator's exact V85 numerical-parity self-test passes. The runner confines output to a direct `dataset*.json` child of the V86 analysis directory. No V86 game, candidate reward/margin, checkpoint, runtime agent, submission, Kaggle upload, or sealed-holdout read occurred before this receipt.
