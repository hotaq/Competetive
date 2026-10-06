# Meta V16-RC5 Robust Base

This directory is the locally adopted robust baseline for the next Kaggriculture
policy generation.

## Policy roles

- Default robust base: V16-RC5 8-cow/4-sheep route with weed recovery and
  premium-market front-running.
- Specialist reference: `top1_v12_ryo_selector_b120_repaired_standalone`.
- The specialist is not safe to splice into this route without a state-compatible
  takeover because their pasture, animal, worker, and inventory geometries differ.

## Provenance

- Public Kaggle notebook:
  <https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead>
- Pulled kernel output:
  `artifacts/kaggle_notebooks/boatlee_v16_rc5/output/main.py`
- Source and local `main.py` SHA-256:
  `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19`

The local `main.py` is an exact extraction from the notebook's public
`submission.tar.gz`; no policy behavior was changed during adoption.

## Evidence at adoption

- Fixed-opponent public replay diagnostic: full V16-RC5 won 11/11; the open-loop
  core alone won 9/11.
- Live Kaggle environment against our v12 specialist: 18/24 wins over 12
  seat-swapped seeds, mean margin +8,169.7. All 24 games completed normally.
- Ryo development replay diagnostic: V16-RC5 won 69/96, while v12 won 89/96.

This is therefore the validated local broad-meta base, not a claim that it
dominates every seed or every Ryo matchup. It has not been submitted from this
directory.
