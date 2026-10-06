# Kaggle Discussion Recon - 2026-05-04

Source: Kaggle internal discussion API using `~/.kaggle/kaggle.json` auth plus XSRF cookie.

Fetched files:

- `discussions_full.json`
- One JSON per topic id in this directory.

## Topics Read

- `696012` - Tire compound changes during a race. Compounds determine the tire's "Personality".
- `696380` - Inconsistencies in the "original" dataset
- `696227` - F1 nerds, watch out because data doesn't make sense
- `696667` - Is there a Covariate shift?
- `696769` - Is this distribution correct, or am I missing something? (Yearly inconsistency in PitNextLap)
- `696347` - CT boost for binary classification.

## Takeaways

1. Compound domain signal matters, but use it as tabular interaction features rather than strict F1 physics.
   - Hard / medium / soft / intermediate / wet encode durability and degradation regimes.
   - Dry races require at least two tire compounds, but wet races do not.
   - Practical feature direction: compound-normalized tyre age, compound x race, compound x stint, compound x race progress.

2. Several users found that the original/source dataset and the competition dataset violate time-series consistency expectations.
   - Claimed consistency checks include `PitNextLap_i == 1` iff `PitStop_{i+1} == 1`, monotonic stint progression, and sensible `LapTime_Delta`.
   - Multiple examples suggest `Stint`, `TyreLife`, and `LapNumber` are not reliable as a coherent driver-race sequence.
   - Practical modeling implication: do not over-trust sequence reconstruction, lag features, or strict race simulation logic.

3. Treat rows as tabular samples first.
   - One discussion explicitly suggests row-independent modeling because driver/race time series may be corrupted.
   - Our starter has some group aggregate features, but future changes should compare them against row-level encoding to avoid adding noisy sequential assumptions.

4. Year `2023` has a very different target distribution.
   - Verified locally:
     - 2022 positive rate: `26.650520%`
     - 2023 positive rate: `0.960726%`
     - 2024 positive rate: `29.531902%`
     - 2025 positive rate: `28.438866%`
   - Practical modeling implication: include `Year` and year interactions, and inspect fold distributions. Random row CV may hide year-specific behavior.

5. Adversarial validation alone is not enough.
   - One post reports train-vs-test adversarial AUC around `0.498`, suggesting no obvious covariate shift.
   - A reply correctly warns this only tests `P(X)` alignment, not whether `P(Y|X)` is stable.
   - Practical modeling implication: still trust OOF only cautiously, especially with synthetic target generation and the 2023 anomaly.

6. CTBoost exists as an experimental option.
   - A discussion points to a `ctboost` GPU starter, but it is slow and a new dependency.
   - Given the repo rule against new dependencies without explicit request, keep this as a research lane, not the local default.

## Immediate Next Model Direction

- Keep CatBoost as the local default.
- Add leak-free target encodings for `Driver_Race_Year`, `Driver_Race`, `Race_Year`, `Driver_Compound`, and `Race_Compound`.
- Add fold reports by `Year`, especially 2023.
- Consider reducing sequential features or making an ablation switch because discussion evidence says full time-series consistency is suspect.
