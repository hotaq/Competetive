# V59/V67 route-oracle analysis on exposed development160

Date: 2026-08-26

Status: post-unblind development analysis only. No policy was changed, no new holdout was opened, and nothing was uploaded.

## Question

How much headroom is available from choosing among the fixed `base`, `a120`, `b120`, and `c120` schedules, can observable state before/at step 120 select the right schedule without team or episode identity, and can a small set of existing distinct routes rescue the remaining failures?

## Ranked synthesis

| Rank | Finding | Confidence | Direct basis |
|---:|---|---|---|
| 1 | Route choice is the main measured source of headroom, but the four-route family is not sufficient for the target. | High | Actual V59/V67 are both 106/160. The hindsight four-route oracle is 134/160. |
| 2 | The V67 front-run overlay is not the reason the oracle improves. | High | V59 and V67 four-route oracles both win 134/160; V59 has the better mean and CVaR. |
| 3 | Observable state contains route signal, especially shared prices and money/inventory state, but current linear LOFO selection extracts only a small fraction of the oracle gap. | Medium-high | Best V59 LOFO result is 108/160; best V67 result is 107/160, far below 134/160. |
| 4 | Six additional existing experts explain 14 more recoverable failures. | High for this development set | Targeted expanded oracle reaches 148/160; a greedy six-expert rescue portfolio covers all 14 added wins. |
| 5 | The exposed set still does not support a decisive promotion claim. | High | Expanded hindsight ceiling is 92.5%, with 12 residual losses; fourth targeted screen adds zero wins. |

## Four-route results

All eight fixed family/route combinations ran on the same 160 keys.

| Family | base | a120 | b120 | c120 | hindsight oracle | Oracle mean | Oracle worst | Oracle CVaR10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V59 | 100 | 24 | 91 | 64 | 134/160 | +11,519.39 | -15,748 | -4,596.00 |
| V67 | 100 | 24 | 94 | 64 | 134/160 | +11,008.91 | -15,718 | -4,989.94 |

The oracle route counts are:

- V59: `base=76`, `a120=4`, `b120=29`, `c120=51`.
- V67: `base=75`, `a120=6`, `b120=32`, `c120=47`.

The frozen V59 selector used only `base=139` and `c120=21` on these rows. Reconstructing that decision from observable features produced exact 160/160 margin parity with both actual V59 and actual V67. This validates the fixed-route counterfactual wiring.

V67 changes the static `b120` result from 91 to 94 wins, but it does not increase oracle wins and reduces oracle mean margin by about 510.48. Evidence therefore supports treating the overlay as a local close-out effect, not the broad source of route headroom.

## Important rescues from the original four routes

The four-route V67 oracle can reverse five deep actual V67 losses:

| Team / episode | Actual V67 | Best route | Counterfactual margin |
|---|---:|---|---:|
| Crop Dusta / 99890702 s0 | -30,337 | a120 | +27,217 |
| Mingkang YAN / 100084756 s1 | -31,605 | b120 | +13,687 |
| Subramanya N / 99920421 s0 | -52,486 | c120 | +335 |
| peikopon / 100134962 s0 | -26,928 | c120 | +995 |
| tyz123456 / 100134962 s1 | -25,383 | c120 | +3,306 |

`c120` is the clearest Yarn/Wool-heavy schedule in the original family: after step 120 its trace contains 464 units of `SELL:WOOL` and six `BUY_ANIMAL:SHEEP` orders, alongside a large wheat buy/sell loop.

## Observable selection at/before step 120

The diagnostic selector used only observable economy state: player seat, clock, candidate/opponent money, shared prices, private shed, seeds, and unit inventory. It did not use team or episode identity.

| Family | Feature history | Best LOFO wins | Worst | CVaR10 | Choice counts |
|---|---|---:|---:|---:|---|
| V59 | pre-120: 24,48,72,96 | 106 | -52,353 | -20,429.63 | base139 / b1 / c20 |
| V59 | step 120 only | 106 | -31,739 | -17,389.75 | base134 / b1 / c25 |
| V59 | pre + at 120 | **108** | -31,739 | -18,731.31 | base113 / b18 / c29 |
| V67 | pre-120: 24,48,72,96 | 106 | -52,486 | -21,605.94 | base134 / b4 / c22 |
| V67 | step 120 only | **107** | -31,605 | -18,399.13 | base134 / b2 / c24 |
| V67 | pre + at 120 | 107 | -31,605 | -19,107.25 | base122 / b12 / c26 |

Strongest route-discriminating associations in the V59 development fit are:

1. Wool price at steps 96/120.
2. Milk price at steps 96/120.
3. Strawberry and fertilizer prices around steps 96/120.
4. Money difference and opponent money at steps 24/120.
5. Wheat shed/seed state for the `b120` and `c120` advantages.

The largest univariate relation is `c120` advantage versus Wool price at step 120 (`r=+0.447`), followed by Milk (`r=-0.377`) and Strawberry (`r=-0.341`). For `b120`, early candidate money and Wheat shed/seed state are the leading associations, but all are weaker (`|r|<=0.209`). These are associations, not causal route rules.

## Expanded expert screen

Only the residual losses at each stage were evaluated. This makes the win count exact over the screened schedules while keeping the experiment bounded; mean-margin statistics are conservative because added experts were not run on already-winning rows.

| Stage | Experts | Rows per expert | Entry wins | Exit wins | Gain |
|---|---:|---:|---:|---:|---:|
| Existing portfolio routes | 6 | 26 | 134 | 144 | +10 |
| Time-aware behavior medoids | 8 | 16 | 144 | 146 | +2 |
| Neighbors of rescuing route 99598660 | 8 | 14 | 146 | 148 | +2 |
| Siblings/neighbors of 99564516 and 99594060 | 8 | 12 | 148 | 148 | +0 |

Expanded targeted oracle:

- Wins: **148/160 (92.5%)**.
- Mean margin lower bound: **+12,149.26**.
- Worst margin: **-7,816**.
- CVaR10: **-2,132.31**.

The minimal greedy rescue portfolio from observed targeted rows contains six added experts:

| Route | New rescues | Broad trace character |
|---|---:|---|
| 99587277 | 7 | Strawberry/Milk/Fertilizer with Cow-oriented mixed production |
| 99605418 | 2 | Fertilizer/Wool/Strawberry, Sheep-oriented |
| 99598660 | 2 | Strawberry/Fertilizer/Wool, Sheep-oriented |
| 99612231 | 1 | Fertilizer/Milk/Strawberry mixed production |
| 99564516 | 1 | Large Wheat product loop plus Fertilizer/Wool |
| 99594060 | 1 | Wool/Fertilizer/Strawberry, Sheep-oriented |

The final zero-gain stage is the bounded stop condition. The remaining population was not brute-forced.

## Residual 12 losses

| Team | Episode / seat | Best screened margin |
|---|---|---:|
| QQ Farming | 100007008 s1 | -7,816 |
| Blu3s | 99986573 s1 | -4,927 |
| Mingkang YAN | 100054967 s1 | -4,119 |
| Blu3s | 99879123 s1 | -3,544 |
| Blu3s | 100075620 s1 | -3,440 |
| Blu3s | 100013966 s0 | -2,763 |
| Mingkang YAN | 100061842 s1 | -2,686 |
| nao.kwmr | 99986486 s1 | -1,693 |
| Arda Ceylan | 100006958 s0 | -1,598 |
| Blu3s | 99979758 s1 | -939 |
| Blu3s | 99856427 s0 | -555 |
| peikopon | 100045947 s1 | -386 |

Six of the 12 residual losses are against Blu3s, which is evidence of a remaining schedule-family coverage gap on this exposed set. It is not evidence that opponent identity should become a runtime feature.

## Validation

- 1,772 counterfactual games evaluated.
- Original eight fixed route/family jobs: 160/160 exact keys each.
- Actual-selected fixed-route alignment: 160/160 for V59 and 160/160 for V67.
- Every evaluated game completed 719 transitions.
- Zero sanitized steps across all fixed and expanded expert screens.
- No policy file changed, no upload, and no new holdout opened.

## Evidence, inference, and limits

Evidence:

- Four-route hindsight selection creates a 28-win ceiling gap over the actual 106/160 policy.
- Thirty additional expert files add 14 more recoverable rows, reaching 148/160.
- Observable linear LOFO models reach only 107-108 wins.

Inference:

- More route diversity is useful, but reliable route identification is now the harder problem than finding one more tape.
- A route gate should emphasize market regime and resource state rather than team/episode memorization.
- The repeated Blu3s residuals suggest a missing behavioral family, not a proven identity rule.

Unknowns:

- Whether a nonlinear, causally safe selector can generalize beyond this exposed development set.
- Whether the expanded expert routes retain their value under live opponent adaptation.
- Whether remaining losses need an earlier branch, a later second-stage switch, or a genuinely reactive controller rather than a fixed tape.

Machine-readable evidence:

- `rl/evaluations/v59_v67_route_oracle_development160_20260826.json`
- `rl/evaluations/v59_v67_expanded_route_oracle_development160_20260826.json`

