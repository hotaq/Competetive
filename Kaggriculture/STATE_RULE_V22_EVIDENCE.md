# V22 State-Rule Promotion Evidence

Date: 2026-08-26

## Decision

V22 is a promotion candidate over V16 for further packaging and live Kaggle
validation. The new logic is based only on the current observable state. It
does not select actions by opponent name, team ID, episode ID, or copied replay
position.

V22 is not an RL policy yet. It is a deterministic state-rule overlay on the
current V16 agent and is the safer baseline for a later RL residual policy.

## Rules

1. Third-land expansion
   - Active only on days 10 through 14.
   - Active only when the player owns exactly two unlocked quadrants.
   - Estimates available cash from current money and the leading `SELL` orders
     already scheduled by V16 at current market prices.
   - Inserts `BUY_LAND` after those funding sales only when the estimated land
     cost (1,000) plus an operating reserve (800) is covered.
   - Does nothing when a land purchase is already scheduled.
2. Terminal liquidation
   - From step 718 onward, appends legal `SELL` orders for unscheduled products
     remaining in the shed, subject to the market-order limit.

Implementation:

- `src/kaggriculture_rl/policy/market_rules.py`
- `submissions/meta_v22_v16_land3_terminal_rule_local/main.py`

## Three-split evidence versus V16

Each split contains four replays for each frozen Top-20 team (80 games). The
offset-8 test split was not used to design or tune the rules.

| Split | V16 wins | V22 wins | V16 mean margin | V22 mean margin | V16 worst | V22 worst | Loss to win | Win to loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development, offset 0 | 43/80 | 49/80 | 403.9 | 3,085.8 | -33,791 | -32,660 | 6 | 0 |
| Holdout, offset 4 | 39/80 | 44/80 | 1,994.2 | 4,262.9 | -32,241 | -32,241 | 5 | 0 |
| Fresh test, offset 8 | 49/80 | 50/80 | 2,265.8 | 2,749.2 | -27,384 | -27,384 | 1 | 0 |

Across the three splits, no evaluated game changed from win to loss and no
game had a negative margin delta. The fresh split improved one loss to a win
and increased mean margin by 483.4 without degrading the worst game.

Fresh-test ablation:

| Candidate | Wins | Mean margin | Worst margin | Change versus previous |
|---|---:|---:|---:|---|
| V16 | 49/80 | 2,265.8 | -27,384 | Baseline |
| V21: land rule only | 50/80 | 2,721.5 | -27,384 | 1 loss to win; no regressions |
| V22: V21 plus liquidation | 50/80 | 2,749.2 | -27,384 | +27.7 mean margin; no regressions |

## Rejected variants

- Buying land on days 8-20 with no cash reserve fell to 45/80 on development
  and worsened the worst margin to -47,984.
- Buying land on days 9-18 with a 400 reserve reached only 47/80 and worsened
  the worst margin to -50,368.
- Replay-tape component swaps and direct policy handoffs produced catastrophic
  losses and are not promotion candidates.

These failures show that the useful behavior is not simply "buy more land".
The timing window and retained operating cash are material parts of the rule.

## Remaining limitation and gate

The three splits use different episodes but the same frozen Top-20 team roster.
They support generalization across replay episodes, not across every future
opponent or meta change. Replay opponents are also sanitized after
counterfactual state divergence, so the scores are comparative evidence rather
than a guaranteed Kaggle result.

Before upload:

1. Build a standalone Kaggle-compatible V22 bundle.
2. Run import, action-schema, size, and local smoke checks on that exact bundle.
3. Submit once and inspect live score plus replays before making another rule
   change.
