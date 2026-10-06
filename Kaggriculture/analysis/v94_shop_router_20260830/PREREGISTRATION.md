# V94 — Does conditioning tape choice on the revealed shops beat the best single tape?

Pre-registered 2026-08-30, **before any evaluation seed was run**. Frozen gates below.

## Origin

`analysis/v85_route_bandit_cv_20260827/` concluded route choice is not selectable from
observable state. That conclusion is unsafe for two reasons found on 2026-08-30:

1. `encode_selector_features`
   (`submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py:87`) never encodes
   `town.unlocked_shops`. The candidate branch variable was absent from the model input.
2. V85 had 32 contexts. The effect size claimed publicly is ~+2,300 mean margin, against a
   per-game margin SD in the tens of thousands. n=32 cannot resolve it.

The public claim under test is Yusuke Hayashi's (rank 3, live 2937), Kaggle notebook
`yhay81/public-match-history-router-rating-2929-aug-30`: routing among fixed 719-action
tapes on the first two revealed shops raised point rate 0.678 -> 0.814 and mean margin
+3,935 -> +6,283 over 4,096 held-out seeds x 3 opponents x 2 seats.

Mechanism independently confirmed on our engine before pre-registration: shop 1 appears at
step 72, shop 2 at step 144, shop 3 at step 216.

## What is being tested

**Not** Yusuke's agent, and **not** the two-shop table. We have 49 candidate tapes against
his 1,446, and 64 shop pairs cannot be populated at our budget. The reduced, answerable
question is:

> Does the identity of the best tape depend on the **first** revealed shop, strongly enough
> that a table fitted on training seeds beats the best single tape on unseen seeds?

A negative result here does not refute Yusuke. A positive result establishes that the
signal V85 declared absent is present and reachable with the tapes we already hold.

## Protocol

- Candidates: the 51 tapes in `artifacts/opponent_tapes_20260829/*/tapes/`, minus the 2 used
  as fixed opponents = 49.
- Opponents: 2 fixed tapes, held constant, excluded from the candidate pool.
- Both seats played for every (candidate, seed, opponent).
- Replay offset applied: the action taken from state `i` is `actions[i+1]`
  (`analysis/live_losses_20260830/REPORT.md`).
- Point rate scores a win 1.0, an exact money draw 0.5, a loss 0.0. Draws are real in this
  game and must not be scored as losses (23% of V16-RC6's live episodes were exact draws).

### Seeds — disjoint, fixed now, never reused across stages

| stage | seeds | purpose |
|---|---|---|
| screen | 13100001-13100040 | cut 49 candidates to 12; 1 opponent only |
| train | 13200001-13200240 | fit the first-shop table AND pick the best single tape |
| holdout | 13300001-13300240 | the only seeds the gates are read from |

The best single tape is chosen on **train**, never on holdout, so both arms get the same
information. No seed appears in two stages.

## Frozen gates — all three must pass

- **G1 (primary).** On holdout, point rate of the first-shop-routed policy is **strictly
  greater** than point rate of the best single tape.
- **G2 (no regression).** On holdout, routed point rate is **>= single** against each of the
  two opponents taken separately.
- **G3 (margin agrees).** On holdout, routed mean margin **>= single** mean margin.

A failed gate is a hard stop. No threshold may be revised after outcomes are seen, and the
holdout may not be re-read after any change to the table.

## Recorded regardless of outcome

- How many of the 8 first-shop cells select a tape other than the best single tape.
- Per-cell holdout n, so any cell too thin to trust is visible.
- Screen and train results in full, so the selection path is auditable.

---

# Amendment 1 — 2026-08-30, before any train or holdout game was run

**No gate has been read.** Only screen seeds (13100001-13100040) have been played. The three
frozen gates above are unchanged; what changes is the measurement apparatus, for two defects
found in the first screen.

## Defect 1: the opponent was too weak, and the metric saturated

The first screen used `rank002_crop_dusta/ep102201447` as the sole opponent. The best
candidate scored **80/80 games, point rate 1.000**, and 27 of 49 candidates scored >= 0.95.
G1 asks whether routing beats the best single tape; against a ceiling of 1.000 it cannot be
answered either way.

Cause: Crop Dusta is rank 1 live but its tapes replay **worst** of any team on foreign seeds
(its 12 tapes occupy screen ranks 36-49, including the only two negative-margin tapes). That
is itself consistent with Crop Dusta being adaptive — its tape with adaptation removed is
poor — and it made it the wrong choice of fixed opponent.

Replacement opponents, chosen on screen seeds only, for spreading candidate point rates
across the middle of the range rather than for any candidate's benefit:

| opponent | mean candidate point rate |
|---|---:|
| `rank003_yusuke_hayashi/ep102583251` | 0.367 |
| `rank001_milan_leonard/ep102297457` | 0.575 |
| `rank004_kenjo1209/ep102449287` | 0.577 |

All three are excluded from the candidate pool. The screen is re-run against all three.

## Defect 2: the candidate pool contained duplicates

Hashing the 51 tapes' action sequences gives **43 unique**. Yusuke Hayashi's 12 harvested
episodes collapse to **4** distinct tapes (one appears 8 times, another twice). Duplicates
would have let one tape occupy several table cells while looking like several candidates.

This also corroborates the method under test from our own data: a rank-3 agent whose 12
public episodes contain 4 distinct action sequences is a small tape set being routed, which
is what `yhay81/public-match-history-router-rating-2929-aug-30` describes.

## Revised protocol

- Candidates: 43 unique tapes minus the 3 opponents = **40**.
- Opponents: the three named above, all three used at screen, train and holdout.
- Seeds, stages and the three gates are unchanged.
