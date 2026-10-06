# V79 step-248 expansion-allocation feasibility — consumed gate8 only

Date: 2026-08-27

## Verdict

**The fixed identity-free price rule fails, so V79 stops before an overlay.**
It selects the COW→SHEEP swap in only one of the four losses. Its
model-conditioned rescue ceiling is therefore **1/4**, below the required
**4/4**. No candidate agent or overlay was created and no action-changing run
was executed.

**This does not kill the whole expansion-allocation family.** A separate,
deliberately generous per-loss structural upper bound clears all four loss
deficits. The bound is too loose to predict a win, but it means an unconditional
or better observable allocation model still has theoretical headroom and must
be tested under a separately declared hypothesis.

## Sealed scope

- Only the eight already-consumed cases in
  `analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json` were
  opened.
- No old80, fresh95, third160, future, newly downloaded, or aggregate replay was
  read.
- Candidate always occupied `agent_seat`; only `opponent_tape_seat` supplied the
  fixed action tape.
- Frozen V67 reproduced all 8 margins exactly over 719 transitions each.
- Aggregate normalization, sanitization, and fallback counts were all zero.
- No V67 or V59 file was modified; no upload was attempted.

## Exact action dependency

All six gate cases that buy COW×2 at step 248 have the same required token
rewrite:

| Step | Component | Frozen V67 | Counterfactual token |
|---:|:---|:---|:---|
| 248 | `market[1]` | `BUY_ANIMAL COW 2` | `BUY_ANIMAL SHEEP 2` |
| 249 | `hands[4]` | `PICKUP COW 2` | `PICKUP SHEEP 2` |
| 256 | `hands[4]` | `PLACE COW 1` at `[3,5]` | `PLACE SHEEP 1` |
| 261 | `hands[4]` | `PLACE COW 1` at `[4,5]` | `PLACE SHEEP 1` |

There are no further COW-specific unit tokens in this expansion chain. All
downstream animal operations at these pastures are generic `FEED`, `CARE`, and
`HARVEST` commands.

The exact frozen schedule feeds both pastures every day from day 10 through day
28. One pasture skips `CARE` on day 24, but the engine-accurate counterfactual
still harvests **43 WOOL** from the two sheep, leaves **0** terminal wool on the
animals, and requires no extra feed, care, or harvest command. Frozen cows
harvest **41 MILK** over the same schedule. The economic upper bound uses the
more favorable daily-care caps of 44 WOOL versus 42 MILK.

## Fixed observable price rule

The identity-free rule was fixed before this V79 bound execution, with no
threshold search:

```text
swap iff 44 × WOOL_price(step 248) − 1,000
      > 42 × MILK_price(step 248) − 800
```

It uses only the shared market observation at step 248 and known animal
production/cost rules. It contains no team, episode, seat, or opponent identity.
The gate was already consumed before V79, so this is a causal diagnostic rule,
not episode-disjoint validation.

| Loss episode | Margin | MILK/WOOL @248 | Predicted SHEEP−COW | Swap? | Model margin ceiling |
|---:|---:|:---:|---:|:---:|---:|
| 100159725 | -6,859 | 185 / 77 | -4,582 | no | -6,859 |
| 100184942 | -5,470 | 203 / 144 | -2,390 | no | -5,470 |
| 100187229 | -21,061 | 158 / 215 | +2,624 | yes | +10,434 |
| 100196432 | -2,479 | 191 / 1 | -8,178 | no | -2,479 |

The rule selects no regression control: two controls already use the SHEEP
expansion and all COW-expansion controls remain COW. Nevertheless, the hard
loss-rescue requirement fails before any action-changing evaluation.

## Per-loss economic bounds

Bounds are terminal **margin deltas**, not expected scores.

- Optimistic structural upper bound: credit all 44 possible wool units at the
  maximum future baseline WOOL price, subtract only the extra 200 animal cost,
  set forgone cow-milk revenue to zero, and add favorable cross-player market
  feedback.
- Pessimistic lower bound: credit zero new wool revenue, debit 42 cow-milk units
  at the maximum future MILK price plus the extra 200 cost, and add the worst
  adverse market feedback.
- Market feedback uses the exact monotone engine price rule, a full 42-unit milk
  scarcity or 44-unit wool-supply shift, and requested order quantities as if
  every request filled. Each transaction is charged its own worst intra-step
  inventory price change. This intentionally favors the hypothesis and is not a
  causal estimate.

| Loss | Direct WOOL upper | Favorable feedback upper | Structural delta [lower, upper] | Optimistic margin ceiling |
|---:|---:|---:|:---:|---:|
| 100159725 | 10,208 | 26,909 | [-28,251, +36,917] | +30,058 |
| 100184942 | 10,120 | 44,812 | [-34,350, +54,732] | +49,262 |
| 100187229 | 10,560 | 21,135 | [-14,972, +31,495] | +10,434 |
| 100196432 | 10,120 | 24,083 | [-25,864, +34,003] | +31,524 |

Without favorable market feedback, the worst loss `100187229` has at most
10,360 direct net headroom, below its 21,061 deficit. Its structural ceiling
clears zero only because the requested-quantity market-feedback envelope is
very generous. Therefore the evidence supports **“family not rejected”**, not
“allocation will rescue the game.”

## Validation and stop condition

- 8/8 frozen margins exact.
- 11,504 engine market-price observations checked; 0 mismatches.
- Exact rewrite signature matched in every COW-expansion case.
- Postulated sheep schedule: 43 harvested, 0 stranded, no incremental commands.
- Normalized steps: 0; sanitized steps: 0; fallback components: 0.
- Fixed price-rule rescue ceiling: 1/4 losses.
- Structural-oracle headroom: 4/4 losses, but not realized evidence.
- **Stop:** no overlay, no candidate package, no broader evaluation, no upload.

## Artifact hashes

- `analyze_feasibility.py`:
  `0ed5ec1e0d7b64e4b4acfedaa7b132217f02c0ad887fc16ae2f8f5ab2576abb8`
- `results.json`:
  `b9fd8670db9940f520ad1d928c68c08903cd9ead2a69f9f3cd8b1bcb3ca52416`
- Frozen V67:
  `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- Frozen V59:
  `64c82b6b01e9ec102433c2c747e04ae6277011ce56de53cf733cce3783593ee0`
- Gate8 manifest:
  `4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2`
- Native Rust engine:
  `315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd`

## TokenSave coverage note

TokenSave MCP was unavailable and `.tokensave/tokensave.db` indexed only the
older Rust engine graph, not V67, gate8, or the current replay harness. If this
coverage gap should be reported, open an issue at
<https://github.com/aovestdipaperino/tokensave> and remove all sensitive or
proprietary code from the issue description first.
