# Market/livestock economics audit — 2026-08-27

## Outcome

The mechanics model is reusable and parity-checked, but **it is not a safe
step-248 COW-to-SHEEP selector**.

On the sealed, already-consumed gate8, observable projected cash value prefers
SHEEP in all four target losses. However, among the two regression controls
that also issue `BUY_ANIMAL COW 2` from observation step 248, it incorrectly
prefers SHEEP in one. That case is episode 100166584, the thin +602 control;
its realized future price-path sanity check instead prefers COW.

The smallest target projected advantage is +844 while the false-positive
control is +770. Selecting a threshold in that 74-coin gap would be pure
threshold mining on the consumed gate, explicitly outside this audit. No
candidate was implemented, promoted, or uploaded.

## Exact engine/runtime findings

The installed reference is `kaggle-environments==1.32.7`; its
`kaggriculture.py` SHA-256 is
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.

### Price curve

The runtime and Rust engine use the same resource-specific, two-sided curve
with a one-coin floor and ties-to-even rounding. Relevant default parameters:

| item | base | T | below | target | above | target |
|---|---:|---:|---|---:|---|---:|
| WHEAT | 25 | 400 | sqrt | 0.80 | log | 0.20 |
| MILK | 160 | 122 | sqrt | 0.60 | linear | 1.60 |
| WOOL | 200 | 105 | log | 0.20 | square | 3.20 |
| FERTILIZER | 100 | 200 | linear | 0.40 | linear | 0.40 |

Sources: runtime `kaggriculture.py:41-78,192-212`; Rust
`rules.rs:121-148,177-221`.

### Interleaved market execution

1. Only the first ten orders per player are retained.
2. Order-list index is the outer loop: index 0 for both players completes
   before index 1 begins.
3. `HIRE` and `BUY_LAND` are atomic and execute once in seat order.
4. `SELL`, `BUY_PRODUCT`, `BUY_SEED`, and `BUY_ANIMAL` execute one unit at a
   time. Both seats are quoted from the same pre-commit inventory for each
   paired unit, then commit in seat order.
5. A failed commit terminates the remaining quantity of that order.
6. Floor-price sales pay one coin but do not add market supply.
7. `BUY_PRODUCT` quotes at post-buy inventory `I-1`, making an otherwise
   unchanged buy-then-sell round trip net zero.

Sources: runtime `kaggriculture.py:544-628,652-685`; Rust
`engine.rs:389-495,626-679`.

### Town timing and demand

- Market orders execute before town demand each turn.
- Shop demand ticks when `step % 4 == 0`; town-center demand ticks when
  `step % 24 == 0`, including step 0.
- Town center removes one unit of every product except FERTILIZER.
- Each unlocked shop instance consumes independently. A one-product shop
  consumes two units; duplicate shops stack.
- At end of each third day, one shop is drawn with replacement until eight
  total instances. The first unlock occurs after action step 71 and is visible
  at observation step 72, where it can immediately consume because 72 is a
  shop tick.

Sources: runtime `kaggriculture.py:103-118,728-749,860-891,894-963`; Rust
`rules.rs:151-173`, `engine.rs:310-345,498-558`.

### Permitted BUY_PRODUCT items

Only WHEAT and FERTILIZER are legal. Every other product, including MILK and
WOOL, is a silent no-op. Sources: runtime `kaggriculture.py:596-605,662-685`;
Rust `engine.rs:478-495,651-678`.

### Livestock lifecycle

| animal | cost | structure | first yield | interval | held cap | product |
|---|---:|---|---:|---:|---:|---|
| COW | 400 | PASTURE | 8 days | 2 days | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 days | 3 days | 6 | WOOL |

- Daily feed resets the missed-feed counter. Two consecutive missed refreshes
  make the animal escape permanently; the empty PASTURE remains.
- A fed production refresh gives one base unit plus previously banked care,
  capped at six held units.
- Care only banks a bonus when both cared and fed. A production refresh first
  consumes the old bonus, then banks the current day's care for the next event.
- Each surviving animal exposes one fertilizer collection per day, but
  collection consumes a unit action.
- Terminal reward is bank money; a surviving animal has no liquidation value.

Sources: runtime `kaggriculture.py:19-23,229-241,381-400,449-472,505-531,805-833`;
Rust `rules.rs:91-116`, `engine.rs:756-765,826-855,886-915,1077-1122`.

## Step-248 COW versus SHEEP

Same-turn buy and placement is impossible because units act before the market.
The fixed-COW route has no empty pasture at observation step 248, buys two cows
through the action recorded at replay index 249, builds pastures, then places
at steps 256 and 261. Placement remains on day 10, so production timing is
identical to a pre-owned animal placed at step 248.

With daily feed/care and prompt harvest:

- one COW: 21 MILK on days 18/20/22/24/26/28;
- one SHEEP: 22 WOOL on days 16/19/22/25/28;
- both: 18 feed + 18 care actions and up to 19 fertilizer collections;
- COW needs six harvests, SHEEP five.

The table below gives observable projected **core net**: optimized own product
portfolio delta minus animal purchase and projected WHEAT opportunity cost,
before labor and before fertilizer. `+fert` adds the separately reported
snapshot liquidation of all 19 fertilizer units, still before labor.

| episode | group | current step-248 animal order | COW core | COW +fert | SHEEP core | SHEEP +fert | SHEEP-COW core | ex-post sanity preference |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 100159725 | loss | COW x2 | -925 | 515 | 4,028 | 5,468 | +4,953 | SHEEP |
| 100184942 | loss | COW x2 | -713 | 750 | 4,042 | 5,505 | +4,755 | SHEEP |
| 100187229 | loss | COW x2 | -885 | 616 | 4,077 | 5,578 | +4,962 | SHEEP |
| 100196432 | loss | COW x2 | -827 | 613 | 17 | 1,457 | +844 | SHEEP |
| 100168881 | control | none | -776 | 664 | 3,997 | 5,437 | +4,773 | SHEEP |
| 100182646 | control | COW x2 | 2,525 | 4,026 | -313 | 1,188 | -2,838 | COW |
| 100143759 | control | none | -764 | 748 | 4,295 | 5,807 | +5,059 | SHEEP |
| 100166584 | control | COW x2 | -887 | 610 | -117 | 1,380 | +770 | **COW** |

The ex-post check uses future prices only after the observable model is frozen;
it is not a policy feature and not a counterfactual replay. Projection-versus-
realized-path net MAE is 251.875 coins for COW and 576.625 for SHEEP. The
100166584 preference flip is larger than the 74-coin separation between its
projected delta and the weakest target delta, so the apparent consumed-gate
separation is not robust.

## Reusable function

The executable specification is `marginal_value.py`:

- exact price: `market_price`, lines 229-248;
- exact aligned-unit sale: `lockstep_sell`, lines 251-276;
- care/feed/yield calendar: `animal_schedule`, lines 290-353;
- visible opponent-farm supply: `_existing_visible_supply`, lines 397-433;
- optional-sale dynamic program: `_project_product_revenue`, lines 445-571;
- public `value_one_animal`, lines 621-732;
- public comparison `compare_cow_sheep`, lines 735 onward.

See `SPEC.md` for formula, assumptions, and labor-shadow integration.

## Validation and boundary

- 11/11 analysis invariants passed against the frozen Python runtime: price
  grid, monotonic floor, same-item lockstep sales, BUY_PRODUCT allowlist,
  animal schedule, two-miss death, step-0 town demand, first shop unlock, and
  unit-before-market same-turn placement.
- Rust engine: 13/13 tests passed (9 unit + 4 legality integration).
- Rust/Python differential: deterministic full-state trace plus 960 randomized
  turns passed.
- All eight replay SHA-256 hashes and the frozen V67 SHA were asserted before
  gate execution.
- Only `analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json`
  and its eight named replays were read. No old80, fresh95, third160, future,
  newly downloaded replay, candidate, or upload was used.

## TokenSave limitation

TokenSave MCP was unavailable. The SQLite graph was queried first, but its
indexed `engine.rs` size is 35,948 bytes while the current file is 45,274 bytes,
and the installed runtime checkout is not indexed. If this extractor/index
freshness gap should be reported, open an issue at
<https://github.com/aovestdipaperino/tokensave> and remove all sensitive or
proprietary code from the issue description first.
