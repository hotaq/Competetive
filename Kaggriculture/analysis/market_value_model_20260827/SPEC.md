# Deterministic observable livestock marginal-value specification

Status: analysis-only executable specification. It is not an agent rule, a
promotion gate, or a learned opponent classifier.

Implementation: `marginal_value.py::value_one_animal` and
`marginal_value.py::compare_cow_sheep`.

## Inputs

The function accepts only:

- the current observation (`market`, `town`, both public `farms`, own
  `private`, player seat, day/hour/step);
- public game configuration;
- animal (`COW` or `SHEEP`) and proposed placement step.

It never accepts episode id, team/opponent name, replay id, seed, hidden
opponent inventory, outcome label, or a fitted decision threshold.

## Exact price primitive

For product `r`, market inventory `I`, equilibrium `I0`, base `B`, throughput
`T`, side shape `f`, and side target `g`:

```text
A = g B / f(T, T)

I < I0:  P_r(I) = max(1, round_even(B + A f(I0 - I, T)))
I >= I0: P_r(I) = max(1, round_even(B - A f(I - I0, T)))
```

Shapes are `linear(x)=x`, `sq(x)=x^2`, `sqrt(x)=sqrt(x)`,
`log(x)=ln(1+x)`, and
`hinge(x,T)=x/T + 8 max(0,x/T-1)^2`.

Authoritative runtime: `.venv/.../kaggriculture.py:41-78,192-212`.
Matching Rust: `rust/kaggriculture-engine/src/rules.rs:121-148,177-221`.

## Exact sale primitive

For one aligned order-list position, quantities are executed unit by unit.
At each unit iteration both active seats receive quotes from the same
pre-commit inventory. Commits then occur in seat order. A sale priced above
one adds one market inventory; a sale at the one-coin floor does not.

This is `lockstep_sell(inventory, (q0, q1), item)` in
`marginal_value.py:251-276` and mirrors
`.venv/.../kaggriculture.py:544-628,652-685` plus
`rust/kaggriculture-engine/src/engine.rs:389-495,626-679`.

## Observable projection

For each product independently, projected market inventory advances in exact
interpreter order:

```text
market sales -> known/expected town demand -> end-of-day shop unlock
```

Known unlocked shop instances are exact, including duplicates. Unknown future
shops use the state-independent expectation of a uniform draw with replacement
from the eight runtime shop types until the eight-instance cap. Expected demand
per new shop tick is `MILK=3/8`, `WOOL=2/8`, and `WHEAT=5/8` units. This is an
expectation, not an oracle for the nonlinear future price.

All animals currently visible on both farms are projected under the same
identity-free upper-service behavior: daily feed, daily care, immediate harvest,
and sale after a two-step logistics lag. The marginal animal's output may be
sold or withheld. Dynamic programming maximizes own projected portfolio
revenue, so forced dumping cannot assign a negative product option value.

Feed is charged at the projected opportunity value of one WHEAT per useful
service day. Future public crop sales are omitted rather than replaced with an
opponent-specific behavior model; the assumption is surfaced in every result.

## Scalar and non-scalar outputs

The core cash marginal value is:

```text
CoreMV_a = max_sale_policy(
             own portfolio revenue with animal
             - own portfolio revenue without animal
           )
           - animal purchase cost
           - projected WHEAT opportunity cost
```

Labor is not assigned a mined constant. Instead, the function returns exact
service counts and the break-even action shadow value. If a caller has an
independently justified alternative-action value `lambda`, it may evaluate:

```text
MV_a(lambda) = CoreMV_a
               - lambda * (feed + care + harvest + minimum logistics)
               + max(0, fertilizer_snapshot_value
                        - lambda * fertilizer_collection_actions)
```

`minimum logistics` is `PICKUP + PLACE`, plus `BUILD_PASTURE` when the current
observable farm has no empty matching structure. Movement cost remains an
explicit state-dependent burden. The market purchase is separately constrained
by cash and the ten-order market budget.

Fertilizer is reported as a separate option because COW and SHEEP generate the
same daily fertilizer opportunity at the same placement day, while collection
uses one additional unit action each time.

## Step-248 mechanics

At default 24 turns/day, step 248 is day 10 hour 8. Unit actions execute before
market orders, so an animal bought in that turn cannot be placed in that same
turn. In the frozen route, the order selected from observation step 248 is
stored at replay record 249; the two animals are actually placed at steps 256
and 261. All are still placed on day 10 and serviced before end of day, so they
share this production schedule:

| animal | production refresh days | cared output units | feed | care | harvest |
|---|---|---:|---:|---:|---:|
| COW | 18, 20, 22, 24, 26, 28 | 21 MILK | 18 | 18 | 6 |
| SHEEP | 16, 19, 22, 25, 28 | 22 WOOL | 18 | 18 | 5 |

Each also exposes at most 19 fertilizer collections. Two consecutive missed
end-of-day feeds remove the animal irrecoverably and leave the structure.

