# V77 causal macro-optionality pre-rewrite diagnostic

Date: 2026-08-27 (Asia/Bangkok)

Verdict: **STOP / FAIL before implementation**

The predeclared smallest causal intervention did not rescue any of the four
target losses. Per the stop condition, no threshold search, cow/sheep rewrite,
candidate packaging, holdout access, or upload followed.

## Scope and evidence

- Gate definition:
  `analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json`
- Gate SHA-256:
  `4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2`
- Frozen V67 baseline:
  `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`
- Frozen V67 SHA-256:
  `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- Candidate occupied each entry's `agent_seat`; the evaluator received the
  corresponding `opponent_tape_seat`.
- Only the eight consumed public live replays in the gate were evaluated.

## Observable-only diagnostic rule

The one-shot screen used no seed, episode, team, opponent name, or identity
field. It activated only after the frozen V67 action exposed the fixed-cow
signature `BUY_ANIMAL COW 4` during observation steps 156-159. From step 216:

1. current MILK price had to be at least its base price, 160;
2. V67 already had to request a MILK sale on the current turn;
3. the first MILK sale was increased only to current shed MILK minus same-turn
   MILK pickup reserve.

This was a diagnostic action wrapper only; no project candidate was written.

## Exact gate result

| Group | Episode | V67 margin | Diagnostic margin | Delta | Sanitized steps | Acceleration events |
|---|---:|---:|---:|---:|---:|---:|
| target | 100159725 | -6,859 | -6,859 | 0 | 0 | 0 |
| target | 100184942 | -5,470 | -5,470 | 0 | 0 | 2 |
| target | 100187229 | -21,061 | -21,061 | 0 | 0 | 0 |
| target | 100196432 | -2,479 | -2,479 | 0 | 0 | 1 |
| control | 100168881 | +18,079 | +18,079 | 0 | 0 | 0 |
| control | 100182646 | +9,765 | +10,064 | +299 | 0 | 17 |
| control | 100143759 | +23,005 | +23,005 | 0 | 0 | 0 |
| control | 100166584 | +602 | +602 | 0 | 0 | 0 |

Target rescue rate was `0/4`; controls remained `4/4` wins and no control
margin decreased. Aggregate mean moved from `+1,947.750` to `+1,985.125`, but
the worst margin remained `-21,061`. The mandatory `8/8` positive-margin gate
therefore failed decisively.

Observed target events:

- `100184942`: step 370, price 201, scheduled 4, accelerated 6; step 385,
  price 183, scheduled 3, accelerated 6.
- `100196432`: step 370, price 199, scheduled 4, accelerated 6.
- `100159725` and `100187229`: no qualifying action.

## Why the cow/sheep rewrite was not attempted

The frozen base route buys `COW x4` at observation step 157 and `COW x2` at
168, then immediately consumes those animals through pickup/build/place
sequences before shop three appears at step 216. Removing those purchases is
not a market-only delta: it requires replacing a long movement and production
suffix. That is exactly the kind of state mismatch that prior V61 option
opening splices warned about. The reviewed V61 append variants achieved only
47-57 wins out of 80 despite zero sanitizer steps.

Because the smaller causal market hypothesis produced zero target rescues, a
larger livestock rewrite would not be justified by this gate and was stopped
before implementation.

## Data boundary

- No fresh95, third160, future holdout, newly downloaded Kaggle replay, or
  upload was used.
- The only historical results additionally inspected were the explicitly
  requested V61 option-opening summaries; no new old80 evaluation was run.
- TokenSave MCP was unavailable and `.tokensave/tokensave.db` contained no
  current V67/V77 entries, so the named current artifacts were inspected
  directly.

