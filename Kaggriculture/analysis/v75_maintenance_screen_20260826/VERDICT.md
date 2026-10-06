# V75 two-replay oracle verdict

Verdict: **HARD STOP — do not promote, expand evaluation, or upload.**

The maintenance-only floor-400 admission is legal and preserves cash, but it does
not solve the closed-loop completion failure.

| Replay | Final money / gate | Land @ 192 | Longest stranded | Sanitizer / fallback |
| --- | ---: | ---: | ---: | ---: |
| median `99614514` | 6,799 / 53,036 | 1 | sheep 434 steps | 0 / 0 |
| worst `99589578` | 16,127 / 53,533 | 1 | sheep 434 steps | 0 / 0 |

Both replays also failed the 48-step deficit-starvation gate. The screen was
stopped immediately after these two authorized replays.

## Causal readout

- Every one of the 576 post-handoff, pre-liquidation controller turns was
  classified as maintenance; expansion turns and no-target turns were both zero.
- Admission rejection and planner-blocked counters were empty, while the minimum
  cash after a fixed purchase remained 4,888 / 5,020. The failure is therefore
  not caused by the floor-400 admission check or by a rejected planner order.
- The policy peaked at four live sheep but ended with three live sheep and one
  sheep stranded in the shed for 434 consecutive steps. A maintenance macro can
  monopolize the scheduler without reaching physical completion.
- With maintenance never resolving, true land expansion was never admitted;
  land remained one in both games and the worst replay failed the land gate.
- Final money improved over V74's diagnostic results (2,925 / 4,167) but remains
  only 12.8% / 30.1% of the required gates. That is not close enough to justify
  a broader replay evaluation.

## Smallest falsifiable next change (not implemented)

Add a physical-progress watchdog for maintenance macros: if live/placed count,
feed state, or crop count does not improve within a short bounded dwell, release
that macro and enter an explicit placement/recovery action selected from current
state. Gate it on these same two replays first: stranded <= 24, starvation <= 48,
land >= 2 at step 192 in the worst replay, money gates met, and zero sanitizer /
fallback. Do not alter admission floors until that completion invariant passes.

## Scope

Only episodes `99614514` and `99589578` from the already-authorized old80
manifest were executed. No fresh, full-old80, third/future holdout, or upload was
used.
