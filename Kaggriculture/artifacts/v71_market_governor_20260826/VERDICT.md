# V71 market governor: rejected at development gate

Scope was restricted to the frozen old80 and fresh95 development cases. No
candidate was promoted, packaged, or uploaded.

## Causal screen

- A single-event `early_exit` intervention moved at most two WOOL units from
  step `t+1` to `t`, with exact repayment at `t+1`.
- It rescued 8 unique episode IDs across 6 teams in the shallow-loss oracle
  screen, with zero sanitized actions.
- Every rescue used only observable state: exact town clock and unlocked-shop
  demand, current and 24-step WOOL price slope, own shed coverage, and own cash.
- The reverse `reserve` intervention rescued 0 of 30 loss cases across 23
  eligible events. It was therefore not activated in the full-league wrapper.

The oracle result was only a causal existence signal. It did not establish that
the same generalized rule was safe when applied repeatedly to every matching
state.

## Frozen wrapper result

| Set | V59 | V71 early-only | Rescues | Regressions | Mean margin delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| old80 | 66/80 | 66/80 | 2 | 2 | -910.55 |
| fresh95 | 79/95 | 76/95 | 4 | 7 | -839.936842 |
| combined | 145/175 | 142/175 | 6 | 9 | -872.22 |

The stop gate required at least 3 rescued cases from at least 3 teams and no
more than 1 regression. The generalized policy failed because it caused 9
regressions. It also missed the required old/fresh win gates of 67/80 and
83/95.

Tail risk regressed:

- old80 CVaR10: `-14385.5` -> `-14536.875`; worst stayed `-21138`.
- fresh95 CVaR10: `-12971.8` -> `-14702.8`; worst stayed `-36180`.
- Sanitized actions remained zero in both sets.

The likely mechanism is non-compositional intervention: a single early WOOL
sale can rescue a selected loss, but applying the state rule on all matching
trajectories changes early cash and market prices. Those changes cascade into
later V59 route/economy decisions. Paired evaluation changed 48/80 old cases
and 63/95 fresh cases, including large negative margin deltas.

Per the pre-registered stop rule, there was no further threshold tuning and the
bidirectional research wrapper was not evaluated as a candidate.

## Fresh94 overlap diagnostic

Episode `99546279` occurs in old80 for one team/seat and in fresh95 for the
opposite team/seat. Removing its fresh occurrence leaves 94 cases. The row was
an unchanged loss (`-11472`) for both V59 and V71, so purging it does not change
the flip counts:

- V59: 79/94, CVaR10 `-11994.2`, mean margin `12754.521277`.
- V71: 76/94, CVaR10 `-13777.3`, mean margin `11905.648936`.
- Paired flips remain 4 rescues and 7 regressions; verdict remains reject.

## Exact evidence and hashes

- old80 league manifest SHA256:
  `1253ef13a3bea8bfd8e741185d27c178b296e3aafde08832972dd5dd16934c57`
- old80 20 team-manifest digest-list SHA256:
  `ee58bd6cbd991dcdefd20b4f1ca10f4e702e9378a100eb2399af0fc0a926ab40`
- fresh95 league manifest SHA256:
  `af7ccd297374a9eaad811820e7bb05ffc71e124e696eeacd746df608dcabb6d1`
- fresh95 `manifest_fresh.json` set: 20 files, 95 rows; digest-list SHA256:
  `97c3a25c00e2649a44435a53cd95cfcf76664352dee9f2853c1b158eb2a4fc2f`
- frozen wrapper SHA256:
  `22a7613871d0857d60da0448257133e8eb94c2d65fed79c61b4d90a365a4353d`
- old80 report SHA256:
  `ac9ec4486feb470d9c5fb1da726d8a1935c8a2d118c63ac7f14b80ef0bf1acbb`
- fresh95 report SHA256:
  `9ec847920840dce04fb879acbd1234bd1faf749bc358397bf19a262a51a4fb8b`
- old80 paired report SHA256:
  `6d55c226e1655f26e01c50062506ab5fad03c7b5ad2d402d8a46a736a43a802c`
- fresh95 paired report SHA256:
  `028c313f0acc3b59902550fbf82f48ab20f055e4d1233c83b491a19feac08fbc`

An earlier command followed the drifted default fresh directory manifests and
ran 160 rows. That aggregate was immediately invalidated, was not used for any
decision or per-case analysis, and the report path was overwritten by the
exact 95-row run using `--team-manifest-name manifest_fresh.json`.

