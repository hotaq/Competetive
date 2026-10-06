# V73 V59-to-V72 late recovery: rejected at loss-only gate

V73 was a development-only splice experiment. Each wrapper used V59 before a
fixed observable branch step and a fresh `ClosedLoopTownQuotaPolicy` afterward.
The four pre-registered branch steps were 480, 600, 648, and 672. The wrappers
contain no opponent identity, episode ID, seed rule, or replay lookup, and reset
late-policy state independently by player and episode.

## Frozen scope

- old80 V59 losses: 14 cases from `manifest.json`.
- fresh95 V59 losses: 16 cases from `manifest_fresh.json`.
- Total loss-only screen: 30 exact team/episode/seat cases.
- No third/future set and no upload were used.

## Loss-only result

| Branch | Rescued | Improved | Mean candidate margin | Mean delta vs V59 | Best delta | Worst margin | Post-branch sanitizer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | 0/30 | 0/30 | -60509.5 | -51650.866667 | -22276 | -95209 | 0 |
| 600 | 0/30 | 0/30 | -28859.0 | -20000.366667 | -12693 | -49383 | 0 |
| 648 | 0/30 | 0/30 | -24735.733333 | -15877.1 | -11581 | -49566 | 0 |
| 672 | 0/30 | 0/30 | -19389.2 | -10530.566667 | -8086 | -44728 | 0 |

Every branch made every baseline loss worse. The loss-only gate required at
least 3 rescued episode IDs from at least 3 teams, with zero post-branch
sanitizer. Although legality passed with zero sanitizer, all branches had zero
rescues and therefore failed decisively.

The evidence supports an objective/continuity mismatch rather than an action
legality failure: the fresh V72 controller produces legal actions from the
handoff state, but abandoning V59's remaining economic schedule destroys value.
Even the latest branch at step 672 lost at least 8086 additional margin on every
screened case. This experiment does not establish one narrower causal mechanism
inside V72 because the pre-registered stop rule forbids further tuning after a
zero-rescue screen.

## Decision

- Stop before all-case evaluation.
- Do not select a branch.
- Do not tune branch thresholds or V72 internals.
- Do not package or upload.

## Validation

- Wrapper invariants: 7/7 tests passed.
- Ruff: passed.
- Exact boundary behavior, copied outputs, player-isolated reset, and rewind
  reset are covered by tests.

## Artifact hashes

- Overlay SHA256:
  `8931f1564cd81cdcaedbd3f70d45b047ad6bbacf8e720a90077bad38b5201ee4`
- Branch 480 wrapper SHA256:
  `140d0190c56fcf39002be3b8d67ac2805d8f01aac4a6d4ff4d6769eee7e453cc`
- Branch 600 wrapper SHA256:
  `39a37c5283f539117ef9bd8ece9dcf2f7474d977ca49a95a123f261e19b8e033`
- Branch 648 wrapper SHA256:
  `10de84c90ef0e5b9dd5c7084ff9942f26645dfd464974bbcb1c4b823f271251f`
- Branch 672 wrapper SHA256:
  `c4723cf635446bb2baee952c46420aaa9dade1f0011dc6b89149dbc612c8e0f7`
- Loss-screen script SHA256:
  `83e5dbc61c8ae7cbd24068db3614e24cb096fe87559c5eebf2441239bdeb86be`
- Invariant test SHA256:
  `98e5d55630dc4d7bcd23ad99b4bc09f3887e8f2e6c011b1f4593709ee6cacb9a`
- Branch 480 report SHA256:
  `dbb3bd03d6ad3ea9d99d5b068a7fe36d662ad61725dee2875e7074ba1f420293`
- Branch 600 report SHA256:
  `04e1ecef3ca649121d055bdcb35a9dfcae75d7eb7582831f2acb3d992a83d6b9`
- Branch 648 report SHA256:
  `d42dfc51be5469913e68104089c052f6d965646bf3754a33d857831d08ec14f6`
- Branch 672 report SHA256:
  `13809eabb347221b1baf1134c90fab28be8a9f55aad53abfeab543cfff73fd83`

The tokensave SQLite graph contained no V72 or
`ClosedLoopTownQuotaPolicy` nodes, so the implementation audit used the live
source and runtime artifacts after the required graph query.

