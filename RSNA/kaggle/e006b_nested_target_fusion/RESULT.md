# E006b result — nested target-selective fusion

- Accepted Kaggle run: `chinnaphatkhun/rsna-knee-e006b-nested-target-fusion`, version 3
- Status: complete, no runtime errors
- Accepted runtime: about 3.2 minutes
- Audit population: 57 official-gold rows covered by both pinned sources
- Leaderboard submission: none

Versions 1 and 2 were rejected engineering runs. Version 1 recursively scanned the huge
competition mount and then correctly stopped on the unmatched 58-versus-57 source
coverage. Version 2 used the wrong new-style Kaggle mount hints. Version 3 used direct,
hash-pinned paths and a strictly paired 57-row comparison.

## Paired nested result

| Metric | Value |
|---|---:|
| Pilkwang macro AUC | 0.870040 |
| Fixed Steven v4 macro AUC | 0.893237 |
| Nested target-selective macro AUC | 0.886156 |
| Nested gain versus Pilkwang | **+0.016117** |
| Minimum target delta | -0.035166 |
| Bootstrap 95% interval | [+0.002189, +0.030645] |
| Bootstrap `P(delta > 0)` | 0.9897 |

The generic nested selector protected Fracture automatically, without a target-specific
rule. Its largest target gains were Synovitis `+0.10062`, Contusion `+0.07194`, and
Lateral Meniscus `+0.03453`. Its largest loss was Effusion `-0.03517`, still above the
predeclared `-0.050` floor.

The all-Gold57 cross-validated deployment diagnostic selected Steven-v4 shares of `1.00`
for Synovitis, `0.75` for Contusion and Lateral Meniscus, and `0.00` for every other
target. Those full-data shares are diagnostic; a future image experiment would have to
repeat selection inside its own outer folds.

## Gate and decision

Coverage, prevalence, no-target-collapse, and bootstrap-direction checks passed. The
macro gain was `+0.01612`, below the frozen `+0.02000` requirement.

**NO-GO — `stop_target_fusion`.** E007 is not authorized under this contract. Do not
round the gain upward or weaken the gate after observing the result.

