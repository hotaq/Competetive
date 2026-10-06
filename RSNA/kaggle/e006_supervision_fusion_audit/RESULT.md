# E006 result — supervision fusion audit

- Kaggle kernel: `chinnaphatkhun/rsna-knee-e006-supervision-fusion-audit`, version 1
- Status: complete, no runtime errors
- Runtime: about 30.4 minutes, dominated by 5,000 paired bootstrap replicates
- Inputs: five hash-pinned CSV files; no DICOM access
- Leaderboard submission: none

## Gold58 macro AUC

| Label source | Macro AUC | Delta vs Pilkwang |
|---|---:|---:|
| Pilkwang | 0.870040 | — |
| Steven v2 | 0.887349 | +0.017310 |
| Lixin | 0.835194 | -0.034846 |
| Steven v4 fixed external blend | **0.892707** | **+0.022668** |
| Equal three-source fusion | 0.834823 | -0.035216 |
| Cross-fitted three-source fusion | 0.828202 | -0.041837 |

The cross-fitted three-source arm failed decisively. Its paired-bootstrap 95% interval
for macro gain was `[-0.06231, -0.02127]`, with zero of 5,000 replicates positive.

Steven v4 passed the macro and bootstrap checks: paired-bootstrap 95% interval
`[+0.00376, +0.04202]`, with `P(delta > 0) = 0.9906`. It nevertheless failed the
predeclared no-target-collapse check because Fracture changed by `-0.07827`, below the
allowed `-0.050` floor. Its largest positive target deltas were Synovitis `+0.09650`,
Contusion `+0.09184`, Effusion `+0.04710`, Lateral OA `+0.04415`, and Lateral Meniscus
`+0.03745`.

**Coverage correction:** Pilkwang omits one official-gold study. E006 allowed each AUC
arm to finite-filter independently, so the fixed-v4 comparison above is not strictly
paired. E006b supersedes that comparison with the 57 rows covered by both sources; its
paired fixed-v4 macro was 0.893237 versus 0.870040 for Pilkwang.

## Decision

**NO-GO — `stop_b3_supervision_gate`.** Do not launch E007 using one global fused label
source. The next safe metadata-only experiment, if pursued, is a newly predeclared nested
target-selective fusion that must estimate the Pilkwang-versus-v4 choice out of fold; it
must not simply hard-code the favorable targets observed here and rescore the same 58 rows.
