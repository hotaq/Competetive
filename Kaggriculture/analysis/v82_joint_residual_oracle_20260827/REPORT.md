# V82 Joint Residual Oracle — 2026-08-27

## Verdict

Status: **STOP_JOINT_TARGET_PREREQUISITE_FAILED**.

This is a consumed-Gate8, development-only exact Rust feasibility screen. It is not a learned policy, promotion result, package, or upload.

## Target results

| Episode | V67 | Sheep v2 | V80 macro | Joint | Joint valid | Joint win |
|---:|---:|---:|---:|---:|:---:|:---:|
| 100159725 | -6859 | -2246 | -3458 | -6826 | yes | NO |
| 100184942 | -5470 | -229 | +27036 | +8982 | yes | yes |
| 100187229 | -21061 | -11473 | -27786 | -18124 | yes | NO |
| 100196432 | -2479 | +603 | +4439 | -1856 | yes | NO |

Joint rescued **1/4** targets; worst joint margin **-18124**.

Every variant used callable V67 on the current engine state and the raw opponent replay action on the fixed opponent seat. Candidate sanitizer, same-state overlay scope, residual phase/counters, and the zero-partial packaging gate are recorded in `results.json`.

## Interaction finding

All four joint executions were mechanically valid: zero candidate sanitizer changes, zero out-of-scope changes, zero partials, and a passing completion gate. The stop is economic, not an executor fault.

- `100159725`: sheep alone reached -2246 and macro alone -3458, but the joint fell back to -6826.
- `100184942`: the joint was the only target rescue at +8982.
- `100187229`: the on-policy macro reversed the earlier V80 tape-oracle signal and scored -27786; the joint remained -18124.
- `100196432`: sheep alone (+603) and macro alone (+4439) both won, yet their joint lost at -1856.

V80 held the candidate continuation tape fixed after its one mutation; V82 instead recalls V67 from every changed state. Therefore the V80 single-mutation margins are an optimistic tape oracle and are not expected to equal these on-policy macro-only margins.

## Gate order

At least one exact joint target remained non-positive or invalid. Per the predeclared fail-fast rule, **no controls were executed**.

Therefore this report makes no control-regression claim.

## Scope and reproducibility

- Data: only four consumed Gate8 losses; controls only if 4/4 target pass.
- No old/fresh/novel/sealed/new replay was opened.
- No PPO, candidate archive, package, submission, or upload was created.
- Input and action-stream SHA-256 hashes are in `results.json`.
- `run_oracle.py`: `c3060b9f97ab5a7970f447fdf77108b3e2e51317ae8ac2a19e1dca8750cb1fc2`
- `results.json`: `854065018b5d3c82ab311377d474b751e54f1b5ba50bb1f86da096fa606e8039`

Final validation: Ruff check PASS, Ruff format check PASS, Python compile PASS, and `tests/policy/test_v67_residual.py` PASS.

```bash
.venv/bin/python analysis/v82_joint_residual_oracle_20260827/run_oracle.py
.venv/bin/ruff check analysis/v82_joint_residual_oracle_20260827/run_oracle.py
.venv/bin/ruff format --check analysis/v82_joint_residual_oracle_20260827/run_oracle.py
```

## TokenSave boundary

The TokenSave MCP surface was unavailable and its SQLite graph had no current V7x/V80 nodes, so named current artifacts had to be read directly. If reporting this at <https://github.com/aovestdipaperino/tokensave>, strip sensitive or proprietary code from the issue description first.
