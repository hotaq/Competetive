# Validation Method For The 4743.93 Pack

## Result

- Kaggle status: `SubmissionStatus.COMPLETE`
- Public score: `4743.93`
- Submitted artifact: `submission.zip`
- ONNX files in the zip: `394`
- `submission.zip` SHA-256: `28f34d909fd570da918c47d37213349549889748038439a34161d492f7756633`
- Task table rows: `400`

## Validation Ladder

The validation ladder used for this pack was:

1. Python/reference exactness on visible examples where a local rule existed.
2. ONNX checker and graph hygiene.
3. ONNX Runtime exactness on visible examples where applicable.
4. `onnx_tool` profile success where applicable.
5. File size and banned-op checks.
6. Pack-level compatibility checks.
7. Full-pack preflight.
8. Kaggle public confirmation.

## Final Pack Evidence

| field | value |
| --- | --- |
| `preflight_ok` | `true` |
| `pack_hygiene_ok` | `true` |
| `compatibility_ok` | `true` |
| `compatibility_bad_count` | `0` |
| `task_count` | `394` |

The explicit compatibility task ids were:

- `task012`
- `task017`
- `task044`
- `task058`
- `task139`
- `task243`
- `task308`

## Interpretation

The final pack is a lineage artifact. A row can be:

- included in the final Kaggle-complete pack
- inherited from a previous complete pack
- sourced from a public kernel
- generated locally
- selected as a local overlay
- selected as a public reliability repair
- dropped after public-confirmed zero-loss checks

That distinction matters because local visible-example exactness is not the same as public-score reliability.

## Main Lesson

The final public result improved while local estimated score went down. Public reliability mattered more than keeping the locally strongest candidate for every task.

