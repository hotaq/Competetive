# Fourth Submission: Native Interactions

Target: exceed the current `0.821` best and move toward `0.826+`.

## Changes

- Native-categorical LightGBM with aligned category domains.
- Fold-local early stopping and numeric infinity/NaN sanitation.
- Leakage-safe nested-OOF frequency/target-encoded interaction specialist.
- Small-data bounded triple interactions; pair-only interactions on larger data.
- Cross-fitted regularized logistic stack over core OOF predictions.
- One guarded core/meta blend slot instead of several correlated blends.
- Family-reserved candidate export with gain, stability, and diversity gates.
- Observable post-fit time check and at most four modeled submissions.

## Validation evidence

The completed pre-stack 16-task simulator selected candidates by supplied Public
split and measured the corresponding Private split:

| Package | Mean Public AUC | Mean Private AUC | Mean overall AUC |
|---|---:|---:|---:|
| `02_portfolio` | `0.80207666` | `0.80409998` | `0.80308908` |
| Native + interactions | `0.80248406` | `0.80479836` | `0.80363657` |
| Gain | `+0.00040740` | `+0.00069838` | `+0.00054749` |

The final cross-fitted stack was additionally checked on task 10. It improved
that task from Public/Private `0.84978/0.86127` to
`0.85091/0.86220`. It remains guarded by a minimum `0.0005` OOF gain and is
omitted when it does not beat the simpler core blend.

## Package

- Agent: `submissions/04_native_interactions/agent`
- Archive: `submissions/04_native_interactions/submission.zip`
- Upload requires explicit approval of the final archive SHA-256.
