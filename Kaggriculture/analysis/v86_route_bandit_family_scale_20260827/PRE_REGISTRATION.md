# V86-A Family-Scale Route Bandit Pre-registration

Status: **FROZEN BEFORE ANY V86 CANDIDATE ECONOMIC OUTCOMES** on 2026-08-27.

V86-A is the final cheap falsifier for the exact V85 ridge route-bandit. It adds opponent-family diversity without changing the V67 baseline, route actions, observable encoder, ridge model, uncertainty rule, or safety threshold.

## Frozen experiment

- Candidate: frozen standalone V67, SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`.
- Actions: `KEEP_V67`, `FORCE_B120_SUFFIX`, `FORCE_C120_SUFFIX`.
- Families: the four consumed V84 implementation families plus four Top20 replay-tape families from four distinct submissions.
- Seeds: exactly `27100200..27100207`, both seats.
- Size: 128 fresh contexts and 384 full Rust-engine games.
- Features: the unchanged V85 observation-only encoder at steps 24, 48, 72, 96, and 120; 405 finite values per context.
- Model: unchanged dual ridge with `alpha=1000`, `beta=2`, and strict switch threshold `2500`; there is no hyperparameter search.
- CV: 64 family-seed cluster folds. Each fit excludes the complete held family and every row sharing the held seed, leaving 98 training rows.

The four new Top20 families were selected using preexisting snapshot rank/team identifiers and static public tape actions. Selection did not read any new V86 rewards or outcomes, nor any V67/B/C rewards or margins. It used an operation-frequency fingerprint over six 120-step bins, deterministic farthest-first selection anchored at rank 1, and one medoid tape per submission. Ordered backups may replace a selected tape only after a mechanical import/call/completion failure and before any margin table is read.

## Gates

Mechanical validity requires all 384 games to reach step 719, exact V67 prefix parity through step 119, route first-difference no earlier than step 120, zero candidate sanitizer/fallback/exception/partial events, a finite identifier-free `128 x 405` matrix, pinned source hashes, and deterministic core-hash reproduction.

The dataset is adequate only with at least 12 V67 loss rows in at least six family-seed clusters and three families, while the B/C oracle rescues at least 75% of those losses across at least three families.

The selector passes only with 1–32 interventions, every intervention producing strictly positive paired delta and final margin, at least 75% of V67 losses and six loss clusters rescued, at least 122/128 wins, at least 15/16 wins in every family, and no family-level win/worst/mean regression against V67.

Any missed gate stops this exact ridge branch. Threshold, model, actions, families, and seeds may not be changed after outcomes are observed. Even a pass authorizes only a separately pre-registered blind validation; V86-A cannot create a checkpoint, open the sealed holdout, create a submission, or upload to Kaggle.

The complete machine-readable contract is in `protocol.json`.
