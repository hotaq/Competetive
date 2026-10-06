# Confirmation on an independent live panel (pre-registered before harvest)

Primary screen result (gates.json): W = `m___m___p___q__ep112738202__seat1`,
holdout 0.730 vs LIVE 0.595, dpoint +0.135, CI95 [−0.027, +0.297] → H1 FAIL,
H2/H3 pass. Verdict of the primary screen stays FAIL; this is a separate,
independent replication and does not re-open it.

## Panel

All public episodes of submission 56361130 (omar) — never used for any
selection in this project. Seed + opponent's recorded actions, our seat as played.
The panel is frozen to the episodes listed at harvest time.

## Arms (both counterfactual, so both carry the same can't-react bias)

- W = `artifacts/tape_pool_20260924_top10/tapes/m___m___p___q__ep112738202__seat1.json` (frozen, not reselected)
- Y = `artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json` (the live 56495539 tape)
- LIVE omar = recorded (sanity: omar tape replay must reproduce it exactly)

## Gates (frozen)

| gate | criterion |
|---|---|
| C0 | omar tape replay reproduces recorded money on every episode |
| C1 | point(W) − point(Y) >= +0.05, paired episode bootstrap CI95 lower bound > 0 |
| C2 | point(W) >= point(Y) on episodes with opponent score >= panel median |

Pass C0-C2 → build W bundle (single-route path of refresh_20260923) + G3;
Slot B (omar) replacement is the human's call. Fail → nothing built.
