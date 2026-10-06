# Mid-game route switch (day 12 / day 18) REPORT — PASS

Stage 0 connectivity (N3-live 136): all 12 splices at steps 288/432 are usable (0 errors,
no game < 20k, money ~98k).
Train (700): fixed switches beat SB682 (0.8414): S432_R120 0.8643, S432_R105 0.8600,
S288_R120 0.8614; R110 hurts (0.72–0.74).

Disclosed correction: the first policy used day-18 features to decide day-12 switches. That
is not implementable (lookahead), and its holdout result (+0.050) is void. The implementable
two-stage policy decides at 288 from day-12 features only (stay value = the fitted stage-2
decision), then at 432 from day-18 features if it stayed. Trees depth 3 (5-fold CV 0.8700),
fitted on train only. When I built it, I had already seen the holdout results of the fixed
options (S432_R120 +0.054, S432_R105 +0.058).

| gate | result | verdict |
|---|---|---|
| M1 holdout 258 vs SB682 0.7016 | 0.7558, +0.054, CI95 [+0.008, +0.101] | PASS |
| embedding check | the built main.py scores 0.7558 on the holdout (identical to offline; 257/258 games match an option run to the coin) | PASS |
| M2 arena, 17 agents, seeds 27100001–27100020 | 0.8882 vs SB682 0.8706 | PASS |
| M3 G3 official env, seed 27200001, both seats vs N3 | 81,151 / 79,368 equal to play.py | PASS |

Two bugs caught before shipping: Kaggle's loader picking a tree function as the agent, and a
deleted helper that silently disabled switching (the holdout check showed 0.7016 = SB682).
Bundle `submissions/n3_sb682_midswitch_standalone/submission.tar.gz` (sha256 57c701e7cafb966b…,
main.py only). Not submitted by this run. Submitting retires N3 (56521684), so the active pair
becomes this bundle + SB682.

Submitted 2026-09-26 on the human's order as **56578534** (tar sha256 57c701e7cafb966b…). Active pair is now 56578534 + SB682 56541172; N3 56521684 retired.
