# Newest public agents vs MS REPORT — PASS: W = Q1 (956a50f9)

Stage 1 (SELECT 164; MS 0.6341), h2h vs MS on seeds 27400001–27400020:
Q1 = Q2 = Q3 = Q4 (ttv1 / master-engine-v4 / harvest-ledger new; 2965 master new; demand-preserving
new; farmer-john idle seller) play identically: 0.7439, 43 up / 25 down, h2h 0.750.
Q5 0.4604 (h2h 0.10) · Q7 Lucky boy 0.3963 (h2h 0.15) · Q6 0.1220 (h2h 0.00). W = Q1.

| gate | Q1 | MS | verdict |
|---|---|---|---|
| F1 CONFIRM 164 (newest, untouched) | 0.7683 | 0.4268 | PASS: +0.341, CI95 [+0.25, +0.43], 67 up / 11 down |
| F2 arena, 17 agents, seeds 27500001–27500020 | 0.8985 | 0.8794 | PASS |
| F3 exceptions / crashed games | 0 / 0 | | PASS |
| G3 official env, notebook's own tarball (LICENSE + NOTICE + main.py), seed 27600001, both seats vs MS | 151,290 / 150,844 equal to play.py | | PASS |

Tarball: artifacts/public_notebooks_20260927/outputs/kaggriculture-ttv1/submission.tar.gz
(sha256 9e145f5fdc871813…). Open rules question (#743890, #743829): this public version was
run after the 23 Sep sharing deadline. N3 (the base of MS and SB682) was pulled on 24 Sep and
carries the same question. No staff ruling yet.

Submitted 2026-09-28 on the human's explicit order (after the rules-question disclosure) as **56614906** (tar sha256 9e145f5fdc871813…). Active pair is now Q1 56614906 + MS 56578534; SB682 56541172 retired.
