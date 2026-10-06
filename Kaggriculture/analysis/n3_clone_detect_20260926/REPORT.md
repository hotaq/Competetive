# N3M clone-aware REPORT — FAIL (B2)

Stage A (N3-live 136): N3M-none 0.6324 (N3 0.5551, SB682 0.6397); the sibling-branch early
dumps all hurt (620: 0.4118, 644: 0.4706, 660: 0.4191). W = N3M-none (mirror branch only).

| gate | result | verdict |
|---|---|---|
| B1 SB682 live panel (173, untouched) | W 0.7312 vs N3 0.6994: +0.032, CI95 [+0.012, +0.058], 8 up / 0 down | PASS |
| B2 same panel vs SB682's recorded live result | W 0.7312 < SB682 0.7514 | FAIL |
| B3 arena, 17 agents, seeds 26300001–26300020 | W 0.9029 = N3 0.9029 (identical in every game; the detector never fires against non-clones) | PASS |
| B4 mirror vs N3, seeds 26400001–26400050 | 0.92 | PASS |
| B5 exceptions | 0 | PASS (G3 not run) |

Verdict: FAIL, nothing proposed. N3M is never worse than N3 anywhere tested, but it did not beat
SB682 on SB682's own opponents. In that 2100–2300 band, dumping at d29h10 also pays against
non-mirror siblings.
