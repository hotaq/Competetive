# Refresh 2026-09-23 REPORT — winner found, ONE bundle built, NOT submitted

Pipeline per `PREREGISTRATION.md` (frozen before any compute) on the frozen
Sept-22 pool — no re-harvest, no tuning, no gate relaxation. Live at start:
omar_base_tape_standalone (56361130, public 1431.8 COMPLETE); top-1 DSM 3166.0.

## Verdict: WINNER — W = yannik_schiffner__ep111952809__seat0 → build

| arm | holdout point | dPoint vs live | dMoney vs live (95% seed-cluster CI) | gate |
|---|---:|---|---|---|
| **W** yannik_schiffner ep111952809 seat0 | **0.8599** | **+0.0664** | **+2206.7 [+1262.9, +3145.9]** | G1 PASS, G2 PASS |
| W2 majkel1337 ep111933074 seat0 | 0.6698 | −0.1237 | −1398.2 [−2247.6, −560.9] | G1 FAIL, G2 FAIL (diagnostic only, never built) |
| live omar ep110807162 seat1 | 0.7935 | — | — | reference (Slot A) |

E0 determinism: 200/200 replayed games identical, 0 mismatches — PASS.
LIN lineage: W prefix-72 identity vs live route 0.0, vs live tape 0.0 (< 0.90) — PASS, not a clone.
G3: `candidate.json` standalone_validated (960/960 action-equivalence checks);
official-env smokes both seats DONE/DONE, 719/719 active steps, 0 invalid-key
steps (seed 8262026; candidate money 135146.0 either seat) — PASS.
Bundle: `submissions/refresh_yannik_standalone/` (main sha bf1fab558ba1…,
tar sha e500c0ce5629…).

## Screen (19500101–19500200; 92 tapes + ROUTER × 10 V100 opp × 2 seats)

ROUTER (live omar) pooled point 0.8670 — top of the board, no pool tape beats
it on this stale panel (same picture as the §8 dry run). Best per team:

| pooled point | team | best tape |
|---:|---|---|
| 0.8125 | Yannik Schiffner | yannik_schiffner__ep111952809__seat0 |
| 0.7190 | mtmr_s1 | mtmr_s1__ep111955057__seat0 |
| 0.6910 | KawattaTaido | kawattataido__ep111931862__seat1 |
| 0.6025 | Otter Vibe | otter_vibe__ep111939633__seat0 |
| 0.5210 | THIRD FARM CLUB | third_farm_club__ep111946168__seat0 |
| 0.5205 | Majkel1337 | majkel1337__ep111933074__seat0 |
| 0.4810 | Kaggledew Valley | kaggledew_valley__ep111943076__seat0 |
| 0.4760 | SpaTaro | spataro__ep111934253__seat0 |
| 0.4690 | THUNDER THUNDER | thunder_thunder__ep111943074__seat1 |
| 0.4470 | M & M & P & Q | m___m___p___q__ep111927190__seat0 |
| 0.4115 | QQ | qq__ep111953944__seat0 |
| 0.3900 | 吃白饭的大肥鱼 | 吃白饭的大肥鱼__ep111931862__seat0 |
| 0.2945 | Vadim Vasilenko | (cut) |
| 0.2655 | DECEM | (cut) |
| 0.1660 | DSM | (cut) |
| 0.1460 | Unknown Mother-Goose | (cut) |
| 0.1440 | Azat Akhtyamov | (cut) |
| 0.0955 | Orbital Terraformer | (cut) |
| 0.0010 | Densike | (cut) |

Top 12 teams forwarded to round-robin (prereg cut).

## Round-robin (19500201–19500400; 13 tapes all-pairs, live games excluded)

| pooled point | tape |
|---:|---|
| 0.8221 | yannik_schiffner__ep111952809__seat0 (**W**) |
| 0.6121 | majkel1337__ep111933074__seat0 (**W2**, diff-team, prefix-72 vs W 0.0) |
| 0.5935 | kawattataido__ep111931862__seat1 |
| 0.5863 | mtmr_s1__ep111955057__seat0 |
| 0.5406 | otter_vibe__ep111939633__seat0 |
| 0.4698 | m___m___p___q__ep111927190__seat0 |
| 0.4619 | qq__ep111953944__seat0 |
| 0.4446 | spataro__ep111934253__seat0 |
| 0.4398 | kaggledew_valley__ep111943076__seat0 |
| 0.3979 | third_farm_club__ep111946168__seat0 |
| 0.3321 | 吃白饭的大肥鱼__ep111931862__seat0 |
| 0.2994 | thunder_thunder__ep111943074__seat1 |

Ranking identical with self-games excluded (0.8514 / 0.6223 / …).

## Holdout per-opponent (19300001–19301000; 20,000 games/arm)

| opponent | W point | W2 point | live point |
|---|---:|---:|---:|
| kaggledew_valley seat0 | 0.9760 | 0.7810 | 0.7917 |
| kawattataido seat1 | 0.6970 | 0.2600 | 0.6960 |
| m___m___p___q seat0 | 0.9405 | 0.7855 | 0.9635 |
| mtmr_s1 seat0 | 0.7555 | 0.5590 | 0.6905 |
| otter_vibe seat0 | 0.9355 | 0.6080 | 0.7045 |
| qq seat0 | 0.6410 | 0.6410 | 0.9935 |
| spataro seat0 | 0.9070 | 1.0000 | 0.7940 |
| third_farm_club seat0 | 0.8605 | 0.8225 | 0.6900 |
| thunder_thunder seat1 | 1.0000 | 0.9325 | 0.8245 |
| 吃白饭的大肥鱼 seat0 | 0.8860 | 0.3085 | 0.7870 |

W weakest vs qq (0.641) and kawattataido (0.697); no opponent below 0.5.
W2 collapses vs kawattataido (0.26) and 吃白饭的大肥鱼 (0.31) — correctly rejected.

## Wall-times

Screen 62 s (186,000 games) → round-robin 26 s (67,600) → holdout 48 s
(120,000 rows) → E0 probe 1 s → build ~1 s → G3 seat0 2 s / seat1 3 s.
Total compute ≈ 2.5 min. Data: `screen.json`, `round_robin.json`,
`holdout.json`, `gates.json`, `e0_run{1,2}.json`, `g3_seat{0,1}.json`.

## Notes / limitations

- `--tape live=` collided by name with the `--routes` live router arm (single
  `live` arm in output). Harmless: verified the live tape actions[1:720] are
  byte-identical (719/719) to the v4 route 0, so either survivor plays the same
  omar replay; base/one/two/live router arms all scored identically (0.7935).
- Seeds 19500101–19500400 and 19300001–19301000 are now burnt — never reuse.
- Local transfer caveat (CLAUDE.md): holdout +0.066 point / +2.2k money over
  live is real but modest; live omar keeps decaying (~1466 → 1431.8 in a day),
  so firing the challenger into Slot B is the prereg hedge, Slot A stays omar.

## Submit command for the human (DOCUMENTED — NOT executed)

```bash
kaggle competitions submit -c kaggriculture -f submissions/refresh_yannik_standalone/submission.tar.gz -m "yannik_schiffner ep111952809 replay refresh 2026-09-23: holdout 0.8599 vs live 0.7935 (dMoney +2207 CI excl 0), G1+G2+E0+LIN+G3 pass, evidence analysis/refresh_20260923"
```

After COMPLETE + public score > omar's: select as Slot B final; Slot A remains
56361130 until then. Never deselect both incumbents in one move.
