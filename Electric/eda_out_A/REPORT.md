# Conditional-Interaction EDA (A): Simpson segments, Env x income crossings, veto combos, calibration

Data: train.csv 668665 rows, base buy rate 0.1746. Test 286571 rows. Read-only; no Kaggle pushes.
Already-exploited (NOT re-proposed): digit features, freq enc, Income TE / IncEnv TE, orig-means,
is_30k / is_deadzone / is_170537 / is_env_hater, buy_score / buy_margin / worry_score,
Afford_Per_Car, NoHome_x_stations, ZeroGate, trailing-zeros, quantization bins.

## 1. Univariate anchors

- Env: 1:0.57% 2:2.14% 3:11.09% 4:24.88% 5:51.83% (dominates).
- Subsidy Yes 27.47% vs No 0.58% (near-necessary gate).
- Home Yes 19.58% vs No 12.71% (reverses inside City strata, see 3).
- RangeAnx High 0.14% (3/2194), Medium 4.17%, Low 18.90%.
- City Rural 19.34% > Suburban 18.09% > Urban 16.11% (overall; flips conditional, see 3).
- Cars / Gender / CarType all <1pp spreads (weak).

## 2. Per-Env income curves: NO crossings, fan-out + two spikes

`env_x_income.csv`, `fig_env_income_curves.png`, `fig_env_income_counts.png`.

- Order Env1<Env2<Env3<Env4<Env5 holds in EVERY income bin (12/12). No crossing to exploit.
- Shape is fan-out, not parallel: Env5 rises 17.7% (0-30k) to 76.9% (140-170k); Env1 stays 0.1-1.5%.
  HighIncome (>=100k) lift grows with Env: +1.2pp (Env1+SubYes) ... +22pp (Env5+SubYes 61.0%->83.2%).
- Dead-zone 30-40k is 0/573 except Env5 1/84; 0-30k bin is NOT dead (0.09-17.7%).
- 170k+ bin is 393/393 Yes in all Envs (min 170537.0) — already covered by banned is_170537 flag; tiny (0.06%), do not duplicate.

## 3. Simpson flip: City effect reverses inside Home strata (TRUE sign flip)

`simpson_*.csv`. No flips within Env strata (all 19 features keep sign in all 5 Envs).
Within Home_Charging_Possible, 6 flips, biggest:

| feat (overall lift) | in Home=Yes | in Home=No |
|---|---|---|
| Urban -2.39pp | +0.60pp (20.04 vs 19.43) | +5.81pp (13.56 vs 7.75) |
| Rural +2.30pp | -0.08pp* | -8.23pp (4.64 vs 12.87) |
| HighHomeSt -1.23pp | flat | +3.72pp |
| ZeroHomeSt +1.17pp | flat | -6.82pp |

*Rural/HomeYes ~19.8 vs 19.9 essentially tied; the HomeNo reversal is the signal.
Mechanism: 60.7% of Urban rows have Home=No vs 3.3% Rural / 10.3% Suburban; Env/Subsidy means by City are flat, so Home access fully explains the overall Urban deficit.
Pocket: Urban&HomeNo = 13.56% (n=175669, 26.3% of train; 75185 / 26.2% of test) vs Rural&HomeNo 4.64% (n=4074). Additive expectation misses it by ~4pp (see 4).

## 4. Non-additive pockets (actual minus additive expectation)

`twoway_additive.csv`, `fig_twoway_resid.png`. Largest |residuals|:

- Env5 x SubNo: actual 2.48% vs expected 34.94% => -32.5pp (n=33567). Env as main effect over-predicts without subsidy.
- Env5 x AnxHigh: 1.32% vs 34.5% => -33.2pp (n=227). Env5 x AnxMed: 17.4% vs 38.5% => -21.1pp.
- Env1 x SubNo: 0.02% vs -16.3%(floor) => +16.3pp truncation artifact; Env1xAnxHigh similar.
- SubNo x HomeNo: 0.40% vs -4.2%(floor) => +4.6pp floor artifact.
- Urban&HomeNo discussed above is the largest POSITIVE non-floor pocket worth modeling.

## 5. Veto / jackpot combos (n>=500)

`veto_combos.txt`. Hard zeros (exactly 0.0000):

- SubNo&AnxHigh 0/887; Env2&AnxHigh 0/509; AnxHigh&Old55 0/777; SubNo&AnxHigh&LongComm 0/779.
- Env2&SubNo&AnxMed 0/5476; SubNo&AnxMed&LoInc50 0/3441; Env1&AnxMed&LoInc50 0/2229.
- Env2&SubNo&ZeroWorkSt 0/1878; Env1&SubNo&ZeroWorkSt 0/2035; Env1&SubNo&Old55 1-2/19815 (~0.00%).
- Broad veto Env1&SubNo 0.02% n=59674 (8.9% of train; test 8.87%) — single biggest deterministic slice.
- AnxHigh overall 3/2194; all 3 are Env5+SubYes. AnxMed&SubNo 25/25283 (0.10%) vs AnxMed&SubYes 6.94%.
Jackpots: Env5&SubYes&HiInc100 83.20% n=35514 (test 5.32%); Env5&SubYes&HomeYes 72.44% n=69500;
Env5&SubYes&ZeroHomeSt still 70.46% n=10065 (stations do not veto the jackpot).

## 6. Calibration Env x Subsidy x Home (20 cells, tight CIs)

`calib_env_sub_home.csv`, `fig_calib.png`.

- Sub=No column: all 10 cells <=2.8% (Env5/No/Yes 2.76% n=22644; Env5/No/No 1.89% n=10923; Env1-2/No/* <=0.06%).
- Sub=Yes column fans by Env: E1 ~0.7-1.1%, E2 ~2.5-4.0%, E3 13.5/20.9%, E4 30.5/43.5%, E5 60.7/72.4% (No/Yes home).
- Home lift conditional on Env+Sub: ~+0.4pp (E1) ... +13pp (E4SubYes) / +11.7pp (E5SubYes). Monotone except noise.
- Use directly as smoothed segment prior (see features).

## 7. Commute 10-20km peak in EVERY Env (inverted-U, new)

`fig_commute_env.png`, `fig_comm_homeSt.png`, `comm_x_homest.csv`.

- Per-Env curves all peak at 10-20km: Env5 62.9% vs 54.8/52.2/51.0/48.5; Env4 34.2% vs 26.6/25.1/24.3/22.5; same shape E1-E3. Peak bin n~7-8.7k per Env, test 5.74%.
- Pooled homeSt columns also peak at homeSt=1 within most commute rows (e.g. 10-20km: st1 31.7% vs st0 25.7% vs 8+ 23.8%).
- Long commutes (40+) buy LEAST in every Env (range-anxiety gradient), but the 10-20 sweet spot is the non-additive part.

## 8. Stations gradient exists ONLY when Home=No (refinement, not re-proposal)

`stations_within_envhome.csv`.

- Home=No pooled: st0 6.0% -> st2 10.7% -> st5 12.2% -> st10 15.3% monotone; per Env same (E5HomeNo 24.3%->48.1%; E4HomeNo 8.8%->21.3%; E3HomeNo 3.1%->9.1%).
- Home=Yes pooled: flat 17.9-20.8% across st0-st10; per Env flat (E5HomeYes 52-57%, E4 25-29%, E3 11-14%).
- Banned NoHome_x_stations product already captures existence; the TIER + Urban&HomeNo cross below refines it.

## 9. Age hump at 50-60 in every Env (small, consistent)

`age_x_env.csv`: 50-60 is max in E1 (0.65), E2 (2.46), E3 (12.02), E4 (26.51), E5 (54.10); 60+ is min in E1/E2/E3/E4 and 2nd-min E5. Spread ~1-4pp. No crossing.

## New-feature formulas (concrete; none overlap banned list)

```python
# F1 subsidy/anxiety veto gate (hard prior; covers 6 exact-zero combos)
anx_no_sub = ((df.Range_Anxiety_Level != "Low") & (df.Subsidy_Available == "No")).astype(int)
# rate: AnxMed&SubNo 0.10%, AnxHigh&SubNo 0/887. Test coverage ~9%.
subno_env1 = ((df.Environmental_Concern_Level == 1.0) & (df.Subsidy_Available == "No")).astype(int)
# rate 0.02% n=59674. Strongest single veto slice.
# F2 Simpson cross: city x home (6-level + smoothed stat; do NOT use city or home alone)
city_home = df.City_Type.str[:3] + "_" + df.Home_Charging_Possible  # Urb_No 13.56% vs Rur_No 4.64%
urban_nohome = ((df.City_Type == "Urban") & (df.Home_Charging_Possible == "No")).astype(int)
# test 26.2%. Smoothed target-mean of city_home (min_samples ~500) recommended.
# F3 commute sweet spot (inverted-U; peak holds in all 5 Envs)
commute_peak = df.Daily_Commute_km.between(10, 20).astype(int)  # +5-8pp vs neighbors
comm_dist15 = (df.Daily_Commute_km - 15).abs()  # monotone-bad distance alternative
# F4 home-no station tier (ordinal, NOT raw product; flat when Home==Yes by construction)
# 0: Home==Yes (any st); 1: Home==No & st<=1; 2: Home==No & 2<=st<=4; 3: Home==No & st>=5
homeno_st_tier = pd.Series(0, index=df.index)
mask = df.Home_Charging_Possible == "No"
homeno_st_tier[mask & (df.Charging_Stations_Near_Home <= 1)] = 1
homeno_st_tier[mask & (df.Charging_Stations_Near_Home.between(2, 4))] = 2
homeno_st_tier[mask & (df.Charging_Stations_Near_Home >= 5)] = 3
# pooled rates tier1~6.5% < tier2~10.7% < tier3~13%; Home==Yes tier0 ~19.6%.
# F5 fan-out + calibration priors (no income TE, no digits, no freq)
env_hiinc = df.Environmental_Concern_Level * (df.Annual_Income_USD >= 100000).astype(int)
# captures HiInc lift growing with Env (+1pp E1 .. +22pp E5|SubYes).
age_50s = df.Age.between(50, 60).astype(int)  # consistent +1-3pp hump in all Envs
seg_env_sub_home_rate = leave-one-out_smoothed_mean(  # 20-cell table in calib_env_sub_home.csv
    df, keys=["Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible"], alpha=50)
# jackpot anchor E5/Yes/Yes 72.4%, veto anchor E1/No/* 0.02%.
```

## Files in eda_out_A/

- cond_eda.py, cond_eda2.py (scripts); console.log, console2.log
- figs: fig_env_income_curves, fig_env_income_counts, fig_twoway_resid, fig_calib, fig_comm_homeSt, fig_commute_env
- tables: env_x_income, simpson_env, simpson_Subsidy_Available, simpson_Home_Charging_Possible,
  twoway_additive, calib_env_sub_home, veto_combos.txt, comm_x_homest, age_x_env,
  stations_within_envhome, univ_*.csv

## Limits

- Additive residuals near 0/1 boundaries are floor/ceiling artifacts (flagged, not modeled).
- 170k+ 100% slice (n=393) intentionally left to existing is_170537 flag.
- No crossing of Env income curves found: do not engineer Env-order flips; model fan-out instead.
