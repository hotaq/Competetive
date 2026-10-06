# EDA-B: Train/Test Drift + Ensemble-Error Analysis (playground-series-s6e9)

Read-only EDA. No Kaggle pushes, no repo edits outside `eda_out_B/`. Row order of train.csv preserved everywhere (never sorted). OOFs assumed aligned to train row order (all length 668665 = train rows; verified).

**Knowns (not re-reported):** income floor 30k, dead-zone 38-42k, 170k cliff, veto combos, commute 10-20km bump. Findings below are NEW.

## Artifacts
- `scripts/eda_drift.py` — single reproducible script for everything below
- `figs/01_adv_cat_pp.png`, `02_adv_numeric.png`, `03_rankcorr.png`, `04_disagree_slices.png`, `05_hard_concentration.png`, `06_rowll_hist.png`, `07_calibration.png`
- CSVs: `adv_categorical_pp.csv`, `adv_numeric_shift.csv`, `adv_numeric_decile_dpp.json`, `adv_logreg_topcoef.csv`, `oof_logloss.json`, `rank_corr.csv`, `pairwise_disagreement.csv`, `disagree_slices_*.csv`, `hard_slices.csv`, `depth2_probe.txt`, `depth2_leaves.csv`

## 1) Adversarial validation: effectively NO drift
- No unseen test levels in any categorical (Gender/City/Car/HomeCharg/Subsidy/RangeAnxiety all overlap 100%).
- Largest categorical gaps (test minus train): RangeAnxiety Medium −0.17pp, Subsidy Yes +0.15pp, City Urban −0.14pp. All < 0.2pp.
- Numeric decile-bin max |pp|: 0.09–0.17pp across all 7 numeric features; mean shifts ≤0.02 train-std (income +30 USD on mean 84.8k).
- Logistic adversarial AUC (50k/50k subsample) = 0.507 ≈ coin flip; top |coef| ≈ 0.02 (noise).
- Model prior check: all 6 calibrated OOF means ≈ 0.174, test-pred means match to ±0.0003 → no prediction shift.
- **Conclusion:** test is same distribution. Do not spend budget on drift adaptation; spend it on the error region (§3).

## 2) Model disagreement: LGBMs identical, diversity lives in MLP/GAM vs GBM on positives
- OOF logloss: lgb_te_s42_d4 0.2185 (best) < lgb_naji_s7_n5 0.2189 ≈ lgb_te2_s42_d4 0.2190 < GAM_h 0.2213 < LGBMte_f 0.2311 < HistGB_e 0.2325. naji_blend raw 0.6010 (see §4).
- Rank corr: LGBM↔LGBM 0.995–0.997; LGBM↔GAM ≈ 0.986–0.990; LGBM↔HistGB/LGBMte ≈ 0.972–0.982. So ordering is near-identical; gains must come from calibration/shape, not reranking.
- P(|Δp|>0.2): LGBM pairs 0.02–0.06% of rows; LGBM vs GAM 0.34–0.44%; LGBM vs HistGB 2.5–2.8%.
- Disagreement slices (best vs HistGB_e, rate of |Δp|>0.2): env 5.0 6.1%, env 4.0 6.0%, Subsidy=Yes 4.4%, commute 10–20km 4.4%, income 60–100k 3.6%, HomeCharg=Yes 3.1%. GAM shows same ordering at ~1/7 the rate. Tiny edge slices (income 170k+ 37.7% of n=393; dead-zone 38–42k 6.9% of n=1277) also light up vs HistGB.
- **Conclusion:** ensemble diversity is concentrated exactly in the high-positive-probability region where the hard rows live — keep GAM/HistGB for shape diversity, but recalibrate before logloss blending.

## 3) Hardest rows (top-decile per-row logloss of best single, thr 0.674, n=66867): one interaction explains them
- 97.9% of hard rows have Subsidy=Yes (hard rate 15.6% vs 10% base, +5.6pp). Next enrichments: income 100–170k 12.9% (+2.9pp, 35% of hard mass), commute 10–20km 12.0% (+2.0pp), Rural 11.0% (+1.0pp), HomeCharg=Yes 10.9% (+0.9pp).
- Easy poles: income <30k 4.4% (−5.6pp), income 42–60k 7.2% (−2.8pp), HomeCharg=No 8.0% (−2.0pp), commute 50km+ 8.8% (−1.2pp).
- Depth-2 probe predicting is_hard from raw features: AUC 0.816 with splits on Environmental_Concern_Level ⊕ Subsidy_Yes only. Leaves: Env≤2.5 & Subsidy=No (n=112490) hard 0.03%, pos 0.03% (free negatives); Env≤2.5 & Subsidy=Yes (n=170119) hard 2.1%; Env>2.5 & Subsidy=No (n=136266) hard 1.0%; Env>2.5 & Subsidy=Yes (n=249790, 37% of data) hard 24.8%, pos 44.7% — this one leaf holds most of the loss.
- **Conclusion:** residual error is not diffuse noise; it is the Env>2.5 × Subsidy=Yes × (100–170k income / 10–20km commute / Rural) positive pocket where P(Yes)≈0.45 but individual outcomes are near-coin-flip given current features.

## Top-5 findings
1. No train/test drift worth acting on (all gaps <0.2pp, adv AUC 0.507, no unseen levels, test priors match).
2. `naji_blend` OOF is rank-excellent (spearman 0.995–0.998) but catastrophically miscalibrated (mean 0.50 vs 0.174; its 0.4–0.6 bin, n=310k, has actual rate 3.3%). Never blend raw; recalibrate first.
3. Hard rows = Env>2.5 × Subsidy=Yes pocket (depth-2 AUC 0.816; top leaf 24.8% hard, 44.7% positive). That is where all logloss lives.
4. Calibrated-model disagreement concentrates on the same pocket (env4/5, Subsidy=Yes, 10–20km, 60–170k) — diversity exists, but only there.
5. Within-pocket ordering is already near-perfect (LGBM rank corr ≥0.995); remaining loss needs better probability shape / new signal, not reranking.

## Concrete ideas to fix errors
1. **Recalibrate-then-blend:** cross-fitted Isotonic (or Platt) on naji_blend + GAM/HistGB OOF before any logloss-weighted averaging; expect the blend's rank value to convert into real logloss gain.
2. **Pocket-specific calibrator / second-stage model:** fit a small calibrator (or shallow tree/GBM on OOF + Env×Subsidy×income×commute interactions) restricted to Env>2.5 & Subsidy=Yes; blend its output only inside the pocket.
3. **Explicit interaction features:** EnvConcern×Subsidy, EnvConcern×income-bin, Subsidy×commute-bin, Subsidy×Rural, HomeCharg×Subsidy — plus finer income cuts at 100k/170k and commute cuts at 10/20km; these target the exact disagreement/hard slices.
4. **New model types for shape diversity in the pocket:** kNN / kernel smoother on (income, commute, age, env) and a shallow MLP or GAM with pairwise interactions — families whose probability shapes differ from GBMs where it matters; validate by pocket logloss, not global.
5. **Loss-aware tuning:** optimize pocket logloss (or focal loss emphasizing p∈[0.3,0.7]) for one farm slot; keep the global best single untouched and ensemble at the pocket level.
6. **Do not:** chase drift adaptation, unseen-level handling, or global reranking — evidence says they are spent bullets here.
