# HANDOFF — EV Purchases (playground-series-s6e9)

อัปเดตล่าสุด: 2026-09-05 (ดู status kernel ล่าสุดด้วยคำสั่งด้านล่างก่อนทำอะไรต่อ)

## 1. การแข่งขัน
- Slug: `playground-series-s6e9` | Title: Predicting Electric Vehicle Purchases
- Deadline: 2026-09-30 | ประเภท Playground, metric คือ AUC (ดูจาก leaderboard ~0.946 + sample เป็น probability)
- LB ล่าสุดที่เช็ค: ที่ 1 Chris Deotte 0.94656, top-20 เกาะกัน 0.94632–0.94656 (gap แค่ 0.00024 = ใกล้ ceiling)
- เช็ค LB: `kaggle competitions leaderboard -c playground-series-s6e9 --show | head -25`

## 2. ข้อมูล (ในโฟลเดอร์นี้)
- `train.csv` 668,665 x 15 | `test.csv` 286,571 x 14 | `sample_submission.csv` | target `Will_Buy_EV` (Yes 17.46%)
- ไม่มี missing/dup, train/test ไม่ shift (diff < 0.3%)
- Features: numeric 7 (Age, Income, Commute, Ncars, HomeStations, WorkStations, Env 1-5) + categorical 6 (Gender, City, Car, Home, Subsidy, Anxiety)
- ไฟล์สคริปต์: `eda.py`, `eda2.py`, `deep_dive.py`, `hill_climb.py`, `veto_test.py`, `rand_screen.json`

## 3. ข้อค้นพบสำคัญ (อย่าเสียเวลาซ้ำ)
- Univariate AUC: Env 0.8435 > Subsidy 0.718 > Income 0.670 > ที่เหลือ ~0.5 (noise: Age, Ncars, station counts)
- มี veto combos 813 กลุ่ม (11.7% ของ train) ที่เป็น No 100% — แต่ LGBM เรียนรู้ไปแล้ว (ทาย veto rows เฉลี่ย 0.0014) เอาไป post-process (zero-out) แล้ว AUC ลด 0.94227 -> 0.94204 อย่าทำ
- Random features 400 ตัว: ตัวแรงคือ Env x {Subsidy, Income, Age, Ncars} ทั้งหมดมีใน FEv1 แล้ว ไม่มีตระกูลใหม่
- Hill climb forward-selection 9 candidates บน LGBM+FEv1: gain +0.00001 -> สรุป LGBM อิ่มที่ ~0.942 ปั้น feature ต่อไม่พาไป 0.946
- LogReg บน tiny data เก่งพอๆ กับ LGBM (0.9404 vs 0.9379) -> ฟังก์ชันจริงค่อนไปทาง additive เก็บ LogReg ไว้ใน ensemble เสมอ

## 3b. Deep-dive EDA 2026-09-05 (สำคัญมาก — เปลี่ยนแผน)
- **Original dataset เจอแล้ว**: Kaggle `itzzomkar/ev-adoption-behavior-and-range-anxiety` (10,000 แถว คอลัมน์เหมือน train เป๊ะ, Yes 17.5%) — copy ไว้ที่ `orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv`; บน kernel ใส่ `dataset_sources: ["itzzomkar/ev-adoption-behavior-and-range-anxiety"]`
- **Income = pointer ไปยังคนใน original**: 97.9% ของ train/test มี Annual_Income_USD ตรงกับค่าใน original (7,089 ค่า) แต่คอลัมน์อื่นไม่ตรง (Age ตรง 2%, Commute 2%) → Income ทำตัวเป็น categorical ID ไม่ใช่ตัวเลข; test income 99.4% เห็นใน train
- **ผลทดลอง (LGBM depth3, 5-fold seed42, full data)**: baseline leaves63 0.94153 → depth2 0.94220 → +Income exact-value target-encoding (m=50, in-fold) **0.94464** → +Income_cnt(train+test) +orig-label-by-income **0.94548** (+0.004 จาก baseline, LB top 0.94656) → +(Income,Env) TE เพิ่มอีก ~+0.0002 | สคริปต์: `eda3_income_te.py`
- Commute/Age exact-value TE ไม่เพิ่ม (Income กินหมดแล้ว); Commute ค่าดึงจาก original 100% แต่ไม่ link กับแถวเดียวกัน
- **ฟังก์ชันจริงเกือบ additive**: stumps(leaves=2) 0.94081 ≈ full LGBM 0.94073, depth-2 0.94190 ดีกว่า → ใช้ต้นไม้ตื้น (depth 2-3), logreg/GAM ควรแข่งได้; leaves=63 overfit
- Feature importance (drop test): Env -0.10, Subsidy -0.052, Income -0.014, Anxiety -0.0017, Commute -0.0005, Age -0.0002, ที่เหลือ ≈0 (Gender/City/Car/Ncars/Stations ตัดได้ไม่เสีย)
- Per-value effect ของ Age/Stations/Ncars เป็นของจริง (split-half corr 0.78-0.99) แต่เล็ก; Age effect รูปเดียวกันทุก Env (additive)
- Artifacts: Income floor 30000 (9.2% แถว, Yes 4.4%), Commute floor 5.0 (21.6%), Income yes-rate ไม่ monotone (dip 38-40k=0%, 48k, 62k=2.5%; bump 68k=20.6%, 128k=42.6%), Commute 10-20km bump 25%, Anxiety High = veto (0% ที่ Env1-4), Subsidy No ≤2.8%
- **Next**: ใส่ Income_te (in-fold!) + Income_cnt + Orig_lbl + (Income,Env)_te เข้า farm_lib.add_features แล้ว push batch ใหม่ด้วย depth 2-4; ลอง logreg one-hot(Income as cat) ด้วย

## 4. Baseline (local, CPU)
- `baseline_lgbm.py` -> OOF 0.94177 (5-fold, raw+cat, lr 0.05/leaves 63) -> `submission.csv` (submit ได้เลยถ้าต้องการคะแนนแรก)
- `v2_push.py` = FEv1(13 features) + LGBM-tuned + CatBoost + XGB blend (ยังไม่เคยรันจบ local)
- FEv1 (`farm_lib.add_features`): Income_log, TotalStations, StationDiff, Commute_per_station, SubsidyYes, HomeYes, AnxietyOrd, Env_x_Subsidy, Income_x_Env, Env_x_Home, HomeSubsidy, City_Car, Env5, AnxietyHigh

## 5. Kernels บน Kaggle (เทรนบน Kaggle เท่านั้น อย่าเทรน local)
- `chinnaphatkhun/ev-s6e9-push-v2` — notebook เบลนด์ 3 โมเดล (v3 ต่อ data ถูกแล้ว) | เช็ค: `kaggle kernels status ...` / `logs ...`
- `chinnaphatkhun/ev-farm-it1` — ensemble farm batch แรก (10 models, script kernel `train.py` ไฟล์เดียว) | โค้ดต้นทาง: `farm_it1/`
- บทเรียน: kernel-metadata.json ของ CLI 2.2.2 ต้องใช้ key `competition_sources` (ไม่ใช่ `competition_data_sources`) ไม่งั้น /kaggle/input ว่างเปล่า; script kernel เห็นแค่ `code_file` ไฟล์เดียว ต้อง bake lib ใส่ไฟล์เดียว (controller ทำให้แล้ว)

## 6. Ensemble farm (ระบบหลักตอนนี้)
- `ensemble_farm/farm_lib.py` — zoo 18 configs (lgb/cat/xgb/hgb/logreg x base/fe x seeds), fixed 5-fold seed 42 (OOF เทียบกันได้ข้าม batch), greedy blend (Caruana)
- `ensemble_farm/kernel_train.py` + `farm_lib.py` — source (อย่า push ตรง ต้องผ่าน controller เพื่อ bake เป็นไฟล์เดียว)
- `ensemble_farm/controller.py` — สั่งงาน: `new <N> [prev_slug]` / `push <N>` / `status-it <N>` / `pull <N> <slug>` / `submit <N>` / `quota`
- BATCHES: it1 = 10 ตัว deep/no-TE (**ถูก cancel** ได้แค่ 3 ตัว OOF 0.9414-0.9418 ยืนยันเพดาน no-TE), it2 = 8 ตัวที่เหลือของ zoo เดิม, **it3 = 12 ตัว shallow+TE เสร็จแล้ว: BLEND OOF 0.94559 → LB 0.94573 (rank 222/776, submit 2026-09-05)**, it4 = 10 ตัว shallow+TE ที่เหลือ
- it3 solo OOF: lgb_te d2/d3/d4 ≈ 0.9455 (ดีสุด), cat_te ≈ 0.94543, xgb_te ≈ 0.9453, hgb 0.94527, logreg_te 0.94217 | blend เลือก 6 ตัว (lgb×4 + cat×2) กำไรจาก blend +0.0001 เท่านั้น → โมเดลใกล้กันมาก ต้องการ diversity/feature ใหม่ ไม่ใช่ seed เพิ่ม | output ที่ `farm_out_it3/`
- **it5 = 12 ตัว (lgb/cat/xgb te) 10-fold — DONE 2026-09-05: BLEND OOF 0.94606 แต่ LB 0.94573 เท่า it3 (10-fold overfit OOF, corr ระหว่างโมเดล 0.999+) | solo lgb_d4 OOF 0.94597 ส่ง solo แล้วรอสกอร์**
- **it6 = 6 ตัว te2 (base + digit 56 + freq 69 + Income TE, 143 feats) 5-fold — DONE 2026-09-06: BLEND OOF 0.94578 (solo lgb_d4 0.94570) ต่ำกว่า it5 — submit แล้วรอ LB (OOF เคยหลอก)** | solo it5 LB 0.94569 (blend 0.94573 ดีกว่านิดเดียว) — digit (+0.00143) + freq (+0.00119) ตาม Naji/Dvorkin; batch: lgb_te2 d3/d3/d2/d4 + cat_te2_d4 + xgb_te2_d3 | ทดลอง local: 5-fold 0.94554 → 10-fold **0.94592** (+0.0004 = lever ใหญ่สุดที่เจอ เพราะ TE คำนวณจาก 90% ของแถว) | fold count ตั้งใน `controller.N_SPLITS_BY_BATCH`, ฉีดเป็น `N_SPLITS_OVERRIDE` ใน train.py
- ทดลองแล้วไม่ช่วย (2026-09-05, ±0.0001): pair TE (Income,Subsidy/Anxiety/Home), orig-row lookup features, income-group stats train+test, nested TE, residual TE, LGBM native cat Income (แย่ -0.0025), ต่อ orig rows เข้า train (แย่), Commute_cnt, stacker บน OOF (= greedy blend) | ช่วยนิดเดียว: Income_te m=10 เพิ่ม (+0.0001, ใส่ใน te set แล้ว)
- Feature sets ตอนนี้: base | fe | te (base + Income_te, IncEnv_te, Income_cnt, Orig_lbl) | tefe (fe + TE) — TE คำนวณ in-fold ด้วย folds คงที่ seed 42 (`farm_lib.add_te_features`), ต้องส่ง y/folds/orig ให้ `build_matrices`
- kernel ต้อง attach `dataset_sources: ["itzzomkar/ev-adoption-behavior-and-range-anxiety"]` (controller ใส่ให้อัตโนมัติ) — kernel_train หา `/kaggle/input/**/EV_Adoption*.csv` เอง
- zoo ตอนนี้ 40 configs; lgb รองรับ `depth` ใน params แล้ว (max_depth)
- Output ต่อ iteration: `farm_out_itN/` (oof_*.npy, test_*.npy, submission_farm.csv, weights.json, manifest.json) + `farm_log.csv`
- **mega (local, ไม่ต้องเทรน): `farm_out_mega/` = greedy blend ข้ามทุก batch (it3+it5+it6 = 30 models → 8 picks, OOF 0.94609) — LB 0.94580 👑 (สูงสุดตอนนี้)** — ทุกครั้งที่ batch ใหม่จบให้ re-run `/tmp/mega_blend.py` แล้ว submit ใหม่
- **it7 = 6 ตัว tefe2 (fe + digit + freq + TE, 171 feats) 5-fold — push 2026-09-06 (`chinnaphatkhun/ev-farm-it7`, RUNNING)**
- **it8 = 6 ตัว naji-set (port `najiama/pure-lgbm` 0.94612: dual sklearn TE ทุก string-cat + orig-means + str-cats + magic flags, 141 feats; lgb lr0.02/col0.3) — push 2026-09-06 (`chinnaphatkhun/ev-farm-it8`, RUNNING)** — smoke local ผ่าน (train 4k/30 rounds ได้ AUC); trainer รองรับ `a`/`es`/`max_bin` แล้ว — batch: lgb_tefe2 d3/d3/d4 + cat_tefe2_d4 + xgb_tefe2_d3 + logreg_tefe2
- Quota guard: ไม่ auto-submit ถ้าวันนี้ submit ไปแล้ว >= 15 (limit ~20/วัน)


## 6b. Code-sharing review 2026-09-15 (อ่าน 18 public notebooks — อย่าอ่านซ้ำ)
- **LB 0.94643-0.94650 ทุกตัวคือ blend ของ public test-prediction** (talhatursun daily rank-average → Taeyang lexsort → beraterolelk) จูน weight บน LB ไม่มี OOF = ไม่ใช่สัญญาณใหม่ | megayak พิสูจน์: source ที่ Spearman >0.998 กับ blend ไม่ช่วยที่ weight ใด; RealMLP (0.9964) ตัวเดียวที่ decorrelated ให้ +0.00002
- **Boundary rules + lexsort tie-break (Taeyang/Berat)**: ทดสอบบน blend OOF ของเรา = cliff≥170537 +0.000001, dead 31004-41970 +0.000005, commute≥83 0, 30k-cell -0.00002, lexsort 0 → **ไม่ต้องทำ** โมเดลเราจัดอันดับพวกนี้ถูกอยู่แล้ว
- **Single model ที่ดีสุดบน board (ทุกคน 5-fold seed42 stratified = fold เดียวกับเรา)**: megayak one-lgbm (10f 0.94626/LB 0.94633) → Mizushima competition-only (5f **0.94610**/LB 0.94636) → jazivxt Zoom Zoom (5f **0.94613**) | สูตร: inc_d1-3, inc_mod100/1000, km_d1, km_mod100, quantisation ladder inc_q{50..5000} km_q{5..50}, flags (30k, cliff, dead 38-42k, env1), fold-local frequency ของทุก key (inc exact/100/1000, km int, ทุก cat + Age/Ncars/stations/Env), **triple TE sklearn (auto/10/100, cv=5) บน key เดียวกันทั้งหมด**, LGBM lr0.02 depth5 leaves32 **min_child 10** subsample0.8 colsample0.3 alpha0.071 lambda2 3500r ES100 (~111 feats) | jazivxt เพิ่ม income neighbor stats (bin q=8192/16384: central/symmetric/left/right/slope/curvature/logcount, inner 5-fold) ได้ +0.00001 เท่านั้น
- port "naji" ของเราได้แค่ 0.94577 vs 0.94610 ของ Mizushima บน fold เดียวกัน → **สิ่งที่สูตรเขามีแต่เราไม่มี**: (1) smooth keys = income//100 และ income//1000 + commute int เป็น key สำหรับ freq+TE (pool ค่าใกล้กัน จับ pointer ที่ถูก perturb) (2) TE channel ที่ 3 smooth=100 (3) quantisation ladder inc_q{50,100,250,500,1000,2500,5000} km_q{5,10,25,50} + inc_mod100/1000 km_mod100 (4) frequency fit เฉพาะ fold-train (ไม่ใช่ train+test) (5) params ของเราตรงกับเขาแล้ว (min_child 10, alpha 0.071, l2 2, bag 0.8) ต่างแค่ max_bin 1024 vs 255 → ช่องว่างมาจาก feature ล้วน | ที่เรามีเกิน: orig means, digit ทุก power | **แต่** naji_v1/Pure LGBM V3 (สายเดียวกัน, OOF 0.94606-0.94618) อยู่ใน blend แล้ว → member ใหม่จากสูตรนี้คาด +0.00001 ไม่คุ้ม kernel 2 ชม.
- **blend v3 + RealMLP (2026-09-15)**: greedy 114 members → 8 picks OOF **0.94639** (naji_v1 .25, it5_lgb_d4 .25, PureV3 .25, realmlp .125, it5_cat_d5 .125) = +0.00001 จาก 0.94638; ft transformer ไม่ถูกเลือก | **submit 2026-09-15 → LB 0.94634** (ต่ำกว่า v4 0.94636 ไป 0.00002 = noise ล้วน; OOF +0.00001 ไม่สะท้อนบน public) → best public ยังเป็น v4 (56102861, oof 0.94638, 3 members)
- **blend v4 + public OOF libraries (2026-09-17, `stack_v4.py` → `farm_out_v4/submission_v4.csv`, ยังไม่ submit)**: 124 members → 8 picks เท่ากัน 0.125: mega6_ensemble, mega6_B, mega6_D, mega_realmlp3, naji_xgb10, naji_v1, it5_lgb_d4, it5_cat_d5 → **OOF 0.94647** (จาก 0.94639) → **submit 2026-09-17 LB 0.94641 = best public ใหม่** (56306870) | nested 5-fold: gain vs naji_v1 บน held fold +23..+37e-5 บวก 5/5 → ของจริง | คาด LB ≈ 0.94645 (megayak OOF-only stack ได้เท่านี้) | หมายเหตุ: สมาชิก mega*/naji_xgb10 เป็น 10-fold OOF (optimistic กว่า 5-fold เล็กน้อย)
- **meta-stack ทดสอบแล้ว 2026-09-17 (`stack_v5_meta.py`, nested 5-fold บน 16 สมาชิกแรงสุด)**: greedy 0.94647 held-mean | Powell weights -0.6e-5 (1/5) | logreg C0.01/C1 -1.1/-0.7e-5 (2/5) | Lasso -1.6..-3.2e-5 (0/5) → **greedy เท่ากับดีสุด อย่าทำ stacker อีก** | ไฟล์ `farm_out_v5/submission_v5_meta.csv` = greedy เต็ม 16 สมาชิก ≈ v4 ไม่ต้อง submit
- **it20 = 4 feature-view members (push 2026-09-17, `chinnaphatkhun/ev-farm-it20`, 5-fold seed42, RUNNING)**: lgb_win (te2−digits + centred-window income±2/10/50/200 commute±1/5 rates m2/30/300 + lift vs original), lgb_nokey (te2 − exact-income keys + ladder //10/50/500/5000 TE m20/200 + windows m20/200), lgb_init (te2 − exact TE; Income_te_m10 เป็น init_score ไม่ใช่ feature), lgb_te2_lin (linear_tree) | ทั้งหมด lr0.02 depth5 leaves32 min_child10 ff0.3 | หลังเสร็จ: `kaggle kernels output chinnaphatkhun/ev-farm-it20 -p farm_out_it20` → `python3 stack_v4.py` (รวม it20 แล้ว) → ถ้า OOF > 0.94647 และ nested บวก ค่อย submit | feature sets ใหม่ใน farm_lib: win/nokey/init (`_build_view`), trainer รองรับ `init_col` และ `linear_tree`
- OOF สาธารณะที่ดาวน์โหลดได้: **RealMLP yekenot** `ext3/oof_realmlp.npy` (OOF 0.94601, id order = train, fold seed42) — ที่เหลือ (Zoom Zoom, Mizushima, megayak, evgendvorkin, sergey focal) ให้แค่ test pred ใน `ext3/` (ใช้ได้เฉพาะ LB-blend)
- **it-ft transformer (ของเรา, T4)**: OOF 0.94221 (fold 0.9424-0.9447 แต่ pooled ต่ำ = calibration ข้าม fold ไม่ตรง) — อ่อนกว่า RealMLP มาก ไม่คุ้มพัฒนาต่อ ใช้ RealMLP เป็น NN member แทน | output `farm_out_ft/`
- **Generator ถูก reconstruct แล้ว (อ่าน 2026-09-17)**: Deotte `cdeotte/fable-5-1-eda-original-data-insights` — original 10k = numpy RandomState(101): Age U[25,69], Income N(85k,35k) floor 30k, Commute N(40,25) floor 5, cat loaded dice (Gender 52/45/3, City 50/35/15, Cars 40/40/15/5, CarType 40/35/15/10, Env uniform 1-5, Subsidy 60/40), chargers+home ขึ้นกับ City, Anxiety = worry score+noise, Buy = 1.2*inc/1e5 + 0.6*env + 2.0*subsidy + ... + wobble (recipe AUC บน original ~สูง) | **แต่ competition data มาจาก generator ที่เทรนบน original อีกที** ไม่ใช่สูตรตรง
- Maldonado `s6e9-the-generator-remembers-the-original-rows`: income ใน train = copy ของ original 97.9% (chance 8.4%); per-value residual จริง ±0.044 (variance 3.7x binomial); label ของ original row อยู่ใน residual (+0.029/-0.009) แต่ **มูลค่า: encoding อย่างเดียว (value, //100, //1000, nearest-original keys, prior 1, colsample 0.3) = OOF 0.94603; label ของ original เพิ่มแค่ 0.00003** → เราใช้หมดแล้ว (Income_te/IncEnv_te/Orig_lbl) ไม่มีอะไรใหม่
- ฉันทามติบอร์ด 09-16/17 (Afshar "6 of 9 models add zero", Mamarin, Maldonado, megayak "where 0.94651 comes from"): เพดานเป็นคุณสมบัติของ data; 0.94651 มาจาก anchor ของ jazivxt ที่ถูกจูนบน public LB ไม่ใช่เทคนิค; tie-break มีค่าสูงสุด 2.5e-7; boundary rules +0.6e-5 บน OOF; public SE ≈ 0.0011 → ช่องว่างทั้ง top-500 เป็น noise; slot ที่ 2 ของ final selection ไม่มีค่าถ้า rho>0.992 | **megayak OOF-only stack (four views + RealMLP 3-seed + naji XGB) = OOF 0.94634-0.94635 / LB 0.94645** — OOF library ดาวน์โหลดไว้ที่ `ext3/ds_*` (10-fold seed42)

## 7. Next steps (ตามลำดับ)
0. [DONE 2026-09-05] it3 → LB 0.94573 | LB แน่นมาก: rank 10 = 0.94638, rank 100 = 0.94616, ที่ 1 = 0.94656 → ต้องการอีก +0.0004 ถึง top-100
1. รอ it5 จบ -> `controller.py pull 5 chinnaphatkhun/ev-farm-it5` -> `submit 5` (หวัง OOF ~0.9460 / LB ~0.9461 ≈ rank ~100) | it4 (5-fold seeds เพิ่ม) ไม่คุ้ม ข้ามได้
2. ต่อจาก it5: (a) 20-fold ถ้าผลทดลอง local ดีกว่า 10-fold (เวลา kernel x2 → ลดโมเดลเหลือ ~6) (b) NN + Income embedding (torch มี MPS) เพื่อ diversity (c) CatBoost native cat Income (ผลทดลองอยู่ใน scratchpad ถ้ายังไม่จบ)
3. อย่าเสียเวลากับ: hand-feature เดิม, veto post-process, deep trees (leaves 63+ แพ้ depth 2-4), เทรน local, pseudo-labeling (-0.001), orig augmentation (-0.00023), LR stacking, optuna micro-tune (อ่านจาก topic andrewleal70/mamarin 2026-09-07)
4. BACKLOG (ยังไม่ทำ): it11 FLAML probe (time_budget ~1.5h บน naji matrix, kernel เปิด internet) — ต่อเฉพาะถ้า MLP v2 ไม่ถึง 0.944; AutoGluon extreme ข้ามไปก่อน (แพง + ข้างในก็ GBM corr 0.99) | it12 coarse param sweep บน naji (lr 0.01/0.02/0.04 x leaves 16/32, 6 ตัว — coarse มีผล, fine optuna = noise)

## 8. คำสั่งด่วน
```bash
cd "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
kaggle kernels status chinnaphatkhun/ev-farm-it3   # batch ใหม่ shallow+TE
kaggle kernels status chinnaphatkhun/ev-farm-it1
kaggle kernels logs chinnaphatkhun/ev-farm-it1  # หา BLEND OOF / <<< model OOF
python3 ensemble_farm/controller.py pull 1 chinnaphatkhun/ev-farm-it1
python3 ensemble_farm/controller.py submit 1
```
