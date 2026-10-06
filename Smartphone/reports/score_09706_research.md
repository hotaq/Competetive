# Research: คะแนนประมาณ 0.9706 ใน Playground S6E8

ค้นเมื่อ 2 สิงหาคม 2026 โดยใช้ Kaggle CLI และ source ของ public notebooks บน Kaggle เป็นหลัก

## คำตอบสั้น

คะแนน `0.9706` มีอยู่จริง และตอนที่ตรวจ leaderboard ล่าสุด frontier ขยับสูงกว่านั้นแล้ว:

| อันดับ ณ เวลาที่ตรวจ | ทีม | Public AUC |
|---:|---|---:|
| 1 | Maher el Ouahabi | 0.97073 |
| 2 | Gigrise | 0.97068 |
| 3 | Don Mani | 0.97067 |
| 4 | Szymon Nowacki | 0.97066 |
| 5 | ALKRIS | 0.97064 |

บริเวณ `0.97060` ยังมี Rayk Kretzschmar, AnthonyTherrien และ Md Abdul Al Hasib โดยตรง ดู leaderboard ปัจจุบันได้ที่ [Kaggle leaderboard](https://www.kaggle.com/competitions/playground-series-s6e8/leaderboard)

อย่างไรก็ตาม ยังไม่มี public source ที่พิสูจน์ว่าอันดับ 1 (`0.97073`) ใช้สูตรอะไร จึงไม่ควรเดาเทคนิคของทีมนั้นจากคะแนนเพียงอย่างเดียว สิ่งที่ยืนยันและทำซ้ำได้จาก public notebooks คือ pipeline ระดับ `0.97060–0.97062` ซึ่งมาจาก **OOF stacking ของโมเดลจำนวนมากและหลากหลาย** ไม่ใช่ CatBoost ตัวเดียว

## หลักฐานที่ทำซ้ำได้

### 1. 54-model honest OOF stack: LB 0.97062

[S6E8 honest OOF blend 54 models; LB 0.97061](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend-54-models-lb-0-97061) รายงาน:

- 54 โมเดลใช้ split เดียวกัน: `StratifiedKFold(5, shuffle=True, random_state=42)`
- best single OOF ประมาณ `0.96866–0.96867`
- logistic regression stack ใน **logit space** ได้ honest OOF `0.96943`
- public LB ของ submission ล่าสุดในคำอธิบาย notebook คือ `0.97062`
- จาก 24 เป็น 54 โมเดล LB เพิ่มประมาณ `0.0016` แต่ผลเพิ่มส่วนใหญ่เกิดจากโมเดลเพียง 2–3 ตัวที่ “ผิดต่างกันจริง”
- model architecture ต่างกันแต่ใช้ feature view เดียวกันมี correlation ราว `0.991`; architecture เดียวกันแต่ feature view ต่างกัน correlation ต่ำกว่ามากราว `0.902`
- logistic regression ที่ยอมให้มีน้ำหนักลบชนะ non-negative hill climb และ rank hill climb ใน OOF

prediction library ถูกเผยแพร่ที่ [S6E8 OOF library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models) และ notebook ตรวจ shape/ลำดับ row ก่อน stack

### 2. 47-model stack + independent rank blend: LB 0.97060

[S6E8 Private Researched OOF Rank Blend](https://www.kaggle.com/code/raykkretzschmar/s6e8-private-researched-oof-rank-blend) รายงานเส้นทางดังนี้:

| วิธี | OOF AUC | Public LB |
|---|---:|---:|
| early tree rank blend | 0.96707 | 0.96842 |
| 25-model mixed stack | 0.96782 | 0.96900 |
| 47-model mixed stack | 0.96942 | 0.97059 |
| stack 90% + independent rank prediction 10% | — | 0.97060 |

ทั้งสอง prediction มี Spearman correlation ประมาณ `0.99045`; disagreement เล็กน้อยยังช่วยได้เพราะ ROC AUC สนใจลำดับ จึงแปลงแต่ละ submission เป็น percentile rank ก่อน blend การลองน้ำหนัก 10% และ 20% ให้คะแนนที่ปัดแล้วเท่ากันคือ `0.97060`

### 3. Strong single model: TabM + constrained imputation + lattice TE

[S6E8 TabM with constrained imputation](https://www.kaggle.com/code/szymonkapiski/s6e8-tabm-with-constrained-imputation) เป็น single model ไม่ใช่ blend และรายงาน OOF `0.96867`, public LB `0.96967` โดยจุดสำคัญคือ:

- TabM ทำ error ต่างจาก GBDT; correlation กับ tree models ราว `0.984`
- ใช้ generator identity ที่พบว่าไม่มี violation ใน complete rows 421,427 แถว:  
  `daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`
- หากค่า `daily_screen_time_hours` หรือ component หาย ใช้ identity นี้เป็นขอบเขตให้ Iterative Imputer แทนการเติมค่ากลางตรง ๆ; รายงาน masked-value MAE `0.679` เทียบกับ unconstrained iterative `0.691` และ median `1.089`
- แปลง numeric ทุกคอลัมน์เป็น categorical lattice key ที่ความละเอียดเต็ม พร้อม rounded/floored keys, 2D pair cells, target mean และ cell count แบบ fold-safe
- average 3 seeds ภายในแต่ละ fold เพิ่ม OOF ราว `0.00019`; การเปลี่ยนขนาด architecture หลายแบบให้ผลต่างน้อยกว่า

โน้ตสำคัญจากผู้เขียน: แม้โมเดลนี้เป็น single ที่ดีขึ้น แต่ไม่เพิ่มคะแนนใน stack ที่มี TabM variants อยู่แล้ว เพราะ correlation สูงกว่า `0.9985`

### 4. Digit/float categorical features เป็นอีก feature view ที่ใช้จริง

[S6E8 CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) และ [S6E8 LGBM](https://www.kaggle.com/code/donmarch14/s6e8-lgbm) สร้าง feature จากแต่ละ numeric เช่น:

- หลัก ones/tenths/hundredths
- floor, round-to-0.1, integer modulo 10, fractional bucket 20 ช่อง
- pair ของ daily/weekend และ social/gaming
- missing flags/count, activity aggregates, ratios และ shares
- CatBoost version มี fold-safe smoothed target encoding ของ categoricals, pair keys และ quantile bins

แนวคิดนี้สอดคล้องกับข้อมูล synthetic ที่ค่าต่อเนื่องถูกวาดแล้วปัดเป็น lattice จึงมี repeated exact values จำนวนมาก ไม่ใช่เพียง domain feature ทั่วไป

### 5. Public hill-climbing และ L2 stack ยืนยันว่า ensemble คือแกนหลัก

- [Hill Climbing Ensemble](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction) รวม OOF/test predictions จาก XGBoost, RealMLP, TabM และ ResNet โดยอนุญาต negative weights
- [Public L2 Stack V1](https://www.kaggle.com/code/ravi20076/playgrounds6e8-public-l2stack-v1) รวม aligned OOF predictions แล้วใช้ Ridge regression เป็น meta-model

ทั้งสองอันไม่ได้พิสูจน์คะแนน 0.97073 แต่ยืนยันว่า public ecosystem มี OOF artifacts หลากหลายเพียงพอให้สร้าง stack ได้

## สิ่งที่น่าจะพาเราจาก 0.96588 ไปหา 0.9706

ลำดับที่คุ้มที่สุด:

1. **หยุดมอง CatBoost run เดียวเป็น final model** — OOF ราว `0.966x` ยังเป็น base learner ที่ดี แต่ gap ถึง 0.9706 ใหญ่เกินกว่าจะปิดด้วย tuning เล็กน้อย
2. **กำหนด CV contract เดียวกันทุกโมเดล** — 5-fold stratified, shuffle, seed 42; save `id`, OOF และ test prediction ทุกตัว เพื่อเอา public OOF ที่ตรวจ alignment แล้วมา stack ได้
3. **สร้าง feature views ที่ต่างกันจริง 3 กลุ่ม**:
   - raw + missing/domain features สำหรับ GBDT
   - digit/lattice/full-resolution fold-safe target encoding
   - constrained-imputation + ratios สำหรับ TabM/RealMLP
4. **ฝึก base families อย่างน้อย TabM/RealMLP + LGBM + XGB + CatBoost** โดยเลือกจาก OOF AUC และ residual correlation ไม่ใช่ AUC อย่างเดียว
5. **fit logistic regression บน clipped logits ของ OOF** และทำ meta-CV ซ้ำภายใน fold; อย่าวัด stack ด้วยการ fit และ score บน OOF matrix ชุดเดียวกัน เพราะเป็น in-sample
6. **ใช้ rank blend เฉพาะ prediction ที่เป็นอิสระจริง** น้ำหนักเล็ก 5–15% แล้วตัดสินจาก OOF; postprocess calibration ทั่วไปแทบไม่ช่วย ROC AUC เพราะ metric สนใจ ranking

เป้าหมายขั้นกลางที่สมเหตุผลคือ single-model LB ประมาณ `0.9695–0.9697`, honest stack OOF ประมาณ `0.9694`, แล้วค่อยคาดหวัง public LB ประมาณ `0.9706` ตามหลักฐานสาธารณะ ไม่ควรจูนตรงกับ public leaderboard เพื่อไล่ `0.0001` เพราะเสี่ยง public overfit สูง

## สิ่งที่ยังไม่ควรทำ

- อย่าสรุปว่า dataset ชื่อคล้ายกันคือ original data: [Generator Forensics notebook](https://www.kaggle.com/code/ryota517/s6e8-generator-forensics-leak-safe-cv) เรียก dataset ภายนอกที่ schema คล้ายกันว่า “candidate mirror” เพราะไม่มี exact row match และ lineage ไม่ได้รับการยืนยัน
- อย่าทุ่ม compute เพิ่ม seeds/hyperparameters ของ GBDT family เดิมจำนวนมาก หาก OOF rank correlation เกินประมาณ `0.995`; stack ต้องการ disagreement มากกว่า model clone
- อย่าคัดลอก submission แล้ว blend โดยไม่มี aligned OOF: จะรู้แค่ public LB และมีโอกาส overfit public split

## ข้อจำกัดของข้อสรุป

คะแนน leaderboard เปลี่ยนได้ตลอดเวลา และวิธีของทีมอันดับต้น ๆ ไม่จำเป็นต้องเปิดเผย สิ่งที่ระบุว่าเป็น “วิธีระดับ 0.97060–0.97062” ข้างต้นอ้างจาก public notebooks ที่มี source/OOF artifacts; ส่วน `0.97073` เป็นเพียงคะแนนที่ยืนยันจาก leaderboard ณ เวลาตรวจ ไม่สามารถผูกกับเทคนิคเฉพาะอย่างมีหลักฐานได้
