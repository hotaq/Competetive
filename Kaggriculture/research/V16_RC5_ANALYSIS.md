# วิเคราะห์ V16-RC5: 8C/4S Premium Market Lead

วันที่ตรวจสอบ: 2026-08-21 (Asia/Bangkok)

Notebook: [boatlee/v16-rc5-high-score-8c-4s-premium-market-lead](https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead)

## ข้อสรุป

V16-RC5 ไม่ใช่ RL แต่เป็น **replay-derived macro policy** ที่มีสามชั้น:

1. ตารางทำฟาร์มแบบ open-loop ที่สร้าง `4 SHEEP` ทันทีและสะสมเป็น `8 COW / 4 SHEEP` ภายใน step 192
2. ตัวซ่อม trajectory เฉพาะกรณี `WEED` ขวาง `PLANT` หรือ `BUILD_PASTURE`
3. ตัวเลื่อนยอดขาย `MELON`, `MILK`, `STRAWBERRY`, `WOOL` จากเทิร์นถัดไปมาเทิร์นปัจจุบัน เมื่อไม่มี town demand และมี stock จริง เพื่อขายก่อนคลื่นตลาดหนึ่งเทิร์น

สาเหตุที่มันแรงคือ **core production ทำค่าเฉลี่ยสูง ส่วนสอง overlay ปิดจุดแพ้ปลายหาง** ไม่ใช่เพราะชื่อ 8C/4S อย่างเดียว การทดสอบอิสระกับ 11 replay ล่าสุดของ submission เราพบว่า core ชนะ 9/11 แต่ full policy ชนะ 11/11; ในเกมกับ `ringbearer` premium front-run พลิก margin จาก `-2,628` เป็น `+177` และในเกมกับ `MeeMee` ต้องใช้ทั้ง front-run และ weed repair จึงพลิกจาก `-2,755` เป็น `+298` ([ablation report](../rl/evaluations/boatlee_v16_rc5_public11_ablation.json)).

อย่างไรก็ตาม **ยังไม่มีหลักฐานใน notebook ว่าไฟล์นี้ได้ leaderboard score ประมาณ 2900 จริง** ณ snapshot ของ Kaggle public leaderboard วันที่ 2026-08-21 16:14:35 ทีม `boatlee` อยู่ rank 832, score `1903.8`; ทีมต้นทาง `Nikita Lugovoy` อยู่ rank 785, score `1927.3`; top score คือ `3151.8` ([public leaderboard](https://www.kaggle.com/competitions/kaggriculture/leaderboard)). ตัวเลข 2900 อาจเป็น historical/associated submission ที่หน้า UI แสดง แต่ไม่อยู่ใน source, metadata หรือ output ของ notebook จึงไม่ควรใช้เป็นข้อเท็จจริงจนกว่าจะมี submission ID หรือ leaderboard snapshot ตรงรุ่น

## 1. Verified notebook facts

ไฟล์ที่ดาวน์โหลดจาก Kaggle API อยู่ที่:

- [notebook source](../artifacts/kaggle_notebooks/boatlee_v16_rc5/v16-rc5-high-score-8c-4s-premium-market-lead.ipynb)
- [kernel metadata](../artifacts/kaggle_notebooks/boatlee_v16_rc5/kernel-metadata.json)
- [exact output main.py](../artifacts/kaggle_notebooks/boatlee_v16_rc5/output/main.py)
- [exact submission.tar.gz](../artifacts/kaggle_notebooks/boatlee_v16_rc5/output/submission.tar.gz)

`main.py` มี SHA-256 `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19` ตรงกับ hash ที่ notebook ตรวจเอง ตาราง `_ACTIONS` มี 720 step และถูกบีบด้วย base85/zlib

### ตารางขยายฟาร์ม

| Step | Market expansion |
|---:|---|
| 0 | ซื้อ `1 COW`, `4 SHEEP` |
| 120 | ซื้อ `1 COW` |
| 160 | ซื้อ land แปลงที่ 2 หลังขาย wool/fertilizer |
| 161 | ซื้อ `2 COW` |
| 168 | ซื้อ `2 COW` หลังคลื่นขาย wool |
| 192 | ซื้อ `2 COW` |
| 240 | ซื้อ land แปลงที่ 3 หลังคลื่นขาย wool |

รวมเป้าหมายเป็น `8 COW / 4 SHEEP` และสาม quadrant ตามที่ notebook ระบุ ตารางดิบมีคำสั่ง `HIRE` 264 ครั้ง, `FEED` 290 ครั้ง, `CARE` 967 ครั้ง, `COLLECT_FERTILIZER` 296 ครั้ง, `HARVEST` 390 ครั้ง และ `WATER` 1,010 ครั้ง ตัวเลขนี้คือจำนวนคำสั่งใน tape ไม่ใช่จำนวนคำสั่งที่สำเร็จเสมอเมื่อ state drift

ตารางขายดิบรวม `MELON 126`, `MILK 320`, `STRAWBERRY 300`, `WOOL 154`, `FERTILIZER 300` และ `WHEAT 479` หน่วย จึงเป็นระบบผลิตหลายรายได้ ไม่ได้พึ่งพาสินค้าชนิดเดียว

### ทำไม 8C/4S ทำเงินได้

กติกาทางการกำหนดให้ sheep ราคา 500 ผลิต wool ราคา base 200 ครั้งแรกหลัง 6 วันและทุก 3 วัน ส่วน cow ราคา 400 ผลิต milk ราคา base 160 ครั้งแรกหลัง 8 วันและทุก 2 วัน สัตว์ทุกตัวสร้าง fertilizer หนึ่งหน่วยต่อวัน และ `CARE` ที่ทำในวันที่สัตว์ได้รับอาหารจะสะสม bonus ไปยัง production tick ถัดไป ([official competition README/data](https://www.kaggle.com/competitions/kaggriculture/data)).

ดังนั้น route นี้ใช้เงินเริ่มต้น 3,000 ซื้อ sheep จำนวนมากตั้งแต่เปิดเกมเพื่อให้ wool/fertilizer เริ่มคืนทุนเร็ว แล้วเอาเงินคลื่นแรกไปซื้อ land และ cow เพิ่ม เป็น cash-flow snowball หลัง step 192 มีสัตว์ 12 ตัวที่ให้ทั้ง premium products และ fertilizer พร้อมใช้ farm hands จำนวนมากดูแลสัตว์และพืชคู่ขนาน

จุดสำคัญคือมันรักษา geometry เดิมตลอดเกม การ feed/care จึงชี้ไปยัง pasture เดิม ต่างจาก selector ของเราที่อาจเปลี่ยนไปใช้ tape ซึ่งมีตำแหน่งสัตว์คนละแบบ

## 2. Premium market lead ทำงานอย่างไร

โค้ดที่ [main.py lines 136-229](../artifacts/kaggle_notebooks/boatlee_v16_rc5/output/main.py) ทำดังนี้:

1. ดูยอด `SELL` ของสินค้าพรีเมียมใน `_ACTIONS[step + 1]`
2. คำนวณว่าเทิร์นปัจจุบันมี town-center/shop demand ของสินค้านั้นหรือไม่
3. ตรวจ stock ใน `private.shed` และกันของที่ farmer/hands จะ `PICKUP` หรือมีคำสั่งขายอยู่แล้ว
4. ถ้ามีของเหลือและไม่มี demand ให้เพิ่มยอดขายในเทิร์นปัจจุบัน
5. บันทึก `due` แล้วหักยอดเดียวกันออกจากคำสั่งขายเทิร์นถัดไป

ปริมาณขายรวมสองเทิร์นจึงเท่าเดิม เป็นการเปลี่ยนเวลา ไม่ใช่สร้างสินค้าหรือขายซ้ำ

เหตุผลทางเศรษฐศาสตร์ตรงกับกติกา: ผู้เล่นสองคนขายพร้อมกันแบบ unit-by-unit แต่สินค้าที่ขายเพิ่ม market inventory และทำให้ราคาหน่วยต่อมาลดลง สินค้าพรีเมียมมี oversupply curve รุนแรง เช่น strawberry/milk มี `above_target 1.60` และ melon/wool ใช้เส้นโค้งที่ลงแรงจนถึง floor `$1` ได้ การขายก่อนคลื่นคู่แข่งหนึ่งเทิร์นจึงรับราคาก่อน glut ([official market mechanics](https://www.kaggle.com/competitions/kaggriculture/data)).

เงื่อนไข “ไม่มี town demand ตอนนี้” สำคัญ เพราะ turn order ทางการคือ market orders ก่อน town consumption หาก town กำลังจะกินสินค้าหลังเทิร์นนี้ การรอขายเทิร์นถัดไปจะได้ราคาหลัง inventory ถูกดูดออก โค้ดจึงไม่ front-run ในกรณีนั้น เป็น market timing ที่ค่อนข้างปลอดภัย ไม่ใช่เลื่อนทุกคำสั่งแบบตายตัว

ข้อจำกัดคือมันมองเพียงตารางขายของตัวเอง ไม่ได้ทำนาย action คู่แข่งโดยตรง ประโยชน์สูงสุดเกิดเมื่อ meta ใช้ schedule ใกล้กันและปล่อยคลื่นขายในเวลาเดียวกัน

## 3. Weed repair ลด state drift อย่างไร

โค้ด [main.py lines 81-121](../artifacts/kaggle_notebooks/boatlee_v16_rc5/output/main.py) ตรวจเฉพาะตอน actor กำลังจะ `PLANT` หรือ `BUILD_PASTURE` แต่ tile ปัจจุบันเป็น `WEED`:

- เทิร์นนั้นเปลี่ยนเป็น `DIG`
- เทิร์นถัดไปทำ intended action เดิม
- จากนั้น replay action ของ actor คนนั้นแบบเลื่อนหนึ่ง step ได้นานสูงสุด 8 step

มันไม่ใช่ state-aware planner เต็มรูปแบบ แต่ดีกว่าการ sanitize action เป็น `PASS` เพราะยังรักษางานที่ตั้งใจและเลื่อนเฉพาะ lane ของ actor ที่เสียจังหวะ นอกจากนี้ `_align_hands` ปรับจำนวน action ให้ตรงกับ hands ที่มีจริง

## 4. Notebook self-reported evaluation

Notebook ระบุว่า:

- reconstruct schedule จาก public replays ของ Nikita Lugovoy submission `55440039`, episodes `92165990`, `92185587`, `92223213`
- field schedule ตรงกัน และ market schedule ตรง 99.91% ของ decision steps
- reconstructed core ชนะ 60/60 จาก 30 seed แบบสลับที่นั่ง, mean margin `+1,911.9`, worst `+68`
- full V16-RC5 ผ่าน reference panels ต่อ Kaito/Rayk/llcc

ตัวเลขเหล่านี้ถูกใส่เป็น JSON table คงที่ใน notebook ไม่มี replay logs, seed list หรือ executable panel ครบชุดให้ reproduce ภายใน notebook ส่วน validation cell ที่รันจริงเป็นเพียง self-play seed `451781128` โดยโหลด `main.py` ไฟล์เดียวกันทั้งสองที่นั่ง เพื่อตรวจ 720 frames, status และ action schema ดังนั้นให้ถือ 60/60 และ 99.91% เป็น **author-reported** ไม่ใช่ independently verified

## 5. Independent local evaluation

| Evaluation | V16-RC5 result | Comparison / meaning |
|---|---:|---|
| 11 public replays จาก submission ล่าสุดของเรา, fixed opponent actions | 11/11, mean `+19,493.5`, worst `+177` | core 9/11; overlays ปิดสอง tail losses ([report](../rl/evaluations/boatlee_v16_rc5_public11_ablation.json)) |
| Live Rust engine vs v12 selector, 12 seed สลับสองที่นั่ง | 18/24, mean `+8,169.7`, worst `-14,946` | V16 แข็งแรงกว่าใน broad live matchup แต่ไม่ชนะทุก seed ([report](../rl/evaluations/boatlee_v16_rc5_vs_v12_selector_live24.json)) |
| Ryo development 96, fixed opponent actions | 69/96, mean `+6,962.5`, worst `-15,850` | v12 specialist ชนะ 89/96, mean `+15,966.8` ([report](../rl/evaluations/boatlee_v16_rc5_vs_ryo_dev96.json)) |

การทดสอบ fixed-opponent replay เป็น counterfactual: action คู่แข่งถูกตรึงจากเกมที่เดิมเล่นกับ agent อื่น คู่แข่งจึงไม่ตอบสนองต่อ state ใหม่ ผล live two-player 24 เกมมีน้ำหนักสูงกว่าสำหรับการเปรียบเทียบ V16 กับ v12 โดยตรง

## 6. ต่างจาก candidate v12 ของเราอย่างไร

| ประเด็น | V16-RC5 | v12 selector ของเรา |
|---|---|---|
| แกน policy | tape เดียวที่ reconstruct จาก 3 replay และคง geometry | เลือก/สลับหลาย Ryo-derived tapes ที่ branch 96/120/144/168 |
| Adaptation | weed repair, hand-count alignment, stock/demand-aware sell timing | exact-safe contextual routes และ action legalization |
| จุดแข็ง | coherent production, broad live meta, premium queue timing | specialist ต่อ Ryo: 89/96 |
| จุดเสี่ยง | ยังเป็น open-loop; แพ้ Ryo 27/96 | spatial alias แล้ว switch ไป tape ที่ geometry ไม่ตรง |
| หลักฐานล่าสุด | ชนะ v12 live 18/24 | เกม `96090591` สัตว์ลดจาก 6 cow/9 sheep เหลือ 3/3 เพราะ feed/care ไม่ตรงตำแหน่ง ([state report](../rl/evaluations/top1_v12_submission55671647_ep96090591_state_vs_ringbearer.json)) |

นี่อธิบายว่าทำไม v12 ซึ่ง local score ต่อ Top1 สูงกว่ายังแพ้เกมจริง: selector ของเราชนะ benchmark เฉพาะ แต่การเปลี่ยน tape อาจทำลายความต่อเนื่องของ pasture/animal positions ส่วน V16 ใช้ route เดียวตั้งแต่ต้นจึงไม่เกิด spatial alias แบบเดียวกัน

## 7. สิ่งที่ควรนำมาใช้

ไม่ควรแทน v12 ด้วย V16 ทั้งก้อน เพราะ V16 แพ้ Ryo corpus มากกว่าอย่างชัดเจน แนวทางที่มี evidence รองรับคือ:

1. เพิ่ม V16 8C/4S เป็น macro candidate/default broad-meta หนึ่งตัวใน portfolio
2. ย้าย `premium front-run + repay` เป็น overlay ทั่วไปเหนือทุก policy เพราะเป็น state-aware และ ablation แสดงว่าช่วย worst case
3. ย้าย weed repair แบบ “DIG แล้วเลื่อน actor lane” มาแทนการ sanitize เป็น `PASS`
4. บังคับ spatial compatibility gate ก่อน switch tape: pasture/animal positions, farmer/hands positions และ feed/care targets ต้องตรง ไม่ใช่ดู aggregate state อย่างเดียว
5. ใช้ v12 เฉพาะเมื่อ detector มั่นใจว่า state/opponent เป็น Ryo-like; fallback ไป coherent 8C/4S route เมื่อไม่มั่นใจ
6. ก่อนส่งรุ่นผสม ให้ทดสอบ live seat-swapped อย่างน้อย 24–48 เกมต่อ V16, v12, Ryo และ opponent ล่าสุด พร้อม gate ทั้ง win rate และ worst-case

## 8. Limitations

- Notebook, metadata และ output ไม่มี submission ID หรือ score 2900 จึงยืนยันความเชื่อมโยงกับคะแนนนั้นไม่ได้
- Evaluation 60/60 และความตรงของ schedule 99.91% เป็นตัวเลขที่ผู้เขียนฝังไว้ใน notebook โดยไม่มี seed list หรือ game logs ครบชุด
- การทดสอบ public11 และ Ryo96 ตรึง action คู่แข่งจาก replay จึงเป็น counterfactual; คู่แข่งไม่ได้ตอบสนองต่อ state ที่ V16 สร้างใหม่
- Live evaluation มีเพียง 24 เกม/12 paired seeds จึงเป็นสัญญาณที่ดีแต่ยังไม่ใช่ promotion proof สำหรับ leaderboard meta ทั้งหมด
- ทั้ง V16 และ v12 ยังมีแกนเป็น action tape ไม่ใช่ live planner เต็มรูปแบบ V16 จึงยังเสี่ยงต่อ drift แบบอื่นนอกเหนือจาก weed และ v12 เสี่ยงต่อ spatial alias
- Leaderboard snapshot และ meta เปลี่ยนตามเวลา ข้อสรุปเรื่องอันดับใช้ได้กับ timestamp 2026-08-21 16:14:35 เท่านั้น

## แหล่งข้อมูลและวิธี reproduce

Primary sources:

- [Kaggle notebook](https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead)
- [Kaggriculture official competition data/README](https://www.kaggle.com/competitions/kaggriculture/data)
- [Kaggriculture public leaderboard](https://www.kaggle.com/competitions/kaggriculture/leaderboard)
- Exact downloaded artifacts และ local evaluation reports ที่ลิงก์ไว้ด้านบน

คำสั่ง retrieval ที่ใช้:

```bash
kaggle kernels pull boatlee/v16-rc5-high-score-8c-4s-premium-market-lead -m
kaggle kernels output boatlee/v16-rc5-high-score-8c-4s-premium-market-lead
kaggle competitions download kaggriculture -f README.md
kaggle competitions leaderboard kaggriculture --download
```

Leaderboard เปลี่ยนได้ตามเวลา ทุกการอ้าง score ในรายงานนี้จึงผูกกับ snapshot วันที่และเวลาที่ระบุ
