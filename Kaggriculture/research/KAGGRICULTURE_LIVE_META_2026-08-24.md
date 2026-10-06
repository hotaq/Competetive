# Kaggriculture Live Meta Snapshot — 2026-08-24

ตรวจข้อมูลสดจาก official Kaggle API เวลา `2026-08-24T14:37:53Z` (21:37 น. เวลาไทย) และวิเคราะห์ replay สาธารณะล่าสุดของทีมอันดับต้น ๆ ด้วย analyzer เดียวกัน ข้อสรุปนี้เป็น snapshot ของ meta ณ เวลาตรวจ ไม่ใช่การรับประกันอันดับในอนาคต

## Executive conclusion

Meta ตอนนี้คือ **fixed/coherent opening + adaptive expansion และตลาด + strawberry-to-wheat transition + endgame liquidation** ไม่ใช่ action tape เดียวตลอด 720 steps และไม่มี animal ratio เดียวที่ชนะทุกสาย

ทีมที่ฟอร์มสดดีที่สุดในตัวอย่างล่าสุดคือ **Crop Dusta (#2)**: 7W–1L จาก newest 8 public episodes ขณะที่อันดับ 1 เดิม Ryo อยู่ที่ 1W–7L ใน newest 8 และ 12W–12L ใน newest 24 แปลว่า leaderboard #1 ยังนำด้วยคะแนนสะสม แต่ build ของเขากำลังโดน meta ใหม่ไล่ทัน/แก้ทาง

สิ่งที่สอดคล้องกันใน Top 1–4:

- ขยายเป็น 3 quadrants ภายในประมาณวัน 9–14 และไม่ลงทุน quadrant ที่ 4
- ช่วงกลางเกมใช้ strawberry เป็น crop หลัก แล้วลด strawberry และเพิ่ม wheat ชัดเจนช่วงวัน 20–24
- ปลายเกมลด inventory/พืชคงค้างและเปลี่ยนเป็นเงินให้มากที่สุด
- ใช้ market อย่างต่อเนื่อง ขายสินค้าหลาย order ขนาดเล็ก/กลางตามราคา แทนการขายก้อนเดียวแบบตายตัว
- จำนวนวัว/แกะต่างกันมาก จึงต้องมองสัตว์เป็น portfolio ที่เข้ากับ route ไม่ใช่ target เดียว เช่น `8 cows / 4 sheep`

## Live leaderboard

| Rank | Team | Score |
|---:|---|---:|
| 1 | Ryo Hasegawa | 3087.7 |
| 2 | Crop Dusta | 3027.9 |
| 3 | Subramanya N | 2975.8 |
| 4 | junseok lee | 2970.4 |
| 5 | TerraLogic Labs | 2905.0 |
| 6 | Arman Tuganbaev | 2900.7 |
| 7 | Yizuki | 2888.9 |
| 8 | taiseiu | 2874.1 |
| 9 | ActiveMusyoku | 2847.5 |
| 10 | Ueddy | 2845.9 |

ช่องว่าง #1 ถึง #2 มีเพียง 59.8 คะแนน และ #1 ถึง #4 เพียง 117.3 คะแนน จึงเป็น leaderboard ที่กำลังเคลื่อนตัวเร็ว ไม่ใช่ dominance ของ strategy เดียว

## Recent public-replay form

แต่ละแถวใช้ newest 8 public episodes ของทีมนั้น ณ เวลาสร้าง snapshot ตัวอย่างแต่ละทีมไม่ได้เป็นคู่แข่งชุดเดียวกัน จึงใช้บอก momentum/พฤติกรรมล่าสุด ไม่ใช่ unbiased head-to-head rating

| Team | Rank | Newest-8 W-D-L | Win rate | Median margin | Exact common prefix | Unique full traces |
|---|---:|---:|---:|---:|---:|---:|
| Ryo Hasegawa | 1 | 1-0-7 | 12.5% | -4,147 | 101 steps | 8/8 |
| Crop Dusta | 2 | 7-0-1 | 87.5% | +4,261 | 24 steps | 8/8 |
| Subramanya N | 3 | 0-0-8 | 0.0% | -9,092.5 | 72 steps | 8/8 |
| junseok lee | 4 | 5-0-3 | 62.5% | +1,206 | 88 steps | 8/8 |
| zero waste (เรา) | — | 3-1-4 | 37.5% wins | -1,543.5 | 1 step* | 7/8 |

`*` exact prefix ของเราแตกเร็วจากรายละเอียด action/market แต่ phase modal-action rate ยังสูงประมาณ 99% ตั้งแต่ expansion ถึง endgame จึงยังเป็น route ที่ตายตัวมากกว่าสายบนอย่างชัดเจน

ใน newest 24 episodes ของ Ryo เขาอยู่ที่ 12W–12L, median margin -910 และเจอ Crop Dusta โดยตรง 12 เกมที่ผล 6–6 จึงไม่ควรตีความอันดับ 1 ว่าเป็น oracle หรือ action tape ที่ลอกแล้วจะชนะเสมอ

## Strategy fingerprints

ค่าด้านล่างเป็น median end-of-day state จาก newest 8 replays

| Team | Day 0 | Day 9 | Day 14 | Day 24 | Day 29 |
|---|---|---|---|---|---|
| Ryo | 2C/2S, 11M/6W | 7C/2.5S, 19St/7.5W, Q2 | 7.5C/5.5S, 27.5St/24.5W, Q3 | 8.5C/7S, 9St/43W | 4.5 plants left |
| Crop Dusta | 3C/2S, 5M/8W | 4C/4.5S, 20St/12.5W, Q3 | 5.5C/7S, 28St/18W | 6.5C/10.5S, 15St/36.5W | 1 plant left; animals reduced to 5C/4S |
| Subramanya | 2C/2S, 8M/3W | 3C/3S, 26.5St/8W, Q2 | 4C/4S, 35St/17.5W, Q3 | 5C/6.5S, 11.5St/32.5W | 4.5 plants left |
| junseok | 0C/4S, 7M/5W | 9C/4S, 21St/8W, Q2 | 11C/4S, 36.5St/12W, Q3 | 11C/4S, 17St/41W | 12 wheat plants left |
| zero waste | 1C/4S, 5M/5W | 8C/4S, 20St/13W, Q2 | 8C/4S, 36St/13W, Q3 | 8C/4S, 16St/44W | liquidated |

คำย่อ: `C` cow, `S` sheep, `M` melon, `St` strawberry, `W` wheat, `Q` unlocked quadrants

### 1. Opening ยังสำคัญ แต่ไม่มี opening เดียว

- Ryo: 5 hires, 2 cows, 2 sheep, 11 melon, 6 wheat และ 4 pasture ในวันแรก
- Crop Dusta: 4 hires, 3 cows, 2 sheep, 5 melon, 8 wheat และ 5 pasture
- Subramanya: 6 hires, 2 cows, 2 sheep พร้อม crop mix ที่มี carrot ช่วงเปิด
- junseok: เปิด 4 sheep ก่อน แล้ว pivot เป็น 9 cows / 4 sheep ภายในวัน 9

ดังนั้น consensus ไม่ใช่จำนวน unit แต่คือการสร้าง cashflow + fertilizer + crop pipeline ที่ coherent ตั้งแต่วันแรก

### 2. Meta leader เชิง momentum ปรับตัวตั้งแต่ expansion เร็วกว่า

Modal exact-action rate ในช่วงวัน 5–9:

- Crop Dusta: 25.5%
- Subramanya: 21.1%
- Ryo: 44.0%
- junseok: 80.8%
- zero waste: 99.6%

Crop Dusta มี exact common prefix เพียง 24 steps (ประมาณหนึ่งวัน) แต่ทั้ง 8 เกมมี full trace ไม่ซ้ำกัน และกำลังฟอร์ม 7–1 นี่เป็น evidence ที่แรงที่สุดว่าการเลือก route/market ตาม state และคู่แข่งกำลังได้เปรียบ fixed schedule

อย่างไรก็ตาม junseok ยังชนะ 5–3 ด้วย route ที่คงรูปมากกว่า แปลว่าความ adaptive อย่างเดียวไม่พอ route พื้นฐานต้องมี production geometry ที่ดีด้วย

### 3. Strawberry กลางเกม, wheat ปลายเกม เป็น consensus ที่ชัดที่สุด

วันที่ 14 ทั้ง Top 1–4 มี strawberry median 27.5–36.5 tiles และ wheat 12–24.5 tiles วันที่ 24 strawberry ลดเหลือ 9–17 ขณะที่ wheat เพิ่มเป็น 32.5–43 นี่คือ transition ที่พบร่วมกันมากกว่าสัดส่วนสัตว์ใด ๆ

### 4. Endgame ต้อง liquidate จริง

Ryo, Crop Dusta และ Subramanya เหลือพืช median เพียง 1–4.5 tiles เมื่อจบวัน 29; junseok เหลือ wheat 12 tiles และมีความเสี่ยง tail มากกว่า แม้ recent median margin ยังบวก การถือ inventory/สัตว์โดยไม่แปลงเป็นคะแนนก่อนจบเกมจึงเป็น worst-case leak สำคัญ

## Implication สำหรับ agent ของเรา

จุดอ่อนปัจจุบันไม่ใช่ว่า crop mix ต่างจาก Top 4 มาก—เราเองก็ไป 3 quadrants, 8C/4S, strawberry กลางเกม, wheat ปลายเกม และ liquidate ได้ แต่ **transition ของเรายังเกือบตายตัว 99%** ขณะที่ราคาตลาด, เงินสด, inventory, ตำแหน่งคู่แข่ง และจังหวะ land purchase เปลี่ยนทุกเกม

ลำดับพัฒนาที่มี evidence รองรับที่สุด:

1. เก็บ route/layout ที่แข็งแรงของ V16 เป็นฐาน ไม่ splice action tape ของ Ryo
2. สร้าง route portfolio อย่างน้อย 3 แบบ: cow-heavy, sheep-heavy และ balanced
3. ให้ policy เลือกเฉพาะ macro decision ที่ phase boundary (ประมาณวัน 5, 9, 14, 20, 25) จาก cash, prices, inventory, unlocked land และ opponent state
4. ให้ deterministic executor จัดการ movement/layout เพื่อไม่เสียความเสถียร
5. ทำ market controller ที่เลือก batch/hold/sell ตามราคา และบังคับ liquidation ก่อนจบ
6. promote เฉพาะ candidate ที่ดีขึ้นทั้ง win rate, median margin และ worst-case บน unseen replay league

สิ่งที่ไม่ควรทำต่อคือ optimize ให้ชนะ Ryo tape เดียว 24/24 เพราะ Ryo ล่าสุดเองกำลังแพ้ meta ใหม่ และ full-route/splice experiment ก่อนหน้านี้ก็ไม่ generalize

## Confidence and limitations

- **สูง:** leaderboard ณ timestamp, W-D-L/margins ของ replay ที่โหลด, state medians, action-diversity metrics
- **กลาง:** การเรียก Crop Dusta ว่า momentum leader เพราะ newest 8 เป็นตัวอย่างเล็กและ matchmaking ไม่เหมือนกันทุกทีม
- **กลาง:** strategy fingerprint อนุมานจาก state/action logs ไม่ใช่ source code ของคู่แข่ง
- **ต่ำ:** การทำนายว่าใครจะจบอันดับ 1 เพราะคะแนนและ opponent pool เปลี่ยนต่อเนื่อง

## Sources and reproducibility

- [Official Kaggriculture leaderboard](https://www.kaggle.com/competitions/kaggriculture/leaderboard)
- [Top-1 24-replay manifest](../analysis/top1_ryo_20260824_current/manifest.json) — official API timestamp, submission ID, episode IDs, rewards และ SHA-256
- [Top-1 24-replay move analysis](../analysis/top1_ryo_20260824_current/move_analysis.json)
- [Ryo newest-8 manifest](../analysis/live_meta_20260824/r01_16644287_ryo/manifest.json) และ [analysis](../analysis/live_meta_20260824/r01_16644287_ryo/move_analysis.json)
- [Crop Dusta newest-8 manifest](../analysis/live_meta_20260824/r02_16714457_crop_dusta/manifest.json) และ [analysis](../analysis/live_meta_20260824/r02_16714457_crop_dusta/move_analysis.json)
- [Subramanya newest-8 manifest](../analysis/live_meta_20260824/r03_16705390_subramanya/manifest.json) และ [analysis](../analysis/live_meta_20260824/r03_16705390_subramanya/move_analysis.json)
- [junseok newest-8 manifest](../analysis/live_meta_20260824/r04_16734551_junseok/manifest.json) และ [analysis](../analysis/live_meta_20260824/r04_16734551_junseok/move_analysis.json)
- [Our newest-8 manifest](../analysis/live_meta_20260824/ours_55710081/manifest.json) และ [analysis](../analysis/live_meta_20260824/ours_55710081/move_analysis.json)
- Analyzer: [`scripts/analyze_top1_moves.py`](../scripts/analyze_top1_moves.py)

