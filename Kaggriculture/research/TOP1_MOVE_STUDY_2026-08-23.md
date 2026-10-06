# ศึกษาการเดินอันดับ 1 Kaggriculture — snapshot 2026-08-23

วันที่ตรวจสอบ: 2026-08-23 20:14–20:31 (Asia/Bangkok), หรือ 13:14–13:31 UTC

## สรุปสั้น

อันดับ 1 ณ snapshot คือ **Ryo Hasegawa**, team ID `16644287`, leaderboard score `3137.8`, จาก submission `55614463` วันที่ 2026-08-19 โดย Kaggle API แสดง public episodes ที่จบแล้ว 204 เกม ([leaderboard](https://www.kaggle.com/competitions/kaggriculture/leaderboard), [Top-1 manifest](../replays/top1_snapshot_20260823/manifest.json)).

สิ่งสำคัญที่สุดจาก replay ไม่ใช่ “copy ทุก action ของ Ryo” แต่เป็นโครงสร้างสามช่วง:

1. **Opening มาตรฐานและเร็วมาก** — ในตัวอย่าง 20 เกมล่าสุด action JSON เหมือนกันทุกเกมถึง step 72 แม้สลับ seat และเปลี่ยนคู่แข่ง อย่างไรก็ตาม เมื่อขยายหน้าต่างเป็น 24 เกมพบการแตกแขนงครั้งแรกที่ step 45 จึงควรมองนี่เป็น modal opening ที่เสถียรมาก ไม่ใช่ universal tape ที่เหมือนกันทุกเกม
2. **Midgame แตกแขนงตาม state** — หลัง opening สัดส่วนพืช/สัตว์เปลี่ยนจริงตามเงิน ราคา inventory และสภาพฟาร์ม ไม่ได้ล็อกไว้ที่ `8 COW / 4 SHEEP`
3. **ขายของ premium ตามราคาและบีบ 10 market orders ให้คุ้ม** — โดยเฉพาะ strawberry, milk และ wool; ส่วนท้ายเกมเป็น liquidation ที่ตอบสนองต่อการขายของคู่แข่ง

ดังนั้นแนวทางที่มีโอกาสไต่ Top-1 มากที่สุดคือ **hybrid agent**: deterministic/behavior-cloned opening + state-conditioned macro RL สำหรับ allocation, market timing และ endgame response โดยให้ Rust planner รับผิดชอบ route และ legality

## แหล่งข้อมูลและวิธีตรวจ

ใช้เฉพาะแหล่งข้อมูล first-party:

- Kaggle `get_leaderboard`, `competition_team_submissions`, `competition_list_episodes` และ `get_episode_replay`; เส้นทาง API ที่ใช้ดูได้ใน [download_top1_replays.py](../scripts/download_top1_replays.py)
- replay JSON เต็ม 20 เกมล่าสุดของ submission `55614463` พร้อม SHA-256 ใน [Top-1 manifest](../replays/top1_snapshot_20260823/manifest.json)
- replay JSON เต็ม 8 เกมล่าสุดของ submission เรา `55710081`, เลือก seat 0 และ seat 1 อย่างละ 4 เกม พร้อม SHA-256 ใน [candidate manifest](../replays/submission_55710081_snapshot_20260823/manifest.json)
- action schema และ observation schema ที่ Kaggle ฝังไว้ใน replay แต่ละเกม เช่น [episode 97619249](../replays/top1_snapshot_20260823/97619249.json): action แบ่งเป็น `farmer`, `hands`, `market`; observation ของ agent ไม่มีชื่อทีมคู่แข่ง มีเพียง public farms/market/town และ private state ของตัวเอง

ช่วงเวลาที่ใช้ในรายงานเป็น bin สำหรับวิเคราะห์ ไม่ใช่นิยามทางการของ Kaggle:

- opening: steps 1–143, days 0–5
- midgame: steps 144–575, days 6–23
- endgame: steps 576–719, days 24–29

ราคา “pre-action” ในรายงานคือราคาจาก observation ก่อนส่ง action ถัดไป จึงเป็นราคาที่ policy มองเห็นตอนตัดสินใจ ไม่ได้อ้างว่าเป็น realized execution price ทุก order

## Snapshot เชิงปริมาณ

การนับ public episode metadata ทั้งหมด ณ 13:31 UTC ได้ผลดังนี้:

| Submission | Score | Public games | W-D-L | Raw win rate | นับ draw ครึ่งแต้ม |
|---|---:|---:|---:|---:|---:|
| Ryo `55614463` | 3137.8 | 204 | 162-0-42 | 79.4% | 79.4% |
| zero waste `55710081` | 1618.5 | 78 | 38-15-25 | 48.7% | 58.3% |

แยก seat ของ Ryo คือ seat 0 `69W/20L` และ seat 1 `93W/22L`; ของเราคือ seat 0 `20W/7D/15L` และ seat 1 `18W/8D/10L` ตัวเลขมาจาก Kaggle `competition_list_episodes(submission_id)` ที่กรอง `PUBLIC` + `COMPLETED` ตาม [ตัวดาวน์โหลด](../scripts/download_top1_replays.py#L105-L116). Matchmaking ของสอง submission ไม่เหมือนกัน จึงใช้ตารางนี้เป็นผลจริงบน Kaggle แต่ไม่ใช้เป็น causal A/B test

Ryo เจอคู่แข่งระดับบนบ่อยและไม่ได้ชนะทุกเกม:

- เจอ Subramanya N: `14W/9L` (60.9%)
- เจอ Arman Tuganbaev: `15W/5L` (75.0%)
- เจอ Crop Dusta: `16W/9L` (64.0%)

ตัวอย่าง full replay 20 เกมล่าสุดของ Ryo ได้ `14W/6L`, mean margin `+2,846.1`, median `+4,416.5`, best `+14,799`, worst `-9,198`; seat 0 `6W/3L`, seat 1 `8W/3L` ([manifest และ rewards](../replays/top1_snapshot_20260823/manifest.json)). ตัวอย่างนี้เป็น newest slice ไม่ใช่ random sample

## 1. Opening: สูตรเดียวกัน แต่ใช้เงินและพื้นที่มีประสิทธิภาพกว่าเรา

### Step 1 ของ Ryo

ในทั้ง 20 เกม Ryo ส่ง action เดียวกัน:

```text
farmer: BUILD_PASTURE
market: HIRE x5,
        BUY_ANIMAL SHEEP 2,
        BUY_ANIMAL COW 2,
        BUY_SEED MELON 11,
        BUY_SEED WHEAT 6,
        BUY_PRODUCT WHEAT 4
```

นี่ใช้ market-order cap ครบ 10 order ตั้งแต่ตาแรก และสร้าง pasture พร้อมกันด้วย farmer ([seat 0 episode 97619249](../replays/top1_snapshot_20260823/97619249.json), [seat 1 episode 97569024](../replays/top1_snapshot_20260823/97569024.json)). Step 2 เขาขาย wheat 4 หน่วยกลับทันทีหลัง shared wheat price ที่เห็นเปลี่ยนจาก 25 เป็น 27 แล้วค่อยซื้อ wheat ทีละหน่วยเพื่อ feed/route ต่อไป นี่เป็น buy→sell financing cycle ที่ผูกกับตลาด ไม่ใช่แค่ตารางเดินบนฟาร์ม

### State ที่ได้จาก opening

| Checkpoint | Ryo: farm state | zero waste `55710081`: farm state |
|---|---|---|
| step 24 / day 1 | 2 cow, 2 sheep, 4 pasture, 17 crop tiles (`11 melon + 6 wheat`), money 138 | 1 cow, 4 sheep, 5 pasture, 10 crop tiles (`5 melon + 5 wheat`), money 9 |
| step 72 / day 3 | 2 cow, 2 sheep, 4 pasture, 21 crop tiles, money 354 | 1 cow, 4 sheep, 5 pasture, 10 crop tiles, money 18 |
| step 144 / day 6 | 2 cow, 2 sheep, 5 pasture, 27 crop tiles, money ราว 159–182, เปิด `NE` แล้ว | 2 cow, 4 sheep, 6 pasture, 19 crop tiles, median money 22.5, ยังมีเพียง `NW` |

หลักฐาน Ryo: [97619249](../replays/top1_snapshot_20260823/97619249.json), [97518737](../replays/top1_snapshot_20260823/97518737.json), [97213078](../replays/top1_snapshot_20260823/97213078.json). หลักฐานของเรา: [97642053](../replays/submission_55710081_snapshot_20260823/97642053.json), [97628298](../replays/submission_55710081_snapshot_20260823/97628298.json)

submission เราเริ่มด้วย `farmer: PASS`, ซื้อ product wheat 14, cow 1, sheep 4 และ seeds เพียง melon 5 + wheat 5 จากนั้น stock wheat ไว้มากกว่า แต่ Ryo เปลี่ยนเงินเป็น productive crop tiles เร็วกว่าอย่างชัดเจน ([candidate episode 97642053](../replays/submission_55710081_snapshot_20260823/97642053.json)).

Ryo ซื้อ land แรกตรง step `121` ครบทุก 20 เกม และ land ที่สองระหว่าง steps `228–255`; เราซื้อคงที่ที่ steps `161` และ `241` ครบทั้ง 8 เกม ตัวเลขนี้ทำให้ **เลื่อน land แรกจาก 161 → 121** เป็น ablation ลำดับแรกที่ควรทดลอง แต่ยังไม่ถือว่า causal จนกว่าจะทดสอบ same-seed live games

## 2. การแตกแขนง: ตอบสนองต่อ state ไม่ใช่จำชื่อคู่แข่ง

20 replay ของ Ryo มี exact action sequence เหมือนกันถึง step 72 และเริ่มแตกแขนงจริงที่ step 124 ตัวอย่าง branch ซื้อ wheat seed:

| Episode | คู่แข่ง | State ก่อน step 124 | step 124 | step 141 | รวม |
|---|---|---|---:|---:|---:|
| [97151458](../replays/top1_snapshot_20260823/97151458.json) | Subramanya N | money 146, wheat price 30 | 2 | 6 | 8 |
| [97619249](../replays/top1_snapshot_20260823/97619249.json) | Subramanya N | money 159, wheat price 31 | 3 | 5 | 8 |
| [97569024](../replays/top1_snapshot_20260823/97569024.json) | Crop Dusta | money 169, wheat price 31 | 4 | 4 | 8 |

ทั้งสาม branch ยังซื้อรวม 8 seed เท่ากัน แต่เลื่อนจำนวนระหว่างสองจุดตาม cash/state นี่คือรูปแบบ **target เดิม + state-aware scheduling** ไม่ใช่ replay tape แบบตายตัว และการที่คู่แข่งชื่อเดียวกันเกิดได้หลาย branch แสดงว่าไม่ใช่ named-opponent lookup

ข้อจำกัดสำคัญ: observation schema ไม่ส่ง `TeamNames` ให้ agent ดังนั้น Ryo ไม่สามารถรู้ชื่อคู่แข่งโดยตรงจาก action input; สิ่งที่เรียกว่า adaptation ในรายงานคือการตอบสนองทางอ้อมต่อ public farm, market, town และ trajectory ที่คู่แข่งสร้าง ([embedded observation schema](../replays/top1_snapshot_20260823/97619249.json)).

## 3. Midgame: allocation เปลี่ยนตามเกมจริง

ที่ step 360 / day 15 ตัวอย่าง 20 เกมของ Ryo มี animal composition 9 แบบและ crop composition 15 แบบ เช่น:

- [97619249](../replays/top1_snapshot_20260823/97619249.json): `10 cow / 2 sheep`; `36 strawberry / 6 tomato / 21 wheat`
- [97518736](../replays/top1_snapshot_20260823/97518736.json): `7 cow / 9 sheep`; `30 strawberry / 29 wheat`
- [97386449](../replays/top1_snapshot_20260823/97386449.json): `8 cow / 2 sheep`; ใช้ carrot, melon, strawberry, tomato และ wheat พร้อมกัน

submission เรา 8 เกมมี animal composition แบบเดียวคือ `8 cow / 4 sheep` และ crop composition เพียง 2 แบบ ซึ่งต่างกันแค่ strawberry 35 หรือ 36 tile ([candidate replay set](../replays/submission_55710081_snapshot_20260823/manifest.json)). นี่เป็นหลักฐานตรงว่าตัวเราเป็น schedule เดิมมากกว่า Ryo แม้ชื่อ submission จะมี adaptive market overlay

ค่ากลางที่ step 360:

| Metric | Ryo, 20 games | zero waste, 8 games |
|---|---:|---:|
| Money | 19,611 | 11,843.5 |
| Crop tiles | 63 | 61 |
| Pastures | 12 | 12 |
| Animals | 12 | 12 |

จำนวน asset หลักใกล้กันแล้ว แต่เงินต่างกันมาก จึงชี้ว่าปัญหาหลัง opening อยู่ที่ **ชนิด asset + timing รายได้/ตลาด** มากกว่าการเพิ่มจำนวน tile แบบดิบ ๆ ตารางนี้เป็น descriptive comparison เพราะ seed และคู่แข่งไม่ตรงกัน

## 4. Market timing: จุดต่างที่แรงที่สุด

ค่าเฉลี่ยราคาที่ policy เห็นก่อนออก `SELL`, weighted ด้วย quantity:

| Product | Ryo midgame | เรา midgame | Ryo endgame | เรา endgame |
|---|---:|---:|---:|---:|
| Strawberry | 184.6 | 132.4 | 109.3 | 37.4 |
| Milk | 153.8 | 117.5 | 88.0 | 93.0 |
| Melon | 218.2 | 150.4 | — | — |
| Fertilizer | 66.0 | 66.0 | 29.0 | 23.8 |
| Wheat | 40.4 | 40.1 | 44.7 | 41.9 |
| Wool | 163.3 | 147.0 | 115.1 | 146.0 |

Ryo ไม่ได้ชนะด้านราคาทุกสินค้า แต่เหนือกว่ามากใน premium crop/milk ช่วงกลางเกม โดยเฉพาะ strawberry ความต่างนี้สอดคล้องกับคะแนน แม้เราขายจำนวนหน่วยมากกว่าในหลายสินค้า

Ryo ใช้ market action ครบ cap 10 orders เฉลี่ย `25.15` steps ต่อเกม (503 ครั้งใน 20 เกม) เทียบกับเรา `5.0` steps ต่อเกม (40 ครั้งใน 8 เกม). รูปแบบของ Ryo คือรวม `SELL` สินค้าหลายชนิดก่อน แล้วใช้ slot ที่เหลือ `HIRE` ทำให้ cash-out และเริ่มวันใหม่เกิดใน action เดียว ตัวอย่างชัดคือ [episode 97569024, step 577](../replays/top1_snapshot_20260823/97569024.json): ขาย strawberry 43, milk 18, wheat 20 แล้ว HIRE 7 รวม 10 orders

## 5. Endgame และจุดพลิก

### Comeback ที่ดี: episode 97569024, seat 1

Ryo ตามหลังสูงสุด `-7,011` ที่ step 526 และยังตาม `-3,914` ที่ step 576 แต่ step 577 ขาย bundle ขนาดใหญ่และเปลี่ยน gap เป็น `+9,300`; หลังแกว่งกับการขายของ Crop Dusta เขานำถาวรตั้งแต่ step 652 และจบ `+7,483` ([replay 97569024](../replays/top1_snapshot_20260823/97569024.json)). นี่แสดงว่า money gap ก่อน liquidation ไม่เท่ากับโอกาสชนะ และ policy ต้องมอง stock/production ที่ยังไม่ขายด้วย

### Early recovery: episode 97619249, seat 0

Ryo ตาม Subramanya N ถึง `-2,285` ที่ step 239 แต่กลับมานำถาวรที่ step 241 และจบ `+6,494`; cash jump ใหญ่สุดของเขาอยู่ step 457 จาก bundle strawberry + milk ก่อนเติม workers ([replay 97619249](../replays/top1_snapshot_20260823/97619249.json)).

### จุดอ่อนของอันดับ 1: episode 97317971, seat 0

Ryo นำถึง `+8,109` ที่ step 695 และขาย wheat/milk/fertilizer พร้อม HIRE ที่ step 697 แต่ Crop Dusta ทยอย liquidate tomato, carrot, milk, wool และ strawberry แทบต่อเนื่องตั้งแต่ steps 698–719; gap ข้ามเป็นลบที่ step 708 และ Ryo จบ `-2,472` ([replay 97317971](../replays/top1_snapshot_20260823/97317971.json)).

กรณีนี้บอกเป้าหมายที่มีค่ากว่า copy Ryo ตรง ๆ: สร้าง **opponent latent-inventory estimator** จาก public crops, animals, harvest cadence และยอดขายที่ผ่านมา เพื่อประเมิน cash ที่คู่แข่งยังซ่อนไว้ใน shed แล้วปรับเวลาขายช่วง 24 steps สุดท้าย

อีก tail loss คือ [97422923](../replays/top1_snapshot_20260823/97422923.json): เคยนำ `+9,553` ที่ step 300 แต่เป็น deficit ถาวรหลัง step 530 และจบ `-9,198` จึงไม่มีหลักฐานรองรับคำกล่าวว่า Top-1 ต้องชนะทุกเกม

## 6. Contrast กับ agent ปัจจุบันของเรา

| ด้าน | Ryo `55614463` | zero waste `55710081` | ช่องว่างที่ควรแก้ |
|---|---|---|---|
| Opening farmer | `BUILD_PASTURE` ตั้งแต่ step 1 | `PASS` | ใช้ action economy ต่ำกว่า |
| Day-1 crops / cash | 17 / 138 | 10 / 9 | production snowball ช้า |
| First land | step 121 | step 161 | ขยายช้า ~40 steps |
| Day-15 animal/crop diversity | 9 / 15 patterns | 1 / 2 patterns | allocation ยังเป็น fixed schedule |
| Mid strawberry decision price | 184.6 | 132.4 | sell gate พลาด premium window |
| 10-order steps/game | 25.15 | 5.0 | packing sell + hire ต่ำกว่า |
| Full public outcome snapshot | 162W/42L | 38W/15D/25L | robustness gap ใหญ่ |

เกมล่าสุดของเรายืนยันพฤติกรรม schedule เดิม: [97642053](../replays/submission_55710081_snapshot_20260823/97642053.json) แพ้เพียง 254, [97639733](../replays/submission_55710081_snapshot_20260823/97639733.json) แพ้ 16,201, [97637451](../replays/submission_55710081_snapshot_20260823/97637451.json) ชนะ 7,236 และ [97635154](../replays/submission_55710081_snapshot_20260823/97635154.json) เสมอ แต่ farm composition กลางเกมแทบเหมือนกันทั้งหมด ต่างจาก Ryo ที่เปลี่ยน production mix

## 7. Solution ที่ควรสร้างเพื่อไล่ Top-1

### ชั้น 1 — Ryo-compatible opening, ไม่ใช้ RL

สร้าง semantic opening target ถึง step 72 จาก trajectory ของ Ryo แต่ให้ Rust planner route ตาม state จริง:

- step 1 target: 2 cow, 2 sheep, 11 melon seeds, 6 wheat seeds, wheat product 4, 5 hires และสร้าง pasture พร้อมกัน
- reproduce buy→sell wheat financing cycle โดยมี legality/budget guard
- target state step 24 และ 72 ตามตารางด้านบน
- unlock land แรก target step 121; land สองเป็น state gate ช่วง 228–255
- split wheat-seed target ที่ steps 124/141 ตาม cash เหมือน branch `2+6`, `3+5`, `4+4`

การ copy เป็น **semantic target** สำคัญกว่า copy coordinates เพราะ weed หรือ action ที่ fail หนึ่งครั้งจะทำให้ raw action tape เลื่อนทั้งเกม

### ชั้น 2 — Macro allocation policy, ใช้ BC แล้วค่อย RL

State ควรมี day/hour, money, shed/seeds, market price/inventory, crop/animal counts, unlocked land, public opponent farm และ recent action/history. Output ไม่ควรเป็น raw movement ทุก hand แต่เป็น target เช่น:

- desired crop mix และ animal mix สำหรับ 24–48 steps ถัดไป
- buy-land / buy-animal budget
- desired shed reserve สำหรับ feed และ liquidation
- market sell threshold ต่อสินค้า

เริ่ม behavior cloning จาก public replays ของ active Top-1 ทั้ง 204 เกมและ previous Ryo submission `55588237` แล้ว fine-tune ด้วย RL/self-play บน Rust engine การใช้ทั้งสองรุ่นช่วยให้เห็น evolution ของ policy แทนการ overfit snapshot เดียว

### ชั้น 3 — Market response head

ให้ policy เลือก `SELL now / hold / partial sell` จากราคาเทียบ rolling baseline, town demand, opponent sell pressure และเวลาที่เหลือ จากนั้นใช้ deterministic packer เรียง:

1. premium sells ที่หมด window ก่อน
2. feed/seed buys ที่จำเป็น
3. `HIRE` เติมจนครบ cap 10

เป้าหมาย ablation แรกคือยกระดับ strawberry/milk pre-action sell price โดยไม่ลด quantity และไม่ทำให้ worst-case ขาด feed

### ชั้น 4 — Endgame adversarial liquidation

ฝึก head แยกสำหรับ days 24–29 ให้ประมาณ latent inventory คู่แข่งจาก public production แล้ว optimize ทั้ง expected margin และ CVaR/worst-case margin. Episode 97317971 ควรเป็น hard-negative หลัก: policy ต้องตรวจว่าคู่แข่งยังมี carrot/tomato stock มาก แม้เรานำเงินสด 8k อยู่

### Training/evaluation gate

อย่า promote จาก replay-tape win อย่างเดียว เพราะเมื่อเราเปลี่ยน state action ที่บันทึกของคู่แข่งอาจ stale หรือ illegal; ข้อจำกัดนี้ระบุไว้ใน [evaluate_replay_tape.py](../scripts/evaluate_replay_tape.py). Gate ที่ควรใช้:

- exact Rust/Python transition validation และ 720-step legality
- paired live seeds สลับทั้งสอง seat กับ opponent pool: Ryo-like, Subramanya-like, Arman-like, current `55710081`, V16-RC5 และ perturbations
- รายงาน W-D-L, half-draw rate, mean/median margin, worst margin และ CVaR10
- chronological holdout: ห้าม train จาก replay ที่ใช้ตัดสิน promotion
- promote เมื่อชนะ current agent แบบ paired live อย่างมีนัยสำคัญ, seat gap เล็ก และ worst-case ไม่ถอย; จากนั้นค่อยส่ง Kaggle เพื่อวัด meta จริง

## ผล implementation ของ Candidate A

ทดลองจริงแบบ live และสลับทั้งสอง seat แล้ว Candidate A **ไม่ผ่าน promotion gate**:

- ต่อ Ryo opening 0–72 เข้ากับ V16 suffix ตรง ๆ แพ้ 0/8 เพราะ coordinate plan คนละ layout
- ใช้ full legalizer ก็ยังแพ้ 0/8; legality repair ไม่สามารถสร้าง production plan ใหม่แทน suffix เดิมได้
- ใช้ Rust semantic executor หลัง opening แพ้ 0/8 และเหลือเงินเฉลี่ยเพียง 488 เพราะ executor ปัจจุบันยังรักษาฟาร์มหลาย crop/animal ระดับ Top-1 ไม่ได้
- ใช้เส้นทาง Ryo เต็มเกมเพียงเส้นเดียวได้ 7W/17L บน validation 24 เกม
- adaptive route reselection ได้ 6W/18L บน validation 24 เกม
- เลื่อน land อย่างเดียวแพ้ 0/8; ขาย wheat 15 เพื่อ fund land ลดความเสียหายแต่ยังแพ้ 0/8
- budget tune ที่ดูเหมือนชนะ mirror 16W/8L มี mean margin เพียง +3 และถอย −2/−4 เมื่อเทียบสอง validation opponents

หลักฐานรวมอยู่ใน [V17 Candidate A gate](../rl/evaluations/v17_candidate_a_gate_20260823.json). ข้อสรุปใหม่คือห้าม splice action tape และห้ามบังคับ land เร็วกับ V16 เดิม; ต้องทำ state-aware executor ที่ทำเงินได้ระดับแข่งขันก่อนจึงค่อยนำ Ryo opening และ macro RL กลับมาทดลอง

## ข้อสรุปเชิงตัดสินใจ

ทางไป Top-1 ไม่ใช่ “train PPO นานขึ้น” และไม่ใช่ “เลียนแบบ Ryo ทั้ง 720 steps” แต่คือ:

> **ล็อก opening ที่พิสูจน์แล้ว → ให้ RL เลือก production/market macro ตาม state → ใช้ Rust executor เดินและ legalize → ฝึก endgame แบบ adversarial**

ลำดับทดลองที่ให้ข้อมูลเร็วที่สุดคือ:

1. เปลี่ยน opening ของ `55710081` เป็น Ryo-compatible ถึง step 72
2. เลื่อน first land 161 → 121
3. เพิ่ม premium-price sell gate + 10-order packer
4. เพิ่ม opponent liquidation estimator สำหรับ 48 steps สุดท้าย
5. ทำ paired live ablation หลังเพิ่มทีละชั้น

ผลวิจัยนี้ชี้เส้นทางที่มีหลักฐานรองรับ แต่ไม่รับประกันอันดับ 1 เพราะ leaderboard, matchmaking และ meta เปลี่ยนตามเวลา ทุก identity/score ในรายงานผูกกับ snapshot ที่ระบุเท่านั้น

## ระดับความเชื่อมั่นของหลักฐาน: แชร์ได้โดยมีข้อควรระวัง

- ยืนยัน identity, submission, score และ W-D-L ซ้ำจาก Kaggle API ทางการแล้ว
- คำนวณรูปแบบการเดินของหน้าต่าง 24 เกมซ้ำแบบอิสระจาก replay JSON และ `move_analysis.json`
- matchmaking ของ Ryo และของเราไม่เหมือนกัน ดังนั้น public win rate ใช้บอกช่องว่างเชิงพฤติกรรมได้ แต่ไม่ใช่ causal A/B test
- ไม่ใช้ผล “สู้กับ replay tape” เป็น promotion evidence เพราะเมื่อ state เปลี่ยน action ที่บันทึกไว้อาจ stale หรือ illegal
- snapshot นี้อ้างอิงข้อมูล ณ 2026-08-23; leaderboard และ meta อาจเปลี่ยนภายหลัง

## Reproduction

```bash
python scripts/download_top1_replays.py \
  --output replays/top1_snapshot_RECHECK \
  --episodes 20 \
  --expected-team-id 16644287 \
  --expected-submission-id 55614463
```

ตรวจ `leaderboard_checked_at`, team ID, submission ID, SHA-256 และ rewards ใน manifest ก่อนใช้ทุกครั้ง เพื่อไม่ให้ leaderboard race ทำให้วิเคราะห์คนละ agent
