# Paired causal trace: V59 vs V74

ขอบเขตการวิเคราะห์นี้มีเพียง old80 สอง episode ที่ได้รับอนุญาต:

- `99614514` — replay `r10_16670032_kenjo1209`, top1 seat 0, candidate seat 1
- `99589578` — replay `r09_16748154_taiseiu`, top1 seat 0, candidate seat 1

## Verdict

V74 ไม่ได้แพ้เพราะ action ผิดกติกาหรือมีสัตว์ค้าง แต่แพ้เพราะหลัง handoff ที่ step 120
มันไม่มีแรงงานรายวันเลย แม้ scheduler จะรายงาน target ว่า `PLANNED` อยู่เรื่อย ๆ
ผลคือ strawberry เริ่มหายที่ step 144, พืชทั้งหมดหายที่ step 192, sheep ลดที่ step 240
และสัตว์ทั้งหมดหายภายใน step 360

V59 ทำตรงข้าม: ขายของก่อนเพื่อจ้าง 6 hands ใน action เดียวกันที่ step 120 แล้วจ้างใหม่ทุกวัน
วงจรแรงงานนี้รักษาพืชและสัตว์ไว้จนยอดขาย fertilizer, melon, milk, strawberry และ wool ทบต้น

| Episode | V59 candidate / top1 / margin | V74 candidate / top1 / margin |
|---|---:|---:|
| 99614514 | 106,073 / 94,407 / **+11,666** | 2,925 / 141,575 / **-138,650** |
| 99589578 | 107,066 / 107,039 / **+27** | 4,167 / 183,263 / **-179,096** |

เงินของ top1 เปลี่ยนตาม candidate เพราะใช้ shared market จึงควรให้น้ำหนักกับ trajectory
ของ candidate มากกว่าเทียบ final margin อย่างเดียว

## ข้อจำกัดเชิงเหตุผล

นี่ไม่ใช่ clean A/B ตั้งแต่ step 120 เพราะ V59 กับ V74 ใช้ opening คนละตัว
และ market/action แรกต่างกันตั้งแต่ step 0; pre-action economic state ต่างครั้งแรกที่ step 1
V59 เลือก route `base` ในทั้งสอง episode ส่วน V74 ใช้ V16 opening ก่อน handoff
ดังนั้นผลนี้ชี้กลไกความล้มเหลวได้ แต่ห้ามอ้างว่า quota controller เป็นสาเหตุเดียวแบบ randomized causal test

## First economic divergence

ที่ step 0:

- V59: buy sheep 4, product wheat 4, melon seed 7, wheat seed 5, hire 2; เงิน 3,000 → 280
- V74: buy cow 1, sheep 4, product wheat 14, melon seed 5, wheat seed 5, hire 5; เงิน 3,000 → 4

ที่ step 120 รูปแบบเหมือนกันในทั้งสอง episode:

| ก่อน/หลัง action | V59 | V74 |
|---|---|---|
| เงินก่อน action | 9 / 10 | 359 / 359 |
| land; cow; sheep | 1; 1; 4 | 1; 1; 4 |
| crop M/S/W | 7/3/9 | 5/3/6 |
| shed | F4, W24 | F5, W28; seed W1 |
| market prefix | sell F5, sell W1, **hire 6** | sell F5, **hire 0** |
| farmer | pickup W5 | pickup W1 |
| เงินหลัง action | 400 / 400 | 828 / 828 |

V74 introspection ที่ step 120 เลือก cow maintenance ซึ่ง `baseline=goal=1` และรายงาน
animal target กับ selling target เป็น `PLANNED` แต่ไม่ได้สร้าง workforce หรือ productive progress

## Timeline checkpoints

สถานะ step 120–600 เป็น pre-action ที่ต้นวัน จึงมี hands ปัจจุบันเท่ากับ 0 ทั้งคู่
คอลัมน์ `H` แสดง `current/peak-in-previous-interval`; step 719 เป็น final state หลัง action
สัตว์เขียนเป็น `live/owned/stranded`; crop ใช้ `M/S/W`; inventory คือ shed ที่สำคัญ
โดย `Cr/F/M/S/W` หมายถึง carrot/fertilizer/melon/strawberry/wheat

### Episode 99614514

| Step | V59 money/L/H | V59 C; S | V59 crop; inventory | V74 money/L/H | V74 C; S | V74 crop; inventory |
|---:|---|---|---|---|---|---|
| 120 | 9/1/0-3 | 1/1/0; 4/4/0 | 7/3/9; F4 W24 | 359/1/0-5 | 1/1/0; 4/4/0 | 5/3/6; F5 W28, seed W1 |
| 144 | 629/1/0-6 | 1/1/0; 4/4/0 | 7/7/6; W23 | 828/1/0-0 | 1/1/0; 4/4/0 | 5/0/5; W23, seed W1 |
| 192 | 280/2/0-6 | 6/8/2; 4/4/0 | 7/20/10; C2 F3 W14 | 3,524/1/0-0 | 1/1/0; 4/4/0 | 0/0/0; W14 wool5, seed W1 |
| 240 | 1,257/2/0-8 | 9/9/0; 4/4/0 | 7/21/5; F9 W19 wool4 | 5,245/1/0-0 | 1/1/0; 3/3/0 | 0/0/0; W6 wool4, seed W1 |
| 360 | 11,038/3/0-13 | 11/11/0; 4/4/0 | 4/40/12; Cr20 F11 milk2 S5 W15 wool9 | 2,845/3/0-0 | 0/0/0; 0/0/0 | 0/0/1; wool8 |
| 480 | 45,189/3/0-12 | 11/11/0; 4/4/0 | 4/37/15; F17 milk6 S20 W39 wool6 | 2,845/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; W1 wool8 |
| 600 | 73,602/3/0-13 | 11/11/0; 4/4/0 | 0/19/41; F11 milk12 S23 W30 wool16 | 2,845/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; W1 wool8 |
| 719 | 106,073/3/9-15 | 11/11/0; 4/4/0 | 0/0/12; liquidated | 2,925/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; liquidated |

### Episode 99589578

| Step | V59 money/L/H | V59 C; S | V59 crop; inventory | V74 money/L/H | V74 C; S | V74 crop; inventory |
|---:|---|---|---|---|---|---|
| 120 | 10/1/0-3 | 1/1/0; 4/4/0 | 7/3/9; F4 W24 | 359/1/0-5 | 1/1/0; 4/4/0 | 5/3/6; F5 W28, seed W1 |
| 144 | 624/1/0-6 | 1/1/0; 4/4/0 | 7/7/6; W23 | 828/1/0-0 | 1/1/0; 4/4/0 | 5/0/5; W23, seed W1 |
| 192 | 292/2/0-6 | 6/8/2; 4/4/0 | 7/20/10; C2 F3 W14 | 3,525/1/0-0 | 1/1/0; 4/4/0 | 0/0/0; W14 wool5, seed W1 |
| 240 | 985/2/0-8 | 9/9/0; 4/4/0 | 7/21/7; F9 W19 wool4 | 5,362/1/0-0 | 1/1/0; 3/3/0 | 0/0/0; W6 wool4, seed W1 |
| 360 | 11,063/3/0-13 | 11/11/0; 4/4/0 | 4/40/12; Cr20 F11 milk3 S5 W18 wool9 | 4,120/3/0-0 | 0/0/0; 0/0/0 | 0/0/1; empty shed |
| 480 | 49,990/3/0-12 | 11/11/0; 4/4/0 | 4/37/15; F17 milk6 S20 W42 wool6 | 4,120/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; W1 |
| 600 | 84,518/3/0-13 | 11/11/0; 4/4/0 | 0/19/41; F11 milk12 S23 W33 wool16 | 4,120/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; W1 |
| 719 | 107,066/3/9-15 | 11/11/0; 4/4/0 | 0/0/12; liquidated | 4,167/3/0-0 | 0/0/0; 0/0/0 | 0/0/0; liquidated |

Unit inventory ว่างที่ checkpoint ที่ร้องขอทั้งหมดสำหรับทั้งสอง agent
V59 มี cow ค้างชั่วคราว 2 ตัวที่ step 192 แต่ลงฟาร์มครบที่ step 240
V74 มี stranded เป็นศูนย์เพราะไม่ซื้อ replacement เพิ่ม ไม่ใช่เพราะ completion ดีกว่า

## Buy/sell flow ระหว่าง checkpoints

ค่าตัวเลขเงินเรียงเป็น `99614514 / 99589578`; รายการที่ต่างกันระบุด้วย `/`

| Interval | V59 orders หลัก; money delta | V74 orders; money delta |
|---|---|---|
| 0–119 | buy C1 S4 product W25, seed M7 S3 W14, H14; sell F12; **-2,991/-2,990** | buy C1 S4 product W37, seed M8 S3 W14, H15; sell F15 W9; **-2,641/-2,641** |
| 120–143 | seed S4, H6; sell F11 W6; **+620/+614** | sell F5; **+469/+469** |
| 144–191 | buy C7 L1 product W3, seed S13 W5, H12; sell F9 wool24; **-349/-332** | sell W1 wool15; **+2,696/+2,697** |
| 192–239 | buy C1 product W14, seed S1, H15; episode 995 เพิ่ม seed W2; sell F18 wool13; **+977/+693** | sell milk5 wool5; **+1,721/+1,837** |
| 240–359 | buy cow2 L1 product W41, seed Cr8 M4 S19 W12, H57; sell Cr1 F70 M42 milk12 S5 wool11; **+9,781/+10,078** | buy L2 product W2; sell milk3 และ episode 995 ขาย wool8; **-2,400/-1,242** |
| 360–479 | buy product W21, seed W22, H57; sell Cr20 F32 milk92/98 S94 wool35; **+34,151/+38,927** | ไม่มี action ตลาด; **0/0** |
| 480–599 | seed W47, H61; sell F45 M24 milk77 S154 wool22; **+28,413/+34,528** | ไม่มี action ตลาด; **0/0** |
| 600–718 | seed Cr12 W30, H60; sell Cr36 F106 milk90 S77 W226/228 wool37; **+32,471/+22,548** | final liquidation: W1 และ episode 996 เพิ่ม wool8; **+80/+47** |

## Profitable loop ของ V59

1. Step 120 ขาย F5 + W1 ก่อน HIRE6 จึงเปลี่ยนเงิน 9/10 เป็น 400 โดยไม่รอสะสม cash
2. จ้างใหม่ทุกวันและยอมใช้เงินลงทุน: ก่อน step 192 ซื้อ cow 7, land 1 และ strawberry seed 13
3. แก้ cow stranded จาก 2 เป็น 0 ภายใน step 240 พร้อมมี land 2, cow 9, sheep 4
4. ภายใน step 360 มี land 3, cow 11, sheep 4 และ crop 56 ช่อง
5. Workforce รักษา recurring output หลายชนิด; premium sales ช่วง step 360–599 เพิ่มเงิน
   62,564 / 73,455 ก่อน liquidation ท้ายเกม

กำไรจึงไม่ได้มาจากสินค้า premium ตัวเดียว แต่เป็นวงจร
`sell prefix -> hire/maintain -> multi-product production -> premium sales`

## Smallest causal delta สำหรับ V76 ถ้า V75 fail

ไม่ควรย้าย V74 executor มาทับ V59 และไม่ควร rewrite route ที่ชนะอยู่แล้ว ให้เริ่มจาก action ของ V59
แล้วเพิ่มเพียง **sale-funded maintenance fallback**:

1. รักษา movement, SELL, HIRE และ BUY_LAND ของ V59 เดิมแบบ byte-for-byte เมื่อ fallback ไม่ทำงาน
2. เปิด fallback เฉพาะเมื่อ productive capacity ลดต่อเนื่อง และ action เดิมของ V59 ไม่มีคำสั่งแก้ชนิดนั้น
3. เพิ่ม maintenance ได้สูงสุดหนึ่ง order/action ต่อ turn
4. คำนวณ affordability จากเงินหลัง SELL ที่อยู่ก่อนหน้าใน market prefix จริง
5. ห้ามกัน `next_land_cost` จาก replacement seed/feed/animal สำหรับโครงสร้างที่มีอยู่แล้ว
6. ห้ามใช้ episode id, replay id หรือตัวบ่งชี้ identity ใด ๆ

Positive-control gate ต้องเป็น fallback activation = 0 และ action parity **719/719** กับ V59
ในทั้งสอง trace นี้ก่อนทดสอบ failure case ของ V75 เพราะ episode 99589578 ชนะเพียง +27

## Validation และ artifacts

- ทั้งสี่ run จบ step 719
- normalization changes = 0 และ sanitizer changes = 0 ทุก run
- V59 final results ตรง frozen old80 report; V74 final results ตรง V74 iter2 screen
- `trace_pairs.py` เป็น reproducer ที่ hard-code เฉพาะสอง replay ที่ได้รับอนุญาต
- `result.json` เก็บ actions, states, policy metadata, checkpoints, interval orders และ first divergences ทุก step
