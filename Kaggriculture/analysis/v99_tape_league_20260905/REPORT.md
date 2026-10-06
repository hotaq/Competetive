# V99 — Tape league: lineage ปัจจุบัน (ShopForge/v97) ตกรุ่นแล้ว; CemBas/YUK/Lagrangian คือคลื่นใหม่

วันที่ 2026-09-05 Pre-registration: `PREREGISTRATION.md` (rule: pooled ≥ ROUTER และไม่แพ้ ROUTER เกิน 0.05 ต่อ opponent)
Data: `league.json` (pool 30 ส.ค. 1,390 ม้วน, 5.0M เกม), `league_20260905pool.json` (harvest 5 ก.ย. 313 ม้วนจาก
ทีม top-20 ปัจจุบัน, 1.1M เกม), `ranking_aug30.json`, `ranking_sep05.json` seeds 17100001–17100200 × 2 seats,
opponents 9 ตัวชุด V97, reference ROUTER = v97 (`artifacts/shop_router_v2/routes.json`) = **0.8025**

## ผล

| pool | tapes | ผ่าน rule | ดีสุด (pooled / เงิน / margin) |
|---|---:|---:|---|
| 30 ส.ค. (top-200 ตอนนั้น) | 1,390 | **0** | A Poor Vul ep103015872 0.886 / 88k / +12.9k แต่แพ้ v97 ต่อ kenjo165685 −0.135, ueddy −0.10 |
| 5 ก.ย. (top-20 ตอนนี้) | 313 | **27** | LagrangianLocomotive ep105793292 **0.997 / 107k / +39k**, YUK 0.996, CemBas 0.973 |

ทีม top-20 ตอนนี้ (keiz, Dmytro Maliarenko, peikopon, Jesse Bullard, LagrangianLocomotive, YUK, CemBas, …) **ไม่มีสักทีม
ใน pool 30 ส.ค.** meta หมุนภายใน 6 วันตามที่ discussion (sobameshi, topic 739273) วัดไว้: "newest strong economy wins",
clone ของ episode สาธารณะที่แรงที่สุดจะขึ้น top 10% แล้วเสื่อมใน 2–3 วัน

Head-to-head 200 seeds × 2 seats (แถวชนะคอลัมน์):

| | v97 | Lagrangian | YUK | CemBas |
|---|---:|---:|---:|---:|
| v97 | – | 0.01 (−30k) | 0.00 (−30k) | 0.06 (−12k) |
| Lagrangian | 0.99 | – | 0.22 | 0.01 |
| YUK | 1.00 | 0.78 | – | 0.00 (−117k) |
| **CemBas** | 0.94 | **0.98** | **1.00** | – |

## Lineage ของคลื่นใหม่

- YUK / LagrangianLocomotive / CemBas เป็นครอบครัวเดียวกัน (prefix 0–72 ตรงกัน 97%, 0–144 93–99%) ไม่ได้สืบจาก
  ShopForge (ตรง route 0 แค่ 41%) YUK เกือบ open-loop (self-identity 0.92, แตกที่ 73), Lagrangian กึ่ง adaptive (0.84)
- **CemBas = shop router**: 13 จาก 14 ม้วน share prefix เป๊ะถึง step 144, segment 72–144 เหมือนกัน 12/13, segment
  144–216 มี 3 แบบ (แบบ `cb19c47805` ใช้เมื่อ shop2 = YARN_STORE), แตกอีกที่ 216 (shop 3) replay เกมต้นฉบับใน
  engine เราได้เงินตรงทุกเหรียญ 13/13 → deterministic ตาม state, ใช้เป็น donor ได้
- peikopon (n=17, prefix เหมือนกันถึง 144) คือ ShopForge รุ่นใหม่ (ตรง route 0 90%) ได้ 0.94 — ดีกว่า v97 แต่แพ้ครอบครัว CemBas
- Andrey Tikhomirov, AI是我的豆包 เป็น lineage อื่น (0.06 / 0.09 vs route 0)

## ข้อสรุป

1. Live agent ของเรา (56027851) จะแพ้คลื่น CemBas/YUK/Lagrangian ราว 100% เมื่อเจอ ทำอะไรบน ShopForge ต่อก็ไม่ทัน
2. Rule ผ่านที่ 27 ม้วน stage 2 ตาม pre-registration: สร้าง shop router บน lineage ใหม่ — เลือก **CemBas** เพราะชนะ
   ทั้ง Lagrangian และ YUK และมี donor 13 ม้วนที่ prefix ตรงเป๊ะ (V100)
3. Opponent set ต้องเปลี่ยนเป็นยุคปัจจุบัน: tape ชุด 30 ส.ค. แพ้คลื่นใหม่ 100% จึงแยกแยะอะไรไม่ได้อีก
4. ตามกติกา final (Bradley-Terry หลัง deadline, ทีม = ตัวที่ดีกว่าใน 2 slot) slot ที่สองเป็น hedge ฟรี — ควรใช้
