# V98 — Day-18 handover สู่ closed-loop manager: ผลลบ (ไม่ได้ pre-register, ไม่มี gate ถูกอ่าน)

วันที่ 2026-09-05 งานนี้เป็น **development probe** เพื่อดูว่า design มีโอกาสก่อน pre-register: ผลแพ้ baseline
ชัดเจนจนไม่ต้องรัน holdout จึงบันทึกเป็น negative result ตาม convention ของ repo (เหมือน `metered_overlay`)
Code: `src/kaggriculture_rl/policy/late_game_manager.py`, `src/kaggriculture_rl/policy/idle_hand_overlay.py`,
harness `scripts/evaluate_shop_router_policy.py --router NAME=PATH` suffix `+late[STEP]` / `+idle`
Data: `probe_handover_vs_tape.json` (120 เกม), `probe_idle_overlay.json` (5,400 เกม), script `probe_handover_vs_tape.py`

## คำถาม

ช่องว่างที่แพ้ทั้งหมดเกิดในวันที่ 18–30 (memory `kaggriculture-endgame-gap-is-production-not-liquidation`):
คู่แข่งที่ชนะทำรายได้ 103k vs เรา 63k จากนม/สตรอว์เบอร์รี ถ้าให้ router v97 เล่นถึง step 431 แล้วส่งต่อให้
ตัวจัดการฟาร์มแบบ closed-loop จะได้เงินช่วงท้ายมากกว่า tape ที่บันทึกไว้หรือไม่

## สิ่งที่สร้าง

`LateGameManager`: แผนรายวันที่ hour 0 (จ้าง 12, ซื้อข้าวสาลีให้พอเลี้ยงสัตว์, ซื้อเมล็ด), ทุก unit หยิบข้าวสาลี/ปุ๋ยที่
shed แล้วเดินทัวร์ nearest-neighbour ประจำวันตามลำดับคุณค่า (สัตว์ → เก็บเกี่ยว/สตรอว์เบอร์รี → รดน้ำ → ปลูก)
ทำ HARVEST/FEED/CARE/COLLECT_FERTILIZER บนสัตว์, HARVEST/WATER/FERTILIZE บนพืช, ปลูกข้าวสาลีซ้ำถึงวันที่ 26,
ขายของใน shed ทุก step เป็นก้อนละ 6, วันสุดท้าย DROP กลับ shed ก่อนตลาดปิด
`IdleHandOverlay`: แทน PASS ด้วย CARE/WATER/COLLECT บน tile ที่ยืนอยู่ (ไม่ขยับ unit)

## บั๊กที่ไล่แก้ (แต่ละข้อเปลี่ยนผลอย่างมีนัย — เป็นความรู้เรื่อง engine ที่ควรจำ)

1. FEED ใช้ข้าวสาลีจาก inventory ของ unit ที่ทำ ต้องหยิบที่ shed (4 tile กลางกระดาน) และต้องมีข้าวสาลีใน shed
   พอ 17+/วัน ไม่งั้นสัตว์อดตายใน 2 วัน
2. คิว market 10 order/turn: HIRE 12 ตัดคำสั่งซื้อข้าวสาลีทิ้ง ต้องแยกจ้างเป็น hour 0/1
3. นม/ขนแกะสะสมบน tile สัตว์ ต้อง HARVEST ถึงจะได้ของ (FEED/CARE อย่างเดียวได้ 0)
4. Atomic planting: ถ้า PLANT รวมทุก unit ใน step เกินเมล็ดที่มี **ทุกคำสั่งล้ม** ต้องจำกัดจำนวน PLANT ต่อ step
5. tile ว่างใน quadrant ที่ยังไม่ซื้อดูเหมือน tile ว่าง แต่ PLANT ล้ม (ใน engine obs เป็น "LOCKED" ใน live เป็น null)
6. ต้นที่ปลูกใหม่ต้องรดน้ำวันนั้นเลย ไม่งั้นนับเป็นไม่รดน้ำ 1 วันทันที
7. สตรอว์เบอร์รีให้ 2 หน่วยต่อรอบเมื่อ **รดน้ำและใส่ปุ๋ย** (ไม่งั้น 1) และให้แค่ 4 รอบแล้วตาย → ปุ๋ยจาก
   COLLECT_FERTILIZER มีค่ามากกว่าราคาปุ๋ยในตลาด

## ผล

| probe | ผล |
|---|---|
| handover vs tape, 2 opponents × 60 seeds, seat 0 | manager ดีกว่า **0/120** เกม; gain d18→30 median +46k vs +62k; ชนะ 65 vs 110 |
| idle overlay, 9 opponents × 100 seeds × 2 seats | point −0.006 [−0.043, +0.032], margin −1.4k [−2.5k, −0.3k] |

Per-crop (seed 16700001 vs kenjo): นม/ขนแกะเท่ากัน (108/176) สตรอว์เบอร์รี 171 vs 209 ข้าวสาลี 320 vs 402
tape ยังเทข้าวสาลี ~250 และปุ๋ย ~185 หน่วยลงตลาด ซึ่งกดรายได้คู่แข่งลง ~7k (คู่แข่งได้ 87.6k vs 80.7k)
ฟาร์มของ tape ที่วันที่ 18 สมบูรณ์ทุก seed (แปลง 52–62, สัตว์ 10–17, วัชพืช ≤ 13) จึงไม่มี "เกมพัง" ให้ rescue

## ข้อสรุป

late game ของ tape คือบันทึกของ agent ระดับ expert ตัวจัดการ greedy ทำได้ ~75% และต่อให้เท่ากันก็ไม่ปิด
ช่องว่าง 40k ของผู้ชนะ ซึ่งมาจากจำนวนวัวที่ตัดสินตอนเปิดเกม สิ่งที่เหลือ: (1) route-level — donor ที่เปิดเกมด้วยวัว
มากกว่า วัดทั้งเกมด้วย `play_batch` (การสลับแกะ→วัวใน tape ตรงๆ ล้มเหลว: เงินร่วง 91k→49k เพราะเงินขนแกะวันที่ 6
เป็นตัวจ่ายซื้อของถัดไป) (2) market leverage — เทของกดราคาคู่แข่งอย่างจงใจ

Overlay ที่แตะการเปลี่ยนแปลงของ tile (ปลูก/เก็บ) จะ re-roll RNG วัชพืชของทั้งสองฟาร์ม ผลจึงวัดได้เฉพาะเชิงสถิติ
บน seed จำนวนมากเท่านั้น (`probe_idle_overlay.json`: seed เดียวเงินแกว่ง 60k ↔ 127k)
