# V102 — v9/4 t0 opening override: FAIL ชัดเจน, ไม่ build (22 ก.ย.)

Gates freeze ใน `PREREGISTRATION.md` Data: `screen.json` (4800 เกม),
`gates.json` Verdict: **OVERALL FAIL — G1∧G2∧G3 ตกทั้งหมด ไม่ build bundle**

## ผล

| arm | point | mean margin |
|---|---|---|
| base (omar, BUY_PRODUCT step 0) | 0.8588 | +29,319 |
| opening (BUY_SEED step 0) | 0.0017 | −82,712 |

Paired (opponent, seed, seat) n=2400: mean diff **−112,031**,
95% CI [−114,013, −110,038] ชนะ 0/2400 (เสมอ 343 — เกมที่แพ้ทั้งคู่)

| gate | ผล |
|---|---|
| G1 (mean diff > 0, CI ไม่คร่อม 0) | **FAIL** (−112k, CI ห่าง 0 มาก) |
| G2 (point diff ≥ −0.01) | **FAIL** (−0.8571) |
| G3 (ทุก opp mean diff ≥ −200) | **FAIL** (แย่สุด ymg_aq −179,835; ดีสุด ebi −88,110) |

Seat symmetry: seat0 −111,870 / seat1 −112,193 — พังทั้งสอง seats เท่าๆ กัน

## ตีความ

Source claim (+107/-0 จาก 6648 openings) ไม่ transfer มาที่ omar tape นี้เลย
ทิศทางกลับกันแบบสุดขั้วและสม่ำเสมอทุก opponent (0 wins) กลไกที่เป็นไปได้:
BUY_SEED WHEAT 20 ใช้เงินสดมากกว่า BUY_PRODUCT มาก (หรือเปลี่ยน cash flow
day-0) จน tape open-loop ที่เหลือทั้ง 719 steps ซึ่ง calibrate มากับเงินตั้งต้น
แบบเดิมพังเป็น cascade — แต่レポートนี้ไม่ claim กลไกเพราะไม่ได้วัด state กลางเกม
routes files verify แล้วว่าต่างกันแค่ step-0 market จริง (byte-identical ที่เหลือ)
ดังนั้น effect นี้เป็น causal ไม่ใช่ wiring bug

Assumption ที่บันทึกไว้ใน prereg: ตีความ "+107/-0" เป็น context ไม่ใช่ gate —
การตัดสินใจจึงยึด gate ทิศทางที่ prereg ไว้เท่านั้น ไม่มีการผ่อนปรน

## ของที่ส่งมอบ (ไม่รวม bundle ตาม decision rule)

- `src/kaggriculture_rl/policy/opening_override.py` — overlay `OpeningOverride` /
  `apply(obs, action)` / `apply_to_routes_data` (ruff clean, มี test 4 ข้อ)
- `artifacts/shop_router_v4/base_tape_routes_opening.json` — routes ที่ bake
  override ไว้ (ต่างจาก base แค่ step-0 market, verify ตรงกับ overlay ทั้ง 2 seats)
- `tests/policy/test_opening_override.py` — 4 passed
- ไม่มี `submissions/` ใหม่ (rule: ไม่ผ่าน gate = ไม่ build) และไม่มีไฟล์ใดใต้
  `submissions/` ถูกแตะ

## ข้อจำกัด / ต่อยอด (ถ้าจะทำ — ต้อง prereg ใหม่ ห้าม tune บน seeds 18400xxx)

- Pyramid นี้วัดแค่ full 20-seed-quantity swap; ไม่ได้แยกว่า BUY_SEED แพงกว่า
  หรือ SELL side effect — mutation เล็กลง (เช่น BUY_SEED qty น้อย) เป็นคำถามใหม่
- Opponent เป็น tape open-loop 6 ตัว; closed-loop opponents ไม่ได้วัด
