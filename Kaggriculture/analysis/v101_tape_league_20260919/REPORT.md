# V101 — Refresh 19 ก.ย.: สอง slot ที่ live เสื่อมแล้ว; Omar Althobaiti tape ผ่านทุก gate → build

Gates freeze ใน `PREREGISTRATION.md` Data: `screen.json` (380 ม้วน × 10 opp × 100 seeds), `round_robin.json`
(12 ทีม + CemBas live, 200 seeds), `holdout.json` (60,000 เกม, อ่านครั้งเดียว), `gates.json`

- Live ก่อนเริ่ม: 56034703 (CemBas) 1337, 56027851 (V97) 1225, rank 2951/9540, top-1 3278
- Field top-20 ปัจจุบันเกือบทั้งหมดเป็น lineage เดียว (prefix 0–72 ตรงกัน 0.96) CemBas live ได้ **0.035** กับพวกนี้ (−44k/เกม)
- ทีม top-20 จริง (Arda Ceylan, Sida Zuo, … — RL/closed-loop ตาม discussion) tape พังเมื่อเล่น open-loop; tape ที่ดีคือของคู่แข่งพวกเขา

| arm | holdout point | − cembas_live | G1 | G2 | G3 |
|---|---:|---|---|---|---|
| **W** omar_althobaiti ep110807162 seat1 | **0.753** (+10.3k) | +0.719 [+0.710, +0.727] | PASS | PASS | PASS (official env ตรงทั้งสอง seat) |
| W2 kaggriculture_agent ep110814298 | 0.437 | +0.402 | PASS | **FAIL** | – (ไม่ build) |

W ต่อ opponent: 0.61–0.996 (ต่ำสุด suyash_verma 0.610, nilochan 0.641) ไม่มีตัวไหนต่ำกว่า 0.5
Bundle: `submissions/omar_base_tape_standalone/` (main sha 06d2a4d66656d83e, tar sha 37b3481c69c6243b)
Router ไม่ทำ: ไม่ได้ตรวจ donor ≥ 8 และ V100 ให้แค่ +0.011 Hedge คนละ lineage ยังไม่มี (ymg_aq 0.21) — slot 2 คง CemBas ไว้หรือใส่อันดับ 2 (nilochan) ก็ได้
ข้อจำกัด: opponent เป็น tape ของ agent ที่หลายตัว adaptive; ตามประวัติคลื่นแบบนี้เสื่อมใน 2–5 วัน → ต้อง refresh อีกครั้ง ~28–29 ก.ย. ก่อน deadline
