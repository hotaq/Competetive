# V103 — CAPHARV micro-rule screen (22 ก.ย.): harvest-now ป้องกัน overnight overflow waste

Pre-registered 2026-09-22 ก่อนรัน seed ใดๆ ด้านล่าง Unit-level verification ผ่านแล้วที่นี่
(`tests/policy/test_unit_rules.py` + `CapHarvRule` ใน `src/kaggriculture_rl/policy/unit_rules.py`);
league screen ด้านล่างยังไม่รัน — รันแยกหลัง prereg นี้ freeze

## Hypothesis (freeze)

v9/4 วัด CAPHARV +20/-0 บน 1000 เกม top-25: กลไกคือ midnight dump ทิ้งของ —
ทุก carried inventory ถูกเทเข้า shed ตาม `shedCapacity` (100, non-seed items) ส่วนเกินถูก discard
(`drop_inventories_to_shed`, `rust/kaggriculture-engine/src/engine.rs:1146`;
ground truth จาก engine ไม่ใช่ notebook เพียงอย่างเดียว)

กฎ `CapHarvRule`: ทุก COOP/PASTURE tile ที่มี CARE หรือ COLLECT_FERTILIZER อยู่ในคืนนี้
และ projected shed+carried (shed stock + carried รวมทุก unit + tile yield ที่ harvest-now จะยกขึ้นมา)
จะเกิน shedCapacity (100) โดยไม่มี HARVEST อื่นใน action เดียวกัน → แทนด้วย HARVEST ทันที
เหตุผล: banked-care มีค่าแค่ +1/day จ่ายที่ production รอบถัดไป (ต้อง fed ในวันผลิต ไม่งั้น bank หาย,
tile cap 6 cow/sheep และ 4 goose) ซึ่งน้อยกว่า concrete yield ที่เสี่ยงถูก discard ที่ midnight —
harvest ตอนนี้ให้ downstream DROP/SELL มี steps เหลือมากที่สุดก่อน midnight dump

Facts จาก engine (ground truth, `rules.rs` + `engine.rs`):
FEED keeps alive (missed >= 2 วันติด → tile ถูกล้าง), CARE ตั้ง `cared_today` จ่าย bonus ที่ production
ถัดไปเฉพาะเมื่อ fed (`daily_refresh_animals`, `engine.rs:1087`), production ต่อรอบ = 1 + bonus
(ถ้า fed) cap ที่ max_held (goose 4 ทุก 1 วัน, cow 6 ทุก 2 วัน, sheep 6 ทุก 3 วัน)

## Scope freeze (unit-level ที่ทำแล้ว — ห้ามขยาย)

- Additive only: ไม่แตะ rule classes เดิม, ไม่แตะ `submissions/`, `rust/`, `bindings/`
- Rule pure (no I/O, no randomness), handle ทั้งสอง seat ผ่าน observation player/farms indexing
- Proxy ที่รู้ข้อจำกัด: "no later harvest scheduled before midnight" ดูจาก HARVEST ใน action
  ปัจจุบันเท่านั้น (pure single-step view — ถ้า league ไม่ผ่าน ค่อยพิจารณา hour gate / schedule lookahead)

## Stages (seed ใหม่ทุก stage, engine config เดิม, `play_batch`, ทั้งสอง seat)

1. **Screen** seeds 19100001–19101000: base + CapHarvRule vs opponents ชุด V100 (10 ตัว)
   reference ROUTER = CemBas live (`artifacts/shop_router_v3/base_tape_routes.json`) → 1000 เกม top-25
   (ประกบ v9/4 protocol เดิม: วัดบน top-25 games เท่านั้น ไม่ใช่ทุกเกม)
2. **Holdout** seeds 19300001–19301000: arms capharv, cembas_live vs opponents = อีก 10 ม้วนจาก stage 2
   ของ V101 (ไม่รวม W, W2) — reuse opponent set เดิมเพื่อ comparability กับ V101

## Gates (freeze)

| gate | เกณฑ์ |
|---|---|
| G1 | point(capharv) − point(cembas_live) ≥ +0.20, 95% CI (bootstrap ตาม seed) ไม่คร่อม 0 |
| G2 | point(capharv) ≥ 0.50 pooled บน holdout |
| G3 | bundle ของ capharv เล่นใน official kaggle_environments ได้เงินตรง `play_fast` ทั้งสอง seat |

Decision: ผ่าน G1∧G2∧G3 → promote CapHarvRule เข้า default policy stack (slot เดียวกับ v9/4);
ตัวที่ไม่ผ่านไม่ build — กลับมาแก้ projection (เช่น รวม COLLECT +1 fertilizer, hour gate) แล้ว prereg ใหม่
ห้ามแก้ gate หลังเห็นผล Scripts: evaluate_tape_league.py beb116b0112ef49f, evaluate_shop_router_fast.py 6efc2e40160ce726
