# V101 — Refresh lineage (19 ก.ย.): tape ไหนใน top-20 ปัจจุบันควรแทน 2 slot ที่เสื่อมแล้ว

Pre-registered 2026-09-19 ก่อนรัน seed ใดๆ ด้านล่าง Live ตอนนี้: 56034703 (CemBas) 1337 / 56027851 (V97) 1225, rank 2951/9540,
top-1 3278 — ตรงกับ V99: คลื่น replay เสื่อมในไม่กี่วัน Deadline 2026-09-30; final = Bradley-Terry หลัง deadline, ทีม = ตัวที่ดีกว่าใน 2 slot

## Pool (freeze)
`artifacts/tape_pool_20260919.tar.gz` sha 833151c8f8c6c6bc — 380 tapes, top-20 teams ณ 2026-09-19 (`harvest_tape_pool.py --teams 20 --episodes-per-team 12`)

## Stages (seed ใหม่ทุก stage, engine config เดิม, `play_batch`, ทั้งสอง seat)
1. **Screen** seeds 18100001–18100100: ทั้ง 380 ม้วน vs opponents ชุด V100 (10 ตัว) reference ROUTER = CemBas live
   (`artifacts/shop_router_v3/base_tape_routes.json`) → เก็บ tape ดีสุดต่อทีม, เอา 12 ทีมบนสุด
2. **Round-robin** seeds 18200001–18200200: 12 ม้วนนั้น + CemBas live เล่นกันเองทุกคู่ → W = pooled point สูงสุดในหมู่ 12 ม้วน
   (ไม่นับเกมกับ CemBas), W2 = อันดับ 2 ที่ **ต่างทีมและ prefix 0–72 ตรงกับ W < 0.9** (hedge ต้องเป็นคนละ lineage)
3. **Holdout** seeds 18300001–18301000: arms W, W2, cembas_live vs opponents = อีก 10 ม้วนจาก stage 2 (ไม่รวม W, W2)

## Gates (freeze)
| gate | เกณฑ์ |
|---|---|
| G1 | point(W) − point(cembas_live) ≥ +0.20, 95% CI (bootstrap ตาม seed) ไม่คร่อม 0 |
| G2 | point(W) ≥ 0.50 pooled บน holdout |
| G3 | bundle ของ W เล่นใน official kaggle_environments ได้เงินตรง `play_fast` ทั้งสอง seat |
| G1'/G2' | เกณฑ์เดียวกันสำหรับ W2 (ตัดสินแยก) |

Decision: build standalone bundle ของตัวที่ผ่าน G1∧G2∧G3 (W → slot 1, W2 → slot 2) ตัวที่ไม่ผ่านไม่ build
Router บน lineage ของ W ทำเฉพาะถ้ามี donor prefix-ตรง-ถึง-144 ≥ 8 ม้วน (V100 ให้แค่ +0.011 จึงไม่ใช่ทางหลัก)
Upload เป็นการตัดสินใจของผู้ใช้ Scripts: evaluate_tape_league.py beb116b0112ef49f, evaluate_shop_router_fast.py 6efc2e40160ce726
