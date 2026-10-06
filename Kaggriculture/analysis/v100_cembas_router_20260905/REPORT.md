# V100 — Router บน lineage CemBas: routing ให้ +0.011 (ไม่ถึงเกณฑ์ +0.02) แต่ lineage ใหม่ชนะ live +0.56 → build base tape

วันที่ 2026-09-05 Gates freeze ใน `PREREGISTRATION.md` (sha d346ee9d0c690cbb) Data: `train.json` (390k เกม fit),
`holdout.json` (100,000 เกม อ่านครั้งเดียว), `fit.json`, `gates.json` Scripts: pipeline V97 ที่ generalize
(`build_shop_router_donors.py --base/--pool`, `search_shop_router_cells.py` incumbent optional, `evaluate_shop_router_fast.py --tape`)

## Verdict

| gate | เกณฑ์ | วัดได้ | ผล |
|---|---|---|---|
| G1 | v100 − base ≥ +0.02, CI ไม่คร่อม 0 | **+0.0113** [+0.0089, +0.0138] | FAIL (บวกจริงแต่ไม่ถึงเกณฑ์) |
| G2 | ทุก opponent v100 ≥ base − 0.02 | แย่สุด +0.000 | PASS |
| G3 | routing == table | 0 violation / 20,000 | PASS |
| G4 | base − v97 ≥ +0.20; base ≥ 0.5 vs yuk, lagrangian | **+0.557** [+0.546, +0.568]; 1.000; 0.993 | PASS |

**Decision rule → build `base` tape (CemBas ep105807822 seat 0 เล่นตรงทั้งเกม)**
bundle: `submissions/cembas_base_tape_standalone/` main.py 20 KB (sha 56ab7d927b4565c1), tar.gz sha b643a1bbcd83f617,
equivalence 960 checks, เล่นใน official `kaggle_environments` ได้เงินตรง Rust loop ทั้งสอง seat

## Holdout (1,000 seeds × 10 opponents ยุคปัจจุบัน × 2 seats)

| arm | point | เงินเฉลี่ย | margin |
|---|---:|---:|---:|
| **v100** (CemBas router 8 cells) | 0.848 | 98,479 | +25,000 |
| **base** (CemBas tape เดี่ยว) | 0.836 | 97,965 | +24,280 |
| yuk (tape เดี่ยว) | 0.599 | 75,546 | −18,152 |
| lagrangian (tape เดี่ยว) | 0.592 | 89,671 | +7,802 |
| **v97 (live ตอนนี้)** | **0.280** | 83,892 | −10,621 |

ต่อ opponent: base ชนะ a_poor_vul 0.998 (v97 0.32), keiz 0.999, yuk 1.000, lagrangian 0.993, ai是我的豆包 0.963, yukino 0.974,
giulio 0.756, andrey_tikhomirov 0.646, peikopon 0.598, **baran_akyıldız 0.440** (แพ้ตัวเดียว) v97 แพ้เกือบทุกตัว
ที่เป็นคลื่นใหม่ (0.00–0.35)

## การ fit

8/64 cells เปลี่ยน แทบทั้งหมดคือ YARN_STORE__* → cembas ep105788591 (เกมต้นฉบับ YARN_STORE/ICE_CREAM 110k) in-sample
0.835 → 0.845 (+0.010) ≈ holdout +0.011 → ไม่ overfit แต่ผลเล็ก เพราะ CemBas เองก็ route ตาม shop อยู่แล้วใน tape ต้นฉบับ
donor 13 ม้วนต่างกันแค่ suffix; v100 ไม่แพ้ base ต่อ opponent ใดๆ (Pareto ≥) จึงเป็นทางเลือกที่ผู้ใช้เลือกได้
(`artifacts/shop_router_v3/routes.json`) แม้ rule ที่ freeze ให้ build base

## ข้อควรรู้ก่อน upload

1. agent นี้คือ clone ของ episode สาธารณะของทีม CemBas ทั้งเกม (host ยืนยันใน discussion ว่าอนุญาตและสนับสนุน) — ระบุใน
   `artifacts/shop_router_v3/PROVENANCE.md`
2. ตาม V99/discussion คลื่นแบบนี้ peak ภายใน 1–2 วันหลังเผยแพร่แล้วเสื่อมภายในไม่กี่วัน — ต้อง harvest ใหม่ทุก 2–3 วัน
   (`scripts/harvest_tape_pool.py` + `scripts/evaluate_tape_league.py`) และ upload ตัวที่ดีที่สุด ณ วันสุดท้าย
3. ยังแพ้ Baran Akyıldız (0.44) และ peikopon (0.60) — lineage ที่ควรเป็น donor/base ในรอบถัดไป
4. slot 2 เป็น hedge ฟรี (ทีม = ตัวที่ดีกว่า) ควรวางตัวนี้แทน 56016840 (1556) และเก็บ 56027851 ไว้จนกว่าจะมีตัวใหม่
