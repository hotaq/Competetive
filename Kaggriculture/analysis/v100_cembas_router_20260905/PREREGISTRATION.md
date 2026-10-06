# V100 — Shop router บน lineage CemBas (คลื่นใหม่จาก V99): router ดีกว่า tape เดี่ยวไหม และควร upload ตัวไหน

Pre-registered 2026-09-05 ก่อนรัน seed ใดๆ ในช่วงด้านล่าง (ไม่มี smoke บนช่วงนี้)

## ที่มา (V99)

tape ของ CemBas/YUK/LagrangianLocomotive (harvest 5 ก.ย.) ชนะ v97 ~100% (−30k/เกม) และชนะ opponent ชุด 30 ส.ค. ~100%
CemBas ชนะ Lagrangian 0.98 และ YUK 1.00 ใน head-to-head และเป็น shop router ในตัว: 13 ม้วน share prefix เป๊ะถึง step 144
แล้วแตกกิ่งที่ 144/216 ตามคู่ shop replay เกมต้นฉบับใน engine ตรงทุกเหรียญ 13/13

## Donors (freeze `artifacts/shop_router_v3/donors.json` sha 982225dea342d902)

13 ม้วน CemBas, donor 0 = base `cembas__ep105807822__seat0` (ดีสุดใน league) ทุก donor family 0 (prefix = base ถึง 144)
สร้างด้วย `scripts/build_shop_router_donors.py --base ... --pool artifacts/tape_pool_20260905` (sha 665b35013c807f9e)
incumbent = donor 0 ทุก cell (ไม่มี table เดิม)

## Opponents (10 tapes ยุคปัจจุบัน, `opponents/`)

| opponent | ที่มา | sha256[:16] |
|---|---|---|
| yuk__ep105790347__seat0 | คลื่นใหม่ lineage เดียวกัน (league #2 อันดับ 2) | e19cc0214e6a91b3 |
| lagrangianlocomotive__ep105793292__seat0 | คลื่นใหม่ lineage เดียวกัน (อันดับ 1) | 1563e8ededf669aa |
| peikopon__ep105807824__seat0 | ShopForge รุ่นใหม่ (0.94) | 332959455e698e81 |
| andrey_tikhomirov__ep105796705__seat0 | lineage อื่น (0.90) | fbe0f061839c8239 |
| ai是我的豆包__ep105803010__seat0 | lineage อื่น (0.94) | d4c4cabdbb2e53dd |
| baran_akyıldız__ep105805663__seat1 | lineage peikopon-like (0.92) | c6bdc01f7d7a65ed |
| giulio_ravasio__ep105784996__seat1 | (0.89) | eb073860e050c96e |
| keiz__ep105793106__seat0 | rank-1 ทีมปัจจุบัน (0.88) | 3852927dc27b7891 |
| a_poor_vul__ep102980038__seat1 | ยุคเก่า adaptive ที่ยากสุดสำหรับเรา | bee71df46cb83c0d |
| yukino__ep103129686__seat0 | ตัวแทน ShopForge clone ที่ยังอยู่บน ladder | 5139fa6cc3843437 |

## Seeds — ใหม่ (ไม่ทับ 171/172xxxxx ของ V99)

| stage | seeds |
|---|---|
| train (fit table) | 17300001–17301500 |
| holdout (อ่าน gate ครั้งเดียว) | 17400001–17401000 |

Fit params เหมือน V97: min_cell_games 30, min_opponent_games 8, tolerance 0.05, min_gain 0.01 (`scripts/search_shop_router_cells.py`
sha ac7215381ecd3abd) holdout ผ่าน `scripts/evaluate_shop_router_fast.py` (sha 6efc2e40160ce726) arms: **v100** (table ที่ fit),
**base** (CemBas ep105807822 เล่นตรงทั้งเกม), **v97** (live ตอนนี้), **lagrangian**, **yuk** (tape เดี่ยวคลื่นใหม่)
engine config เดิม, ทั้งสอง seat, หน่วยวิเคราะห์ (opponent, seed) bootstrap ตาม seed 10,000 รอบ

## Gates — freeze

| gate | เกณฑ์ |
|---|---|
| **G1** | point(v100) − point(base) ≥ +0.02 และ 95% CI ไม่คร่อม 0 |
| **G2** | ทุก opponent: point(v100) ≥ point(base) − 0.02 |
| **G3** | ทุกเกม v100: route ที่เล่น == table lookup |
| **G4** | point(base) − point(v97) ≥ +0.20 และ point(base) ≥ 0.50 ต่อ yuk และต่อ lagrangian แยกกัน (ยืนยันว่าเปลี่ยน lineage คือกำไรหลัก) |

**Decision rule:** ถ้า G1 ∧ G2 ∧ G3 ∧ G4 → build bundle จาก v100 ถ้า G4 ผ่านแต่ G1 หรือ G2 ไม่ผ่าน → build bundle จาก
**base tape เดี่ยว** (routing ไม่เพิ่มค่าแต่ lineage ใหม่ดีกว่าที่ live อยู่มาก) ถ้า G4 ไม่ผ่าน → ไม่ build
Bundle ที่ได้ตั้งใจใช้กับ slot ที่ 2 (hedge ฟรีตามกติกา final) การ upload เป็นการตัดสินใจของผู้ใช้

## บันทึกเสมอ

cell ที่เปลี่ยน, donor ที่ใช้, in-sample vs holdout, ตาราง point ต่อ opponent ทั้ง 5 arms, mean margin, ผลใน cell ที่เปลี่ยน
