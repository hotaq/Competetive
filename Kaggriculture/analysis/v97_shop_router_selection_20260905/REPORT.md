# V97 — เลือก route ต่อ cell แบบ Yusuke บน donor ของเรา: ผ่านทั้ง 5 gate → build bundle

วันที่ 2026-09-05 Gates freeze ใน `PREREGISTRATION.md` (sha d1c538916d107b3a) ก่อนรัน seed ใดๆ
Scripts: `scripts/build_shop_router_donors.py`, `scripts/search_shop_router_cells.py`,
`scripts/evaluate_shop_router_fast.py` (Rust loop), `scripts/evaluate_shop_router_policy.py --no-sanitize`
(V21), `analyze_gates.py` Artifacts: `screen.json`, `train.json` (1.35M เกม), `fit.json`,
`holdout.json` (72,000 เกม), `holdout_v21.json` (2,400 เกม), `gates.json`

## Verdict

**OVERALL PASS — G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5 → build `submissions/shop_router_v2_standalone/`**
(holdout อ่านครั้งเดียว table ไม่ถูกแตะหลังอ่าน)

| gate | เกณฑ์ | วัดได้ | ผล |
|---|---|---|---|
| G1 | v97 − live ≥ +0.02, CI ไม่คร่อม 0 | **+0.1365** [+0.1256, +0.1476] | PASS |
| G2 | v97 − two (port V96) CI lower > −0.01 | **+0.0263** [+0.0216, +0.0313] | PASS |
| G3 | ทุก opponent: v97 ≥ live − 0.02 และ ≥ two − 0.02 | แย่สุด −0.0005 (vs two, 2 opponent) | PASS |
| G4 | route ที่เล่น == table lookup | 18,000 เกม 0 violation | PASS |
| G5 | vs boatlee V21 (400 seeds × 2 seats, official-env semantics) | v97 0.969, two 0.970, live 0.881 | PASS |

## Holdout: point rate ต่อ opponent (1,000 seeds × 2 seats ต่อช่อง)

| opponent | base | live | two | **v97** | v97 − live | v97 − two |
|---|---:|---:|---:|---:|---:|---:|
| a_poor_vul ep102980038 (adaptive) | 0.039 | 0.169 | 0.313 | 0.312 | +0.144 | −0.001 |
| kenjo ep102165685 | 0.648 | 0.575 | 0.859 | **0.860** | +0.285 | +0.001 |
| kenjo ep102449287 | 0.700 | 0.838 | 0.801 | **0.867** | +0.030 | +0.066 |
| milan ep102297457 | 0.709 | 0.836 | 0.793 | **0.863** | +0.027 | +0.070 |
| mtn ep102598799 (adaptive) | 0.926 | 0.713 | 0.897 | 0.906 | +0.193 | +0.009 |
| one_for_all ep102721568 | 0.783 | 0.665 | 0.822 | 0.822 | +0.157 | −0.001 |
| subramanya ep102288474 | 0.755 | 0.861 | 0.900 | **0.925** | +0.064 | +0.025 |
| ueddy ep103145299 | 0.516 | 0.502 | 0.752 | **0.760** | +0.258 | +0.008 |
| yusuke ep102775250 (lineage-mate) | 0.671 | 0.764 | 0.777 | **0.837** | +0.073 | +0.061 |
| boatlee V21 (adaptive, G5) | – | 0.881 | 0.970 | 0.969 | +0.088 | −0.001 |
| **pooled (9 tapes)** | 0.638 | 0.658 | 0.768 | **0.794** | +0.137 | +0.026 |

Mean margin: base +5,331 / live +6,373 / two +10,788 / **v97 +11,792**

สอง opponent ที่ทำให้ V96 ตก G3 (kenjo ep102449287, milan) ตอนนี้ v97 ชนะ live ทั้งคู่ (+0.03, +0.027)

## การ fit (train 1,500 seeds, in-sample)

- screen (500 seeds): เก็บ 45 จาก 142 donors — top 24 family 0 (yukino/ringbearer ครองอันดับต้น
  0.70) + family 1 ทั้ง 8 + donor ของ Yusuke ทั้งหมด
- **12/64 cells เปลี่ยน**, donor ที่ใช้จริง 13 ตัว: routes 0-9 ของ Yusuke, **ep102574290** (tape seat 0
  ของ 56016840) ยึด 7/8 cell ของ YARN_STORE + PET_CAFE__PET_CAFE, ringbearer ep103201048 1 cell,
  yukino ep103199121 1 cell, ที่เหลือเป็น route ของ Yusuke สลับกันเอง 2 cell
- `first_routes` ทุก shop = family 0: กิ่ง YARN_STORE ของ Yusuke (route 1) ถูกแทนด้วย
  ep102574290 ภายใต้ prefix เดียวกัน — ตรงกับที่ V96 พบว่า YARN_STORE__* คือ cell ที่ port แพ้ live
- in-sample pooled 0.7608 → 0.7914 (+0.031); holdout v97 − two = +0.026 → winner's curse เล็ก
- บน holdout: cell ที่เปลี่ยน two 0.786 → v97 **0.931** (n=3,269); cell ที่ไม่เปลี่ยนเท่ากันเป๊ะ 0.7641
  — gain ทั้งหมดมาจาก selection ไม่ใช่ noise

## Bundle

`scripts/build_shop_router_v1_submission.py --routes artifacts/shop_router_v2/routes.json --name shop_router_v2_standalone`
→ `submissions/shop_router_v2_standalone/` main.py 151 KB (sha f98ae37d6c246ca5), submission.tar.gz 119 KB
(sha 8300f4455afd8ec5), stdlib-only, 960 equivalence checks กับ `ShopRouterV1`, smoke บน observation จริงจาก
replay live ผ่าน, และ **เล่นเต็มเกมใน official `kaggle_environments` ได้เงินเท่ากับ Rust loop เป๊ะทั้งสอง seat**
(seed 16400001 seat 0: [94,947, 69,984]; seed 16400002 seat 1: [124,659, 139,830])

## ข้อจำกัดที่ต้องรู้ก่อน upload

1. ผลทั้งหมดเป็น local vs tape ของ top-20 + V21; local ไม่เคยทำนาย rating ได้ตรง
   (memory `kaggriculture-local-eval-does-not-predict-rating`) ตัวเปรียบเทียบที่ยุติธรรมที่สุดคือ
   `live` ซึ่งอยู่บน leaderboard จริงตอนนี้ (1338 หลัง 19 เกม) — v97 ชนะมัน +0.137 pooled และไม่แพ้มันต่อ
   opponent ใดเลย
2. A Poor Vul (adaptive) ยังชนะทุก arm ของเรา ≥ 0.69 — lineage ShopForge ทั้งตระกูลไม่มีคำตอบให้ agent แบบนี้
3. donor 3 ตัวที่ถูกเลือกมาจาก public replays ของทีมอื่น (Yusuke, ringbearer, yukino) — ตรวจ licence/กติกา
   ก่อนส่ง (`artifacts/shop_router_v2/PROVENANCE.md`)
4. Kaggle ให้ active 2 submission: ตอนนี้คือ V93c (1603, ตกต่อเนื่อง) และ 56016840 (1338) ถ้า upload
   ตัวนี้ต้องเลือกว่าจะแทนตัวไหน
