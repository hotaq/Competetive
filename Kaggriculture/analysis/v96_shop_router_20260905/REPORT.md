# V96 — Yusuke two-shop router (port ตรง) ชนะ tape ที่ live อยู่ แต่ไม่ผ่าน no-regression

วันที่ 2026-09-05 Gates freeze ใน `PREREGISTRATION.md` (Amendment 1 แก้แค่ ROOT lookup,
Amendment 2 บันทึกก่อนอ่าน gate ว่า instrument ไม่ตรง oracle และจะอ่านซ้ำแบบ faithful เคียงกัน)
Scripts: `scripts/evaluate_shop_router_policy.py` (harness ที่ freeze), `parity_fastplay.py`
(อ่านซ้ำผ่าน Rust loop), `analyze_gates.py` Artifacts: `games.json` (38,400 เกม),
`games_fast_tapes.json` (33,600 เกม tape opponents, oracle-faithful), `gates.json`,
`gates_fast_tapes.json`, `parity_fastplay.json`

## Verdict

**OVERALL FAIL ตาม decision rule (G1 ∧ G3 ∧ G4) — ไม่ build/ไม่ upload** ทั้งสองการอ่านให้ผลเดียวกัน

| gate | เกณฑ์ | harness ที่ freeze (8 opp) | oracle-faithful (7 tape opp) | ผล |
|---|---|---|---|---|
| G1 | two − live ≥ +0.02, CI ไม่คร่อม 0 | **+0.0478** [+0.037, +0.058] | +0.0351 [+0.024, +0.046] | **PASS** |
| G2 | two − base > 0, CI ไม่คร่อม 0 | **+0.0899** [+0.074, +0.106] | +0.0906 [+0.074, +0.107] | **PASS** |
| G3 | ทุก opponent: two ≥ live − 0.02 | kenjo ep102449287 **−0.041**, milan ep102297457 **−0.033** | เท่ากันเป๊ะ | **FAIL** |
| G4 | route ที่เล่นจริง == table lookup ทุกเกม | 28,800 เกม 0 violation | 25,200 เกม 0 violation | **PASS** |

การอ่านซ้ำแบบ faithful สำหรับ boatlee V21 (`--no-sanitize`) ถูกผู้ใช้หยุดกลางคันที่ 490/4,800 เกม
จึงรายงานเฉพาะ tape opponents; ตัวเลขบน tape ทั้ง 7 ตัวต่างจากการอ่านที่ freeze ไม่เกิน 0.02 และ
verdict ทุก gate เหมือนกัน

## ตาราง point rate ต่อ opponent (harness ที่ freeze, 1,200 เกม/ช่อง)

| opponent | base | live | one | two | two − live |
|---|---:|---:|---:|---:|---:|
| boatlee V21 (adaptive) | 0.881 | 0.852 | 0.942 | **0.964** | +0.113 |
| kenjo ep102165685 | 0.623 | 0.558 | 0.743 | **0.841** | +0.283 |
| kenjo ep102449287 | 0.704 | **0.838** | 0.748 | 0.798 | **−0.041** |
| milan ep102297457 | 0.719 | **0.835** | 0.759 | 0.802 | **−0.033** |
| subramanya ep102288474 | 0.777 | 0.875 | 0.833 | **0.911** | +0.036 |
| yusuke ep102775250 (lineage-mate) | 0.672 | 0.754 | 0.728 | **0.779** | +0.025 |
| crop_dusta Sept ×2 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| **pooled** | 0.797 | 0.839 | 0.844 | **0.887** | +0.048 |

Mean margin: base +37,644 / live +38,863 / one +39,867 / two +42,652; two − live +3,789
[+3,353, +4,241] — router ชนะทั้งบ่อยกว่าและมากกว่า

## สิ่งที่ยืนยันได้

1. **Routing ทำงานจริงบน opponent/seed ที่ Yusuke ไม่เคยเห็น** two − base = +0.09 pooled และ
   **+0.125 [+0.090, +0.160]** เฉพาะเกมที่ตกใน specialist cell (โดนครบ 23/23 cell, 1,645 units)
   ชั้นแรก (`one`) ให้ +0.047 ชั้นสอง (`two`) เพิ่มอีก +0.043 ตรงกับที่ notebook รายงาน (0.678 → 0.814)
2. **Router ชนะ agent ที่ adapt ได้** boatlee V21: 0.964 vs live 0.852 — เกมที่แพ้บน leaderboard
   จริงแพ้ agent ประเภทนี้ทั้งหมด (ดู memory `kaggriculture-56016840-yusuke-tape-diagnosis`)
3. **G3 ไม่ผ่านเพราะ tape seat 0 ของ live (ep102574290) แข็งผิดคาด** per-seat บน faithful read:

   | opponent | live seat0 (ep102574290) | live seat1 (base) | two |
   |---|---:|---:|---:|
   | kenjo ep102449287 | **0.973** | 0.703 | 0.798 |
   | milan ep102297457 | **0.952** | 0.718 | 0.802 |
   | subramanya | **0.975** | 0.780 | 0.913 |
   | yusuke lineage-mate | **0.877** | 0.673 | 0.779 |
   | kenjo ep102165685 | 0.488 | 0.625 | **0.838** |

   tape ที่เคยวินิจฉัยว่าเป็น "minority branch" กลับเป็น tape เดี่ยวที่แรงที่สุดต่อ 4 ใน 5 opponent
   แต่พังกับ kenjo ep102165685 — นี่คือ "styles make fights" แบบเดียวกับที่ phased_router_v3 เจอ
   cell ที่ `two` แพ้ live-s0 มากสุดคือ **YARN_STORE__*** ทุกคู่ (route 1 ของ Yusuke) และ
   BAKERY__PET_CAFE (route 3): กิ่ง YARN_STORE ของ Yusuke ไม่ดีกับ opponent ชุดนี้
4. Crop Dusta Sept tapes (2 ตัว) แพ้ 100% ทุก arm → ไม่ให้ข้อมูล (tape ของ agent ที่ adapt สูง
   เล่นบน seed อื่นแล้วพัง) ควรตัดออกจาก opponent set รอบหน้า

## ค้นพบเรื่อง instrument (Amendment 2)

`engine.sanitize_action` ที่ harness V94/V95/V96 ใช้กับทุก action **เข้มกว่า official env**: ตัด
market order ทั้งใบเมื่อของ/เงินไม่พอ ขณะที่ `kaggle_environments` 1.32.7 ขายทีละหน่วยจนหมด
ตรวจตรงกับ official env บนเกม live-vs-subramanya seed 15300001: env และ Rust loop ให้
[67,751, 40,755] เท่ากันเป๊ะ harness เดิมให้ 74,097 ผลต่อ V96: 11,647/33,600 เกมมีเงินต่าง
แต่ point พลิกแค่ **35 เกม** — gate ไม่เปลี่ยน แต่ตัวเลข V94/V95 เดิมควรอ่านโดยรู้ bias นี้
(memory `kaggriculture-sanitize-harness-not-env-faithful`)

## Deliverable ที่ได้แม้ verdict จะ FAIL

- `src/kaggriculture_rl/policy/shop_router_v1.py` + `artifacts/shop_router_v1/` (routes + PROVENANCE)
- `scripts/build_shop_router_v1_submission.py` (tests ผ่าน; **ยังไม่ build ลง submissions/** ตาม rule)
- **Rust game loop** `kaggriculture_rs.play_batch` (`rust/kaggriculture-python/src/fastplay.rs`):
  677 เกม/วินาที/thread, 3,552 เกม/วินาทีบน 10 threads (เทียบ harness Python 2.3 เกม/วินาที/core)
  ตรงกับ official env ทุกเหรียญ (`bindings/python/tests/test_fastplay.py`) และ engine ได้
  `Engine::step_refs` ที่ไม่ clone action
- `scripts/evaluate_shop_router_fast.py` harness เร็วสำหรับ tape opponents

## ขั้นถัดไป (ต้อง pre-register ใหม่)

ทำ **selection แบบ Yusuke บน donor pool ของเรา** ด้วย fast loop: donors = 10 routes ของ Yusuke
+ ep102574290 + branch อื่นใน `rank003_yusuke_hayashi` (prefix ร่วมถึง step 144 ทั้งหมด) เลือก
route ต่อ cell ด้วย train seeds และรับเฉพาะที่ไม่ regress ต่อ opponent ใดๆ (Yusuke ข้อ 4) แล้ว
วัด holdout — งบ 64 cell × ~12 donor × 6 opponent × 600 seed × 2 seat ≈ 5.5M เกม ≈ 26 นาที
บน 10 threads; ตัด Crop Dusta Sept tapes ออก ใส่ boatlee V21 ผ่าน harness `--no-sanitize`
