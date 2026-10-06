# V102 — v9/4-style t0 opening override บน omar base tape (screen)

Pre-registered 2026-09-22 **ก่อน** รัน eval ใดๆ (smoke/equivalence check ใช้แค่
agent-call ตรงๆ ไม่มีการเล่นเกมวัดผล) Live ตอนนี้: slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162, V101 ผ่านทุก gate)

## Hypothesis

ที่ step 0, market `[["BUY_SEED","WHEAT",20],["SELL","WHEAT",15]]` (v9/4
V9_OPENING_STEP0, source notebook อ้าง +107/-0 จาก 6648 openings) ดีกว่า market
ของ omar base tape `[["BUY_PRODUCT","WHEAT",20],["SELL","WHEAT",15]]`
สมมติฐาน: BUY_SEED ซื้อ seed ตรงสำหรับปลูก ส่วน BUY_PRODUCT ซื้อ wheat product
กลับซึ่งไม่ใช่ seed — order แรกของ base จึงเป็น no-op ที่เสียเปล่า

Assumption (เลือกแบบเล็ก/ปลอดภัย): ตีความ "+107/-0" เป็น context ไม่ใช่ gate;
gate ใช้แค่ทิศทาง (> 0) เพราะ local screen เคย fail to transfer หลายครั้ง

## Candidate / control (freeze)

| arm | routes file | sha256[:16] |
|---|---|---|
| base (control) | `artifacts/shop_router_v4/base_tape_routes.json` | fc0b8f040a56ae3c |
| opening (candidate) | `artifacts/shop_router_v4/base_tape_routes_opening.json` | 8ad87b5b28801df4 |

Candidate = base ที่ step-0 market ของทุก route ถูกแทนด้วย V9_OPENING_STEP0 ผ่าน
`opening_override.apply_to_routes_data` (overlay Python ใน
`src/kaggriculture_rl/policy/opening_override.py` verify แล้วว่าให้ action ตรงกับ
routes file ที่ engine observations จริงทั้งสอง seat) ต่างกันแค่ step 0;
step 1+ byte-identical ทั้งสอง seats ใช้ router depth 2 ทั้งคู่

## Opponents (6 tapes, fixed; คนละทีม, ไม่รวม omar lineage; copy ไว้ที่ `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Pool source: `/tmp/tapepool/tape_pool_20260919/tapes/` (== `artifacts/tape_pool_20260919.tar.gz`)

## Seeds (ใหม่ ไม่ทับ V97/V101: 161-163x, 181-183x)

Screen: **18400001–18400200** (200 seeds × 2 seats × 6 opp × 2 arms = 4800 เกม)
Engine config `{"episodeSteps":720,"seed":<seed>,"weedSpawnChance":0.005}` ทุกเกม
ผ่าน `scripts/evaluate_shop_router_fast.py` (`play_batch`) ทั้งสอง seat

## หน่วยวิเคราะห์

หน่วย = (opponent, seed, seat) แบบ paired: diff = margin(opening) − margin(base)
ในเกมเดียวกัน สถิติหลัก = mean paired diff + 95% percentile bootstrap CI
(10,000 resamples, cluster ตาม seed)

## Gates — freeze ห้ามแก้หลังเห็นผล

| gate | เกณฑ์ |
|---|---|
| **G1** | mean paired margin diff (opening − base) > 0 **และ** 95% CI lower bound > 0 |
| **G2** | point(opening) − point(base) ≥ **−0.01** (no regression ภาพรวม) |
| **G3** | ทุก opponent แยกกัน: mean paired margin diff ≥ **−200** (no collapse รายตัว) |

**Decision rule:** build standalone bundle ใหม่ (`submissions/omar_opening_standalone/`
via `scripts/build_shop_router_v1_submission.py --routes <patched>` ไม่ hand-edit)
ก็ต่อเมื่อ **G1 ∧ G2 ∧ G3** ผ่านทั้งหมด ไม่ผ่านข้อใด = OVERALL FAIL ไม่ build
ไม่มี holdout tuning — screen seeds นี้ไม่ใช้ tune เด็ดขาด

## บันทึกเสมอ

- `screen.json` (4800 rows), `gates.json`, `REPORT.md` รวมผลลบถ้ามี
- point ต่อ opponent ทั้ง 2 arms + mean/median margin
- seat symmetry ของ diff (seat 0 vs seat 1)

## คำสั่งรัน

```
uv run python scripts/evaluate_shop_router_fast.py \
  --routes artifacts/shop_router_v4/base_tape_routes.json \
  --router opening=artifacts/shop_router_v4/base_tape_routes_opening.json \
  --arms base opening \
  --opponents analysis/v102_opening_20260922/opponents/*.json \
  --seed-start 18400001 --seed-count 200 \
  --output analysis/v102_opening_20260922/screen.json
```
