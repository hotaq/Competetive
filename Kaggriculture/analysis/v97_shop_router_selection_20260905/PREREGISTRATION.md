# V97 — เลือก route ต่อ cell แบบ Yusuke บน donor pool ของเรา (screen → fit → holdout)

Pre-registered 2026-09-05 **ก่อน** รัน seed ใดๆ ในช่วงด้านล่าง (smoke test ใช้ seed 1-6 และ 100-105
กับ opponent เดียว เพื่อเช็คว่า pipeline รันได้เท่านั้น) Commit ฐาน `d9adff6` + untracked files ตาม hash

## ที่มา

V96 (`analysis/v96_shop_router_20260905/REPORT.md`): port router ของ Yusuke ตรงๆ ชนะ tape ที่ live อยู่
+0.048 และชนะ base +0.090 แต่ **ไม่ผ่าน no-regression** ต่อ kenjo ep102449287 (−0.041) และ milan
ep102297457 (−0.033) เพราะ tape seat 0 ของ live (ep102574290) เป็น suffix ที่แรงกว่าในหลาย cell
โดยเฉพาะ YARN_STORE__* (route 1 ของ Yusuke) ep102574290 พิสูจน์แล้วว่าใช้ prefix เดียวกับ route 0
(state ที่ step 144 เหมือนกันทุก seed, splice = whole) จึงเป็น donor ที่ถูกต้อง

V97 ทำสิ่งที่ Yusuke ทำแต่บน donor pool ของเรา: เลือก route ต่อ cell จาก donor ที่ share prefix
รับเฉพาะที่ **ดีขึ้นและไม่แย่ลงต่อ opponent ใดๆ** (Yusuke ข้อ 4) fit บน train seeds วัดบน holdout seeds

## Donors (freeze: `artifacts/shop_router_v2/donors.json` sha 7992e59187083be7)

`scripts/build_shop_router_donors.py` สแกน 1,441 tapes (pool 20260830 + harvest 20260829 + v1 routes)
พบ 154 on-prefix (canonical เท่ากับ route 0 ถึง step 144 หรือ route 1 family) → **142 distinct donors**
(family 0: 134, family 1: 8) จาก ringbearer 58, yukino 55, Yusuke 10+5, Boredom 5, Umataro Tenma 5,
Hai Dang 3, Milan 1 — donor 0-9 คือ route 0-9 ของ v1 ตามลำดับ (incumbent)

## Opponents (9 tapes, fixed; ตัด Crop Dusta Sept ที่แพ้ 100% ทุก arm ใน V96 ออก)

| opponent | sha256[:16] |
|---|---|
| a_poor_vul__ep102980038__seat1 | bee71df46cb83c0d |
| kenjo1209__ep102165685__seat1 | 257446490a6b4bc2 |
| kenjo1209__ep102449287__seat1 | cc738009b3109b3e |
| milan_leonard__ep102297457__seat1 | 4e2ac38979f12288 |
| mtn__ep102598799__seat1 | 9b9cb8dad327a051 |
| one_for_all__ep102721568__seat0 | 7d54cbc4f92e7798 |
| subramanya_n__ep102288474__seat1 | db8543b075510a58 |
| ueddy__ep103145299__seat1 | 3698c0f2d63f7f63 |
| yusuke_hayashi__ep102775250__seat0 | fe1ff2d496e6611f |

5 ตัวแรกคือชุด V96 (ไม่รวม Crop Dusta) + 4 ตัวใหม่จากทีม top-20 ต่าง lineage (prefix identity กับ
route 0 ≤ 0.31): A Poor Vul (adaptive), Ueddy (open-loop, margin +81k), One-For-All (open-loop),
MtN (adaptive) boatlee V21 (closed-loop Python) ใช้เฉพาะ holdout G5 ผ่าน harness ช้า `--no-sanitize`

## Seeds — ช่วงใหม่ ไม่ทับ V94/V95/V96 (131-133x, 141-143x, 151x)

| stage | seeds | ใช้ทำอะไร |
|---|---|---|
| screen | 16100001-16100500 | จัดอันดับ 142 donors ตาม pooled point; เก็บ top 24 family 0 + family 1 ทั้งหมด + ทุก donor ชื่อขึ้นต้น `yusuke_` |
| train | 16200001-16201500 | fit table: min_cell_games 30, min_opponent_games 8, tolerance 0.05, min_gain 0.01 |
| holdout | 16300001-16301000 | **อ่าน gate ครั้งเดียว** arms: v97, two (v1 table), live, base |
| holdout-V21 | 16300001-16300400 | G5: v97 / two / live vs boatlee V21, `--no-sanitize`, ทั้งสอง seat |

Engine config `{"episodeSteps":720,"seed":<seed>,"weedSpawnChance":0.005}` ทุก stage tape opponents
เล่นผ่าน `kaggriculture_rs.play_batch` (ตรง official env, ดู `bindings/python/tests/test_fastplay.py`)

## หน่วยวิเคราะห์และ CI

เหมือน V96: หน่วย = (opponent, seed) เฉลี่ย 2 seats, percentile bootstrap 10,000 รอบ cluster ตาม seed

## Gates — freeze ห้ามแก้หลังเห็นผล

| gate | เกณฑ์ |
|---|---|
| **G1** | holdout: point(v97) − point(live) ≥ **+0.02** และ 95% CI ไม่คร่อม 0 |
| **G2** | holdout: point(v97) − point(two) CI lower bound **> −0.01** (การ fit ไม่ทำให้แย่กว่า port) |
| **G3** | holdout ทุก opponent แยกกัน: point(v97) ≥ point(live) − 0.02 **และ** ≥ point(two) − 0.02 |
| **G4** | ทุกเกม v97: route ที่เล่นจริง == table lookup ของ (shop1, shop2) |
| **G5** | vs boatlee V21 (400 seeds × 2 seats): point(v97) ≥ point(live) − 0.02 และ ≥ point(two) − 0.02 |

**Decision rule:** build standalone bundle เพื่อ upload ก็ต่อเมื่อ **G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5** ผ่านทั้งหมด
ไม่ผ่านข้อใด = OVERALL FAIL ไม่ upload table ที่ fit แล้วห้ามแก้หลังอ่าน holdout

## บันทึกเสมอ

- screen ranking ทั้ง 142 (`screen.json`), cell ที่เปลี่ยนจาก incumbent + donor ที่ใช้ (`fit.json`)
- in-sample pooled point (train) ของ v97 vs incumbent เทียบกับ holdout — ขนาดของ winner's curse
- ตาราง point ต่อ opponent ทั้ง 4 arms และ mean margin; ผลใน specialist cells; seat symmetry

## Frozen inputs (sha256[:16])

| file | sha |
|---|---|
| artifacts/shop_router_v2/donors.json | 7992e59187083be7 |
| artifacts/shop_router_v1/yusuke_routes.json (incumbent) | ede06952623ff52d |
| scripts/search_shop_router_cells.py | 264e619d6b394e00 |
| scripts/build_shop_router_donors.py | 366e7733aa0d8c6c |
| scripts/evaluate_shop_router_fast.py | 498bf6ae8172b4ad |
| scripts/evaluate_shop_router_policy.py | de40690557861687 |
| rust/kaggriculture-python/src/fastplay.rs | 7146a33c1be61324 |
| artifacts/yusuke_schedule_s0.jsonl / s1 (live) | 3226cc39970856c0 / dd5823aba61321b9 |
| artifacts/recovered_boatlee_v21_source/main.py | c6f96a8521dc9aa3 |

## คำสั่งรัน

```
A=analysis/v97_shop_router_selection_20260905
uv run python scripts/search_shop_router_cells.py --opponents $A/opponents/*.json \
  --screen-seed-start 16100001 --screen-seed-count 500 --screen-keep 24 \
  --seed-start 16200001 --seed-count 1500 \
  --output-routes artifacts/shop_router_v2/routes.json --output-fit $A/fit.json \
  --output-games $A/train.json --output-screen $A/screen.json
uv run python scripts/evaluate_shop_router_fast.py --arms live base \
  --router v97=artifacts/shop_router_v2/routes.json --router two=artifacts/shop_router_v1/yusuke_routes.json \
  --opponents $A/opponents/*.json --seed-start 16300001 --seed-count 1000 --output $A/holdout.json
uv run python scripts/evaluate_shop_router_policy.py --arms two live --routes artifacts/shop_router_v1/yusuke_routes.json ... (v97 ผ่าน --routes artifacts/shop_router_v2/routes.json) \
  --agent-opponents artifacts/recovered_boatlee_v21_source/main.py --seed-start 16300001 --seed-count 400 --no-sanitize
```
