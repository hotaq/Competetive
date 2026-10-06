# V96 — two-shop tape router แบบ Yusuke (port ตรง ไม่ refit) ชนะตัวที่ live อยู่ไหม?

Pre-registered 2026-09-05 **ก่อน** รัน evaluation seed ใดๆ ในช่วงที่ระบุด้านล่าง
(smoke test ใช้ seed 1-2 กับ 2 opponents เท่านั้น เพื่อเช็คว่า harness รันได้ ไม่ได้อ่านผล)
Commit ฐาน: `d9adff6` + untracked files ที่ freeze ด้วย hash ด้านล่าง

## ที่มา

Submission 56016840 (live ตั้งแต่ 2026-09-04) เล่น tape ของ Yusuke Hayashi 2 ม้วนแบบ open-loop
ไม่สน `town.unlocked_shops` ผล 19 เกมแรก: 14/16 vs rating < 1500, 1/3 vs > 1500 และทุกเกมที่แพ้
มีรูปเดียวกันคือนำที่ day 15 แล้วโดนแซง day 21-24 (ดู memory `kaggriculture-56016840-yusuke-tape-diagnosis`)

Yusuke เผยแพร่ router ตัวเต็ม (notebook `yhay81/public-match-history-router-rating-2929-aug-30`):
10 routes ที่ใช้ prefix ร่วมกัน แตกกิ่งที่ step 72 (shop แรก) และ 144 (คู่ shop) รายงาน point rate
base-alone 0.678 → two-shop 0.814 บน 4,096 seeds × 3 opponents × 2 seats V94/V95 ของเราทดสอบ
*วิธี fit table เอง* แล้ว FAIL ทั้งคู่ (V95: +0.0044 และ overfit opponent ที่ใช้ fit) — รอบนี้จึง
**ไม่ fit อะไรเลย** แค่ port router ของ Yusuke ตรงๆ แล้ววัดบน opponent set และ seeds ของเราที่
Yusuke ไม่เคยเห็น เป็น out-of-sample test ที่สะอาด

## สมมติฐาน 2 ข้อ แยก gate กัน

- **H1 (upload decision):** arm `two` ชนะ arm `live` (tape ที่อยู่บน leaderboard ตอนนี้)
- **H2 (routing hypothesis):** arm `two` ชนะ arm `base` (route 0 เล่นเดี่ยว) — ตรวจว่า
  gain มาจาก routing จริง ไม่ใช่แค่ base tape ดีกว่า

## Arms (closed-loop ทุกตัว ขับด้วย `scripts/evaluate_shop_router_policy.py`)

| arm | policy | หมายเหตุ |
|---|---|---|
| `base` | `ShopRouterV1(max_shops=0)` | route 0 ตลอดเกม |
| `one` | `ShopRouterV1(max_shops=1)` | แตกกิ่งที่ shop แรกเท่านั้น (Yusuke: มีแค่ YARN_STORE ที่แตก) |
| `two` | `ShopRouterV1(max_shops=2)` | **candidate** |
| `live` | `DustaSchedule(YUSUKE_SCHEDULES)` | tape ใน 56016840: seat 0 = ep102574290, seat 1 = ep102589974 |

## Opponents (8 ตัว fixed) — prefix identity กับ route 0 ใน 144 step แรกใส่ไว้เพื่อบอก lineage

| opponent | ชนิด | lineage vs route 0 | sha256[:16] |
|---|---|---:|---|
| kenjo1209__ep102449287__seat1 | tape (V95 opponent) | 0.160 | cc738009b3109b3e |
| kenjo1209__ep102165685__seat1 | tape (margin สูงสุดของ Kenjo ใน pool) | 0.167 | 257446490a6b4bc2 |
| milan_leonard__ep102297457__seat1 | tape (V95 opponent) | 0.451 | 4e2ac38979f12288 |
| yusuke_hayashi__ep102775250__seat0 | tape (V95 opponent, **lineage-mate**) | 0.931 | fe1ff2d496e6611f |
| subramanya_n__ep102288474__seat1 | tape | 0.000 | db8543b075510a58 |
| crop_dusta__ep105391557__seat0 | tape จาก snapshot 2026-09-04 (rank 1 ปัจจุบัน) | 0.014 | 687b76ef80e97019 |
| crop_dusta__ep105366473__seat1 | tape จาก snapshot 2026-09-04 | 0.014 | 996b846726161180 |
| agent:recovered_boatlee_v21_source | **closed-loop file agent** (boatlee V21-R1, rank 12 @ 2026-08-31) | – | c6f96a8521dc9aa3 |

Tape opponents อยู่ใน `opponents/` ของ directory นี้ (pool format: `actions[k+1]` = action ที่ step k)
boatlee V21 เป็น opponent ตัวเดียวที่ adapt ได้ ใส่ไว้เพราะเกมที่แพ้ live ทั้งหมดแพ้ agent ที่ adapt

## Seeds — ช่วงใหม่ ไม่เคยใช้ (V94 ใช้ 131/132/133xxxxx, V95 ใช้ 141/142/143xxxxx)

**15100001-15100600** (600 seeds) × 2 seats × 8 opponents × 4 arms = **38,400 เกม**
ประมาณ 30 นาทีบน 8 workers (tape 0.23 s/เกม, V21 1.27 s/เกม)

ไม่มี train/holdout split เพราะไม่มีการ fit อะไร ทั้ง 4 arms ได้ข้อมูลเท่ากันและถูกอ่านครั้งเดียว

## หน่วยวิเคราะห์และ CI

เกม tape-vs-tape สมมาตรตาม seat (smoke: seat 0 และ 1 ให้ margin เท่ากันทุกคู่) ดังนั้น
**หน่วยอิสระคือ (opponent, seed)** โดยเฉลี่ย 2 seats ก่อน CI ทุกตัวเป็น percentile bootstrap
10,000 รอบ resample ตาม seed (cluster) บน paired difference ของ point ต่อ (opponent, seed)

## Gates — freeze แล้ว ห้ามแก้หลังเห็นผล

| gate | เกณฑ์ | ผูกกับ |
|---|---|---|
| **G1** | point(two) − point(live) pooled **≥ +0.02** และ 95% CI ไม่คร่อม 0 | H1 |
| **G2** | point(two) − point(base) pooled **> 0** และ 95% CI ไม่คร่อม 0 | H2 |
| **G3** | ทุก opponent แยกกัน: point(two) ≥ point(live) − 0.02 | H1 |
| **G4** | ทุกเกมของ `two`: `route_final` == table lookup ของ (shop1, shop2) ที่บันทึกตอน step 72/144 และ == `route_at_second`; ทุกเกมของ `base`: route 0; ทุกเกมของ `one`: `first_routes[shop1]` | implementation |

**Decision rule:** สร้าง standalone bundle เพื่อ upload ก็ต่อเมื่อ **G1 ∧ G3 ∧ G4** ผ่าน
G2 ไม่ block การ upload แต่ถ้า G2 ไม่ผ่านขณะ G1 ผ่าน REPORT ต้องระบุว่า gain มาจาก base tape
ไม่ใช่จาก routing และ arm `base` ต้องถูกรายงานเป็นทางเลือกที่ง่ายกว่า
ถ้า G1 หรือ G3 หรือ G4 ไม่ผ่าน = **OVERALL FAIL** ไม่ upload

## บันทึกเสมอไม่ว่าผลจะเป็นอย่างไร

- point rate / mean margin / median margin ต่อ arm และต่อ (arm, opponent)
- การกระจายของ shop pair ทั้ง 64 cell และจำนวนเกมใน 23 specialist cells ที่โดนจริง
- point rate ของ `two` vs `base` เฉพาะเกมที่ตกใน specialist cell (ที่ routing ทำอะไรได้จริง)
- seat symmetry check บน tape opponents และผลแยก seat บน V21
- เวลาต่อเกม

## Frozen inputs (sha256[:16])

| file | sha |
|---|---|
| artifacts/shop_router_v1/yusuke_routes.json | ede06952623ff52d |
| src/kaggriculture_rl/policy/shop_router_v1.py | 8c5085f2602df270 |
| scripts/evaluate_shop_router_policy.py | 560c9665417a9901 |
| artifacts/yusuke_schedule_s0.jsonl (live seat 0) | 3226cc39970856c0 |
| artifacts/yusuke_schedule_s1.jsonl (live seat 1) | dd5823aba61321b9 |
| artifacts/recovered_boatlee_v21_source/main.py | c6f96a8521dc9aa3 |

Engine config: `{"episodeSteps": 720, "seed": <seed>, "weedSpawnChance": 0.005}` เหมือน V94/V95
Action ทั้งสองฝั่งผ่าน `clean` → `normalize_legacy_action` → `engine.sanitize_action`
(canonicalisation เดียวกับ tape harness เดิม; env จริง default PICKUP/PLACE count = 1)

## คำสั่งรัน

```
uv run python scripts/evaluate_shop_router_policy.py \
  --arms base one two live \
  --opponents analysis/v96_shop_router_20260905/opponents/*.json \
  --agent-opponents artifacts/recovered_boatlee_v21_source/main.py \
  --seed-start 15100001 --seed-count 600 --workers 8 \
  --output analysis/v96_shop_router_20260905/games.json
```

## Amendment 1 — 2026-09-05 ก่อนรัน seed ใดๆ ในช่วง 15100001-15100600

แก้ `shop_router_v1.py` ให้ `ROOT` ไม่ raise IndexError เมื่อโมดูลถูก inline ใน standalone bundle
(`Path("<embedded-module:…>").resolve().parents[3]` ล้มบน cwd ตื้นๆ อย่าง /kaggle/working)
ไม่แตะ logic การ route หรือ action ใดๆ hash ใหม่ = 8c5085f2602df270 (ตารางด้านบนอัปเดตแล้ว) gates ไม่เปลี่ยน

## Amendment 2 — 2026-09-05 ระหว่าง run (arm two/live ยังไม่จบ) **ก่อนอ่าน gate ใดๆ**

พบว่า instrument ไม่ตรง oracle: `engine.sanitize_action` (สืบทอดจาก harness V94/V95) ตัด market
order ทั้งใบเมื่อของไม่พอ ขณะที่ official env (`kaggle_environments` 1.32.7) ขายทีละหน่วยจนหมด
ตรวจบนเกม live-vs-subramanya seed 15300001: official env และ Rust loop ให้ [67,751, 40,755]
เท่ากันเป๊ะ แต่ harness ที่ freeze ไว้ให้ 74,097 ฝั่งเรา

การตัดสินใจ (ก่อนเห็นผล arm two/live):
1. **Gate ทั้ง 4 อ่านบน harness ที่ freeze ไว้ตามสัญญาเดิม** (games.json) ไม่เปลี่ยนเกณฑ์
2. รายงาน **การอ่านซ้ำแบบ oracle-faithful** เคียงข้างกัน: tape opponents ผ่าน `play_batch` (Rust,
   `games_fast_tapes.json`) และ V21 ผ่าน harness เดิมด้วย `--no-sanitize` (`games_v21_nosanitize.json`)
   seeds/opponents/arms เดิมทั้งหมด ถ้าสองการอ่านให้ verdict ต่างกัน REPORT ต้องระบุและ **ไม่ upload**
3. harness เพิ่ม flag `--no-sanitize` (default เดิม) hash ใหม่ = de40690557861687

harness รอบที่กำลังรันโหลดโค้ดเดิมไว้ใน worker แล้ว การแก้ไฟล์ไม่กระทบผลของ run นี้
