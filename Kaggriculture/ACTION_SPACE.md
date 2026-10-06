# Kaggriculture Action Space

เอกสารนี้สรุป action space ของ Kaggriculture จาก environment เวอร์ชัน `0.1.0`
ที่ติดตั้งอยู่ในโปรเจกต์ โดยใช้สำหรับออกแบบ state-aware planner, action masking
และ macro RL policy ต่อไป

## รูปแบบ action ต่อ turn

Agent ต้องคืนค่า dictionary รูปแบบนี้:

```python
{
    "farmer": [op, ...args],
    "hands": [[op, ...args], ...],
    "market": [[op, ...args], ...],
}
```

- `farmer`: หนึ่ง action สำหรับ farmer หลัก
- `hands`: หนึ่ง action ต่อ farm hand ตามลำดับใน observation
- `market`: ordered market actions สูงสุด `maxMarketOrdersPerTurn` รายการ
  (ค่าเริ่มต้น 10)
- เกมมาตรฐานมี 24 turns ต่อวัน, 30 วัน รวม 720 turns
- Action ที่ malformed หรือผิดเงื่อนไขจะเป็น silent no-op

## Unit actions

Interpreter รองรับ unit-operation ทั้งหมด 18 แบบ

### Movement

| Action | Arguments | ผลลัพธ์และเงื่อนไข |
|---|---|---|
| `NORTH` | ไม่มี | เดินขึ้นหนึ่งช่อง |
| `SOUTH` | ไม่มี | เดินลงหนึ่งช่อง |
| `EAST` | ไม่มี | เดินขวาหนึ่งช่อง |
| `WEST` | ไม่มี | เดินซ้ายหนึ่งช่อง |
| `PASS` | ไม่มี | ไม่ทำอะไร |

- เดินเข้าและเดินผ่าน locked tiles ได้
- การเดินออกนอกขอบแผนที่จะเป็น no-op
- Tile actions ส่วนใหญ่ใช้บน locked tile ไม่ได้

### Shed and inventory

| Action | Arguments | ผลลัพธ์และเงื่อนไข |
|---|---|---|
| `PICKUP` | `<item> [n]` | หยิบของจาก shed เข้า inventory; `n` มีค่าเริ่มต้น 1 |
| `DROP` | ไม่มี | เท inventory ทั้งหมดเข้า shed |
| `PLACE` | `<item> [n]` | วางสัตว์บน structure หรือฝากของจาก inventory เข้า shed |

- `PICKUP`, `DROP` และการฝากของด้วย `PLACE` ต้องยืนบนหนึ่งในสี่ช่องที่ติด shed
- Seeds อยู่ใน `private["seeds"]`; ไม่สามารถหยิบด้วย `PICKUP`
- `DROP` ทิ้งของส่วนที่เกิน `shedCapacity`
- `PLACE` เข้า shed ฝากเท่าที่มีพื้นที่ว่าง ส่วนเกินยังอยู่ใน inventory
- เมื่อใช้ `PLACE` วางสัตว์ ค่า `n` จะถูกละเว้นและวางครั้งละหนึ่งตัว

### Plants and terrain

| Action | Arguments | ผลลัพธ์และเงื่อนไข |
|---|---|---|
| `PLANT` | `<crop>` | ใช้ seed หนึ่งหน่วยและปลูกบนช่องว่างที่ปลดล็อกแล้ว |
| `WATER` | ไม่มี | รดต้นไม้บนช่องปัจจุบัน; ทำได้อย่างมีผลวันละครั้ง |
| `HARVEST` | ไม่มี | เก็บผลผลิตจากพืชหรือสัตว์บนช่องปัจจุบัน |
| `FERTILIZE` | ไม่มี | ใช้ fertilizer หนึ่งหน่วยกับพืช; มีผลวันปัจจุบันและอีกสองวัน |
| `DIG` | ไม่มี | ลบ plant, weed หรือ coop/pasture ที่ยังไม่มีสัตว์ |

Crop arguments ที่ถูกต้อง:

```text
WHEAT, CARROT, TOMATO, STRAWBERRY, MELON
```

กฎสำคัญของ `PLANT`:

- ต้องยืนบน empty unlocked tile และมี seed ชนิดนั้น
- ถ้าจำนวนคำสั่ง `PLANT` ของ crop เดียวกันจากทุก unit ใน turn นั้นมากกว่า seed
  ที่มีอยู่ คำสั่ง `PLANT` ของ crop ชนิดนั้นทั้งหมดจะถูกแทนด้วย `PASS`
- Seed ที่ซื้อจาก market ใน turn ปัจจุบันยังใช้ปลูกใน turn เดียวกันไม่ได้

### Animals and structures

| Action | Arguments | ผลลัพธ์และเงื่อนไข |
|---|---|---|
| `BUILD_COOP` | ไม่มี | สร้าง coop บน empty unlocked tile สำหรับ Goose |
| `BUILD_PASTURE` | ไม่มี | สร้าง pasture บน empty unlocked tile สำหรับ Cow หรือ Sheep |
| `PLACE` | `<animal> [n]` | วางสัตว์หนึ่งตัวบน structure ที่ชนิดตรงกันและยังว่าง |
| `FEED` | ไม่มี | ใช้ Wheat หนึ่งหน่วยจาก unit inventory เพื่อให้อาหารสัตว์ |
| `CARE` | ไม่มี | ดูแลสัตว์และสะสมโบนัสสำหรับรอบผลิตถัดไป |
| `COLLECT_FERTILIZER` | ไม่มี | เก็บ fertilizer หนึ่งหน่วยจากสัตว์เมื่อพร้อม |
| `HARVEST` | ไม่มี | เก็บ Egg, Milk หรือ Wool ที่สัตว์ผลิตไว้ |

Animal arguments ที่ถูกต้อง:

```text
GOOSE, COW, SHEEP
```

Structure compatibility:

| Animal | Structure | Product |
|---|---|---|
| `GOOSE` | `COOP` | `EGG` |
| `COW` | `PASTURE` | `MILK` |
| `SHEEP` | `PASTURE` | `WOOL` |

## Market actions

Market รองรับ operation ทั้งหมด 6 แบบ

| Action | Arguments | ผลลัพธ์และเงื่อนไข |
|---|---|---|
| `BUY_SEED` | `<crop> <n>` | ซื้อ seed จำนวน `n` |
| `BUY_PRODUCT` | `<item> <n>` | ซื้อ `WHEAT` หรือ `FERTILIZER` จำนวน `n` |
| `BUY_ANIMAL` | `<animal> <n>` | ซื้อสัตว์จำนวน `n` เข้า shed |
| `SELL` | `<item> <n>` | ขาย product จาก shed จำนวน `n` ตามราคาตลาดปัจจุบัน |
| `HIRE` | ไม่มี | จ้าง farm hand สำหรับวันปัจจุบัน |
| `BUY_LAND` | ไม่มี | ปลดล็อก quadrant ถัดไป |

ตัวอย่าง:

```python
[
    ["BUY_SEED", "WHEAT", 8],
    ["BUY_PRODUCT", "FERTILIZER", 2],
    ["BUY_ANIMAL", "COW", 1],
    ["SELL", "MILK", 6],
    ["HIRE"],
    ["BUY_LAND"],
]
```

Products ที่ขายได้:

```text
WHEAT, CARROT, TOMATO, STRAWBERRY, MELON,
EGG, MILK, WOOL, FERTILIZER
```

ข้อจำกัดสำคัญ:

- `BUY_PRODUCT` ซื้อกลับได้เฉพาะ `WHEAT` และ `FERTILIZER`
- Market orders ถูกประมวลผลตามลำดับใน list และจำกัดค่าเริ่มต้นที่ 10 รายการต่อ turn
- Orders ของผู้เล่นสองคนถูกประมวลผลแบบ per-unit lockstep กับตลาดร่วมกัน
- ราคาอาจเปลี่ยนหลังซื้อหรือขายแต่ละหน่วย
- `HIRE` มีราคารายวันตามลำดับ Fibonacci: 1, 1, 2, 3, 5, 8, 13, ...
- `BUY_LAND` ปลดล็อก `NE`, `SW`, `SE` ตามลำดับ ในราคา 1,000, 2,000 และ 4,000

## Execution order

ในแต่ละ turn environment ทำงานตามลำดับสำคัญดังนี้:

1. ตรวจ atomic `PLANT` requests เทียบกับ seed ที่มีอยู่ก่อนเริ่ม turn
2. ประมวลผล farmer แล้วตามด้วย farm hands
3. ประมวลผล market orders
4. ประมวลผล town consumption และกลไก decay/end-of-turn ที่เกี่ยวข้อง

ผลที่ตามมา:

- ของหรือ seed ที่ซื้อใน market ยังใช้โดย unit ใน turn เดียวกันไม่ได้
- Hand ที่จ้างใน market ยังไม่มี unit action ใน turn ที่เพิ่งจ้าง
- ลำดับ unit actions มีผลเมื่อหลาย unit แย่ง inventory, tile หรือ task เดียวกัน

## RL implications

Action space นี้ไม่ควร flatten เป็น discrete action เดียวโดยตรง เพราะจำนวน combination
เพิ่มขึ้นตามจำนวน hands, item arguments, quantities และ market-order sequences

แนวทางเริ่มต้นที่แนะนำ:

1. ใช้ legality/action mask ปิด action ที่เป็นไปไม่ได้จาก state ปัจจุบัน
2. ให้ deterministic executor จัดการ movement, inventory และ unit-level coordination
3. ให้ RL เลือก macro decisions เช่น crop mix, animal target, worker target,
   selling policy และ wheat reserve
4. เก็บ primitive-action policy ไว้เป็นงานระยะถัดไปหลัง macro policy เสถียรแล้ว

## Source of truth

- `.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/README.md`
- `.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`
- `.venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.json`

หมายเหตุ: `DROP` ถูก implement และระบุใน README แต่ตกหล่นจาก action-description text
ใน `kaggriculture.json`
