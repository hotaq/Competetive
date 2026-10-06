# Rust Engine Migration Decision

สถานะ: core migration พร้อม shadow validation — 15 สิงหาคม 2026

## Problem Statement

Python/Kaggle environment เป็นคอขวดของการสร้าง rollout จำนวนมากสำหรับ RL และผูก simulation
เข้ากับ dynamic dictionaries ใน Python ขณะเดียวกันผลการแข่งขันขึ้นกับรายละเอียดเล็กมาก เช่น
ลำดับ market orders, silent no-op, inventory overflow และ seeded randomness การ rewrite แบบสลับใช้ทันที
จึงเสี่ยงทำให้ train กับ simulator ที่ไม่ตรงกับ Kaggle จริง

## Solution

ใช้ Rust เป็น state-transition core แบบ strongly typed และ deterministic แต่คง Python reference
เป็น behavioral oracle ระหว่าง migration ตัว Rust รองรับ typed in-process API, persistent JSON-lines
protocol และ PyO3 adapter เพื่อให้โค้ด RL เดิมค่อย ๆ ย้ายโดยไม่ต้องเปลี่ยน policy interface พร้อมกัน

หลักการ cutover คือ shadow comparison ก่อนเสมอ: action trace เดียวกันต้องให้ public state, private state,
market, town, day/hour และ terminal reward ตรงกันทุก turn จึงอนุญาตให้ training ใช้ Rust เป็นค่าเริ่มต้น

## Commits

ลำดับ commit ที่ปลอดภัยและแต่ละจุดยัง build/test ได้:

1. เพิ่ม Cargo workspace และ crate เปล่าที่มี smoke test
2. เพิ่ม configuration และ JSON-compatible domain model โดยยังไม่มี transition logic
3. เพิ่ม crop, animal, shop และ market rule tables พร้อม price-function tests
4. เพิ่ม CPython-compatible seeded RNG พร้อม known-vector tests
5. เพิ่ม initialization และ observation projection พร้อม initial-state parity test
6. เพิ่ม movement, shed และ inventory operations พร้อม silent-no-op tests
7. เพิ่ม crop/terrain operations และ atomic planting validation
8. เพิ่ม animal operations และ daily lifecycle
9. เพิ่ม simultaneous market processor, hiring และ land purchase
10. เพิ่ม town demand, weeds, end-of-day reset และ terminal rewards
11. เพิ่ม persistent JSON-lines adapter และ turn-by-turn differential harness
12. เพิ่ม randomized differential sweep และ full-season parity gate
13. เพิ่ม optional PyO3 adapter โดยยังคง JSON contract
14. เพิ่ม benchmark และ migration documentation
15. เมื่อ Python RL source กลับมา ให้เพิ่ม shadow environment adapter โดย default ยังเป็น Python
16. รัน representative policy replays และแก้ mismatch จนเป็นศูนย์
17. สลับ local training default เป็น Rust แต่คง oracle test ใน CI
18. หลังผ่านหลาย training seeds จึงลบ duplicate Python transition path ที่โปรเจกต์เป็นเจ้าของ

## Decision Document

- Rust เป็นเจ้าของ game state transitions; Python เป็น compatibility และ orchestration layer
- official Kaggle environment ยังเป็น oracle สูงสุด ไม่ใช่ Rust implementation
- public/private visibility boundary ต้องคงเดิม ห้ามให้ policy เห็น opponent private state
- invalid actions ยังคงเป็น silent no-ops เพื่อให้ train/evaluation semantics ตรงการแข่งขัน
- market orders ประมวลผลแบบ lockstep ต่อหน่วย และ quote ก่อน commit ของผู้เล่นทั้งสอง
- RNG ต้องตรง CPython สำหรับ seed เดียวกัน ไม่ใช่แค่มี distribution คล้ายกัน
- inventory ใช้ insertion-preserving map เพราะลำดับมีผลเมื่อ shed capacity ไม่พอ
- Python boundary รอบแรกใช้ JSON เพื่อความง่ายในการตรวจ parity; tensor/batch ABI เป็น optimization ภายหลัง
- ไม่แก้ dependency ภายใน virtual environment และไม่ vendor source ของ package มาทับโดยตรง
- default seed ของ Rust เมื่อไม่ระบุเป็นศูนย์เพื่อ reproducibility; production adapter ต้อง resolve/record seed
  ก่อนสร้าง engine เช่นเดียวกับ Kaggle framework

## Testing Decisions

- ทดสอบ external behavior และ serialized state ไม่ผูก test กับ private helper implementation
- unit tests ครอบคลุม RNG vectors, initialization, atomic planting และ buy/sell price symmetry
- differential tests เทียบทั้งสอง players หลังทุก turn ไม่ใช่เทียบเฉพาะ final score
- ต้องมี deterministic scenario, randomized traces, full-season trace และ market override
- parity failure ต้องรายงาน JSON path แรกที่ต่างเพื่อให้หาสาเหตุได้เร็ว
- benchmark แยกเป็น Rust core และ Python-facing adapter และต้องระบุ workload ชัดเจน
- quality gate คือ formatting, tests, Clippy แบบ warnings-as-errors และ Python syntax validation

## Out of Scope

- เปลี่ยน policy, reward, macro action design หรือ training algorithm
- เขียน Kaggle submission agent เป็น Rust; Kaggle execution contract ยังเป็น Python
- เปลี่ยนกฎการแข่งขันเพื่อให้ implementation ง่ายขึ้น
- ลบ Python oracle ก่อน representative policy traces ผ่าน parity
- ทำ tensor-native vectorized ABI ใน migration แรก
- เชื่อมเข้ากับ RL stack เดิมจนกว่า source ชุดนั้นจะกลับมาอยู่ใน checkout

## Current Evidence and Remaining Gate

Rust core และ PyO3 module compile แล้ว Differential validation ผ่าน full season 719 turns และ randomized
6,000 turns โดยเทียบทุก state field ของผู้เล่นทั้งสอง Local pass-action benchmark วัด Python-facing Rust
adapter เร็วกว่า installed reference ประมาณ 110 เท่าใน workload นี้ (median จาก 3 runs)

สิ่งที่ยังทำไม่ได้ใน checkout ปัจจุบันคือเปลี่ยน caller ของ RL เดิม เพราะ source ที่ดัชนี TokenSave อ้างถึง
ไม่ได้อยู่ใน working tree เมื่อ source กลับมา ขั้นต่อไปคือสร้าง shadow adapter แล้ว replay trajectories จาก
policy จริงก่อนเปิด Rust เป็น default backend
