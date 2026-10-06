# V99 — Tape league: มี lineage ไหนใน pool ที่เล่นแบบ open-loop แล้วชนะ v97 บน seed ใหม่?

Pre-registered 2026-09-05 ก่อนรัน seed ในช่วงด้านล่าง (ไม่มี smoke บนช่วงนี้)

## คำถาม

56027851 (V97) นิ่งแถว 1700–1750 และ lineage ShopForge ทั้งตระกูล (Yusuke 2214, boatlee 2285) หล่นมาอยู่ระดับ
เดียวกัน ช่องว่างอยู่ที่ late game ซึ่ง overlay/handover แก้ไม่ได้ (V98) คำถามคือ **มี tape ของทีมอื่นที่เล่นตรงๆ
แล้วดีกว่า v97 ไหม** ถ้ามี lineage นั้นคือฐานใหม่สำหรับ shop router รอบถัดไป

## Design (screen — ไม่มี upload จาก stage นี้)

- Candidates: ทุก tape ใน `artifacts/tape_pool_20260830.tar.gz` (1,390 ม้วน, 189 ทีม) เล่นแบบ open-loop
  ทั้งเกม + reference `ROUTER` = v97 (`artifacts/shop_router_v2/routes.json`, sha 0e9e005496ccc284)
- Opponents: 9 tapes ชุด V97 (`analysis/v97_shop_router_selection_20260905/opponents/`) ไม่เปลี่ยน
- Seeds: **17100001–17100200** (200, ใหม่) × 2 seats; engine config เดิม; ผ่าน `play_batch` (official-env semantics)
- Metric ต่อ tape: pooled point rate (9 opp × 200 × 2 = 3,600 เกม, SE ≈ 0.007), mean margin, point ต่อ opponent
- Script: `scripts/evaluate_tape_league.py` (sha 8430e6d5212fabd6) เก็บ aggregate ต่อ (tape, opponent)

## Decision rule (freeze)

tape "ผ่าน screen" เมื่อ (a) pooled point ≥ point ของ ROUTER บน seed เดียวกัน และ (b) ไม่แพ้ ROUTER เกิน 0.05 ใน
opponent ใดๆ รายงาน top 20 เสมอ พร้อมทีม/episode/final money ที่บันทึกไว้ และ self-identity ของทีม (open-loop หรือไม่)
ถ้าไม่มี tape ผ่าน → lineage ปัจจุบันคือดีที่สุดใน pool นี้ ทางที่เหลือคือ market leverage หรือ harvest pool ใหม่
ถ้ามี → stage 2 (pre-register ใหม่): หา donor ที่ share prefix ของ tape นั้น แล้วทำ shop router ด้วย pipeline V97

## ข้อควรระวังที่รู้ล่วงหน้า

- tape ของ agent ที่ adapt สูง (Crop Dusta, Milan, A Poor Vul) พังบน seed อื่น (V96) — คะแนนต่ำของพวกนี้ไม่ได้บอกว่า
  agent จริงอ่อน
- opponent set เป็น tape ทั้งหมด ผลไม่รับประกัน rating จริง (memory `kaggriculture-local-eval-does-not-predict-rating`)
