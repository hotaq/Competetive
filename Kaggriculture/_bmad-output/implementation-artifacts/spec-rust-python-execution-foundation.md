---
title: 'Validated Rust-Python Execution Foundation'
type: 'feature'
created: '2026-08-16'
status: 'done'
baseline_commit: 'NO_VCS'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture.md'
  - '{project-root}/ACTION_SPACE.md'
  - '{project-root}/RUST_ENGINE_MIGRATION.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Rust engine มี parity baseline ที่ผ่านแล้ว แต่ Python package ยัง bootstrap ไม่สมบูรณ์, ทุก engine step ผ่าน JSON และยังไม่มี semantic executor หรือ canonical cross-language schemas จึงยังไม่ใช่ฐานที่ปลอดภัยสำหรับ RL training

**Approach:** สร้าง vertical foundation หนึ่งชุดจาก strict Python contract ผ่าน typed PyO3 ไปยัง deterministic Rust executor และ Rust engine โดย freeze target families ทั้ง 7 กลุ่ม แต่ implement end-to-end เฉพาะ `SELLING` ใน slice นี้ ส่วน target ที่ยังไม่รองรับต้องรายงาน `BLOCKED` อย่างชัดเจน

## Boundaries & Constraints

**Always:**

- รักษา Kaggle engine semantics และ existing deterministic/random parity เป็น regression gate
- ใช้ dependency direction `kaggriculture-engine <- kaggriculture-executor <- kaggriculture-python`; engine เท่านั้นที่ mutate `GameState`
- ให้ `schemas/v1/*.schema.json` เป็น canonical wire contract, wire enums เป็น `UPPER_SNAKE_CASE`, unknown fields/variants และ incompatible `schema_version` ต้อง fail closed
- Planner ต้อง deterministic จาก snapshot/target เดิม, ห้าม mutate state ระหว่าง planning และต้องเปิดเผย blocked reason/fallback ใน `ExecutionReport`
- คง PyO3 `0.23.5`; typed target/plan/step path ห้าม serialize JSON ต่อ turn ส่วน JSON path เดิมเก็บไว้เพื่อ parity/debug
- ถือ untracked worktree ปัจจุบันเป็น user-approved baseline และรักษาไฟล์ที่อยู่นอก scope

**Ask First:** เปลี่ยน game rules, ลด parity/schema gates, เปลี่ยน canonical wire shape หลังอนุมัติ, เพิ่ม runtime dependency ที่ไม่อยู่ใน architecture หรือขยาย scope ไปยัง RL training/full macro planner

**Never:** duplicate game rules ใน executor/Python, silent no-op สำหรับ valid semantic target ที่ execute ไม่ได้, อัปเกรด PyO3, ลบ Python oracle, เพิ่ม BC/PPO/reward/opponent league/checkpoint/export/Kaggle packaging หรืออ้างว่า smoke benchmark คือ release evidence

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Selling succeeds | `SELLING(item, quantity)` และ player shed มีของเพียงพอ | planner คืน ordered `JointAction` ที่มี `SELL`; report เป็น `PLANNED`; snapshot ไม่เปลี่ยนจน engine apply | ไม่มี fallback |
| Inventory conflict | ปริมาณขายมากกว่า inventory หรือ reservation คงเหลือ | primitive plan เป็น legal pass/empty market; report เป็น `BLOCKED` พร้อม stable reason และ `fallback_count=1` | ห้ามปล่อยให้ engine silent no-op เป็นตัวตรวจ legality |
| Deferred target | target family อื่นใน 6 กลุ่มมี payload ที่ schema-valid | deterministic legal fallback และ `KG-EXEC-TARGET-DEFERRED` | ไม่ panic และไม่ mutate state |
| Invalid contract | version, enum, field หรือ quantity ไม่ถูกต้อง | Rust/Pydantic ปฏิเสธตรงกัน | structured `KG-SCHEMA-*` error ไม่มี stack trace/secrets |
| Repeated planning | snapshot, targets และ seed เหมือนเดิม | byte-equivalent canonical `JointAction`/`ExecutionReport` | divergence ทำให้ test fail พร้อม first differing path |
| Typed execution | Python ส่ง typed mappings ผ่าน binding | Rust plan/apply ทำงานโดยไม่เรียก JSON codec และคืน typed mappings | conversion error ต้อง fail closed ก่อน mutation |

</frozen-after-approval>

## Code Map

- `Cargo.toml` และ `rust/kaggriculture-executor/` — workspace member, semantic types, reservation, legality, fallback, planner และ reports
- `rust/kaggriculture-engine/src/{action,lib}.rs` — symmetric canonical primitive-action serialization และ public engine-owned facts ที่ executor ต้องใช้
- `rust/kaggriculture-python/` กับ `bindings/python/` — typed executor/engine adapter, package metadata, type surface และ contract tests
- `pyproject.toml`, `.python-version`, `uv.lock`, `src/kaggriculture_rl/` — uv root workspace และ strict Python protocol/schema layer
- `schemas/v1/` กับ `tests/fixtures/v1/` — canonical observation, macro action, joint action, execution report schemas และ valid/invalid golden fixtures
- `tests/contracts/`, `tests/parity/` และ `scripts/compare_rust_engine.py` — cross-language fixtures, typed-vs-JSON equivalence และ oracle regression gate

## Tasks & Acceptance

**Execution:**

- [x] `pyproject.toml`, `.python-version`, `uv.lock`, `src/kaggriculture_rl/` — bootstrap Python 3.13 uv workspace, pin foundation dependencies และ expose importable package/version
- [x] `schemas/v1/*.schema.json`, `src/kaggriculture_rl/schemas/`, `tests/fixtures/v1/` — define strict version-1 contracts สำหรับ observation, 7-family macro action, joint action และ execution report พร้อม golden cases
- [x] `rust/kaggriculture-engine/src/{action,lib}.rs` — เพิ่ม symmetric action wire conversion/public read-only facts เท่าที่ executor ต้องใช้ โดยไม่ย้าย planning หรือเปลี่ยน transition rules
- [x] `rust/kaggriculture-executor/` — implement deterministic `SELLING` planner, ordered reservations, legality checks, explicit deferred-target fallback และ report; เพิ่ม Rust unit/integration tests
- [x] `rust/kaggriculture-python/`, `bindings/python/` — expose typed mapping plan/apply surface, release GIL ในช่วง Rust-only work และเพิ่ม import/engine/executor contract tests
- [x] `tests/contracts/`, `tests/parity/`, `scripts/compare_rust_engine.py` — verify Serde/Pydantic fixture agreement, typed-vs-JSON equivalence, first-divergence diagnostics และ baseline oracle parity

**Acceptance Criteria:**

- Given clean locked environment, when sync/build/import และ test commands ทำงาน, then root package และ `kaggriculture_rs` import ได้โดยไม่พึ่ง raw-library loader
- Given valid/invalid golden fixtures, when Pydantic และ Rust Serde อ่านข้อมูล, then accept/reject, enum spelling และ canonical serialization ต้องตรงกันทั้งหมด
- Given identical snapshot และ `SELLING` target, when plan ซ้ำหรือผ่าน Python binding, then action/report เหมือนเดิม, plan ไม่ mutate state และ engine เป็นผู้ apply เพียงส่วนเดียว
- Given conflicts, deferred targets หรือ malformed inputs, when executor/binding ประมวลผล, then legal fallback หรือ structured error ปรากฏชัดและไม่มี silent semantic failure
- Given existing deterministic trace และ randomized 960-turn sweep, when รันหลังแก้ไข, then mismatch เท่ากับศูนย์; Rust workspace tests, fmt และ clippy warnings-as-errors ผ่าน
- Given typed hot-path contract test, when JSON codec ถูกทำให้เรียกแล้วล้มเหลว, then typed plan/apply ยังผ่าน แสดงว่าไม่มี per-turn JSON serialization

## Spec Change Log

## Design Notes

Version 1 ใช้ tagged semantic payload เช่น `{"schema_version":1,"kind":"SELLING","item":"MILK","quantity":3}`. ทั้งเจ็ด `kind` ถูก freeze ใน schema แต่สถานะ executable ของ slice นี้มีเฉพาะ `SELLING`; การรองรับที่เหลือเป็นงานถัดไปและต้องเพิ่มโดยไม่เปลี่ยน wire meaning เดิม

Release gates `100,000` randomized turns, full-season oracle parity และ representative `>=50x` benchmark ยังคงบังคับก่อนเปิด Rust training backend แต่ไม่ถูกนับว่าผ่านจาก foundation smoke suite

## Verification

**Commands:**

- `uv lock --check && uv sync --frozen` — lockfile สอดคล้องและ packages/build backend ติดตั้งได้
- `uv run pytest` — Python import, schemas, binding, typed path และ parity tests ผ่าน
- `cargo fmt --all -- --check` — Rust formatting ผ่าน
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — ไม่มี compiler/clippy warnings
- `cargo test --workspace` — engine/executor contract และ deterministic tests ผ่าน
- `uv run python scripts/compare_rust_engine.py` — existing deterministic trace และ 960 randomized turns mismatch เป็นศูนย์
