---
title: State-Aware Route Controller over the Rust Semantic Executor
type: quick-dev-spec
created: 2026-08-24
status: done
baseline_commit: NO_VCS
baseline: submissions/meta_v16_rc6_adaptive_market_standalone/main.py
baseline_sha256: c8afcbb8f90861cd92131012146e573752820b80fbd939ace3c7503aea83cd5e
---

# Intent

สร้าง vertical slice ตาม flow นี้ โดยรักษา V16 เป็น baseline ที่แก้ย้อนกลับได้:

```text
day/hour + farm state + inventory + shared market
                         |
                         v
               macro/route decision
                         |
                         v
          ordered semantic macro targets
                         |
                         v
          deterministic Rust planner/executor
                         |
              +----------+----------+
              v                     v
        farmer + hands          market orders
```

เป้าหมายรอบนี้คือพิสูจน์ว่า route เปลี่ยนตาม state ได้โดยไม่ splice action tape ต่าง layout และไม่ทำลายความ deterministic ก่อนเปลี่ยน selector เป็น RL ในรอบถัดไป

# Frozen decisions

## Always

- Pin V16 RC6 safe rollback ตาม hash ด้านบน ห้ามแก้ artifact เดิม
- ใช้ raw observation ที่มี step/day/hour, farm, private inventory และ shared market เป็น input
- เปิดเกมด้วย V16 steps `0..119` แล้ว handoff ที่ step `120`; หลัง handoff semantic executor เป็นเจ้าของ action จนจบเกม
- เลือก route เฉพาะ phase gates `120/240/480/696` และคง route ระหว่าง gate
- merge route targets กับ market targets ก่อนเรียก Rust planner
- ให้ Rust planner สร้าง farmer, hands และ market พร้อมกัน เพื่อให้ reservation, inventory และ market ledger สอดคล้องกัน
- state ของ controller ต้องแยกตาม player และ reset เมื่อ step ย้อนกลับหรือเริ่ม episode ใหม่
- output เดิมซ้ำต้องได้ action เดิมแบบ byte-equivalent
- เก็บ route id, gate, reason และ planner status เพื่อวิเคราะห์ replay

## Ask first

- เพิ่ม macro/schema ใหม่ใน Rust
- เปลี่ยน handoff checkpoint หรือ opening prefix หลังเห็นผล benchmark
- train RL, package/upload หรือ promote candidate ไปแทน submission หลัก
- ใช้ route/action tape จาก agent คนละ layout

## Never

- แก้ hands หรือ market หลัง Rust planner คืน action แล้ว
- กลับไปใช้ suffix ของ V16 หลัง semantic route เคยเปลี่ยน world state
- ใช้ fallback agent กลบ illegal action ระหว่าง evaluation
- promote จาก smoke test 8 เกม หรือจาก training seeds

# Initial controller

สร้าง rule-based teacher ที่เลือกหนึ่งใน route profiles: `BALANCED`, `COW_PREMIUM`, `SHEEP_PREMIUM`, `LIQUIDATE` โดยดู normalized price premium เทียบ base price, เงิน, จำนวน land/worker/livestock, inventory และเวลาที่เหลือ Premium route ต้องผ่าน preconditions ก่อน commit; ถ้าไม่ผ่านให้เลือก `BALANCED` ส่วน `LIQUIDATE` ใช้ที่ gate 696 หลัง handoff หาก planner blocked ให้คืน legal PASS/recovery จาก semantic planner ไม่ย้อนกลับเข้าคนละ tape

`market_target_overlay(observation, route)` สร้าง SELLING/INVENTORY_BUFFER/RECOVERY targets ตาม stock, reserve, capacity และราคาปัจจุบัน จากนั้น `_merge_targets()` เรียง priority และตัด target ซ้ำ ก่อนส่งครั้งเดียวเข้า `plan_observation()`

# Code map

- Add `src/kaggriculture_rl/policy/hierarchical_semantic.py`: state extraction, gated route state, target merge, reset และ public policy callable
- Reuse `src/kaggriculture_rl/policy/semantic_schedule.py` และ `production_profile.py`: semantic route targets
- Reuse `bindings/python/kaggriculture_rs/__init__.py` และ `rust/kaggriculture-executor/src/planner.rs`: deterministic planning/sanitization
- Add `tests/policy/test_hierarchical_semantic.py`: branch, determinism, reset, merge, hand/market invariants
- Add a standalone builder only after source policy passes unit and live gates; generated candidate must wrap—not modify—the pinned baseline
- Reuse `scripts/evaluate_file_agent_vs_replay.py` and `scripts/evaluate_standalone_agents_live.py` for promotion evidence

# Implementation tasks

- [x] Implement pure feature extraction and sticky phase-gated route selection, including player/episode isolation.
- [x] Implement route and market target providers plus stable priority/deduplication; call `plan_observation()` exactly once per semantic step.
- [x] Add tests proving each route can be selected, the same observation is deterministic, rollback resets state, no duplicate unit assignment exists, market orders respect inventory/capacity, and `sanitize_action(proposed) == proposed`.
- [x] Add opening-handoff integration using an immutable copy of the pinned V16 actions. Assert prefix parity and validate all branch preconditions at handoff.
- [x] Build a separate candidate artifact and record provenance/hash. Do not change RC5/RC6 files.
- [x] Run the frozen CPU gate sequence; Gate 0 failed score, so Gate 1/2 were not opened and the stop decision is recorded.

# Acceptance criteria

- Given identical observations and controller state, route, targets and primitive action are identical.
- At least two observations differing only in shared-market regime select different expected routes at a phase gate.
- Hands count equals hired hands; no unit receives duplicate/conflicting work; market has at most 10 legal orders.
- Every proposed semantic action at or after the handoff equals its sanitized form; fallback, normalized, and post-handoff sanitized counters are zero. Opening counters are reported separately because the pinned V16 prefix must not be rewritten.
- Steps before handoff are byte-identical to pinned V16 in both seats. Each replay contains 720 recorded frames, 719 executable transitions, and reaches final state step 719.
- Replay gate: 8 contexts, zero fallback components, zero normalized steps, and zero post-handoff sanitized steps. The score screen is recorded separately and can stop promotion even when correctness passes.
- If and only if the replay score screen passes, run the live screen: 4 paired seeds/8 games, paired mean margin > 0, at least 2/4 paired wins, worst margin >= -15,000, max action < 1,000 ms.
- If and only if the live screen passes, run the fresh holdout: 12 paired seeds/24 games, paired wins > losses, paired mean > 0, game wins >= losses, paired worst mean >= -10,000, CVaR10 >= -12,000 และ worst margin >= -15,000.
- หาก holdout ไม่ผ่าน ให้เก็บผลเป็น experiment และไม่ promote; selector นี้จะเป็น teacher/data generator สำหรับ RL รอบถัดไปเท่านั้น

# Review resolution

- `patch`: fixed shared product reservations, executable product quotes, shed-capacity accounting, duplicate workforce/land targets, large-count conversion, phase-gate re-routing, terminal liquidation spending, and malformed planner-output rejection.
- `patch`: expanded opening parity to every step `0..119` in both seats and added selector/precondition coverage for all four routes.
- `bad_spec`: narrowed the zero-sanitization requirement to semantic steps because byte-identical V16 opening actions already sanitize on the replay states; phase counters now make this distinction auditable.
- `bad_spec`: replaced the ambiguous “720 turns” wording with 720 replay frames / 719 executable transitions / final state step 719.
- `bad_spec`: made Gate 1 and Gate 2 conditional, so a completed failed experiment is not falsely reported as a promotable candidate.
- `defer`: the generated candidate imports local Python/native dependencies. Bundle them or add runtime hash enforcement only after a future candidate passes Gate 0 and becomes upload-eligible.
- `defer`: replay-based causal diagnosis remains a hypothesis until a candidate earns a live-policy screen; recorded Top-1 actions may become stale after state divergence.

# Out of scope

- PPO/self-play training, neural inference และ Kaggle GPU
- การรับประกันชนะอันดับ 1 หรือคะแนน leaderboard
- การเพิ่ม action/macro contract ใหม่ใน Rust
- การ submit/upload โดยอัตโนมัติ

## Suggested Review Order

**Controller flow**

- Entry point preserves V16 opening, gates routes, and invokes Rust once.
  [`hierarchical_semantic.py:533`](../../src/kaggriculture_rl/policy/hierarchical_semantic.py#L533)

- Eligibility uses observable capital, workforce, land, capacity, livestock, and time.
  [`hierarchical_semantic.py:236`](../../src/kaggriculture_rl/policy/hierarchical_semantic.py#L236)

- Route and market targets merge deterministically before execution.
  [`hierarchical_semantic.py:447`](../../src/kaggriculture_rl/policy/hierarchical_semantic.py#L447)

**Executor correctness**

- Shared ledger owns funds, stock, capacity, hires, land, units, and purchases.
  [`planner.rs:21`](../../rust/kaggriculture-executor/src/planner.rs#L21)

- Product buys reserve exact executable quotes and available shed slots.
  [`planner.rs:194`](../../rust/kaggriculture-executor/src/planner.rs#L194)

- Animal, workforce, land, and buffer planners share one reservation state.
  [`planner.rs:643`](../../rust/kaggriculture-executor/src/planner.rs#L643)

**Evidence and stop decision**

- Evaluator records phase counters and route decisions at every gate.
  [`evaluate_file_agent_vs_replay.py:118`](../../scripts/evaluate_file_agent_vs_replay.py#L118)

- Builder pins baseline and dependency hashes without marking candidate upload-ready.
  [`build_v18_state_macro_submission.py:20`](../../scripts/build_v18_state_macro_submission.py#L20)

- Gate decision keeps failed experiment and blocks promotion/upload.
  [`v18_state_macro_gate_decision.json:27`](../../rl/evaluations/v18_state_macro_gate_decision.json#L27)

**Regression coverage**

- Rust tests reproduce reservation, quote, capacity, duplicate-target, and overflow failures.
  [`tests/planner.rs:199`](../../rust/kaggriculture-executor/tests/planner.rs#L199)

- Policy tests cover re-routing, preconditions, liquidation, and malformed planner output.
  [`test_hierarchical_semantic.py:243`](../../tests/policy/test_hierarchical_semantic.py#L243)

- Submission tests verify every opening step in both seats before handoff.
  [`test_v18_state_macro_submission.py:68`](../../tests/submission/test_v18_state_macro_submission.py#L68)
