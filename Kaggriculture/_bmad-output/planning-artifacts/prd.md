---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
  - step-12-complete
inputDocuments:
  - RESEARCH_SUSTAINABLE_RL.md
  - ACTION_SPACE.md
  - RUST_ENGINE_MIGRATION.md
documentCounts:
  briefs: 0
  research: 1
  brainstorming: 0
  projectDocs: 2
workflowType: 'prd'
workflowStatus: complete
completedAt: '2026-08-15'
classification:
  projectType: developer_tool
  domain: scientific
  complexity: medium
  projectContext: brownfield
---

# Product Requirements Document - Kaggriculture

**Author:** Chinnaphat
**Date:** 2026-08-15

## Executive Summary

Kaggriculture RL เป็นระบบฝึกและประเมิน reinforcement-learning agent แบบ end-to-end สำหรับสร้าง Kaggle submission ที่แข่งขันได้ ผู้ใช้หลักคือทีมพัฒนา agent ซึ่งต้องการทดลอง policy ได้รวดเร็ว ทำซ้ำได้ และมั่นใจว่าพฤติกรรมระหว่างการฝึกตรงกับ Kaggle environment จริง

ระบบแก้ข้อจำกัดของแนวทางเดิม ได้แก่ Python simulation ที่ช้า, fixed actions ที่ปรับตัวไม่ได้, illegal actions แบบ silent no-op และ policy ที่ overfit ต่อคู่แข่งไม่กี่แบบ Rust engine ทำหน้าที่เร่ง rollout โดยต้องผ่าน behavioral parity ส่วน RL เรียนรู้กลยุทธ์จาก state ปัจจุบันและประเมินกับ seeds และคู่แข่งที่ไม่อยู่ในชุดฝึก

ความสำเร็จวัดจาก Kaggle score และ win rate โดยใช้ lower-decile และ worst-case score เป็น promotion gate โมเดลที่เพิ่มค่าเฉลี่ยแต่ลดความทนทานจะไม่ถูกเลื่อนเป็นตัวหลัก

### What Makes This Special

ระบบแยกความรับผิดชอบชัดเจน: RL เรียนรู้ว่า “ควรทำอะไรและเมื่อไร” ผ่าน semantic macro targets ส่วน deterministic executor รับผิดชอบว่า “ทำอย่างไรให้ถูกกฎ” รวมถึง movement, legality, inventory และการจองทรัพยากรร่วมกันระหว่าง hands

จุดแตกต่างหลักประกอบด้วย:

- Rust simulation ที่เร็วและตรวจ parity กับ Kaggle แบบ turn-by-turn
- State-aware policy ที่ไม่ผูกกับ fixed action tape
- Action masking และ reservation ledger เพื่อลด illegal actions และ silent no-op
- Opponent league และ held-out cross-play เพื่อป้องกันการ overfit
- Promotion gate ที่พิจารณาทั้งคะแนนเฉลี่ย, win rate และ worst-case performance

## Project Classification

- Project Type: `developer_tool`
- Domain: `scientific` — reinforcement learning, simulation และ game AI
- Domain Complexity: `medium`
- Engineering Complexity: `high`
- Project Context: `brownfield`
- Scope: ระบบเฉพาะ Kaggriculture ไม่ใช่ RL framework อเนกประสงค์

## Terminology

- **Agent:** Kaggle submission ที่ประกอบด้วย learned policy และ deterministic executor
- **Policy:** โมเดลที่เลือก semantic targets และ timing จาก state ปัจจุบัน
- **Candidate:** Versioned checkpoint ที่อยู่ระหว่าง evaluation
- **Champion:** Candidate ล่าสุดที่ผ่าน promotion gate และใช้เป็น baseline
- **Python oracle:** Kaggle reference environment ที่ใช้ตัดสิน behavioral correctness
- **Rust simulator:** ตัวจำลองประสิทธิภาพสูงที่ต้องผ่าน parity gate ก่อนใช้ฝึก
- **Held-out pool:** ชุด seeds และ opponents แบบ immutable ที่ training process เข้าถึงไม่ได้
- **Parity gate:** การตรวจว่า Rust simulator ให้ transition ตรงกับ Python oracle
- **Promotion gate:** เกณฑ์รวมที่ candidate ต้องผ่านก่อนแทน champion

## Success Criteria

### User Success

- ทีมสามารถฝึก ประเมิน และสร้าง Kaggle submission ผ่าน pipeline เดียวที่บันทึก config, seed, engine version และ model checkpoint ครบถ้วน
- ผลทดลองสามารถทำซ้ำและเปรียบเทียบกับ champion เดิมบน held-out seeds และ opponent pool เดียวกัน
- ผู้ใช้เห็นเหตุผลที่ model ผ่านหรือไม่ผ่าน promotion gate จาก score, win rate, lower-decile, worst-case และ execution diagnostics
- RL สามารถออกจากพฤติกรรม fixed policy และปรับกลยุทธ์ตาม state กับคู่แข่งได้จริง

### Business Success

- Candidate มี held-out win rate อย่างน้อย `60%` เทียบกับ champion และ opponent mixture
- Mean final score สูงกว่า champion อย่างน้อย `10%`
- Public Kaggle score ใช้ยืนยันความสามารถภายนอก แต่ไม่ใช้เป็น training signal หรือ promotion metric เพียงตัวเดียว
- ระบบลดรอบการทดลองด้วย Rust rollout โดยไม่แลกกับความถูกต้องของ environment

### Technical Success

- Rust engine มี `0 parity mismatches` กับ Python oracle จากอย่างน้อย `100,000 randomized turns` และ representative full-season traces
- Rollout throughput เร็วกว่า Python อย่างน้อย `50×`
- Executed illegal-action/silent-no-op rate ต่ำกว่า `1%`
- ผลการฝึกผ่านเกณฑ์จากอย่างน้อย `3 independent training seeds`
- Training และ evaluation ใช้ legality masks, deterministic executor และ joint resource reservation
- ทุก checkpoint เชื่อมโยงกลับไปยัง config, seed, code version, opponent pool และ evaluation report ได้

### Measurable Outcomes

Candidate จะถูก promote เมื่อผ่านทุกข้อ:

1. Held-out win rate `≥ 60%`
2. Mean final score improvement `≥ 10%`
3. Lower-decile score ไม่ต่ำกว่า champion
4. ไม่มี opponent subgroup ที่ win rate ต่ำกว่า `45%`
5. Executed illegal/no-op rate `< 1%`
6. ผ่าน Rust–Python parity gate
7. ผลลัพธ์คงเส้นคงวาจากอย่างน้อย `3` training seeds

## Product Scope & Phased Development

### MVP — Phase 1

MVP เป็น validated-learning system ที่ต้องพิสูจน์ว่า semantic macro policy ซึ่งใช้ deterministic executor และ parity-validated Rust simulator สามารถฝึกด้วย BC + minimal masked PPO และเอาชนะ champion อย่าง robust ได้

**Must-Have Capabilities:**

1. Audit source เดิมและ freeze rule champion เป็น baseline
2. Semantic macro target schema
3. Enriched state representation และ legality masks
4. Deterministic executor พร้อม joint reservation ledger
5. Rust–Python parity gate และ deterministic replay
6. Macro BC warm start
7. Minimal masked PPO fine-tuning
8. Fixed training opponent mix และ immutable held-out pool
9. Cross-play evaluation และ promotion gates
10. Versioned config, checkpoint, trace และ artifact metadata
11. Minimal CLI: `parity`, `train`, `evaluate`, `export`
12. Kaggle-compatible submission artifact

**Explicitly Out of MVP:**

- DAgger และ recurrent policy
- Automated opponent league/PSRO
- Learned termination หรือ learned low-level executor
- Tensor-native batch API
- PyPI, Windows และ IDE plugin
- Dashboard หรือ distributed training platform

### Growth — Phase 2

- Queryable teacher และ macro DAgger
- Recurrent PPO และ potential-based reward shaping
- Historical policy snapshots และ opponent league
- Automated ablation และ robustness reporting
- `resume` และ advanced `compare` workflows
- Batched Rust rollout API

### Vision — Phase 3

- Learned option termination
- Learned low-level control เฉพาะ executor bottlenecks
- PSRO-style exploiters และ adaptive league
- Automated continuous agent selection
- PyPI/Windows support หากมีความต้องการจริง

## User Journeys

### 1. นนท์ — RL Researcher: จากแนวคิดสู่ Candidate ที่แข็งแรง

นนท์พบว่า fixed policy เดิมเริ่มตัน เขาจึงกำหนด semantic macro actions, state representation, opponent mixture และ training seed ผ่าน experiment config เดียว ระบบตรวจ Rust parity ก่อนเริ่มฝึก จากนั้นรัน BC และ RL พร้อมบันทึก checkpoints กับ metrics อัตโนมัติ

จุดสำคัญเกิดขึ้นเมื่อ candidate ชนะ champion บน held-out evaluation และผ่านทุก promotion gate นนท์มั่นใจว่าผลลัพธ์ไม่ได้เกิดจาก seed หรือคู่แข่งที่ง่ายเพียงชุดเดียว

**ความสามารถที่ต้องมี:** versioned experiment config, reproducible training, checkpointing, metric tracking และ held-out evaluation

### 2. เมย์ — Competition Lead: เลือก Submission โดยไม่หลงค่าเฉลี่ย

เมย์ได้รับ candidate หลายตัว บางตัวมีคะแนนเฉลี่ยสูงแต่แพ้หนักต่อ market bot บางชนิด ระบบแสดง cross-play matrix, win rate, lower-decile, worst-case และผลจากสาม training seeds

เมย์เลือก promote เฉพาะ candidate ที่ผ่านทุก gate ระบบจึง freeze checkpoint, engine version และ submission artifact ชุดเดียวกัน ทำให้มั่นใจว่าไฟล์ที่ส่งคือ policy ที่ผ่านการประเมินจริง

**ความสามารถที่ต้องมี:** champion registry, promotion gate, cross-play report, model comparison และ reproducible submission export

### 3. ก้อง — Engine Developer: เปลี่ยน Engine โดยไม่ทำลายความตรงกับ Kaggle

ก้องเพิ่ม optimization ใน Rust engine ก่อนเปิดใช้กับ training ระบบรัน randomized parity turns และ full-season traces เทียบ Python oracle หาก state, market, RNG หรือ terminal result ต่างแม้เพียงจุดเดียว การ cutover จะถูกปฏิเสธพร้อม trace แรกที่ divergence

หลังแก้สาเหตุ ก้องเห็น parity mismatch เป็นศูนย์และ throughput ยังมากกว่า Python อย่างน้อย 50 เท่า จึงเผยแพร่ engine version ใหม่ให้ experiment ถัดไปใช้ได้

**ความสามารถที่ต้องมี:** differential harness, first-divergence report, deterministic replay, benchmark และ engine version pinning

### 4. แพรว — Experiment Operator: กู้คืน Training ที่หยุดกลางทาง

งานฝึกระยะยาวหยุดเพราะ process หรือเครื่องขัดข้อง แพรวเปิด run metadata เพื่อดู config, seed, checkpoint ล่าสุด, opponent pool และ engine version จากนั้น resume โดยไม่สร้าง experiment ที่เทียบกับของเดิมไม่ได้

เมื่อ job จบ ระบบตรวจ artifact completeness และส่ง candidate เข้าสู่ evaluation โดยอัตโนมัติ แพรวไม่ต้องประกอบไฟล์หรือจดค่าด้วยมือ

**ความสามารถที่ต้องมี:** resumable jobs, artifact validation, structured run metadata, deterministic resume และ pipeline status

### 5. ต้น — Troubleshooter: หาสาเหตุของ Regression

Candidate ใหม่เพิ่ม mean score แต่ lower-decile ลดลงและ no-op rate สูงขึ้น ต้นเปิด episode trace ที่แย่ที่สุด ดู raw action proposal, mask, executor decision, reservation conflict และ `blocked_reason`

เขาพบว่า hands หลายตัวแย่ง seed เดียวกัน จึงแก้ reservation logic แล้ว replay state เดิมเพื่อยืนยันผล Candidate ใหม่ต้องผ่าน evaluation ทั้งชุดอีกครั้งก่อนมีสิทธิ์ promote

**ความสามารถที่ต้องมี:** searchable episode traces, action-mask diagnostics, executor telemetry, deterministic replay และ regression comparison

### Journey Requirements Summary

Journeys ทั้งหมดกำหนดให้ระบบมี:

- Training pipeline ที่ reproducible และ resume ได้
- Versioned configs, seeds, checkpoints, engine และ opponent pools
- Rust–Python parity gate พร้อม first-divergence diagnostics
- Held-out evaluation, cross-play matrix และ automated promotion
- Detailed action/executor traces สำหรับค้นหา illegal no-op และ resource conflicts
- Submission export ที่ผูกกับ checkpoint ซึ่งผ่าน gate แล้ว
- Artifact registry และผลเปรียบเทียบย้อนหลังระหว่าง experiments

## Domain-Specific Requirements

### Compliance & Regulatory

- ไม่มีข้อกำกับเฉพาะอุตสาหกรรมใน scope
- Submission, dependencies, datasets และ generated artifacts ต้องเป็นไปตามกติกาและ license ของ Kaggle
- ทุก model ต้องระบุ provenance: code version, engine version, config, seeds, opponent pool และ training artifacts
- Runtime, memory และ dependency limits ของการแข่งขันเป็น `TBD` และเป็น release gate ก่อนส่ง submission

### Technical Constraints

- Rust engine ต้อง replay แบบ deterministic เมื่อ initial state, seed และ action trace เหมือนกัน
- Rust–Python parity ต้องไม่มี mismatch จากอย่างน้อย 100,000 randomized turns และ representative full seasons
- Training ต้อง reproducible เชิงสถิติจากอย่างน้อย 3 independent seeds
- Train, validation และ held-out evaluation ต้องใช้ seed/opponent sets ที่แยกและ version ไว้ชัดเจน
- รายงานผลต้องมี mean, win rate, lower-decile, worst-case และผลแยกตาม opponent
- Public leaderboard score ห้ามเป็น training signal หรือเกณฑ์คัดเลือกเพียงตัวเดียว
- Engine throughput ต้องเร็วกว่า Python อย่างน้อย 50 เท่า

### Integration Requirements

- Rust core ต้องรองรับ typed in-process API
- Python integration ต้องรองรับ PyO3 และ compatibility adapter
- Input/output ต้องรักษา Kaggle observation, action และ silent-no-op semantics
- Python environment ยังคงเป็น behavioral oracle สำหรับ shadow validation
- Submission artifact ต้อง self-contained และผูกกับ checkpoint ที่ผ่าน promotion gate

### Risk Mitigations

- Simulator mismatch → บังคับ parity gate และ first-divergence trace
- Reward hacking → ใช้ terminal competition objective และตรวจ reward decomposition
- Seed/opponent overfit → แยก held-out pools และใช้ opponent league
- Nondeterminism → บันทึก seed, versions และรองรับ deterministic replay
- Illegal actions/no-op → legality masks, reservation ledger และ execution telemetry
- Evaluation leakage → immutable split registry และห้าม candidate เข้าถึง held-out pool ระหว่าง training
- Compute escalation → บันทึก throughput/resource usage และกำหนด budget ต่อ experiment
- Kaggle incompatibility → submission preflight เมื่อยืนยันข้อจำกัด runtime, memory และ dependencies แล้ว

## Innovation & Novel Patterns

### Detected Innovation Areas

นวัตกรรมหลักเป็นการเปลี่ยน control paradigm จาก fixed-policy offsets หรือ full primitive control ไปเป็น:

**state-aware semantic targets → masked conditional policy → deterministic executor**

RL ตัดสินใจด้านกลยุทธ์และ timing ส่วน executor จัดการ movement, legality, inventory และ joint resource reservation ใหม่จาก state ทุก turn แนวทางนี้มุ่งลด horizon และ silent no-op โดยยังรักษาความสามารถในการปรับตัว

องค์ประกอบเสริมคือ parity-gated Rust simulation และ robustness-first promotion ซึ่งทำให้ความเร็ว การทดลอง และความน่าเชื่อถือของผลลัพธ์อยู่ในระบบเดียวกัน

### Market Context & Competitive Landscape

แนวคิด hierarchical RL, options, action masking, imitation learning และ opponent leagues มีงานวิจัยรองรับแยกกันอยู่แล้ว Innovation ของโปรเจกต์นี้จึงเป็นการผสานแนวคิดเหล่านั้นให้ตรงข้อจำกัดของ Kaggriculture ไม่ใช่การอ้างว่าอัลกอริทึมแต่ละส่วนเป็นสิ่งใหม่ระดับโลก

ทางเลือกเดิมมีข้อจำกัดต่างกัน:

- Fixed rule policy: เสถียรแต่ปรับตัวได้น้อย
- Fixed-offset macro PPO: train ง่ายแต่ติดเพดานของ rule เดิม
- Primitive full-control RL: ยืดหยุ่นแต่ horizon และ action space ใหญ่
- Python-only simulation: ตรงกับ oracle แต่ rollout ช้า
- Rust-only cutover: เร็วแต่เสี่ยง simulator mismatch หากไม่มี parity gate

### Validation Approach

- Ablation: semantic targets เทียบ fixed offsets
- Ablation: enriched state เทียบ coarse state
- Ablation: deterministic executor เทียบ execution เดิม
- Differential validation: Rust เทียบ Python oracle
- Training comparison: BC, BC+DAgger และ masked recurrent PPO
- Robustness comparison: fixed opponents เทียบ opponent league
- ยอมรับ innovation เมื่อเพิ่ม held-out score/win rate โดย lower-decile และ worst-case ไม่ถดถอย

### Risk Mitigation

- Semantic abstraction สูญเสีย control ที่สำคัญ → เพิ่ม event-triggered replanning และค่อยขยาย target heads
- Executor เป็นคอขวด → วัด option success, blocked reasons และ fallback rate ก่อนทดลอง learned low-level control
- Rust divergence → ใช้ Python oracle และปิด Rust training backend ทันทีเมื่อ parity fail
- RL ไม่ชนะ rule champion → ใช้ rule/BC champion ต่อได้โดยไม่ทิ้ง evaluator และ executor
- Opponent league ใช้ compute สูง → เริ่มจาก PSRO-lite และ frozen snapshots
- ความซับซ้อนเพิ่มโดยไม่มีผล → ทุกองค์ประกอบต้องผ่าน ablation และ promotion gate แยกกัน

## Developer Tool Specific Requirements

### Project-Type Overview

Kaggriculture RL เป็น developer tool แบบ Python-first สำหรับนักพัฒนาและนักวิจัย RL โดยใช้ Rust เป็น simulation core ผู้ใช้ต้องสามารถฝึก ประเมิน debug และ export agent ผ่าน interface ที่ reproducible โดยไม่ต้องจัดการรายละเอียดภายในของ engine เอง

### Technical Architecture Considerations

- แยก Rust engine, Python binding, RL pipeline และ submission adapter ออกจากกัน
- Rust core ต้องไม่มี dependency ต่อ Python และรองรับ typed in-process API
- Python binding ต้องรักษา observation/action semantics ของ Kaggle
- Config, schema และ artifact formats ต้องมี version
- Breaking changes ต้อง fail พร้อมข้อความ migration ที่ชัดเจน
- Kaggle Python/runtime compatibility เป็น release gate

### Language Matrix

| Component | Language/Platform | Requirement |
|---|---|---|
| Simulation core | Rust stable | Pin toolchain/MSRV ต่อ release |
| Training and evaluation | Python 3.13 local | รองรับ type hints และ reproducible environments |
| Kaggle submission | Python version `TBD` | ต้องตรวจจาก runtime ทางการ |
| Competition runtime | Linux x86_64 | Required |
| Local development | macOS arm64 | Required |
| Windows | — | ไม่อยู่ใน MVP |

### Installation Methods

- Rust workspace สร้างด้วย Cargo
- Python wheel สร้างจาก PyO3 ด้วย `maturin`
- ติดตั้ง local wheel ผ่าน `uv` หรือ `pip`
- Dependency versions ต้อง lock ได้
- MVP ไม่ต้อง publish สู่ PyPI
- Submission build ต้องตรวจว่าไม่มี development-only dependency หลุดเข้า artifact

### API Surface

- Rust typed API สำหรับ initialization, state transition, inspection และ deterministic replay
- Python API สำหรับ environment lifecycle, training, evaluation และ artifact loading
- Compatibility adapter สำหรับ Kaggle-compatible JSON actions/observations
- CLI commands:
  - `parity`
  - `train`
  - `evaluate`
  - `resume`
  - `compare`
  - `export`
- ทุก command ต้องรองรับ versioned config และ machine-readable output
- Error reports ต้องระบุ run ID, seed, engine version และ first failing state เมื่อเกี่ยวข้อง

### Code Examples

เอกสารต้องมีตัวอย่างที่รันได้จริงสำหรับ:

- Rust–Python parity smoke test
- Minimal semantic-macro BC training
- PPO fine-tuning
- Resume จาก checkpoint
- Held-out cross-play evaluation
- Candidate/champion comparison
- Kaggle submission export

ตัวอย่างทั้งหมดต้อง pin config และ seed พร้อม expected validation outcome

### Migration Guide

1. ใช้ Python environment เป็น behavioral oracle
2. เปิด Rust shadow mode และเปรียบเทียบทุก transition
3. ใช้ Rust เป็น training default หลังผ่าน parity gate
4. เก็บ Python fallback จน representative workloads ผ่านครบ
5. แปลง fixed-offset macros เป็น semantic targets ผ่าน compatibility adapter
6. ถอด adapter หลัง policy และ executor ใหม่ผ่าน promotion gate

### Implementation Considerations

- รองรับ Linux x86_64 และ macOS arm64 ใน MVP
- ใช้ standard IDE support, type hints และ debugger; ไม่สร้าง IDE plugin
- รักษา backward compatibility เฉพาะ schema versions ที่ประกาศ
- แยก experimental APIs ออกจาก stable interfaces
- Documentation, examples และ migration tests ต้องผ่าน CI
- PyPI และ Windows support พิจารณาหลัง MVP เท่านั้น

## Delivery Strategy & Risk Controls

### MVP Strategy & Philosophy

**MVP Approach:** Problem-solving และ validated-learning MVP

MVP ต้องพิสูจน์สมมติฐานเดียว: semantic macro policy ที่ใช้ deterministic executor และ parity-validated Rust engine สามารถฝึกด้วย BC + masked PPO และเอาชนะ champion อย่าง robust ได้

### Resource Model

- นักพัฒนา 1–2 คนที่ครอบคลุม Python/RL และ Rust
- ใช้ local CPU สำหรับ parity, executor tests และ evaluation ขนาดเล็ก
- ใช้ local GPU สำหรับ smoke training ที่ไม่หนัก
- ใช้ Kaggle GPU สำหรับ full training, multi-seed runs และงานที่ใช้ compute สูง
- ทุกงานต้องใช้ config เดียวกันข้าม local/Kaggle และ resume จาก checkpoint ได้

### Risk Mitigation Strategy

**Technical Risks:**

- Source `rl/` เดิมไม่อยู่ใน workspace → MVP ไม่พึ่ง source เดิม; reuse เฉพาะส่วนที่กู้คืนและทดสอบได้
- Rust mismatch → Python oracle และ mandatory parity gate
- Executor abstraction จำกัด policy → event-triggered replanning และ executor telemetry
- PPO ไม่เสถียร → BC warm start, fixed curriculum และสาม training seeds

**Competition Risks:**

- Candidate overfit ต่อ opponent หรือ seed → immutable held-out evaluation
- Local metric ไม่สัมพันธ์กับ Kaggle → ใช้ public score เป็น external validation โดยไม่ใช้ train
- Mean ดีแต่ tail risk แย่ → lower-decile/worst-case promotion gates

**Resource Risks:**

- Local compute ไม่พอ → smoke test local แล้วส่ง full run ไป Kaggle GPU
- Kaggle quota จำกัด → progressive gates หยุด candidate ที่ไม่ผ่าน BC/evaluation ขั้นต้น
- ทรัพยากรบุคคลลดลง → รักษาเฉพาะ parity, executor, BC+PPO และ evaluator; เลื่อน automation ทั้งหมด

## Functional Requirements

### Experiment Lifecycle & Configuration

- FR1: RL Researcher สามารถสร้าง experiment specification ที่ระบุ policy, environment, opponents, seeds และ training stages ได้
- FR2: RL Researcher สามารถตรวจสอบความถูกต้องและ compatibility ของ experiment ก่อนเริ่มงานได้
- FR3: Experiment Operator สามารถใช้ experiment specification เดียวกันระหว่าง local และ remote compute โดยไม่เปลี่ยน semantics ได้
- FR4: Experiment Operator สามารถเริ่ม หยุด และ resume experiment จาก checkpoint ได้
- FR5: สมาชิกทีมสามารถตรวจสอบและเปรียบเทียบ configuration, provenance และผลลัพธ์ของ experiments ได้

### State, Decision & Action Execution

- FR6: RL Researcher สามารถกำหนดและ version semantic macro target schema ได้
- FR7: Policy สามารถรับ state ที่ครอบคลุม economy, entities, hands, inventory, market, time และ permitted history ได้
- FR8: Policy สามารถทราบว่า target และ arguments ใด valid ใน state ปัจจุบันได้
- FR9: Policy สามารถเลือกเป้าหมายด้าน crops, livestock, workers, selling, reserves, expansion และ recovery ได้
- FR10: ระบบสามารถแปลง semantic targets เป็น primitive actions ที่ถูกกฎได้
- FR11: ระบบสามารถประสานงาน active units และจองทรัพยากรร่วมกันก่อน execute actions ได้
- FR12: ระบบสามารถ replan เมื่อ target สำเร็จ ถูก block หรือ state สำคัญเปลี่ยนได้
- FR13: Executor สามารถรายงาน success, progress, blocked reason, fallback count และ turns used ได้
- FR14: ระบบสามารถใช้ legal fallback เมื่อ target ไม่สามารถ execute ได้

### Policy Training & Curriculum

- FR15: RL Researcher สามารถนำเข้าและตรวจสอบ labelled demonstrations ได้
- FR16: RL Researcher สามารถฝึก behavior-cloning policy จาก semantic demonstrations ได้
- FR17: RL Researcher สามารถ fine-tune candidate ด้วย masked reinforcement learning ได้
- FR18: RL Researcher สามารถให้ queryable teacher label learner states สำหรับ DAgger ได้ `[Growth]`
- FR19: RL Researcher สามารถฝึกและเปรียบเทียบ recurrent policy ได้ `[Growth]`
- FR20: RL Researcher สามารถกำหนด terminal objective และ potential-based shaping ได้
- FR21: RL Researcher สามารถกำหนด opponent mixture และ curriculum ต่อ experiment ได้
- FR22: ระบบสามารถจัดการ historical policies และ opponent league ได้ `[Growth]`
- FR23: ระบบสามารถฝึก candidate ด้วยหลาย independent seeds ภายใต้ configuration เดียวกันได้
- FR24: ระบบสามารถ checkpoint และ resume model, optimizer และ training state ได้

### Simulation Validation

- FR25: Engine Developer สามารถรัน action trace เดียวกันบน Rust simulator และ Python oracle ได้
- FR26: ระบบสามารถเปรียบเทียบ public state, private state, market, town, time และ terminal outcome ทุก turn ได้
- FR27: Engine Developer สามารถดู first-divergence report เมื่อ simulator results ต่างกันได้
- FR28: สมาชิกทีมสามารถ replay episode แบบ deterministic จาก initial state, seed และ actions ได้
- FR29: ระบบสามารถป้องกัน simulator version ที่ไม่ผ่าน parity gate จากการใช้ฝึกได้
- FR30: Engine Developer สามารถ benchmark rollout throughput ด้วย representative workloads ได้

### Evaluation, Robustness & Promotion

- FR31: Competition Lead สามารถสร้างและ version train, validation และ immutable held-out sets ได้
- FR32: ระบบสามารถรัน cross-play ระหว่าง candidate, champion และ opponent pool ได้
- FR33: ระบบสามารถคำนวณ mean, median, win rate, lower-decile, worst-case และ execution metrics ได้
- FR34: ทีมสามารถดูผลแยกตาม opponent, environment seed และ training seed ได้
- FR35: Competition Lead สามารถเปรียบเทียบ candidate กับ champion บน evaluation cohort เดียวกันได้
- FR36: Competition Lead สามารถกำหนด promotion policy ที่ candidate ต้องผ่านทุก gate ได้
- FR37: ระบบสามารถ freeze champion พร้อม configuration, artifacts และ evaluation report ได้
- FR38: ทีมสามารถบันทึก Kaggle score เป็น external validation โดยไม่ป้อนกลับเข้า training หรือ held-out selection ได้

### Diagnostics & Operations

- FR39: Troubleshooter สามารถตรวจ episode trace ตั้งแต่ raw proposal, mask, executor decision ถึง executed action ได้
- FR40: Troubleshooter สามารถค้นหา episodes ตาม score, opponent และ failure reason ได้
- FR41: Troubleshooter สามารถตรวจ reservation conflicts, blocked targets, fallbacks และ no-op causes ได้
- FR42: Experiment Operator สามารถตรวจ artifact completeness ก่อนส่ง candidate เข้า evaluation ได้
- FR43: Experiment Operator สามารถ rerun stage ที่ล้มเหลวโดยรักษา experiment identity และ provenance เดิมได้

### Artifacts, Interfaces & Distribution

- FR44: ระบบสามารถลงทะเบียน model, config, engine, opponents, traces และ reports เป็น versioned artifacts ได้
- FR45: Competition Lead สามารถ export และ preflight self-contained Kaggle submission จาก promoted checkpoint ได้
- FR46: ผู้ใช้และระบบภายนอกสามารถใช้ programmatic interface และ command interface ที่ให้ machine-readable results ได้
- FR47: ระบบสามารถโหลด supported schema versions และให้ migration guidance เมื่อพบ incompatible artifact ได้
- FR48: RL Researcher สามารถใช้ batched rollout interface สำหรับงานขนาดใหญ่ได้ `[Growth]`
- FR49: RL Researcher สามารถทดลอง learned termination หรือ learned low-level components โดยยังใช้ evaluation contract เดิมได้ `[Vision]`
- FR50: ผู้ใช้สามารถเข้าถึง installation guide, API reference, runnable examples, troubleshooting และ migration documentation สำหรับ platforms ที่รองรับได้

## Non-Functional Requirements

### Performance

- NFR1: Rust simulator ต้องมี median rollout throughput อย่างน้อย `50×` ของ Python oracle บน representative benchmark เดียวกัน
- NFR2: ทุก benchmark ต้องบันทึก hardware, software versions, workload และจำนวน repetitions
- NFR3: Submission ต้องผ่าน official Kaggle runtime limit ซึ่งยังเป็น `TBD`; ห้าม release จนกว่าจะยืนยันเกณฑ์นี้

### Correctness & Reproducibility

- NFR4: Rust simulator ต้องมี `0 mismatches` กับ Python oracle จากอย่างน้อย `100,000 randomized turns`
- NFR5: Representative full-season traces ต้องตรงกันทุก turn รวม state, market, time และ terminal outcome
- NFR6: Engine replay จาก initial state, seed และ action trace เดียวกันต้องให้ผลเหมือนเดิมทุกครั้ง
- NFR7: Training results ต้องรายงานจากอย่างน้อย `3 independent seeds`
- NFR8: Train, validation และ held-out sets ต้องแยกและ version อย่างชัดเจน
- NFR9: Executed illegal-action/silent-no-op rate ต้องต่ำกว่า `1%`

### Reliability & Recovery

- NFR10: Checkpoints ต้องเขียนแบบ atomic เพื่อให้ resume ได้จาก checkpoint ล่าสุดที่สมบูรณ์
- NFR11: การ resume ต้องรักษา run identity, config, engine version, opponent pool และ training state
- NFR12: ระบบต้องตรวจ artifact completeness ก่อน evaluation, promotion และ export
- NFR13: Candidate ที่ metrics หรือ artifacts ไม่ครบต้องไม่สามารถ promote ได้
- NFR14: Pipeline failure ต้องระบุ stage, run ID และสาเหตุในรูปแบบที่เครื่องอ่านได้

### Integration & Portability

- NFR15: Experiment config เดียวกันต้องมี semantics เหมือนกันบน local และ Kaggle
- NFR16: MVP ต้องรองรับ Linux x86_64 และ macOS arm64
- NFR17: Observation, action, config และ artifact schemas ต้องมี version
- NFR18: Unsupported schema หรือ runtime ต้อง fail ก่อนเริ่มงาน พร้อม migration guidance
- NFR19: Python oracle ต้องยังใช้เป็น fallback ได้จนกว่า Rust simulator จะผ่าน release gates

### Security & Artifact Integrity

- NFR20: Kaggle credentials และ secrets ต้องไม่ปรากฏใน source control, configs, logs, checkpoints หรือ submission artifacts
- NFR21: External credentials ต้องรับผ่าน runtime secret/environment mechanism เท่านั้น
- NFR22: Models, configs และ submission artifacts ต้องมี checksum สำหรับตรวจ corruption หรือการสลับไฟล์
- NFR23: Dependency versions และ licenses ต้องตรวจสอบได้จาก release artifact

### Compute Scalability

- NFR24: งาน parity, smoke training และ evaluation ขนาดเล็กต้องรัน local ได้
- NFR25: Full training และ multi-seed workloads ต้องย้ายไป Kaggle GPU ได้โดยไม่เปลี่ยน experiment semantics
- NFR26: Evaluation jobs ต้องแบ่งตาม seed/opponent และรวมผลแบบ deterministic ได้
- NFR27: ทุก run ต้องบันทึก CPU/GPU, memory, elapsed runtime และ rollout throughput
- NFR28: Batched rollout ในอนาคตต้องไม่เปลี่ยนผล transition หรือ evaluation เมื่อเทียบกับ sequential execution

Accessibility, public-user scalability และ uptime/SLA ถูกตัดออกจาก scope โดยตั้งใจ

## Traceability Summary

| Product outcome | Primary journeys | Functional requirements | Quality requirements |
|---|---|---|---|
| ฝึก policy ที่ปรับตัวจาก state และออกจาก fixed actions | RL Researcher | FR1–FR24 | NFR7–NFR9, NFR24–NFR28 |
| เร่ง rollout โดยไม่เปลี่ยน Kaggle semantics | Engine Developer | FR25–FR30 | NFR1–NFR6, NFR15–NFR19 |
| เลือก candidate ที่แข่งขันได้และไม่ overfit | Competition Lead | FR31–FR38 | NFR7–NFR9, NFR12–NFR13, NFR26 |
| ตรวจและกู้คืน experiment ได้อย่างมีหลักฐาน | Experiment Operator, Troubleshooter | FR39–FR44 | NFR10–NFR14, NFR20–NFR23 |
| สร้าง submission ที่ตรวจสอบย้อนกลับและรันใน Kaggle ได้ | Competition Lead, Experiment Operator | FR44–FR50 | NFR3, NFR15–NFR25 |
