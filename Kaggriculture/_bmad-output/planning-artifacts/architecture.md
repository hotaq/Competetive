---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - ACTION_SPACE.md
  - RUST_ENGINE_MIGRATION.md
  - RESEARCH_SUSTAINABLE_RL.md
  - _bmad-output/planning-artifacts/implementation-readiness-report-2026-08-15.md
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-08-15'
project_name: 'Kaggriculture'
user_name: 'Chinnaphat'
date: '2026-08-15'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:** มี 50 FR แบ่งเป็น 7 capability clusters:

- Experiment lifecycle, configuration และ provenance
- Semantic policy, action masking และ deterministic executor
- BC/PPO training และ checkpointing
- Rust–Python parity, replay และ performance
- Held-out evaluation, cross-play และ champion promotion
- Trace diagnostics, recovery และ artifact management
- CLI/API, Kaggle export, schema migration และ documentation

**Non-Functional Requirements:** มี 28 NFR โดยตัวขับเคลื่อน architecture หลักคือ:

- Rust throughput อย่างน้อย `50×` Python oracle
- `0 mismatches` จากอย่างน้อย `100,000 randomized turns`
- Deterministic full-season replay
- Illegal-action/silent-no-op rate `<1%`
- ผลจากอย่างน้อย 3 training seeds
- Immutable evaluation splits
- Atomic checkpoints และ complete artifact provenance
- Semantics เดียวกันระหว่าง local และ Kaggle
- Versioned schemas, checksums, secret isolation และ dependency/license traceability

**Scale & Complexity:**

- Primary domain: scientific developer tool สำหรับ RL simulation, training และ evaluation
- Complexity level: **High**
- Estimated architectural components: ประมาณ 10 logical subsystems
- ไม่มี multi-tenancy, public GUI, SLA หรือ regulated compliance
- Integration และ correctness complexity สูง แม้ user-interaction complexity ต่ำ
- Rollout, trace และ checkpoint volume มีแนวโน้มสูง แต่ควรใช้ local-first modular monorepo ไม่ใช่ distributed platform ใน MVP

### Technical Constraints and Dependencies

- Live Cargo workspace มี `kaggriculture-engine` และ `kaggriculture-python` พร้อม Python binding package แล้ว
- Rust simulation core ต้องไม่พึ่ง Python; PyO3/Kaggle compatibility ต้องอยู่ใน adapter boundary
- Python/Kaggle environment ยังคงเป็น behavioral oracle และ fallback จนกว่า parity gates จะผ่าน
- ต้องรักษา silent no-op, market ordering, shared-resource conflict, inventory ordering และ CPython-compatible seeded randomness
- ใช้ JSON-compatible boundary สำหรับ parity/debugging ก่อน; tensor-native batch API เป็น post-MVP
- Local development รองรับ macOS arm64 และ Linux x86_64; Windows/PyPI ไม่อยู่ใน MVP
- งานเบารัน local CPU/light GPU ส่วน full training และ multi-seed ใช้ Kaggle GPU ด้วย experiment config เดียวกัน
- Kaggle Python version, runtime, memory และ dependency limits ยังเป็น `TBD` และเป็น release blocker
- Source tree `rl/` เดิมไม่มีใน live checkout จึงต้องสร้าง clean RL pipeline หรือ reuse เฉพาะส่วนที่กู้คืนและพิสูจน์ได้
- ทีม 1–2 คนกำหนดให้ architecture ต้องเรียบง่าย แยก boundaries ชัด และหลีกเลี่ยง distributed services/dashboard

### Cross-Cutting Concerns Identified

- Determinism และ replayability
- Rust–Python behavioral parity
- Schema versioning และ migration
- Action legality, joint reservation และ legal fallback
- Experiment provenance และ artifact checksums
- Atomic checkpoint/resume
- Traceability จาก proposal → mask → executor → primitive action
- Train/validation/held-out isolation
- Robustness ต่อ seeds และ opponents
- Performance measurement โดยไม่ลด correctness
- Local/Kaggle portability
- Structured errors และ machine-readable CLI/API results
- Secrets และ supply-chain hygiene
- Brownfield compatibility และ staged cutover
- Developer experience สำหรับ CLI/API-only workflow

## Starter Template Evaluation

### Primary Technology Domain

Brownfield hybrid Rust–Python scientific developer tool:

- Rust: deterministic simulation core
- PyO3/maturin: native Python adapter
- Python: RL training, evaluation, orchestration และ Kaggle export
- CLI-first ไม่มี web/UI starter

### Starter Options Considered

1. **Extend Existing Cargo/PyO3 Workspace + uv Python Control Plane — Recommended**

   รักษา crates และ mixed Python/Rust layout เดิม เพิ่ม root Python package และ uv workspace สำหรับ RL pipeline

2. **สร้างใหม่ด้วย `maturin new -b pyo3`**

   เป็นคำสั่งทางการสำหรับ project ใหม่ แต่ไม่เหมาะกับ brownfield เพราะจะสร้างสิ่งที่มีอยู่แล้วซ้ำ และเสี่ยงทำให้ parity baseline เปลี่ยน [Maturin tutorial](https://www.maturin.rs/tutorial.html)

3. **Pure-Python project แล้วเรียก Rust ผ่าน JSON process**

   เหมาะกับ parity/debug boundary แต่ไม่เหมาะเป็น rollout path หลักเพราะมี serialization/process overhead

### Selected Starter: Extend Existing Workspace with uv

**Rationale for Selection:**

- Cargo workspace, Rust engine และ PyO3 package มีอยู่แล้ว
- ลด structural churn และรักษา parity-tested behavior
- แยก Python RL control plane ออกจาก simulation rules
- ใช้ lockfile และ command semantics เดียวกันระหว่าง local/Kaggle
- รองรับทีมขนาด 1–2 คนโดยไม่เพิ่ม service/container complexity

**Initialization Command:**

```bash
uv init --package --build-backend uv --python 3.13 \
  --name kaggriculture-rl --vcs none --no-readme .
```

คำสั่งนี้ต้องเป็นส่วนหนึ่งของ implementation story แรกและต้องตรวจ diff ก่อนรับ generated files จากนั้นกำหนด `bindings/python` เป็น uv workspace member ตามรูปแบบ workspace ทางการ [uv project initialization](https://docs.astral.sh/uv/concepts/projects/init/), [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)

### Verified Tooling Baseline

ณ วันที่ 2026-08-15:

| Tool | Existing | Current official | Decision |
|---|---:|---:|---|
| Rust | 1.95.0 | 1.97.1 | ยังไม่ upgrade จนผ่าน parity suite |
| uv | 0.11.6 | 0.12.5 | ใช้ 0.12.5 แล้ว lock version |
| maturin | ไม่ติดตั้ง local | 1.14.1 | ใช้ 1.14.x และ pin ใน build environment |
| PyO3 | 0.23.5 | 0.29.2 | แยก migration เป็น story เฉพาะ ห้ามรวมกับ bootstrap |

แหล่งตรวจสอบ: [Rust stable channel](https://static.rust-lang.org/dist/channel-rust-stable.toml), [uv on PyPI](https://pypi.org/project/uv/), [maturin on PyPI](https://pypi.org/project/maturin/), [PyO3 on crates.io](https://crates.io/crates/pyo3)

### Architectural Decisions Provided by Starter

**Language & Runtime**

- Rust edition 2024
- Python 3.13 สำหรับ local training
- Python binding รองรับขั้นต่ำ 3.11 ผ่าน `abi3-py311`
- Kaggle Python version ยังคงเป็น release-gate `TBD`

**Styling Solution**

- ไม่เกี่ยวข้อง เพราะไม่มี GUI ใน MVP

**Build Tooling**

- Cargo workspace สำหรับ Rust
- uv project/workspace และ lockfile สำหรับ Python
- maturin สำหรับ native wheel/development build
- ห้าม silent dependency upgrade โดยเฉพาะ PyO3/Rust engine dependencies

**Testing Framework**

- `cargo test` สำหรับ engine rules/determinism
- Python tests สำหรับ policy, executor และ artifacts
- Rust–Python differential parity suite เป็น integration gate
- ทุก dependency migration ต้องผ่าน randomized turns และ full-season replay

**Code Organization**

- `rust/kaggriculture-engine`: simulation source of truth
- `rust/kaggriculture-python`: PyO3 boundary
- `bindings/python`: native Python package
- `src/kaggriculture_rl`: experiment, policy, executor, training, evaluation และ export
- ห้ามย้าย simulation rules เข้า Python control plane

**Development Experience**

- `uv sync`, `uv run` และ `uv lock` สำหรับ reproducible Python workflow
- Cargo dependencies ใช้ lockfile และ exact versionsใน parity-sensitive components
- JSON boundary เก็บไว้สำหรับ debugging/parity ส่วน hot rollout ใช้ in-process PyO3
- ไม่เพิ่ม Docker, dashboard หรือ distributed services ใน MVP

**Implementation Note:** Project initialization command ข้างต้นต้องเป็น implementation story แรก

## Core Architectural Decisions

### Data Architecture

**Selected approach:** Filesystem artifacts + SQLite metadata registry + Parquet/JSONL analytical data

- Run แต่ละรายการเป็น immutable artifact directory พร้อม `manifest.json` และ SHA-256
- SQLite เก็บ metadata, provenance, status และ artifact pointers เท่านั้น ไม่เก็บ model/trace blobs
- SQLite index ต้อง rebuild ได้จาก manifests และใช้ explicit SQL migrations ผ่าน `PRAGMA user_version`
- Config เขียนเป็น YAML แต่ validate ด้วย Pydantic แล้ว normalize เป็น canonical JSON ก่อนคำนวณ digest
- Episode/parity traces ใช้ JSONL
- Demonstrations, rollout tables และ evaluation metrics ใช้ Parquet
- DuckDB ใช้เป็น read-only analytics engine บน Parquet ไม่ใช่ source of truth
- Model weights ใช้ Safetensors; optimizer/training state เป็น trusted-only checkpoint พร้อม schema version และ checksum
- Artifact migration สร้าง artifact ใหม่ ห้ามแก้ immutable artifact เดิม
- Cache เป็น content-addressed local cache และห้ามทำให้ held-out data รั่วเข้า training

**Version baseline verified on 2026-08-15:**

| Technology | Verified current | Decision |
|---|---:|---|
| SQLite | 3.53.4; local 3.51.0 | ใช้ portable SQL และบันทึก runtime version |
| Pydantic | 2.13.4 | Config/manifest validation |
| PyArrow | 25.0.1 | Parquet I/O |
| DuckDB | 1.5.5 | Optional analytical queries |
| Safetensors | 0.8.0 | Model-weight serialization |

แหล่งตรวจสอบ: [SQLite](https://www.sqlite.org/download.html), [Pydantic](https://pypi.org/project/pydantic/), [PyArrow](https://pypi.org/project/pyarrow/), [DuckDB](https://pypi.org/project/duckdb/), [Safetensors](https://pypi.org/project/safetensors/)

**Rationale:** รูปแบบนี้รองรับ local/Kaggle single-team workflow โดยไม่เพิ่ม network service; metadata index กู้คืนได้จาก immutable manifests และ analytical workloads แยกออกจาก transactional metadata

**Cascading implications:** Security ต้องกำหนด trusted checkpoint boundary, API ต้องใช้ versioned canonical schemas และ infrastructure ต้องรองรับ atomic artifact commit กับ deterministic result merge

### Authentication & Security

**Selected approach:** Local trusted-run security model; ไม่มี application authentication/RBAC ใน MVP

**Authentication and authorization:**

- ใช้ OS process identity และ filesystem permissions
- หากภายหลังเปิด remote service ต้องออกแบบ authentication ใหม่ ห้ามนำ local trust model ไปใช้ตรง ๆ

**Secrets:**

- รับ Kaggle credentials ผ่าน environment variables หรือ Kaggle Secrets เท่านั้น
- Config schema ต้องไม่มีช่องสำหรับบันทึก secret
- Logs, traces, checkpoints และ error reports ต้อง redact ค่า sensitive
- Submission preflight ต้องตรวจหา credentials และ development-only files
- ห้าม dump environment ทั้งชุดลง run metadata

**Checkpoint trust boundary:**

- Distributable model weightsใช้ Safetensors
- Optimizer/training state โหลดได้เฉพาะจาก trusted run directory ที่ระบบสร้างเอง
- ทุกไฟล์ต้องตรงกับ SHA-256 ใน immutable manifest
- ห้ามโหลด arbitrary pickle/`.pt` ที่ได้จากภายนอก
- Kaggle submission ต้องไม่มี optimizer state หรือ training-only artifacts

**Input and artifact security:**

- Pydantic ใช้ strict validation; schema version ที่ไม่รองรับต้อง fail closed
- Rust boundary ต้อง reject unknown enum variants, oversized payloads และ malformed structures
- Artifact paths ต้อง resolve อยู่ใต้ run directory เพื่อป้องกัน path traversal
- เขียน checkpoint แบบ temporary file → fsync → atomic rename
- Checksums ใช้ตรวจ corruption/accidental replacement; digital signatures เลื่อนไป post-MVP

**Supply chain:**

- ใช้ `uv.lock` และ `Cargo.lock`; CI/build ต้องใช้ locked mode
- ตรวจ Python vulnerabilities ด้วย `pip-audit` 2.10.1
- ตรวจ Python licenses ด้วย `pip-licenses` 5.5.5
- ตรวจ Rust advisories, licenses, bans และ sources ด้วย `cargo-deny` 0.20.2
- Release artifact ต้องมี dependency/license inventory
- Rust simulation core ใช้ `#![forbid(unsafe_code)]`; FFI risk จำกัดอยู่ใน PyO3 adapter

แหล่งเวอร์ชัน: [pip-audit](https://pypi.org/project/pip-audit/), [pip-licenses](https://pypi.org/project/pip-licenses/), [cargo-deny](https://crates.io/crates/cargo-deny)

**Network and encryption:**

- Training/evaluation/submission ไม่พึ่ง external service ขณะรัน
- ไม่เพิ่ม application-level encryption เพราะไม่มี PII หรือ persisted secrets
- ใช้ transport/storage protection ของ OS และ Kaggle platform
- ห้าม network download ระหว่าง submission execution

**Cascading implications:** API หลักเป็น typed in-process API และ CLI; errors ต้อง structured/redacted/fail closed และ exporter ต้องแยก trusted training artifacts ออกจาก submission bundle

### API & Communication Patterns

**Selected approach:** Typed in-process API + compatibility adapters; ไม่มี network service ใน MVP

**Canonical boundaries:**

- Rust core เป็นเจ้าของ `Engine`, `GameState`, `JointAction` และ `StepResult`
- Python binding เปิด opaque `PyEngine` พร้อม `reset`, `step`, `snapshot`, `restore` และ `replay`
- RL layer เป็นเจ้าของ `Observation`, `MacroAction`, `ActionMask`, `ExecutionReport` และ training contracts
- ทุก boundary object มี `schema_version` และ engine/build identity

**Hot rollout path:**

- ใช้ in-process PyO3 เท่านั้น
- ส่ง dense features/masks เป็น NumPy arrays และ metadata ขนาดเล็กเป็น typed mappings
- ห้าม serialize JSON ทุก turn
- PyO3 engine step ต้อง release GIL เมื่อไม่แตะ Python objects
- Tensor-native batched ABI ยังเป็น post-MVP

**Compatibility and debug path:**

- เก็บ canonical JSON/JSONL สำหรับ parity, deterministic replay และ first-divergence reports
- Kaggle adapter แปลง Kaggle JSON ↔ typed domain objects
- Gymnasium 1.3.0 เป็น optional adapter ไม่ใช่ canonical domain contract
- Environment API ใช้ semantics แบบ `observation, reward, terminated, truncated, info`

**CLI contract:**

- ใช้ Typer 0.27.1 สำหรับ `parity`, `train`, `evaluate` และ `export`
- Resume ใน MVP ใช้ `train --resume <checkpoint>`; แยกคำสั่ง `resume` และ `compare` เป็น post-MVP
- Human progress/logs เขียน `stderr`
- Machine-readable result เขียน `stdout` ผ่าน `--format json`
- ทุกคำสั่งคืน stable exit code และ structured error envelope
- ไม่มี REST, GraphQL, gRPC หรือ message broker ใน MVP

**Schema and versioning:**

- Python validation ใช้ Pydantic strict models
- Rust ใช้ Serde types พร้อม reject unknown variants/invalid fields
- Canonical schemas เก็บใต้ `schemas/v1/`
- Wire enums ใช้ stable string names ไม่ใช้ numeric ordinal
- Breaking change ต้องเพิ่ม major schema version
- Reader รองรับเฉพาะ versions ที่ประกาศ; incompatible input ต้อง fail พร้อม migration guidance

**Error contract:**

```json
{
  "schema_version": 1,
  "ok": false,
  "error": {
    "code": "KG-PARITY-DIVERGENCE",
    "message": "Human-readable summary",
    "run_id": "…",
    "details": {}
  }
}
```

- Rust errors map เป็น typed Python exceptions และ CLI error codes
- Error details ต้อง deterministic, redacted และระบุ first failing state เมื่อเกี่ยวข้อง
- ห้ามใช้ exception text เป็น machine contract

**Execution and parallelism:**

- API เป็น synchronous และ deterministic
- Parallelism อยู่ระดับ worker process แยกตาม seed/opponent
- Worker เขียน immutable partial artifacts
- Reducer รวมผลตาม stable key ordering
- ไม่ใช้ shared mutable engine ระหว่าง workers

**Version baseline verified on 2026-08-15:**

| Technology | Verified current | Role |
|---|---:|---|
| PyTorch | 2.13.0 | Policy/BC/PPO; pin ต่อ experiment |
| Gymnasium | 1.3.0 | Optional compatibility adapter |
| Typer | 0.27.1 | Local CLI |
| NumPy | 2.5.2 | Observation/mask arrays |
| PyYAML | 6.0.3 | Safe YAML loading ก่อน Pydantic validation |

แหล่งตรวจสอบ: [PyTorch](https://pypi.org/project/torch/), [Gymnasium](https://pypi.org/project/gymnasium/), [Typer](https://pypi.org/project/typer/), [NumPy](https://pypi.org/project/numpy/), [PyYAML](https://pypi.org/project/PyYAML/)

**Cascading implications:** Infrastructure ต้อง build/test ทั้ง JSON parity path และ PyO3 hot path; worker isolation กับ deterministic reduction เป็น execution contract และ Kaggle preflight ต้องตรวจ schema/runtime/dependency compatibility

### Infrastructure & Deployment

**Selected approach:** Provider-neutral local quality gates + Kaggle GPU execution; CI provider เป็น adapter ที่เรียก canonical scripts

**Execution environments:**

| Environment | Purpose |
|---|---|
| macOS arm64 | Development, unit tests, parity smoke, light training |
| Linux x86_64 CPU | Reference CI, wheel build, parity และ export preflight |
| Local GPU | PPO smoke training |
| Kaggle GPU | Full training และ multi-seed workloads |
| Kaggle submission runtime | Offline promoted-agent execution; constraints ยังเป็น `TBD` |

Experiment config, schema versions และ resolved config digest ต้องเหมือนกันทุก environment

**CI quality gates:**

1. Fast gate: formatting, lint, type checking, Rust/Python unit tests
2. Integration gate: deterministic scenarios, PyO3 import, JSON-vs-hot-path equivalence
3. Release gate: 100,000 randomized parity turns, full-season traces, `50×` benchmark, supply-chain checks
4. Promotion gate: three training seeds, immutable held-out evaluation และ artifact completeness
5. Full training ไม่รันใน CI; CI ตรวจเฉพาะ smoke workflow

CI commands ต้องเป็น provider-neutral scripts ก่อน แล้ว workflow provider เป็นเพียงตัวเรียก scripts เหล่านั้น

**Build and environment reproducibility:**

- Pin Rust ด้วย `rust-toolchain.toml`
- ใช้ `Cargo.lock` และ `uv.lock`
- Build native package ด้วย maturin บน Linux x86_64
- Wheel target/ABI ตัดสินหลังยืนยัน Kaggle runtime
- ทุก run บันทึก OS, CPU/GPU, memory, Python, Rust, CUDA, dependencies และ engine checksum
- Environment variables ใช้เฉพาะ secrets/resource paths; semantic config overrides ต้องถูกบันทึกใน resolved config

**Parallelism and scaling:**

- ใช้ process-level workers แยกตาม seed/opponent
- Start method เป็น `spawn`; ไม่แชร์ mutable engine state
- CPU rollout workers และ GPU learner แยก lifecycle
- Worker count และ memory budget ระบุใน experiment config
- Reducer รวม shards ตาม stable ordering
- ไม่ใช้ Ray, Dask, Kubernetes หรือ distributed scheduler ใน MVP

**Observability and recovery:**

- Structured run events เป็น JSONL
- Metrics และ resource samples เป็น Parquet
- ใช้ psutil และ PyTorch CUDA metrics สำหรับ CPU/GPU/memory
- ทุก stage มี heartbeat, status และ failure record
- Checkpoint เขียนแบบ atomic และ resume ต้องรักษา run identity
- ไม่มี external telemetry/monitoring service

**Kaggle deployment:**

- Heavy run ใช้ CLI และ experiment config เดียวกับ local
- Kaggle-specific notebook/script ทำหน้าที่ bootstrap environment เท่านั้น ห้ามเปลี่ยน semantics
- Export bundle มีเฉพาะ promoted weights, executor, schemas, engine/binding และ manifest
- Submission preflight รันใน clean offline environment
- Export ต้องตรวจ checksum, imports, dependency inventory, secrets และ runtime budget
- Official Kaggle runtime/memory/dependency limits ยังเป็น release blocker

**Tooling baseline verified on 2026-08-15:**

| Tool | Verified current | Role |
|---|---:|---|
| pytest | 9.1.1 | Python tests |
| Ruff | 0.16.3 | Formatting/linting |
| mypy | 2.3.1 | Static type checking |
| psutil | 7.2.2 | Resource metrics |

แหล่งตรวจสอบ: [pytest](https://pypi.org/project/pytest/), [Ruff](https://pypi.org/project/ruff/), [mypy](https://pypi.org/project/mypy/), [psutil](https://pypi.org/project/psutil/)

### Frontend Architecture

ไม่เกี่ยวข้องกับ MVP; ไม่มี dashboard หรือ public UI การใช้งานเป็น CLI/API-only developer experience

### Decision Priority Analysis

**Critical Decisions — Block Implementation**

- รักษา Rust engine เป็น deterministic simulation source of truth
- ใช้ Python/Kaggle environment เป็น behavioral oracle และ fallback
- Versioned state/action/config/artifact schemas
- PyO3 typed in-process hot path แยกจาก JSON parity path
- Semantic macro policy + deterministic executor + joint reservation
- Immutable filesystem artifacts และ rebuildable SQLite registry
- Strict checkpoint trust boundary และ atomic writes
- Process-isolated workers กับ deterministic reduction
- Local/Kaggle experiment semantics เดียวกัน
- Parity, promotion และ Kaggle compatibility gates

**Important Decisions — Shape Architecture**

- Parquet สำหรับ rollout/evaluation tables
- DuckDB เป็น read-only analytical tool
- Safetensors สำหรับ promoted model weights
- Gymnasium เป็น optional adapter
- Typer CLI พร้อม JSON output contract
- Provider-neutral CI scripts
- Resource telemetry ผ่าน psutil/PyTorch
- Dependency, license และ vulnerability checks

**Deferred Decisions — Post-MVP**

- PyO3 major-version migration หลังสร้าง parity regression suite
- Tensor-native batched ABI
- PostgreSQL/object storage
- Remote API authentication/RBAC
- Digital artifact signatures
- Ray/Dask/Kubernetes
- Automated opponent league/PSRO
- Dashboard/public UI
- Windows/PyPI distribution

### Decision Impact Analysis

**Implementation Sequence**

1. Bootstrap root uv package/workspace และ pin Rust/Python toolchains
2. Freeze Python oracle, Rust engine และ existing parity baseline
3. กำหนด canonical schemas, config normalization และ artifact identity
4. สร้าง deterministic parity/replay harness และ release gates
5. เพิ่ม typed PyO3 hot path โดยคง JSON compatibility path
6. สร้าง semantic observation/action masks และ deterministic executor
7. เพิ่ม joint reservation, legal fallback และ execution telemetry
8. สร้าง run registry, immutable artifacts และ atomic checkpoints
9. สร้าง demonstrations/BC pipeline แล้วจึง minimal masked PPO
10. สร้าง immutable splits, cross-play evaluator และ promotion gates
11. สร้าง champion registry, Kaggle exporter และ offline preflight
12. ยืนยัน official Kaggle constraints และผ่าน release suite ทั้งหมด

**Cross-Component Dependencies**

- Schema versions และ config digest เป็น identity ร่วมของทุก subsystem
- Rust engine version ใช้ฝึกไม่ได้จนกว่าจะผ่าน parity gate
- Policy สร้าง semantic targets เท่านั้น; executor เป็นเจ้าของ primitive legality
- Training และ evaluation ใช้ executor contract เดียวกัน
- Checkpoint ต้องอ้าง engine, schema, config, seed และ opponent-pool versions
- Evaluator อ่านเฉพาะ immutable candidate artifacts และ held-out registry
- Exporter รับเฉพาะ promoted champion ที่ artifacts ครบ
- Workers เขียน partial artifacts แยกกัน; reducer รวมด้วย stable ordering
- Kaggle bootstrap เปลี่ยน resources ได้ แต่ห้ามเปลี่ยน experiment semantics

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

พบ conflict points หลัก 24 จุด ครอบคลุม schema fields, enum serialization, identifiers, database naming, files/modules, tests, configs, fixtures, response/error envelopes, timestamps, units, nulls, ordering, events, logs, state mutation, validation, retries, checkpoints และ pipeline status

### Naming Patterns

**Database Naming Conventions**

- Tables: plural `snake_case` เช่น `runs`, `artifacts`, `evaluation_results`
- Columns: `snake_case`
- Primary/foreign keys: `<entity>_id`
- Indexes: `idx_<table>__<column_list>`
- Unique constraints: `uq_<table>__<column_list>`
- ห้ามใช้ ORM-generated names เป็น persisted contract

**API and Schema Naming**

- JSON, YAML, Parquet และ CLI JSON fields ใช้ `snake_case`
- Kaggle primitive operations และ wire enum values ใช้ `UPPER_SNAKE_CASE`
- Rust enum variants ใช้ `PascalCase` แล้ว serialize เป็น `UPPER_SNAKE_CASE`
- CLI commands/options ใช้ `kebab-case`
- Error codes ใช้ `KG-<DOMAIN>-<REASON>`

**Code Naming**

- Python files/functions/variables: `snake_case`
- Python classes/protocols: `PascalCase`
- Rust modules/functions/variables: `snake_case`
- Rust structs/enums/traits: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- `run_id`: `run_<uuid4-hex>`
- Artifact digest: `sha256:<64-lowercase-hex>`
- ห้ามใช้คำกว้างอย่าง `data`, `manager`, `helper` เมื่อมีชื่อ domain ที่ชัดกว่า

### Structure Patterns

**Project Organization**

- จัด Python ตาม domain: `schemas`, `executor`, `policy`, `training`, `evaluation`, `artifacts`, `cli`
- Rust simulation rules อยู่ใน `kaggriculture-engine` เท่านั้น
- PyO3 conversion/error mapping อยู่ใน adapter crate เท่านั้น
- ห้ามสร้าง `utils.py` หรือ `common.rs` เป็นที่รวมโค้ดไร้เจ้าของ
- Dependency direction: CLI → application workflow → domain → adapters
- Domain layer ห้าม import CLI, filesystem registry หรือ Kaggle-specific bootstrap

**Tests**

- Python: `tests/<domain>/test_<subject>.py`
- Rust unit testsอยู่ร่วม module ด้วย `#[cfg(test)]`
- Rust integration testsอยู่ `<crate>/tests/<subject>.rs`
- Golden parity fixturesอยู่ `tests/fixtures/<schema_version>/`
- ห้ามแก้ golden fixture เพื่อทำให้ test ผ่านโดยไม่มี oracle evidence และเหตุผลใน review

**Configuration and Documentation**

- Authoring configs อยู่ `configs/<workflow>/`
- Resolved config ถูกเขียนเป็น immutable `resolved_config.json` ใน run artifact
- Canonical schemas อยู่ `schemas/v<major>/`
- Architecture/contract docs อยู่ `docs/architecture/`
- Runtime output ห้ามเขียนปะปน source directories

### Format Patterns

**CLI Response Envelope**

```json
{
  "schema_version": 1,
  "ok": true,
  "data": {}
}
```

Errors ใช้ envelope เดิมโดยเปลี่ยนเป็น `ok: false` และเพิ่ม `error.code`, `message`, `run_id`, `details`

**Data Exchange**

- Timestamp: RFC 3339 UTC ลงท้าย `Z`
- Duration: integer พร้อมหน่วยในชื่อ เช่น `elapsed_ms`
- Game money/counts/seeds: integer
- Ratios/scores: finite floating point
- ห้ามเขียน `NaN` หรือ `Infinity` ใน JSON/manifests
- Boolean ใช้ `true`/`false`
- Optional stable field ต้องมีค่า `null`; ห้ามใช้ empty string แทน missing
- Collection ที่ไม่มี semantic ordering ต้อง sort ก่อน serialize/hash
- Hands, market orders และ action traces ต้องรักษา domain ordering

### Communication Patterns

**Events and Logs**

- Event names: `<domain>.<past_tense>` เช่น `run.started`, `stage.completed`, `executor.blocked`
- ทุก event มี `schema_version`, `run_id`, `sequence`, `timestamp`, `event_type`, `payload`
- ใช้ monotonic `sequence` สำหรับ ordering; timestamp ใช้แสดงเวลาเท่านั้น
- Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- Human logs ไป `stderr`; machine result ไป `stdout`
- ห้าม parse human log text เป็น API contract

**State Management**

- Engine state เปลี่ยนได้ผ่าน `step` เท่านั้น
- Observation, masks และ snapshots ที่ออกจาก engine ถือเป็น immutable
- Resolved experiment config immutable หลัง run เริ่ม
- Reservation ledger มีขอบเขตหนึ่ง primitive turn และมี owner เดียว
- Policy ห้ามแก้ action mask หรือ executor state
- Held-out registry ห้ามถูกส่งเข้า training process

### Process Patterns

**Validation**

- Validate ที่ config load, artifact read, PyO3 boundary, checkpoint resume และ export
- Reject unknown schema versions และ enum variants
- ห้าม silently coerce string เป็น number หรือ malformed action เป็น valid target
- Legal fallback ต้องปรากฏใน `ExecutionReport`; ห้ามกลืนเป็น no-op เงียบ ๆ

**Errors and Retries**

- Domain/parity/schema errors ไม่ retry
- Retry เฉพาะ transient filesystem/process failures
- จำนวนครั้งและ backoff ต้องอยู่ใน resolved config
- ทุก retry สร้าง structured event
- First deterministic failure ต้องถูกเก็บเป็น diagnostic artifact

**Checkpoint and Stage State**

- Checkpoint: temp write → flush/fsync → checksum → atomic rename → manifest commit
- Stage states: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`
- ห้ามเปลี่ยนจาก terminal state กลับเป็น `RUNNING`
- Resume สร้าง attempt ใหม่ภายใต้ `run_id` เดิม
- Reducer อ่านเฉพาะ shards ที่สถานะ `SUCCEEDED` และ checksum ถูกต้อง

### Enforcement Guidelines

Agent ทุกตัวต้อง:

- ตรวจ architecture และ canonical schema ก่อนเพิ่ม public type
- ใช้ naming/error/event envelopes ที่กำหนด
- เพิ่ม contract test เมื่อเปลี่ยน boundary
- รักษา deterministic ordering ก่อน serialize/hash
- ไม่เพิ่ม dependency, fallback หรือ schema field แบบเงียบ ๆ
- ไม่แก้ oracle/golden data เพื่อกลบ mismatch
- แยก policy strategy ออกจาก executor legality

การบังคับใช้:

- Python: Ruff, mypy, pytest และ schema contract tests
- Rust: `cargo fmt`, `cargo clippy`, `cargo test`
- CI ตรวจ generated schemas, golden fixtures และ resolved-config determinism
- Pattern change ต้องแก้ architecture ก่อน แล้วจึงเปลี่ยน implementation
- การละเมิดต้องบันทึกใน review พร้อมเหตุผลและ migration impact

### Pattern Examples

**Good**

```python
event = {
    "schema_version": 1,
    "run_id": run_id,
    "sequence": 42,
    "event_type": "executor.blocked",
    "payload": {"blocked_reason": "INSUFFICIENT_SEED"},
}
```

```rust
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum MacroTarget {
    CropCycle,
    RecoverLiquidity,
}
```

**Anti-Patterns**

- `utils.py` ที่รวม config, hashing และ executor logic
- ใช้ timestamp เป็นลำดับ event
- ส่ง `None`, `""` และ missing field ด้วยความหมายเดียวกัน
- Retry parity mismatch
- โหลด checkpoint ภายนอกด้วย pickle
- เปลี่ยน config ผ่าน environment variableโดยไม่บันทึก resolved config
- ให้ policy ส่ง primitive action เพื่อเลี่ยง executor contract
- ใช้ JSON serialization บน PyO3 hot path

## Project Structure & Boundaries

สัญลักษณ์: `[E]` มีอยู่แล้ว, `[P]` ต้องสร้าง, `[G]` generated/ignored

### Complete Project Directory Structure

```text
Kaggriculture/
├── Cargo.toml                              [E] Rust workspace
├── Cargo.lock                              [E]
├── rust-toolchain.toml                     [P]
├── pyproject.toml                          [P] Python control plane
├── uv.lock                                 [P]
├── .python-version                         [P]
├── deny.toml                               [P]
├── .env.example                            [P] ชื่อ secrets เท่านั้น
├── .gitignore                              [E] ต้องเพิ่ม artifacts/dist
├── README.md                               [P]
├── AGENTS.md                               [E]
│
├── rust/
│   ├── kaggriculture-engine/               [E] Game simulation เท่านั้น
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── examples/
│   │   │   └── throughput.rs
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs
│   │       ├── action.rs
│   │       ├── rules.rs
│   │       ├── engine.rs
│   │       ├── python_random.rs
│   │       ├── snapshot.rs                [P]
│   │       ├── inspection.rs              [P]
│   │       ├── error.rs                   [P]
│   │       ├── version.rs                 [P]
│   │       └── bin/
│   │           └── kg-engine.rs
│   │
│   ├── kaggriculture-executor/             [P] Agent execution; ไม่ใช่ game rules
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── target.rs
│   │   │   ├── planner.rs
│   │   │   ├── reservations.rs
│   │   │   ├── legality.rs
│   │   │   ├── fallback.rs
│   │   │   └── report.rs
│   │   └── tests/
│   │       ├── joint_reservations.rs
│   │       └── legal_fallbacks.rs
│   │
│   └── kaggriculture-python/               [E] PyO3 adapter
│       ├── Cargo.toml
│       ├── build.rs
│       └── src/
│           ├── lib.rs
│           ├── engine.rs                   [P]
│           ├── executor.rs                 [P]
│           ├── conversion.rs               [P]
│           ├── errors.rs                   [P]
│           └── version.rs                  [P]
│
├── bindings/python/                        [E]
│   ├── pyproject.toml
│   ├── kaggriculture_rs/
│   │   ├── __init__.py
│   │   ├── _native.pyi                     [P]
│   │   └── py.typed                        [P]
│   └── tests/                              [P]
│       ├── test_import.py
│       ├── test_engine_contract.py
│       └── test_executor_contract.py
│
├── src/kaggriculture_rl/                   [P] Python RL control plane
│   ├── __init__.py
│   ├── __main__.py
│   ├── version.py
│   ├── cli/
│   │   ├── app.py
│   │   ├── parity.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── output.py
│   ├── workflows/
│   │   ├── parity.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   └── export.py
│   ├── schemas/
│   │   ├── base.py
│   │   ├── experiment.py
│   │   ├── observation.py
│   │   ├── actions.py
│   │   ├── execution.py
│   │   ├── artifacts.py
│   │   ├── events.py
│   │   └── evaluation.py
│   ├── config/
│   │   ├── loader.py
│   │   ├── canonical.py
│   │   └── migrations.py
│   ├── engine/
│   │   ├── protocols.py
│   │   ├── native.py
│   │   ├── oracle.py
│   │   ├── kaggle_adapter.py
│   │   ├── parity.py
│   │   └── replay.py
│   ├── executor/
│   │   ├── protocols.py
│   │   └── native_adapter.py
│   ├── policy/
│   │   ├── model.py
│   │   ├── heads.py
│   │   ├── masking.py
│   │   └── inference.py
│   ├── training/
│   │   ├── datasets.py
│   │   ├── demonstrations.py
│   │   ├── behavior_cloning.py
│   │   ├── ppo.py
│   │   ├── rewards.py
│   │   ├── checkpointing.py
│   │   └── runner.py
│   ├── evaluation/
│   │   ├── splits.py
│   │   ├── opponents.py
│   │   ├── cross_play.py
│   │   ├── metrics.py
│   │   ├── promotion.py
│   │   └── reducer.py
│   ├── artifacts/
│   │   ├── identity.py
│   │   ├── manifest.py
│   │   ├── store.py
│   │   ├── registry.py
│   │   ├── checksums.py
│   │   └── migrations/
│   │       └── 001_initial.sql
│   ├── observability/
│   │   ├── events.py
│   │   ├── logging.py
│   │   └── resources.py
│   └── export/
│       ├── bundle.py
│       ├── preflight.py
│       └── agent_entry.py
│
├── schemas/v1/                             [P]
│   ├── experiment.schema.json
│   ├── observation.schema.json
│   ├── macro_action.schema.json
│   ├── joint_action.schema.json
│   ├── execution_report.schema.json
│   ├── artifact_manifest.schema.json
│   ├── checkpoint_manifest.schema.json
│   ├── event.schema.json
│   ├── evaluation_report.schema.json
│   └── cli_result.schema.json
│
├── configs/                                [P]
│   ├── parity/
│   │   ├── smoke.yaml
│   │   └── release.yaml
│   ├── train/
│   │   ├── bc_smoke.yaml
│   │   ├── ppo_smoke.yaml
│   │   └── ppo_full.yaml
│   ├── evaluate/
│   │   ├── validation.yaml
│   │   └── held_out.yaml
│   ├── opponents/
│   │   └── fixed_mix_v1.yaml
│   └── export/
│       └── kaggle.yaml
│
├── registries/                             [P] Versioned immutable inputs
│   ├── splits/
│   │   ├── train_v1.json
│   │   ├── validation_v1.json
│   │   └── held_out_v1.json
│   └── opponents/
│       ├── training_v1.json
│       └── held_out_v1.json
│
├── tests/                                  [P]
│   ├── contracts/
│   ├── parity/
│   ├── executor/
│   ├── training/
│   ├── evaluation/
│   ├── artifacts/
│   ├── export/
│   ├── integration/
│   │   └── test_smoke_pipeline.py
│   └── fixtures/v1/
│       ├── deterministic_scenarios/
│       ├── divergence_cases/
│       └── manifest.json
│
├── scripts/
│   ├── compare_rust_engine.py              [E] เปลี่ยนเป็น CLI wrapper
│   ├── benchmark_engines.py                [E] เปลี่ยนเป็น CLI wrapper
│   ├── ci/                                 [P]
│   │   ├── fast.sh
│   │   ├── integration.sh
│   │   └── release.sh
│   └── kaggle/                             [P]
│       ├── bootstrap_train.py
│       └── bootstrap_evaluate.py
│
├── kaggle/                                 [P]
│   ├── README.md
│   ├── kernel-metadata.json
│   └── submission-template/
│       └── agent.py
│
├── docs/
│   ├── architecture/
│   │   └── README.md                       [P] Link ไป planning artifact
│   ├── adr/                                [P]
│   ├── contracts/                          [P]
│   ├── examples/                           [P]
│   ├── troubleshooting.md                  [P]
│   └── migration.md                        [P]
│
├── artifacts/                              [G]
│   ├── registry.sqlite
│   ├── runs/<run_id>/
│   ├── champions/
│   └── cache/
├── dist/                                   [G]
├── target/                                 [G]
├── _bmad/                                  [E]
└── _bmad-output/planning-artifacts/
    ├── prd.md                              [E]
    ├── architecture.md                     [E]
    └── implementation-readiness-report-2026-08-15.md [E]
```

### Architectural Boundaries

**Rust dependency direction**

```text
kaggriculture-engine
        ↑
kaggriculture-executor
        ↑
kaggriculture-python
```

- Engine ห้าม import executor หรือ Python
- Executor ใช้ public typed engine API เท่านั้น
- PyO3 adapter แปลง types/errors แต่ไม่มี simulation หรือ planning rules

**Python dependency direction**

```text
CLI → workflows → policy/training/evaluation
                    ↓
         engine/executor protocols
                    ↓
              native adapters
```

- CLI ห้ามมี domain logic
- Policy ห้ามเรียก private Rust implementation
- Python executor package เป็น protocol/adapter; planning และ reservation implementation อยู่ Rust
- Kaggle bootstrap ห้ามกำหนด training semantics

**Data Boundaries**

- Immutable manifests/artifacts เป็น source of truth
- SQLite เป็น rebuildable index
- Held-out registriesเปิดให้ evaluation process เท่านั้น
- DuckDB อ่าน Parquet โดยไม่แก้ artifacts
- Training workersเขียนเฉพาะ run directory ของตนเอง
- Champion pointer เปลี่ยนผ่าน promotion transaction เท่านั้น

### Requirements Structure Mapping

| Requirements | Primary location |
|---|---|
| FR1–FR5, FR23–FR24 | `config/`, `workflows/`, `artifacts/`, `training/` |
| FR6–FR14 | `schemas/actions.py`, `kaggriculture-executor`, `executor/` |
| FR15–FR24 | `policy/`, `training/`, `configs/train/` |
| FR25–FR30 | `kaggriculture-engine`, `engine/parity.py`, `tests/parity/` |
| FR31–FR38 | `evaluation/`, `registries/`, `configs/evaluate/` |
| FR39–FR44 | `observability/`, `artifacts/`, execution reports |
| FR45–FR50 | `export/`, `cli/`, `schemas/`, `docs/` |
| NFR1–NFR6 | Rust benchmarks, parity/replay tests |
| NFR7–NFR14 | Training/evaluation gates, artifacts/checkpoints |
| NFR15–NFR19 | Binding contracts, schemas, environment matrix |
| NFR20–NFR23 | Security preflight, lockfiles, supply-chain checks |
| NFR24–NFR28 | CI scripts, Kaggle bootstrap, deterministic reducer |

### Integration Points

**Internal flow**

```text
Experiment YAML
  → strict validation
  → resolved config + digest
  → engine/executor rollout
  → events/traces/checkpoints
  → cross-play evaluation
  → promotion transaction
  → champion export
  → offline Kaggle preflight
```

**External integrations**

- Kaggle Python environment ผ่าน `engine/oracle.py`
- Kaggle observation/action ผ่าน `engine/kaggle_adapter.py`
- Kaggle GPU bootstrap ผ่าน `scripts/kaggle/`
- Kaggle Secrets อ่านเฉพาะ runtime boundary
- ไม่มี database, telemetry หรือ model-registry service ภายนอกใน MVP

### File Organization Patterns

- Production Python code อยู่ `src/kaggriculture_rl`
- Rust domain code อยู่ crate ตาม ownership
- Compatibility scripts delegate เข้า CLI ห้าม duplicate logic
- Tests แยกตาม domain และมี integration suite กลาง
- Large generated traces/checkpoints อยู่ `artifacts/` และไม่ commit
- Golden fixtures ต้องมี manifest/checksum และมีขนาดพอสำหรับ regression เท่านั้น
- BMad artifacts เก็บใน `_bmad-output`; ADRs สำหรับ implementation อยู่ `docs/adr`

### Development Workflow Integration

- `uv run` เรียก Python CLI/tests
- `cargo` เรียก Rust checks/tests
- maturin build/develop เชื่อม binding package
- Fast/integration/release scripts เป็น canonical CI entrypoints
- Kaggle scripts ทำ bootstrap แล้วเรียก CLI เดียวกับ local
- Exporter สร้าง `dist/` จาก promoted champion เท่านั้น

## Architecture Validation Results

### Coherence Validation

#### Decision Compatibility

✅ การตัดสินใจหลักทำงานร่วมกันได้โดยไม่มี dependency cycle:

- ลำดับ dependency คือ `kaggriculture-engine <- kaggriculture-executor <- kaggriculture-python`
- Python policy เลือก semantic target; Rust executor รับผิดชอบ joint reservation, legality และ deterministic fallback
- Rust engine เป็นส่วนเดียวที่ mutate `GameState`
- Typed PyO3 call เป็น hot path; JSON/JSONL ใช้เฉพาะ parity, replay และ debugging
- PyO3 `0.23.5` เดิมยังคงอยู่จนกว่า parity suite จะผ่านก่อนแยก migration ไปสู่รุ่นใหม่
- Local CPU/GPU และ Kaggle GPU ใช้ semantics ชุดเดียวกัน โดยเปลี่ยนได้เฉพาะ resource configuration

ชุด dependency รุ่นปัจจุบันที่บันทึกไว้เป็นข้อมูลตั้งต้น ไม่ถือว่าผ่าน compatibility จนกว่า `uv lock`, import smoke test และ binding build จะผ่านบน compatibility matrix ที่กำหนด

#### Pattern Consistency

✅ Naming, versioned schema, error envelope, event ordering, seed handling, atomic checkpoint และ deterministic reduction ใช้กฎเดียวกันทั้งระบบ

✅ ห้าม silent fallback หรือ retry เมื่อเกิด parity, schema หรือ integrity error

✅ Workers แยกตาม seed/opponent และไม่แชร์ mutable training state ข้าม process

#### Structure Alignment

✅ Project structure แยก engine, executor, binding, policy, training, evaluation, artifact store และ export ตาม architectural boundaries

✅ MVP ไม่เพิ่ม REST API, message broker, dashboard, Kubernetes หรือ distributed framework ที่ไม่จำเป็น

### Requirements Coverage Validation

#### Feature Coverage

ตรวจหมายเลข requirements จาก PRD ครบถ้วน:

| Requirement | พบใน PRD | Architecture coverage |
|---|---:|---:|
| Functional Requirements | 50/50 | 50/50 |
| Non-Functional Requirements | 28/28 | 28/28 |

#### Functional Requirements Coverage

- FR1-FR5 และ FR23-FR24: configuration, workflows, training และ artifacts
- FR6-FR14: semantic actions และ deterministic Rust executor
- FR15-FR24: policy และ RL training
- FR25-FR30: Rust engine และ parity validation
- FR31-FR38: evaluation, promotion และ opponent registry
- FR39-FR44: observability, traces และ execution reports
- FR45-FR50: export, CLI, schemas และ documentation

#### Non-Functional Requirements Coverage

- NFR1-NFR6: engine performance benchmarks และ parity tests
- NFR7-NFR14: reproducibility, evaluation, artifact integrity และ checkpoints
- NFR15-NFR19: binding, schema และ environment compatibility matrix
- NFR20-NFR23: secret handling, dependency locks และ supply-chain checks
- NFR24-NFR28: CI, Kaggle portability และ deterministic aggregation

### Implementation Readiness Validation

#### Decision Completeness

✅ Critical decisions ระบุ owner, boundary, persistence format, communication path และ validation gate แล้ว

✅ Decision ที่ตั้งใจ defer เช่น PyO3 migration และ automated opponent league มีเงื่อนไขก่อนเริ่มชัดเจน

#### Structure Completeness

✅ ระบุ workspace tree, crate/package ownership, integration points, test locations, artifact boundaries และ CLI entrypoints แล้ว

สิ่งที่ยังไม่มีใน checkout เช่น root Python package และ `kaggriculture-executor` เป็น implementation work ที่กำหนดตำแหน่งไว้แล้ว ไม่ใช่ช่องว่างของ architecture

#### Pattern Completeness

✅ มี enforceable rules สำหรับ schema, serialization, errors, event ordering, retries, concurrency, checkpoints และ tests

✅ Common conflict points ระหว่าง Rust/Python และ local/Kaggle มี single source of truth และ release gate

### Gap Analysis Results

#### Critical Gaps

ไม่มี critical gap ที่ขวางการทำ epic/story decomposition หรือการเริ่ม core MVP implementation

#### Release Blockers

- Official Kaggle Python ABI, dependency policy, memory limit และ execution timeout ยังเป็น TBD
- ต้องยืนยันข้อจำกัดเหล่านี้และผ่าน export smoke test ก่อนสร้าง submission release แต่ไม่ขวาง engine, executor และ local training pipeline

#### Important Gaps Resolved by This Validation

1. `schemas/v1/*.schema.json` เป็น canonical wire contract
2. Pydantic สร้าง candidate schema และ CI ตรวจ diff กับ canonical schema
3. Rust Serde types ตรวจ contract ด้วย golden fixtures ชุดเดียวกัน; MVP ไม่มี bidirectional code generation
4. Coordinator เป็นผู้เขียน `registry.sqlite` เพียงรายเดียว; workers เขียนเฉพาะ immutable run artifacts
5. Executor สร้าง joint primitive action plan จาก immutable snapshot; engine เท่านั้นที่ apply action และ mutate state
6. `uv lock`, binding build/import smoke test และ parity suite เป็น bootstrap acceptance gates
7. Kaggle bootstrap เปลี่ยน resource configuration ได้ แต่ห้ามเปลี่ยน environment, executor หรือ policy semantics
8. Internal optimizer state ต้อง versioned และ trusted-only; submission artifact ใช้ Safetensors และข้อมูลที่ตรวจสอบได้เท่านั้น

#### Nice-to-Have Gaps

- ADR สำหรับ major decisions
- Sequence diagrams สำหรับ training, evaluation และ promotion
- Default resource budgets สำหรับ local/Kaggle profiles
- Benchmark fixture specification ที่ละเอียดขึ้น

### Validation Issues Addressed

- ปิดความกำกวมเรื่อง canonical schema ownership ด้วย versioned JSON Schema และ cross-language golden fixtures
- ปิดความกำกวมเรื่อง concurrent metadata writes ด้วย single-writer coordinator
- ปิดความกำกวมเรื่อง executor/engine state ownership โดยให้ engine เป็น mutation authority เพียงส่วนเดียว
- แยก latest-version discovery ออกจาก compatibility proof; lock และ import/build tests เป็นหลักฐานบังคับ
- แยก core implementation readiness ออกจาก Kaggle release readiness อย่างชัดเจน

### Architecture Completeness Checklist

#### Requirements Analysis

- [x] Project context analyzed
- [x] Scale and complexity assessed
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped

#### Architectural Decisions

- [x] Critical decisions documented with versions or explicit validation gates
- [x] Rust/Python boundaries and dependency direction specified
- [x] Data persistence and artifact strategy specified
- [x] Security and supply-chain controls specified
- [x] Infrastructure and Kaggle deployment approach specified

#### Implementation Patterns

- [x] Naming, schema and serialization conventions defined
- [x] Error and event contracts defined
- [x] Determinism, concurrency and checkpoint rules defined
- [x] Test and enforcement patterns defined

#### Project Structure

- [x] Complete target directory structure defined
- [x] Component and integration boundaries defined
- [x] FR1-FR50 mapped to architectural locations
- [x] NFR1-NFR28 mapped to validation mechanisms

### Architecture Readiness Assessment

**Overall Status:** READY FOR EPIC/STORY DECOMPOSITION; CONDITIONALLY READY FOR MVP IMPLEMENTATION

**Confidence Level:** Medium-High

**Key Strengths:**

- Deterministic mechanics แยกจาก adaptive RL อย่างชัดเจน
- รองรับ speed, Kaggle fidelity, win rate และ worst-case score เป็น first-class evaluation concerns
- Schema และ golden fixtures ลด contract drift ระหว่าง Rust/Python
- Evaluation isolation และ deterministic promotion ลด leakage และผลลัพธ์ที่ทำซ้ำไม่ได้
- MVP scope หลีกเลี่ยง infrastructure ที่หนักเกินความจำเป็น

**Required Gates:**

1. ก่อน baseline training: root package bootstrap, dependency lock, binding import และ parity suite ต้องผ่าน
2. ก่อน implementation scheduling: สร้าง epics/stories พร้อม acceptance criteria และ dependency order
3. ก่อน Kaggle release: ยืนยัน official runtime constraints และผ่าน Kaggle-compatible export smoke test
