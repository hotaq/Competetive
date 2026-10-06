# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

**Reply to the user in Thai.** Every user-facing message — explanations, summaries, questions,
plans — is written in Thai. This matches the repo's own decision records (`RUST_ENGINE_MIGRATION.md`,
`ACTION_SPACE.md`, `research/*.md`), which are already Thai prose with English technical terms.

Keep in English: code, identifiers, file paths, shell commands, commit messages, and technical
terms that have no settled Thai equivalent (tape, seed, seat, margin, parity, rollout, gate,
standalone, closed-loop). Do not translate them mid-sentence — mixing them into Thai prose is the
established style here.

## What this repo is

An agent-development repo for the Kaggle **Kaggriculture** competition: a 2-player,
simultaneous-move farming game (24 turns/day × 30 days = 720 turns). The deliverable is a
single `main.py` uploaded to Kaggle; everything else here exists to *decide which agent to
upload*. The optimisation target is **live leaderboard rating**, not any local score — see
"Evidence discipline" below, because local wins have repeatedly failed to transfer.

Most prose docs (`RUST_ENGINE_MIGRATION.md`, `ACTION_SPACE.md`, `STATE_RULE_V22_EVIDENCE.md`,
`research/*.md`) are written in Thai with English technical terms. They are decision records,
not tutorials, and they are the authority on *why* things are shaped the way they are.

## Commands

Python is managed by `uv` (Python 3.13 pinned); Rust by a Cargo workspace. There is no
Makefile — invoke tools directly.

```bash
# Tests
uv run pytest                                   # 255 tests: tests/ + bindings/python/tests/
uv run pytest tests/policy/test_v67_residual.py # one file
uv run pytest tests/policy/test_market_rules.py::test_name -x   # one test
cargo test --workspace                          # Rust engine + executor
cargo test -p kaggriculture-executor planner    # filter by test-name substring

# Lint / format
uv run ruff check src bindings tests scripts    # src/ and bindings/ are clean; keep them that way
cargo fmt --all --check                         # currently clean
cargo clippy --workspace --all-targets -- -D warnings   # currently clean; the migration doc's gate

# Rebuild the Rust extension after touching rust/  (~13s; nothing else picks up the change)
uv sync --reinstall-package kaggriculture-rs

# Engine as a subprocess (JSON-lines protocol, one request per line on stdin)
cargo run --release -p kaggriculture-engine --bin kg-engine
cargo run --release -p kaggriculture-engine --example throughput

# Smoke-test a submission bundle end to end
uv run python scripts/smoke_submission_bundle.py submissions/<bundle>/main.py < obs.json
```

Do **not** run bare `uv run ruff check` at the repo root: `artifacts/` holds recovered
competitor source and built bundles and produces ~1,300 findings. Lint explicit paths.
`tests/submission/test_v16_rc6_submission_entrypoint.py` has one pre-existing `I001`.

Ruff is used as a **linter only**. Do not run `ruff format` — the Python here is not
ruff-formatted (37 files under `src/` would be rewritten), so it would bury a real diff in churn.
Rust, by contrast, is `cargo fmt`-clean and should stay that way.

Scripts in `scripts/` are run as files, not modules (`uv run python scripts/foo.py`), and they
**import each other by bare name** (`from evaluate_file_agent_vs_replay import ...`), relying on
`sys.path[0]` being `scripts/`. Running them via `python -m` breaks those imports.

## Architecture

### The layer cake

```
rust/kaggriculture-engine     state transitions, rules tables, CPython-compatible RNG
rust/kaggriculture-executor   semantic target planner + fail-closed contract validation
rust/kaggriculture-python     PyO3 cdylib (abi3-py311) -> kaggriculture_rs._native
bindings/python/…             thin Python adapter: Engine, plan_observation
src/kaggriculture_rl/         policies, learning, imitation, evaluation harnesses
submissions/<name>/main.py    the actual Kaggle artifacts
```

**Rust owns state transitions; Python owns orchestration and policy.** The official Kaggle
environment — not the Rust port — remains the ultimate oracle. Invariants that the Rust side
exists to preserve exactly (from `RUST_ENGINE_MIGRATION.md`, and worth knowing before you touch
`rules.rs` or `engine.rs`):

- Invalid/malformed actions are **silent no-ops**, never errors. Training and evaluation
  semantics depend on this matching the competition.
- RNG must reproduce CPython's `random` bit-for-bit for a given seed (`python_random.rs`), not
  merely match in distribution.
- Inventory is an insertion-ordered map — order changes outcomes when shed capacity is short.
- Market orders resolve lockstep per unit, quoting both players before committing.
- The public/private visibility boundary is load-bearing: a policy must never observe opponent
  private state.

Two ways into the engine from Python, and the difference matters for hot loops:

- **JSON path** — `Engine.reset/step/state_json` serialise dicts across the boundary. Convenient,
  used by parity checks.
- **Typed path** — `plan_typed`, `step_typed`, `unit_action_mask_typed`, `market_order_mask_typed`,
  `sanitize_action_typed`. `bindings/python/tests/test_executor_contract.py` asserts the typed hot
  path never touches the Python JSON codec; keep it that way when extending.

### Action shape

Every turn a policy returns exactly three keys — this triple is the contract everywhere in the
repo (`ACTION_SPACE.md` has the full 18-op table):

```python
{"farmer": [op, *args], "hands": [[op, *args], ...], "market": [[op, *args], ...]}
```

### Policy families and the "tape" vocabulary

`src/kaggriculture_rl/policy/` holds hand-written and hybrid policies, versioned by generation
(`v16_*`, `v19_*`, `v22`, `v67_residual`, `closed_loop_town_quota_v74/75/76`). `learning/` holds
the trainable selectors on top of them.

A **tape** is a fixed, open-loop 719-action sequence — a recorded game replayed verbatim,
ignoring observations. Much of the current work is *routing*: picking which tape to play based on
observable state. `learning/tape_selector.py`, `receding_tape_selector.py`,
`temporal_tape_selector.py`, `staged_tape_selector.py`, `dynamic_tape_retriever.py` are variants
of that selector; `artifacts/opponent_tapes_*` and `artifacts/*tape_population*` are the pools
they choose from. Understanding tape-vs-closed-loop is a prerequisite for reading almost any
recent analysis directory.

### Submissions: `_local` vs `_standalone`

`submissions/` holds ~200 bundles and the suffix is the whole story:

- `*_local` (126) — a ~30-line `main.py` that imports `kaggriculture_rl`. Fast to iterate,
  **cannot be uploaded**; Kaggle has no access to this package.
- `*_standalone` (33) — every dependency inlined into one 5,000-line `main.py`. These are the
  uploadable artifacts, generated by `scripts/build_*_submission.py`. **Never hand-edit a
  standalone bundle**; change the source policy and regenerate, or the two silently diverge.

Bundles are loaded by `runpy.run_path` and must expose `agent(observation) -> action`.

### Evaluation harnesses

`src/kaggriculture_rl/evaluation/` plus ~139 scripts in `scripts/`, following a naming grammar
worth learning before writing a new one: `build_*` (make a submission/dataset), `train_*`,
`evaluate_*` (score candidates), `analyze_*` (read results), `harvest_*`/`download_*` (pull Kaggle
replays), `search_*`/`screen_*` (sweeps), `validate_*` (gate checks).

Evaluation is almost always *round-robin against opponent tapes over seed ranges and both seats*,
emitting one JSONL/JSON row per `(candidate, opponent, seed, seat)`. `scripts/evaluate_shop_router.py`
is a compact, current example of the shape. Seat swapping is not optional — margins are strongly
seat-dependent.

## Evidence discipline

This is the repo's most important convention and it is not visible from the code.

Experiments live in `analysis/<vNN>_<topic>_<YYYYMMDD>/` and follow a fixed protocol:

1. `PREREGISTRATION.md` written **before** the evaluation run, naming the hypothesis, the exact
   candidate/opponent/seed sets, and numeric pass/fail gates.
2. Data files (`train.json`, `screen.json`, `holdout.json`, `gates.json`) from frozen inputs.
3. `REPORT.md` with the outcome, including negative results.

Gates are never relaxed after seeing outcomes, and a held-out split is never used for tuning.
Several past conclusions were overturned precisely because someone re-checked whether the claimed
feature was even present in the encoder (`v85` → `v94`), so: before concluding a signal is
unidentifiable, verify the candidate variable actually reaches the model, and that n is large
enough to resolve the claimed effect.

Local results have systematically failed to predict live rating — treat a local sweep as a
screen, not proof. Durable cross-session findings live in
`~/.claude/projects/-Users-chinnaphat-Desktop-cedt-Super-AI-competetive-Kaggriculture/memory/`;
read `MEMORY.md` there before re-deriving anything about lineage, standing, or refuted claims.

`.omx/` is an external agent-runtime ledger (goals, plans, specs, verdicts). Read it for history;
don't hand-edit it.

## Data that is not in git

`.gitignore` excludes `replays/`, `rl/checkpoints/`, `rl/datasets/`, `target/`, `*.so`, and raw
Kaggle replay JSON (32 MB each — only `manifest.json` and reduced tapes are committed). If a
script wants a replay you do not have, re-fetch it with the matching `scripts/download_*.py`
rather than regenerating a substitute.

## Tooling in this environment

Three token-reduction tools are wired up. Prefer them; they change which commands you should reach
for.

### tokensave — code graph (MCP)

The project is indexed (`.tokensave/`, ~17.9k nodes over 1,351 files). **Use `tokensave_context`
as the first step for any code exploration**, not agents, glob, or grep. `tokensave_search` finds
symbols; `tokensave_callers`/`tokensave_callees`/`tokensave_impact` trace relationships. For
structural questions the tools don't cover, query `.tokensave/tokensave.db` directly (tables:
`nodes`, `edges`, `files`).

A hook **blocks** `grep` in Bash when the pattern looks like a symbol name. That is usually
correct. When you genuinely need text search (Thai prose in docs, JSON payloads, log files),
prefix the call with `TOKENSAVE_DISABLE_GREP_HOOK=1`.

### rtk — CLI output filter

`/opt/homebrew/bin/rtk` proxies shell commands and strips boilerplate before output reaches
context. Use `rtk ls`, `rtk read <file>`, `rtk find`, `rtk git`, `rtk diff`, `rtk grep`,
`rtk pytest`, `rtk cargo`, `rtk ruff`, `rtk json` in place of the bare commands. `rtk tree` is a
proxy to native `tree`, which is **not installed here** — use `find -maxdepth N -type d` instead.
`rtk gain` reports savings.

### headroom — proxy-level compression

Requests are routed through a local proxy (`ANTHROPIC_BASE_URL=http://127.0.0.1:8787`, kept alive
by the `SessionStart` hook in `.claude/settings.local.json`). It compresses tool output in flight;
`headroom_retrieve` with the `hash=…` from a `[N items compressed…]` marker recovers the original
text when a truncated detail matters. `headroom_stats` reports savings.

**Known discrepancy:** `AGENTS.md` instructs agents to call `memory_search`/`memory_save` on a
`headroom_memory` MCP server. That server is not connected — only `headroom_compress`,
`headroom_retrieve`, and `headroom_stats` are exposed, and every headroom memory database under
`~/.headroom/memories/` is empty (0 rows). Ignore that instruction; persistent memory is the
`~/.claude/projects/…/memory/` directory described above.
