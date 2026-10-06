# Kaggriculture Rust Engine

Deterministic Rust port of the local Kaggriculture simulation engine. The reference oracle is
`kaggle-environments==1.32.7`, source SHA-256
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`.

## What is migrated

- JSON-compatible configuration, public farm state, private inventory, market and town state
- all farmer/farm-hand operations and silent invalid-action no-ops
- atomic same-crop planting validation across units
- simultaneous per-unit market order processing, dynamic price curves, hiring and land purchase
- crop growth, watering, fertilizer, harvest, decay and weeds
- animal feeding, care bonuses, production, fertilizer and escape
- end-of-day reset, shop unlocks, town demand and terminal rewards
- CPython-compatible seeded RNG, including `random()` and `choice()` behavior used by the oracle

The crate exposes three integration surfaces:

1. A typed Rust library for high-throughput training.
2. A persistent JSON-lines binary named `kg-engine` for differential testing and tooling.
3. An optional PyO3 extension consumed by the thin package under `bindings/python`.

## Validation

```bash
rtk cargo test --workspace --offline
rtk cargo clippy --workspace --all-targets --all-features --offline -- -D warnings
rtk cargo build -p kaggriculture-engine --bin kg-engine --offline
rtk .venv/bin/python scripts/compare_rust_engine.py
```

The committed differential harness compares every public and private state field for both players
after every action. The implementation has passed:

- one complete 719-action season;
- a deterministic scenario covering unit actions, market ordering and silent no-ops;
- 6,000 randomized action turns over multiple seeds;
- stochastic parity for weeds and repeated shop unlocks;
- sparse market-parameter overrides.

## Python extension

The extension uses the stable CPython 3.11 ABI and accepts/returns JSON strings at its native
boundary. Build it through Maturin from `bindings/python`, or validate the native library directly:

```bash
rtk cargo build -p kaggriculture-python --release --offline
```

The JSON boundary is deliberately conservative for the first cutover. A later optimization can
expose batched tensor observations without changing game semantics.

## Local benchmark

On the current Apple Silicon workspace with `kaggle-environments==1.32.7`, the median of five
10,000-turn pass-action runs after 100 warm-up turns produced:

| Engine path | Turns/second | Speedup vs Python |
|---|---:|---:|
| Installed Python/Kaggle reference | 1,283 | 1.00× |
| Rust through PyO3 with JSON state returned each turn | 145,030 | 113.08× |
| Rust through the typed PyO3 RL path | 209,149 | 163.08× |

Re-run `scripts/benchmark_engines.py` on the target training machine. This narrow pass-action
benchmark measures simulator throughput, not end-to-end policy inference or RL training speed.

## Cutover rule

Keep the Python engine as an oracle until representative policy traces pass turn-by-turn parity.
Do not train against one engine and evaluate against the other while a state mismatch remains.
