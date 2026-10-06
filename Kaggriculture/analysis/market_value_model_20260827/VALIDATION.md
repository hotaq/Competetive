# Validation evidence

## Frozen sources

```text
kaggle-environments: 1.32.7
kaggriculture.py sha256:
bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e
gate8 manifest sha256:
4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2
frozen V67 sha256:
300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b
Rust rules.rs sha256:
a0ca59b411fe8fecb03c1365cfb70d9e4f2dd705a4f5e74ac478adf92b4280a5
Rust model.rs sha256:
dd28d09e15c20d788ed017a2583b52a2d9bfd8004b8b02555c9ca275090f2979
Rust engine.rs sha256:
b32ea5d7a7dc7ee2a601af724723523707b0888b6ba99d468ed5c70dad6f9d63
```

## Commands and results

```text
.venv/bin/python analysis/market_value_model_20260827/test_invariants.py -q
Ran 11 tests — OK

cargo test -p kaggriculture-engine --offline
9 unit tests + 4 legality integration tests — all passed

.venv/bin/python scripts/compare_rust_engine.py
Rust/Python parity passed: deterministic full-state trace plus 960 randomized turns.

.venv/bin/ruff check analysis/market_value_model_20260827/{marginal_value.py,run_gate8_audit.py,test_invariants.py}
All checks passed.

.venv/bin/ruff format --check analysis/market_value_model_20260827/{marginal_value.py,run_gate8_audit.py,test_invariants.py}
3 files already formatted.

.venv/bin/python analysis/market_value_model_20260827/run_gate8_audit.py
read_only_gate8_sanity_complete
target losses: projected SHEEP 4/4
controls: projected SHEEP 3/4; among current COW-order controls, switch 1/2
```

Machine-readable evidence: `gate8_results.json`.
