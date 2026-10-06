"""Build the local V18 state-aware macro candidate from pinned V16 RC6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v18_state_macro_local"
POLICY_SOURCE = ROOT / "src/kaggriculture_rl/policy/hierarchical_semantic.py"
EXECUTOR_SOURCE = ROOT / "rust/kaggriculture-executor/src/planner.rs"
NATIVE_BINDING = ROOT / "bindings/python/kaggriculture_rs/_native.abi3.so"
EXPECTED_BASE_SHA256 = (
    "c8afcbb8f90861cd92131012146e573752820b80fbd939ace3c7503aea83cd5e"
)

WRAPPER_SOURCE = r'''


# V18 local prototype: pinned V16 opening, then state-aware semantic routing.
from kaggriculture_rl.policy.hierarchical_semantic import (
    HierarchicalSemanticPolicy as _HierarchicalSemanticPolicy,
)

_V16_AGENT = agent
del agent
_V18_POLICY_SOURCE_SHA256 = "__POLICY_SOURCE_SHA256__"
_V18_EXECUTOR_SOURCE_SHA256 = "__EXECUTOR_SOURCE_SHA256__"
_V18_NATIVE_BINDING_SHA256 = "__NATIVE_BINDING_SHA256__"
_V18_POLICY = _HierarchicalSemanticPolicy(opening_policy=_V16_AGENT)


def agent(obs, configuration=None):
    return _V18_POLICY(obs, configuration)
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build(base: Path, output_dir: Path) -> dict[str, object]:
    base_hash = _sha256(base)
    if base_hash != EXPECTED_BASE_SHA256:
        raise ValueError(
            f"refusing unpinned V16 base: expected {EXPECTED_BASE_SHA256}, got {base_hash}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main.py"
    policy_hash = _sha256(POLICY_SOURCE)
    executor_hash = _sha256(EXECUTOR_SOURCE)
    native_hash = _sha256(NATIVE_BINDING)
    wrapper = (
        WRAPPER_SOURCE.replace("__POLICY_SOURCE_SHA256__", policy_hash)
        .replace("__EXECUTOR_SOURCE_SHA256__", executor_hash)
        .replace("__NATIVE_BINDING_SHA256__", native_hash)
    )
    output.write_text(base.read_text(encoding="utf-8").rstrip() + wrapper + "\n")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "name": "meta_v18_state_macro_local",
        "agent_file": _project_path(output),
        "sha256": _sha256(output),
        "base": {
            "agent_file": _project_path(base),
            "sha256": base_hash,
        },
        "policy": {
            "family": "fixed_opening_state_aware_semantic_routes",
            "is_rl": False,
            "handoff_step": 120,
            "route_gates": [120, 240, 480, 696],
            "routes": [
                "BALANCED",
                "COW_PREMIUM",
                "SHEEP_PREMIUM",
                "LIQUIDATE",
            ],
            "executor": {
                "api": "kaggriculture_rs.plan_observation",
                "source_file": _project_path(EXECUTOR_SOURCE),
                "source_sha256": executor_hash,
                "native_file": _project_path(NATIVE_BINDING),
                "native_sha256": native_hash,
            },
            "source": {
                "file": _project_path(POLICY_SOURCE),
                "sha256": policy_hash,
            },
        },
        "status": "experimental_local_native_required",
        "upload_ready": False,
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
