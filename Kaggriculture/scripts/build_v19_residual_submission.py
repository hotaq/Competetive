"""Build local V19 portfolio/residual candidates from pinned V16 and V12 agents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V16 = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
DEFAULT_SEAT1 = (
    ROOT
    / "submissions/top1_v12_ryo_b96_dual_b144_retriever_exact_safe_local/main.py"
)
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v19_residual_local"
POLICY_SOURCE = ROOT / "src/kaggriculture_rl/policy/v19_residual.py"

EXPECTED_V16_SHA256 = "c8afcbb8f90861cd92131012146e573752820b80fbd939ace3c7503aea83cd5e"
EXPECTED_SEAT1_SHA256 = "6ac20f3e2736e4d94968401acb693656fb58e61a4174175c243db845adf6c7f4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build(
    v16_path: Path,
    seat1_path: Path,
    output_dir: Path,
    *,
    late_hire_cutoff: int,
    hire_boost_start: int = 720,
    hire_boost_end: int = 719,
    minimum_hires: int = 0,
    market_front_run_steps: tuple[int, ...] = (),
    front_run_items: tuple[str, ...] = ("STRAWBERRY", "MILK", "WOOL"),
    front_run_when_trailing: bool = True,
    front_run_reclaim_hires: int = 0,
    seat0_policy: str = "v16",
) -> dict[str, object]:
    v16_hash = _sha256(v16_path)
    seat1_hash = _sha256(seat1_path)
    if v16_hash != EXPECTED_V16_SHA256:
        raise ValueError(
            f"refusing unpinned V16 base: expected {EXPECTED_V16_SHA256}, got {v16_hash}"
        )
    if seat1_hash != EXPECTED_SEAT1_SHA256:
        raise ValueError(
            f"refusing unpinned seat-1 agent: expected {EXPECTED_SEAT1_SHA256}, got {seat1_hash}"
        )
    if not 0 <= late_hire_cutoff <= 720:
        raise ValueError("late_hire_cutoff must be within 0..720")
    if not 0 <= hire_boost_start <= 720:
        raise ValueError("hire_boost_start must be within 0..720")
    if not 0 <= hire_boost_end <= 719:
        raise ValueError("hire_boost_end must be within 0..719")
    if not 0 <= minimum_hires <= 10:
        raise ValueError("minimum_hires must be within 0..10")
    if not 0 <= front_run_reclaim_hires <= 10:
        raise ValueError("front_run_reclaim_hires must be within 0..10")
    if seat0_policy not in {"v16", "v12"}:
        raise ValueError("seat0_policy must be v16 or v12")
    if any(step < 0 or step > 719 for step in market_front_run_steps):
        raise ValueError("market_front_run_steps must be within 0..719")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main.py"
    source = f'''"""Generated local V19 portfolio/residual candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from kaggriculture_rl.policy.v19_residual import V19ResidualPolicy

ROOT = Path(__file__).resolve().parents[2]


def _load_agent(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent: {{path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.agent


_V16_AGENT = _load_agent({_project_path(v16_path)!r}, "v19_v16_base")
_SEAT1_AGENT = _load_agent({_project_path(seat1_path)!r}, "v19_v12_seat1")
_SEAT0_AGENT = {"_V16_AGENT" if seat0_policy == "v16" else "_SEAT1_AGENT"}
_V19_POLICY = V19ResidualPolicy(
    _SEAT0_AGENT,
    _SEAT1_AGENT,
    late_hire_cutoff={late_hire_cutoff},
    hire_boost_start={hire_boost_start},
    hire_boost_end={hire_boost_end},
    minimum_hires={minimum_hires},
    market_front_run_steps={market_front_run_steps!r},
    front_run_items={front_run_items!r},
    front_run_when_trailing={front_run_when_trailing!r},
    front_run_reclaim_hires={front_run_reclaim_hires},
)


def agent(observation, configuration=None):
    return _V19_POLICY(observation, configuration)
'''
    output.write_text(source, encoding="utf-8")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "name": output_dir.name,
        "agent_file": _project_path(output),
        "sha256": _sha256(output),
        "dependencies": {
            "v16": {"path": _project_path(v16_path), "sha256": v16_hash},
            "seat1": {"path": _project_path(seat1_path), "sha256": seat1_hash},
            "policy": {"path": _project_path(POLICY_SOURCE), "sha256": _sha256(POLICY_SOURCE)},
        },
        "policy": {
            "family": "contextual_portfolio_with_residual",
            "seat_0": seat0_policy,
            "seat_1": "V12 dual retriever from step zero",
            "late_hire_cutoff": late_hire_cutoff,
            "hire_boost_start": hire_boost_start,
            "hire_boost_end": hire_boost_end,
            "minimum_hires": minimum_hires,
            "market_front_run_steps": list(market_front_run_steps),
            "front_run_items": list(front_run_items),
            "front_run_when_trailing": front_run_when_trailing,
            "front_run_reclaim_hires": front_run_reclaim_hires,
        },
        "status": "local_experiment",
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--seat1", type=Path, default=DEFAULT_SEAT1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--late-hire-cutoff", type=int, default=696)
    parser.add_argument("--hire-boost-start", type=int, default=720)
    parser.add_argument("--hire-boost-end", type=int, default=719)
    parser.add_argument("--minimum-hires", type=int, default=0)
    parser.add_argument("--market-front-run-steps", default="")
    parser.add_argument(
        "--front-run-items",
        default="STRAWBERRY,MILK,WOOL",
    )
    parser.add_argument(
        "--front-run-unconditional",
        action="store_true",
        help="apply front-run even while candidate money leads",
    )
    parser.add_argument("--front-run-reclaim-hires", type=int, default=0)
    parser.add_argument("--seat0-policy", choices=("v16", "v12"), default="v16")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.v16,
                args.seat1,
                args.output_dir,
                late_hire_cutoff=args.late_hire_cutoff,
                hire_boost_start=args.hire_boost_start,
                hire_boost_end=args.hire_boost_end,
                minimum_hires=args.minimum_hires,
                market_front_run_steps=tuple(
                    int(value)
                    for value in args.market_front_run_steps.split(",")
                    if value.strip()
                ),
                front_run_items=tuple(
                    value.strip()
                    for value in args.front_run_items.split(",")
                    if value.strip()
                ),
                front_run_when_trailing=not args.front_run_unconditional,
                front_run_reclaim_hires=args.front_run_reclaim_hires,
                seat0_policy=args.seat0_policy,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
