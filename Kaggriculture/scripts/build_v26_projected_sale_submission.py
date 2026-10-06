"""Build the standalone V26 projected-wool-sale Kaggriculture bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py"
DEFAULT_RULES = ROOT / "src/kaggriculture_rl/policy/unit_rules.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v26_v22_projected_wool_sale_standalone"

ENTRYPOINT = r"""
_V26_BASE_AGENT = agent
# Kaggle's raw loader chooses the last newly inserted callable in the module.
del agent

_V26_SHED_RETURN = AdjacentShedReturnController(
    minimum_inventory_value=2_500.0,
    maximum_interventions_per_game=1,
    liquidation_items=("WOOL",),
)


def agent(observation, configuration=None):
    del configuration
    return _V26_SHED_RETURN.apply(observation, _V26_BASE_AGENT(observation))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embeddable_rule_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return source.replace("from __future__ import annotations\n", "")


def build(base: Path, rules: Path, output_dir: Path) -> dict[str, object]:
    base = base.resolve()
    rules = rules.resolve()
    output_dir = output_dir.resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    if not rules.is_file():
        raise FileNotFoundError(rules)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    source = "\n\n".join(
        (
            base.read_text(encoding="utf-8").rstrip(),
            _embeddable_rule_source(rules).strip(),
            ENTRYPOINT.strip(),
        )
    ) + "\n"
    compile(source, str(main_path), "exec")
    main_path.write_text(source, encoding="utf-8")

    archive_path = output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "name": "meta_v26_v22_projected_wool_sale_standalone",
        "agent_file": str(main_path.relative_to(ROOT)),
        "archive_file": str(archive_path.relative_to(ROOT)),
        "sha256": _sha256(main_path),
        "archive_sha256": _sha256(archive_path),
        "base": {
            "agent_file": str(base.relative_to(ROOT)),
            "sha256": _sha256(base),
        },
        "policy": {
            "family": "observable_state_overlay",
            "minimum_inventory_value": 2500.0,
            "maximum_interventions_per_game": 1,
            "liquidation_items": ["WOOL"],
            "transaction": "enter shed, drop, and sell projected wool in the same turn",
        },
        "status": "built_pending_final_gate",
    }
    candidate_path = output_dir / "candidate.json"
    candidate_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.rules, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
