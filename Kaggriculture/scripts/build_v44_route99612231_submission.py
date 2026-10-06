"""Build the standalone V44 route-99612231 Kaggriculture submission."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAPE = ROOT / "artifacts/top20_live_tapes80_20260826/tape_99612231_s0.py"
DEFAULT_LEGALIZER = ROOT / "src/kaggriculture_rl/policy/primitive_legalizer.py"
DEFAULT_UNIT_RULES = ROOT / "src/kaggriculture_rl/policy/unit_rules.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v44_route99612231_orphan_standalone"

HEADER = '''\
"""Standalone route 99612231 with deterministic state recovery."""

from __future__ import annotations

import base64
import copy
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
'''

ENTRYPOINT = r"""
_V44_TAPE_AGENT = agent
# Kaggle's raw loader chooses the last newly inserted callable in the module.
del agent

_V44_WEED_RECOVERY = WeedRecoveryController()
_V44_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    proposed = _V44_TAPE_AGENT(observation, configuration)
    weed_repaired = _V44_WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _V44_ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _top_level_source(
    path: Path,
    *,
    include_assignments: set[str] | None = None,
    include_classes: set[str] | None = None,
    include_functions: set[str] | None = None,
) -> str:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    definitions: list[str] = []

    for node in tree.body:
        include = False
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names: set[str] = set()
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
            include = include_assignments is None or bool(names & include_assignments)
        elif isinstance(node, ast.ClassDef):
            include = include_classes is None or node.name in include_classes
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            include = include_functions is None or node.name in include_functions

        if include:
            first_line = node.lineno
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                decorator_lines = [decorator.lineno for decorator in node.decorator_list]
                if decorator_lines:
                    first_line = min(first_line, *decorator_lines)
            if node.end_lineno is None:
                raise ValueError(f"cannot locate end of {type(node).__name__} in {path}")
            segment = "".join(source_lines[first_line - 1 : node.end_lineno]).rstrip()
            definitions.append(segment)

    if not definitions:
        raise ValueError(f"no embeddable definitions found in {path}")
    return "\n\n".join(definitions)


def build(
    tape: Path,
    legalizer: Path,
    unit_rules: Path,
    output_dir: Path,
) -> dict[str, object]:
    tape = tape.resolve()
    legalizer = legalizer.resolve()
    unit_rules = unit_rules.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tape_source = _top_level_source(
        tape,
        include_assignments={"SOURCE", "_PAYLOAD", "TRACE_ACTIONS"},
        include_classes=set(),
        include_functions={"agent"},
    )
    legalizer_source = _top_level_source(legalizer)
    orphan_source = _top_level_source(
        unit_rules,
        include_assignments={"_MOVES"},
        include_classes={"OrphanStructureRecoveryController"},
        include_functions=set(),
    )

    main_path = output_dir / "main.py"
    main_path.write_text(
        "\n\n".join(
            (HEADER.rstrip(), tape_source, legalizer_source, orphan_source, ENTRYPOINT.strip())
        )
        + "\n",
        encoding="utf-8",
    )

    archive_path = output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")

    metadata: dict[str, object] = {
        "version": "meta_v44_route99612231_orphan",
        "agent_file": str(main_path.relative_to(ROOT)),
        "archive_file": str(archive_path.relative_to(ROOT)),
        "sha256": _sha256(main_path),
        "archive_sha256": _sha256(archive_path),
        "sources": {
            "tape": str(tape.relative_to(ROOT)),
            "tape_sha256": _sha256(tape),
            "legalizer": str(legalizer.relative_to(ROOT)),
            "legalizer_sha256": _sha256(legalizer),
            "unit_rules": str(unit_rules.relative_to(ROOT)),
            "unit_rules_sha256": _sha256(unit_rules),
        },
        "policy": {
            "route_episode": 99612231,
            "recovery": ["weed", "orphan_structure", "terminal_liquidation"],
            "final_action_legalizer": True,
        },
        "screening_evidence": {
            "old_top20": {"wins": 60, "games": 80},
            "fresh_disjoint_top20": {"wins": 72, "games": 95},
            "combined": {"wins": 132, "games": 175},
        },
        "status": "built_pending_parity_gate",
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, default=DEFAULT_TAPE)
    parser.add_argument("--legalizer", type=Path, default=DEFAULT_LEGALIZER)
    parser.add_argument("--unit-rules", type=Path, default=DEFAULT_UNIT_RULES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.tape, args.legalizer, args.unit_rules, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
