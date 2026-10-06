#!/usr/bin/env python3
"""Merge and validate the eight independently searched V79 gate results."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

EXPECTED = {
    100159725: ("target", -6859.0),
    100184942: ("target", -5470.0),
    100187229: ("target", -21061.0),
    100196432: ("target", -2479.0),
    100168881: ("control", 18079.0),
    100182646: ("control", 9765.0),
    100143759: ("control", 23005.0),
    100166584: ("control", 602.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_episode(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload["episodes"]) != 1:
        raise AssertionError(f"expected one episode in {path}")
    return payload, payload["episodes"][0]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    source_reports: list[str] = []
    common_inputs: dict[str, Any] | None = None

    for episode_id, (category, baseline_margin) in EXPECTED.items():
        path = args.directory / f"episode_{episode_id}.json"
        payload, row = load_episode(path)
        source_reports.append(str(path))
        if common_inputs is None:
            common_inputs = payload["inputs"]
        elif payload["inputs"] != common_inputs:
            raise AssertionError(f"input mismatch: {path}")
        if int(row["episode_id"]) != episode_id:
            raise AssertionError(f"episode mismatch: {path}")
        if row["category"] != category:
            raise AssertionError(f"category mismatch: {path}")
        if float(row["baseline"]["margin"]) != baseline_margin:
            raise AssertionError(f"baseline mismatch: {path}")
        if int(row["search"]["evaluated_schedules"]) != 4000:
            raise AssertionError(f"search budget mismatch: {path}")
        if int(row["oracle"]["proposed_unit_mismatch_steps"]) != 0:
            raise AssertionError(f"proposed unit tape changed: {path}")
        if int(row["oracle"]["non_target_market_mismatch_steps"]) != 0:
            raise AssertionError(f"non-target market tape changed: {path}")
        rows.append(row)

    targets = [row for row in rows if row["category"] == "target"]
    controls = [row for row in rows if row["category"] == "control"]
    target_rescues = sum(bool(row["rescued"]) for row in targets)
    control_no_regression = sum(bool(row["no_regression"]) for row in controls)
    baseline_margins = [float(row["baseline"]["margin"]) for row in rows]
    oracle_margins = [float(row["oracle"]["margin"]) for row in rows]
    failed_targets = [
        int(row["episode_id"]) for row in targets if float(row["oracle"]["margin"]) <= 0
    ]
    accepted_unit_changes = sum(int(row["oracle"]["accepted_unit_change_steps"]) for row in rows)
    sanitized_steps = sum(int(row["oracle"]["sanitized_steps"]) for row in rows)

    if common_inputs is None:
        raise AssertionError("no source reports")
    for name in ("gate", "parity", "agent", "native_engine"):
        entry = common_inputs[name]
        actual = sha256(args.root / entry["path"])
        if actual != entry["sha256"]:
            raise AssertionError(f"{name} SHA mismatch: {actual} != {entry['sha256']}")
    for row in rows:
        actual = sha256(args.root / row["replay_path"])
        if actual != row["replay_sha256"]:
            raise AssertionError(f"replay SHA mismatch: {row['episode_id']}")

    code_paths = [
        args.directory / "Cargo.toml",
        args.directory / "Cargo.lock",
        args.directory / "src/main.rs",
        args.directory / "merge_results.py",
    ]
    artifact_hashes = {
        str(path): sha256(path) for path in [*code_paths, *(Path(path) for path in source_reports)]
    }

    report = {
        "schema_version": 1,
        "analysis": "v79_market_only_hindsight_oracle",
        "status": "STOP_MARKET_ONLY_PATH",
        "stop_reason": (
            "The bounded per-episode hindsight oracle rescued only 2/4 target losses; "
            f"episodes {failed_targets} remained non-positive."
        ),
        "deployable": False,
        "warning": (
            "Per-episode identity-specific selection used completed consumed replays. "
            "This is an empirical ceiling screen, not a live policy or proof of a global optimum."
        ),
        "scope": (
            "Consumed gate8 only. Uploaded V67 farmer/hand action requests and every "
            "non-MILK/WOOL-SELL market request were frozen exactly."
        ),
        "engine_semantics": (
            "Search stepped raw typed actions through the official Rust engine; ineffective "
            "components therefore no-op as in Kaggle. Sanitizer differences are diagnostics "
            "only and were not substituted into the action tape."
        ),
        "inputs": common_inputs,
        "artifact_hashes": artifact_hashes,
        "source_reports": source_reports,
        "search_budget": {
            "evaluated_schedules_per_episode": 4000,
            "evaluated_schedules_total": 4000 * len(rows),
            "top_k": 16,
            "algorithm": (
                "Independent complete item-menu scans, top-16 cross-product, then bounded "
                "deterministic coordinate refinement."
            ),
            "menu_per_item": (
                "907 deterministic baseline/periodic/price-gated/one-shot timing-quantity "
                "rules with append/prepend intra-turn placement."
            ),
            "future_information_used": True,
            "identity_specific_selection": True,
            "global_optimum_proven": False,
        },
        "summary": {
            "games": len(rows),
            "target_games": len(targets),
            "control_games": len(controls),
            "target_rescues": target_rescues,
            "control_no_regression": control_no_regression,
            "baseline_wins": sum(margin > 0 for margin in baseline_margins),
            "oracle_wins": sum(margin > 0 for margin in oracle_margins),
            "baseline_mean_margin": statistics.fmean(baseline_margins),
            "oracle_mean_margin": statistics.fmean(oracle_margins),
            "baseline_worst_margin": min(baseline_margins),
            "oracle_worst_margin": min(oracle_margins),
            "failed_target_episode_ids": failed_targets,
            "all_proposed_unit_mismatch_steps": 0,
            "all_non_target_market_mismatch_steps": 0,
            "all_accepted_unit_change_steps": accepted_unit_changes,
            "all_sanitized_steps": sanitized_steps,
        },
        "episodes": rows,
        "forbidden_data_used": False,
        "candidate_written": False,
        "uploaded": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
