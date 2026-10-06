#!/usr/bin/env python3
"""Validate completed target JSONL rows and emit the fail-fast V80 evidence bundle."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.v80_residual_action_oracle_20260827.run_oracle import (  # noqa: E402
    GATE_PATH,
    NATIVE_ENGINE_PATH,
    REPLAY_DIR,
    V67_PATH,
    canonical,
    enumerate_mutations,
    sha256,
)
from scripts.evaluate_replay_tape import frame_action  # noqa: E402

ANALYSIS_DIR = ROOT / "analysis/v80_residual_action_oracle_20260827"
TARGET_A_PATH = ANALYSIS_DIR / "target_a.jsonl"
TARGET_B_PATH = ANALYSIS_DIR / "target_b.jsonl"
RESULTS_PATH = ANALYSIS_DIR / "results.json"
CATALOG_PATH = ANALYSIS_DIR / "catalog.json"
HASHES_PATH = ANALYSIS_DIR / "HASHES.sha256"
RUNNER_PATH = ANALYSIS_DIR / "run_oracle.py"
REPORT_PATH = ANALYSIS_DIR / "REPORT.md"

EXPECTED_ROW_COUNTS = {
    100159725: 271,
    100184942: 277,
    100187229: 394,
    100196432: 271,
}
EXPECTED_JSONL_HASHES = {
    TARGET_A_PATH: "a557e6280a43db5b3130dde2b63183e6134ef3b25bc9f9e1fbd1441790502657",
    TARGET_B_PATH: "4bd03e2968da1160612114842bcaf2f807f2cbce40eacf9c2060eb4b029d5f41",
}
ALLOWED_FAMILIES = {"market_drop", "market_quantity", "market_item_swap"}
V79_EXPANSION_PATH = ANALYSIS_DIR.parent / "v79_expansion_allocation_20260827/results.json"
V79_MARKET_PATH = ANALYSIS_DIR.parent / "v79_market_only_oracle_20260827/results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--hashes", type=Path, default=HASHES_PATH)
    return parser.parse_args()


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, expected_hash in EXPECTED_JSONL_HASHES.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise AssertionError(f"JSONL hash mismatch: {path}: {actual_hash}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    counts = Counter(int(row["episode_id"]) for row in rows)
    if dict(counts) != EXPECTED_ROW_COUNTS:
        raise AssertionError(f"unexpected target row counts: {dict(counts)}")
    seen: set[tuple[int, str]] = set()
    for row in rows:
        episode_id = int(row["episode_id"])
        descriptor = row["descriptor"]
        identity = (episode_id, row["mutation_id"])
        if identity in seen:
            raise AssertionError(f"duplicate mutation row: {identity}")
        seen.add(identity)
        if row["role"] != "target":
            raise AssertionError("target JSONL contains a non-target row")
        if row["family"] not in ALLOWED_FAMILIES:
            raise AssertionError(f"unexpected family: {row['family']}")
        if descriptor["surface"] != "market" or descriptor["verb"] == "SELL":
            raise AssertionError(f"out-of-profile descriptor: {descriptor}")
        if not 120 <= int(descriptor["step"]) <= 270:
            raise AssertionError(f"out-of-window descriptor: {descriptor}")
        if int(row["mutated_step_sanitizer_changes"]) < 0:
            raise AssertionError(f"invalid sanitizer diagnostic: {identity}")


def best_rows(rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = {
        int(row["episode_id"]): float(row["baseline_margin"]) for row in gate["target_losses"]
    }
    output: list[dict[str, Any]] = []
    for episode_id in sorted(baselines):
        all_episode_rows = [row for row in rows if int(row["episode_id"]) == episode_id]
        episode_rows = [
            row for row in all_episode_rows if int(row["mutated_step_sanitizer_changes"]) == 0
        ]
        candidates = [
            {
                "mutation_id": "ABSTAIN_KEEP_V67",
                "semantic_key": "ABSTAIN_KEEP_V67",
                "family": "abstain",
                "category": "abstain",
                "descriptor": {"operation": "abstain"},
                "margin": baselines[episode_id],
                "delta_margin": 0.0,
                "mutated_step_sanitizer_changes": 0,
            },
            *episode_rows,
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["margin"]),
                int(row["mutated_step_sanitizer_changes"]),
                row["mutation_id"],
            )
        )
        best = candidates[0]
        by_family: dict[str, dict[str, Any]] = {}
        for row in sorted(
            episode_rows,
            key=lambda item: (-float(item["margin"]), item["mutation_id"]),
        ):
            by_family.setdefault(row["family"], row)
        output.append(
            {
                "episode_id": episode_id,
                "baseline_margin": baselines[episode_id],
                "variant_count": len(all_episode_rows),
                "sanitizer_clean_variant_count": len(episode_rows),
                "sanitizer_changed_variant_count": len(all_episode_rows) - len(episode_rows),
                "positive_variant_count": sum(float(row["margin"]) > 0 for row in episode_rows),
                "oracle_margin": float(best["margin"]),
                "oracle_delta_margin": float(best["delta_margin"]),
                "choice": best,
                "best_by_family": by_family,
                "top_variants": sorted(
                    episode_rows,
                    key=lambda item: (-float(item["margin"]), item["mutation_id"]),
                )[:10],
            }
        )
    return output


def common_rule_screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, dict[str, dict[str, Any]]] = {}
    for episode_id in EXPECTED_ROW_COUNTS:
        by_episode[episode_id] = {
            row["semantic_key"]: row
            for row in rows
            if int(row["episode_id"]) == episode_id
            and int(row["mutated_step_sanitizer_changes"]) == 0
        }
    common = set.intersection(*(set(mapping) for mapping in by_episode.values()))
    summaries = []
    for key in common:
        outcomes = [by_episode[episode_id][key] for episode_id in sorted(by_episode)]
        summaries.append(
            {
                "semantic_key": key,
                "family": outcomes[0]["family"],
                "positive_count": sum(float(row["margin"]) > 0 for row in outcomes),
                "worst_margin": min(float(row["margin"]) for row in outcomes),
                "mean_delta_margin": fmean(float(row["delta_margin"]) for row in outcomes),
                "outcomes": [
                    {
                        "episode_id": int(row["episode_id"]),
                        "margin": float(row["margin"]),
                        "delta_margin": float(row["delta_margin"]),
                    }
                    for row in outcomes
                ],
            }
        )
    summaries.sort(
        key=lambda row: (
            -row["positive_count"],
            -row["worst_margin"],
            -row["mean_delta_margin"],
            row["semantic_key"],
        )
    )
    return {
        "common_rule_count": len(summaries),
        "all_four_positive_count": sum(row["positive_count"] == 4 for row in summaries),
        "ranked_top": summaries[:25],
    }


def build_catalog(gate: dict[str, Any]) -> dict[str, Any]:
    episodes = []
    for role, key in (("target", "target_losses"), ("control", "regression_controls")):
        for gate_row in gate[key]:
            episode_id = int(gate_row["episode_id"])
            replay_path = REPLAY_DIR / f"{episode_id}.json"
            if sha256(replay_path) != gate_row["replay_sha256"]:
                raise AssertionError(f"replay hash mismatch: {episode_id}")
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            actions = [
                copy.deepcopy(frame_action(frame, int(gate_row["agent_seat"])))
                for frame in replay["steps"][1:]
            ]
            _, events = enumerate_mutations(actions)
            category_counts = Counter(event["category"] for event in events)
            verb_counts = Counter(event["verb"] for event in events)
            market_events = [event for event in events if event["surface"] == "market"]
            non_sell_market_events = [event for event in market_events if event["verb"] != "SELL"]
            episodes.append(
                {
                    "episode_id": episode_id,
                    "role": role,
                    "candidate_seat": int(gate_row["agent_seat"]),
                    "replay_sha256": sha256(replay_path),
                    "category_counts": dict(sorted(category_counts.items())),
                    "verb_counts": dict(sorted(verb_counts.items())),
                    "non_sell_market_events": non_sell_market_events,
                    "sell_event_count": sum(event["verb"] == "SELL" for event in market_events),
                    "unit_macro_event_count": sum(event["surface"] != "market" for event in events),
                }
            )
    return {
        "schema_version": 1,
        "source": "uploaded V67 action tape in the eight consumed Gate8 replays",
        "catalog_definition": (
            "post-opening step>=120 HIRE, BUY_LAND, BUY_ANIMAL, BUY_SEED, "
            "BUY_PRODUCT, SELL, PLANT, DIG, animal PLACE, structure build/remove"
        ),
        "episodes": sorted(episodes, key=lambda row: row["episode_id"]),
    }


def main() -> None:
    args = parse_args()
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    rows = load_rows()
    validate_rows(rows)
    target_summary = best_rows(rows, gate)
    common_rules = common_rule_screen(rows)

    catalog = build_catalog(gate)
    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    catalog_hash = sha256(args.catalog)

    positive_targets = sum(row["oracle_margin"] > 0 for row in target_summary)
    result = {
        "schema_version": 1,
        "analysis": "v80_v67_residual_action_priority_oracle",
        "status": "STOP_PRIORITY_GRAMMAR_FAILS_TARGET_PREREQUISITE",
        "stop_reason": (
            "Episode 100159725 remained negative under every tested priority macro "
            "deviation, so the 4/4 target rescue prerequisite failed before controls."
        ),
        "deployable": False,
        "candidate_created": False,
        "training_started": False,
        "upload_performed": False,
        "scope": {
            "replays": "four consumed Gate8 target losses only for score simulation",
            "controls": (
                "catalogued but not simulated because the target-only fail-fast prerequisite failed"
            ),
            "opening_preserved_through_step": 119,
            "mutation_window_inclusive": [120, 270],
            "families": sorted(ALLOWED_FAMILIES),
            "eligible_verbs": [
                "HIRE",
                "BUY_LAND",
                "BUY_ANIMAL",
                "BUY_SEED",
                "BUY_PRODUCT",
            ],
            "continuation": (
                "raw uploaded V67 candidate actions and opponent tape preserved "
                "byte-for-byte outside one mutation"
            ),
            "engine": "official Rust binding",
            "forbidden_sets_used": [],
            "live_proxy_seeds_used": [],
        },
        "coverage": {
            "variant_simulations": len(rows),
            "exact_engine_transitions": len(rows) * 719,
            "sanitizer_clean_variants": sum(
                int(row["mutated_step_sanitizer_changes"]) == 0 for row in rows
            ),
            "sanitizer_changed_variants": sum(
                int(row["mutated_step_sanitizer_changes"]) != 0 for row in rows
            ),
            "target_episode_counts": EXPECTED_ROW_COUNTS,
            "target_positive_count": positive_targets,
            "target_worst_oracle_margin": min(row["oracle_margin"] for row in target_summary),
            "regression_controls_simulated": 0,
        },
        "target_hindsight_oracle": {
            "uses_completed_episode_identity": True,
            "warning": (
                "Per-episode best choice is a non-deployable ceiling, not an observable state rule."
            ),
            "rows": target_summary,
        },
        "same_semantic_rule_screen": common_rules,
        "catalog": {
            "path": str(args.catalog.resolve().relative_to(ROOT)),
            "sha256": catalog_hash,
        },
        "related_closed_screens": {
            "v79_market_only": {
                "path": str(V79_MARKET_PATH.relative_to(ROOT)),
                "sha256": sha256(V79_MARKET_PATH),
                "result": "2/4 target rescues; path stopped",
            },
            "v79_fixed_observable_sheep_rule": {
                "path": str(V79_EXPANSION_PATH.relative_to(ROOT)),
                "sha256": sha256(V79_EXPANSION_PATH),
                "result": "1/4 model-conditioned rescue ceilings; fixed rule stopped",
            },
        },
        "inputs": {
            "gate": {"path": str(GATE_PATH.relative_to(ROOT)), "sha256": sha256(GATE_PATH)},
            "agent": {"path": str(V67_PATH.relative_to(ROOT)), "sha256": sha256(V67_PATH)},
            "native_engine": {
                "path": str(NATIVE_ENGINE_PATH.relative_to(ROOT)),
                "sha256": sha256(NATIVE_ENGINE_PATH),
            },
            "target_jsonl": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in EXPECTED_JSONL_HASHES
            ],
        },
        "untested_uncertainty": [
            "Macro deviations after step 270.",
            "Market reorder and adjacent-step shift variants.",
            "Unit PLANT, structure, animal-placement, and multi-event option bundles.",
            "Regression-control score effects for this priority grammar.",
            "Consumed local live-proxy seeds 27082701..27082712.",
            "Any sealed, old/fresh/novel, or future holdout evidence.",
        ],
        "decision": (
            "Do not start PPO from this grammar and do not create a submission candidate. "
            "A future branch needs a predeclared action family that can first rescue "
            "100159725 under the same consumed-only discipline."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    evidence_paths = [
        Path(__file__),
        RUNNER_PATH,
        args.output,
        args.catalog,
        REPORT_PATH,
        TARGET_A_PATH,
        TARGET_B_PATH,
    ]
    args.hashes.write_text(
        "\n".join(f"{sha256(path)}  {path.resolve().relative_to(ROOT)}" for path in evidence_paths)
        + "\n",
        encoding="utf-8",
    )
    print(
        canonical(
            {
                "status": result["status"],
                "variants": len(rows),
                "target_positive_count": positive_targets,
                "target_worst_oracle_margin": result["coverage"]["target_worst_oracle_margin"],
                "common_all_four_positive": common_rules["all_four_positive_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
