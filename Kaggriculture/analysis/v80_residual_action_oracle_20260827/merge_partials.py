#!/usr/bin/env python3
"""Merge disjoint V80 oracle partials and recompute the eight-case screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.v80_residual_action_oracle_20260827.run_oracle import (  # noqa: E402
    HASHES_PATH,
    RESULTS_PATH,
    VARIANT_ROWS_PATH,
    canonical,
    semantic_rule_summary,
    sha256,
)

EXPECTED_IDS = {
    100159725: "target",
    100184942: "target",
    100187229: "target",
    100196432: "target",
    100168881: "control",
    100182646: "control",
    100143759: "control",
    100166584: "control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--variant-rows", type=Path, default=VARIANT_ROWS_PATH)
    parser.add_argument("--hashes", type=Path, default=HASHES_PATH)
    return parser.parse_args()


def best_with_abstention(episode: dict[str, Any]) -> dict[str, Any]:
    baseline_choice = {
        "mutation_id": "ABSTAIN_KEEP_V67",
        "family": "abstain",
        "category": "abstain",
        "semantic_key": "ABSTAIN_KEEP_V67",
        "descriptor": {"operation": "abstain"},
        "margin": episode["baseline"]["margin"],
        "delta_margin": 0.0,
        "mutated_step_sanitizer_changes": 0,
    }
    return sorted(
        [baseline_choice, *episode["variant_results"]],
        key=lambda row: (
            -row["margin"],
            row["mutated_step_sanitizer_changes"],
            row["mutation_id"],
        ),
    )[0]


def main() -> None:
    args = parse_args()
    partials = [json.loads(path.read_text(encoding="utf-8")) for path in args.partials]
    if not partials:
        raise ValueError("at least one partial is required")
    reference = partials[0]
    for partial in partials[1:]:
        if partial["inputs"] != reference["inputs"]:
            raise AssertionError("partial input hashes differ")
        if partial["grammar"] != reference["grammar"]:
            raise AssertionError("partial grammars differ")

    episodes = [episode for partial in partials for episode in partial["episodes"]]
    episodes.sort(key=lambda row: row["episode_id"])
    actual = {episode["episode_id"]: episode["role"] for episode in episodes}
    if actual != EXPECTED_IDS:
        raise AssertionError(f"unexpected episode partition: {actual}")
    if len(actual) != len(episodes):
        raise AssertionError("duplicate episode in partials")

    semantic_rules = semantic_rule_summary(episodes)
    hindsight_rows = []
    for episode in episodes:
        best = best_with_abstention(episode)
        hindsight_rows.append(
            {
                "episode_id": episode["episode_id"],
                "role": episode["role"],
                "baseline_margin": episode["baseline"]["margin"],
                "oracle_margin": best["margin"],
                "delta_margin": best["delta_margin"],
                "choice": best,
            }
        )
    targets = [row for row in hindsight_rows if row["role"] == "target"]
    controls = [row for row in hindsight_rows if row["role"] == "control"]
    strict_common = [
        row
        for row in semantic_rules
        if row["target_positive_count"] == 4 and row["control_no_regression_count"] == 4
    ]

    variant_digest = hashlib.sha256()
    with args.variant_rows.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for row in episode["variant_results"]:
                output_row = {
                    "episode_id": episode["episode_id"],
                    "role": episode["role"],
                    **row,
                }
                line = canonical(output_row)
                handle.write(line + "\n")
                variant_digest.update((line + "\n").encode())

    hindsight = {
        "deployable": False,
        "uses_completed_episode_identity": True,
        "rows": hindsight_rows,
        "target_positive_count": sum(row["oracle_margin"] > 0 for row in targets),
        "control_no_regression_count": sum(row["delta_margin"] >= 0 for row in controls),
        "target_worst_margin": min(row["oracle_margin"] for row in targets),
        "control_worst_delta": min(row["delta_margin"] for row in controls),
        "mean_delta": fmean(row["delta_margin"] for row in hindsight_rows),
    }
    result = {
        "schema_version": 1,
        "analysis": "v80_v67_residual_action_oracle",
        "scope": reference["scope"],
        "inputs": reference["inputs"],
        "grammar": reference["grammar"],
        "hindsight_oracle": hindsight,
        "common_semantic_rule_screen": {
            "definition": (
                "same event-relative semantic mutation applied in all eight episodes; "
                "no observation threshold"
            ),
            "common_rule_count": len(semantic_rules),
            "strict_pass_count": len(strict_common),
            "strict_passes": strict_common[:25],
            "ranked_top": semantic_rules[:50],
        },
        "episodes": [
            {key: value for key, value in episode.items() if key != "variant_results"}
            for episode in episodes
        ],
        "search": {
            "total_variant_simulations": sum(episode["variant_count"] for episode in episodes),
            "variant_rows_path": str(args.variant_rows.resolve().relative_to(ROOT)),
            "variant_rows_sha256": variant_digest.hexdigest(),
            "partial_inputs": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in args.partials
            ],
        },
    }
    result["status"] = (
        "HEADROOM_ONLY_HINDSIGHT"
        if hindsight["target_positive_count"] == 4 and hindsight["control_no_regression_count"] == 4
        else "STOP_NO_GRAMMAR_HEADROOM"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = []
    for path in (
        Path(__file__),
        Path(__file__).with_name("run_oracle.py"),
        args.output,
        args.variant_rows,
    ):
        lines.append(f"{sha256(path)}  {path.resolve().relative_to(ROOT)}")
    for path in args.partials:
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT)}")
    args.hashes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        canonical(
            {
                "status": result["status"],
                "total_variant_simulations": result["search"]["total_variant_simulations"],
                "target_positive_count": hindsight["target_positive_count"],
                "target_worst_margin": hindsight["target_worst_margin"],
                "control_no_regression_count": hindsight["control_no_regression_count"],
                "strict_common_rule_count": len(strict_common),
            }
        )
    )


if __name__ == "__main__":
    main()
