#!/usr/bin/env python3
"""Fail closed unless a downloaded E030 run passes a declared scout path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PASS_DECISIONS = {
    "PASS_REPLACEMENT_CANDIDATE_NO_SUBMISSION",
    "PASS_COMPLEMENT_SIGNAL_PROMOTE_SCALE_NO_SUBMISSION",
}
EXPECTED_REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_WEIGHT_BYTES = 1_664_223_428
EXPECTED_BASELINE_AUC = 0.9174298724699895


def validate(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "e030_summary.json").read_text())
    failures: list[str] = []

    decision = summary.get("decision")
    if decision not in PASS_DECISIONS:
        failures.append(f"decision={decision!r}")

    required = summary.get("required_gates", {})
    failed_required = sorted(name for name, passed in required.items() if passed is not True)
    if not required:
        failures.append("required_gates missing")
    elif failed_required:
        failures.append(f"failed_required={failed_required}")

    identity = summary.get("identity", {})
    if identity.get("counts") != {"train": 512, "validation": 128, "gold": 58}:
        failures.append(f"counts={identity.get('counts')!r}")
    if identity.get("baseline_identity_exact") is not True:
        failures.append("baseline_identity_exact is not true")

    model = summary.get("model", {})
    if model.get("repo_commit") != EXPECTED_REPO_COMMIT:
        failures.append(f"repo_commit={model.get('repo_commit')!r}")
    if model.get("checkpoint_bytes") != EXPECTED_WEIGHT_BYTES:
        failures.append(f"checkpoint_bytes={model.get('checkpoint_bytes')!r}")
    if model.get("encoder_frozen") is not True:
        failures.append("encoder_frozen is not true")
    if model.get("encoder_initial_sha256") != model.get("encoder_final_sha256"):
        failures.append("encoder hash changed")

    input_contract = summary.get("input", {})
    if input_contract.get("shape") != [1, 3, 16, 384, 384]:
        failures.append(f"input_shape={input_contract.get('shape')!r}")
    if input_contract.get("feature_shape_per_slot") != [8, 3072]:
        failures.append(
            f"feature_shape_per_slot={input_contract.get('feature_shape_per_slot')!r}"
        )
    if input_contract.get("grayscale_repeated_rgb") is not True:
        failures.append("grayscale_repeated_rgb is not true")
    if input_contract.get("per_slot_independent") is not True:
        failures.append("per_slot_independent is not true")
    if input_contract.get("cross_series_clip_count") != 0:
        failures.append(
            f"cross_series_clip_count={input_contract.get('cross_series_clip_count')!r}"
        )

    head = summary.get("head", {})
    if head.get("training_mode") != "frozen_encoder_head_only":
        failures.append(f"training_mode={head.get('training_mode')!r}")
    if head.get("optimizer_head_only") is not True or head.get("updated") is not True:
        failures.append(f"head_contract={head!r}")

    gold = summary.get("gold58", {})
    baseline_auc = float(gold.get("coatnet_macro_auc", math.nan))
    vjepa_auc = float(gold.get("vjepa_macro_auc", math.nan))
    delta = float(gold.get("delta", math.nan))
    win_rate = float(gold.get("paired_bootstrap", {}).get("win_rate", math.nan))
    correlation = float(gold.get("prediction_correlation", math.nan))
    min_target_delta = float(gold.get("min_target_delta", math.nan))
    validation_auc = float(head.get("validation_macro_auc", math.nan))
    if not all(
        math.isfinite(value)
        for value in (
            baseline_auc,
            vjepa_auc,
            delta,
            win_rate,
            correlation,
            min_target_delta,
            validation_auc,
        )
    ):
        failures.append("non-finite core metric")
    if abs(baseline_auc - EXPECTED_BASELINE_AUC) > 1e-12:
        failures.append(f"baseline_auc={baseline_auc}")
    if abs(delta - (vjepa_auc - baseline_auc)) > 1e-12:
        failures.append(f"inconsistent_delta={delta}")

    replacement = summary.get("replacement_gates", {})
    complement = summary.get("complement_gates", {})
    replacement_pass = bool(replacement) and all(value is True for value in replacement.values())
    complement_pass = bool(complement) and all(value is True for value in complement.values())
    if decision == "PASS_REPLACEMENT_CANDIDATE_NO_SUBMISSION":
        if not replacement_pass:
            failures.append(f"replacement_gates={replacement!r}")
        if delta < 0.001 or win_rate < 0.65 or min_target_delta < -0.03:
            failures.append("replacement numeric thresholds failed")
    elif decision == "PASS_COMPLEMENT_SIGNAL_PROMOTE_SCALE_NO_SUBMISSION":
        if not complement_pass:
            failures.append(f"complement_gates={complement!r}")
        if (
            validation_auc < 0.75
            or vjepa_auc < 0.88
            or int(gold.get("target_improve_count", -1)) < 3
            or correlation > 0.98
            or min_target_delta < -0.15
        ):
            failures.append("complement numeric thresholds failed")

    xai = summary.get("xai", {})
    if (
        xai.get("target_count") != 12
        or xai.get("nondegenerate_count") != 12
        or int(xai.get("positive_deletion_count", -1)) < 7
        or float(xai.get("median_deletion_advantage", math.nan)) <= 0
    ):
        failures.append(f"xai={xai!r}")

    for key in ("submission_created", "checkpoint_exported", "feature_cache_exported"):
        if summary.get(key) is not False:
            failures.append(f"{key}={summary.get(key)!r}")

    required_files = {
        "e030_gold58_predictions.npz",
        "e030_gold58_target_metrics.csv",
        "e030_head_history.csv",
        "e030_xai_deletion_metrics.csv",
        "e030_gates.csv",
    }
    missing_files = sorted(name for name in required_files if not (output_dir / name).is_file())
    if missing_files:
        failures.append(f"missing_files={missing_files}")

    if failures:
        raise SystemExit("E030 STOP: " + "; ".join(failures))

    return {
        "decision": decision,
        "validation_macro_auc": validation_auc,
        "gold58_macro_auc": vjepa_auc,
        "delta_vs_coatnet": delta,
        "bootstrap_win_rate": win_rate,
        "targets_improved": gold["target_improve_count"],
        "prediction_correlation": correlation,
        "xai_positive_deletion_count": xai["positive_deletion_count"],
        "all_required_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
