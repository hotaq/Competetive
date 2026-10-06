#!/usr/bin/env python3
"""Fail closed unless a downloaded E031 dual-head run passes a declared path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PASS_DECISIONS = {
    "PASS_E031_BLEND_CANDIDATE_NO_SUBMISSION",
    "PASS_E031_DUALHEAD_COMPLEMENT_NO_SUBMISSION",
}
EXPECTED_REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_WEIGHT_BYTES = 1_664_223_428
EXPECTED_BASELINE_AUC = 0.9174298724699895
EXPECTED_BASELINE_PAYLOAD_SHA256 = (
    "7b356156eca446527d89b61cc444ab4622609a63a75eef95d0a3c521fefb99ee"
)
EXPECTED_METHODS = ["probability", "rank"]
EXPECTED_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "e031_summary.json").read_text())
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
    if identity.get("baseline_payload_sha256") != EXPECTED_BASELINE_PAYLOAD_SHA256:
        failures.append(
            f"baseline_payload_sha256={identity.get('baseline_payload_sha256')!r}"
        )

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
    for key in ("grayscale_repeated_rgb", "per_slot_independent"):
        if input_contract.get(key) is not True:
            failures.append(f"{key} is not true")
    if input_contract.get("cross_series_clip_count") != 0:
        failures.append(
            f"cross_series_clip_count={input_contract.get('cross_series_clip_count')!r}"
        )

    heads = summary.get("heads", {})
    mil = heads.get("mil", {})
    transformer = heads.get("transformer", {})
    if heads.get("training_mode") != "frozen_encoder_head_only":
        failures.append(f"training_mode={heads.get('training_mode')!r}")
    if heads.get("parameter_sets_disjoint") is not True:
        failures.append("head parameter sets are not disjoint")
    if heads.get("optimizers_exclude_encoder") is not True:
        failures.append("head optimizers include encoder parameters")
    expected_mil = {
        "type": "target_specific_slot_time_mil",
        "hidden_dim": 256,
        "optimizer_head_only": True,
        "updated": True,
    }
    expected_transformer = {
        "type": "masked_target_token_transformer",
        "hidden_dim": 192,
        "depth": 2,
        "attention_heads": 8,
        "target_tokens": 12,
        "optimizer_head_only": True,
        "updated": True,
    }
    for key, expected in expected_mil.items():
        if mil.get(key) != expected:
            failures.append(f"mil.{key}={mil.get(key)!r}")
    for key, expected in expected_transformer.items():
        if transformer.get(key) != expected:
            failures.append(f"transformer.{key}={transformer.get(key)!r}")
    mil_validation_auc = float(mil.get("validation_macro_auc", math.nan))
    transformer_validation_auc = float(
        transformer.get("validation_macro_auc", math.nan)
    )
    if int(mil.get("best_epoch", -1)) < 0 or int(transformer.get("best_epoch", -1)) < 0:
        failures.append("invalid best head epoch")

    blend = summary.get("blend_selection", {})
    if blend.get("selected_on") != "fixed_non_gold_validation":
        failures.append(f"selected_on={blend.get('selected_on')!r}")
    if blend.get("methods") != EXPECTED_METHODS:
        failures.append(f"methods={blend.get('methods')!r}")
    if blend.get("global_alphas") != EXPECTED_ALPHAS:
        failures.append(f"global_alphas={blend.get('global_alphas')!r}")
    method = blend.get("selected_method")
    alpha = float(blend.get("selected_alpha_transformer", math.nan))
    validation_blend_auc = float(blend.get("validation_macro_auc", math.nan))
    best_single_validation_auc = float(
        blend.get("best_single_validation_macro_auc", math.nan)
    )
    validation_gain = float(blend.get("validation_gain_vs_best_single", math.nan))
    validation_correlation = float(
        blend.get("validation_head_prediction_correlation", math.nan)
    )
    if method not in EXPECTED_METHODS or alpha not in EXPECTED_ALPHAS:
        failures.append(f"undeclared blend={method!r}/{alpha!r}")
    if abs(best_single_validation_auc - max(mil_validation_auc, transformer_validation_auc)) > 1e-12:
        failures.append("inconsistent best validation head")
    if abs(validation_gain - (validation_blend_auc - best_single_validation_auc)) > 1e-12:
        failures.append("inconsistent validation blend gain")

    gold = summary.get("gold58", {})
    baseline_auc = float(gold.get("coatnet_macro_auc", math.nan))
    mil_gold_auc = float(gold.get("mil_macro_auc", math.nan))
    transformer_gold_auc = float(gold.get("transformer_macro_auc", math.nan))
    blend_gold_auc = float(gold.get("blend_macro_auc", math.nan))
    delta_vs_coatnet = float(gold.get("blend_delta_vs_coatnet", math.nan))
    selected_single_name = gold.get("validation_selected_single")
    selected_single_auc = float(gold.get("selected_single_macro_auc", math.nan))
    best_observed_single_auc = float(
        gold.get("best_observed_single_macro_auc", math.nan)
    )
    gain_vs_selected = float(gold.get("blend_gain_vs_selected_single", math.nan))
    gain_vs_best_observed = float(
        gold.get("blend_gain_vs_best_observed_single", math.nan)
    )
    min_target_delta = float(
        gold.get("blend_min_target_delta_vs_coatnet", math.nan)
    )
    gold_head_correlation = float(
        gold.get("mil_transformer_prediction_correlation", math.nan)
    )
    bootstrap_coatnet = float(
        gold.get("paired_bootstrap_vs_coatnet", {}).get("win_rate", math.nan)
    )
    bootstrap_single = float(
        gold.get("paired_bootstrap_vs_selected_single", {}).get("win_rate", math.nan)
    )
    if gold.get("evaluation_count") != 1:
        failures.append(f"gold_evaluation_count={gold.get('evaluation_count')!r}")
    expected_selected_name = (
        "mil" if mil_validation_auc >= transformer_validation_auc else "transformer"
    )
    expected_selected_auc = (
        mil_gold_auc if expected_selected_name == "mil" else transformer_gold_auc
    )
    if selected_single_name != expected_selected_name:
        failures.append(f"selected_single={selected_single_name!r}")
    if abs(selected_single_auc - expected_selected_auc) > 1e-12:
        failures.append("inconsistent selected single Gold AUC")
    if abs(best_observed_single_auc - max(mil_gold_auc, transformer_gold_auc)) > 1e-12:
        failures.append("inconsistent best observed single Gold AUC")
    if abs(delta_vs_coatnet - (blend_gold_auc - baseline_auc)) > 1e-12:
        failures.append("inconsistent blend delta vs CoAtNet")
    if abs(gain_vs_selected - (blend_gold_auc - selected_single_auc)) > 1e-12:
        failures.append("inconsistent blend gain vs selected single")
    if abs(gain_vs_best_observed - (blend_gold_auc - best_observed_single_auc)) > 1e-12:
        failures.append("inconsistent blend gain vs best observed single")
    if abs(baseline_auc - EXPECTED_BASELINE_AUC) > 1e-12:
        failures.append(f"baseline_auc={baseline_auc}")

    core_values = (
        mil_validation_auc,
        transformer_validation_auc,
        validation_blend_auc,
        validation_gain,
        validation_correlation,
        baseline_auc,
        mil_gold_auc,
        transformer_gold_auc,
        blend_gold_auc,
        delta_vs_coatnet,
        gain_vs_selected,
        gain_vs_best_observed,
        min_target_delta,
        gold_head_correlation,
        bootstrap_coatnet,
        bootstrap_single,
    )
    if not finite(*core_values):
        failures.append("non-finite core metric")

    blend_gates = summary.get("blend_gates", {})
    complement_gates = summary.get("complement_gates", {})
    blend_pass = bool(blend_gates) and all(value is True for value in blend_gates.values())
    complement_pass = bool(complement_gates) and all(
        value is True for value in complement_gates.values()
    )
    if decision == "PASS_E031_BLEND_CANDIDATE_NO_SUBMISSION":
        if not blend_pass:
            failures.append(f"blend_gates={blend_gates!r}")
        if (
            not 0 < alpha < 1
            or validation_gain < 0.002
            or validation_correlation > 0.995
            or gain_vs_selected < 0.001
            or gain_vs_best_observed < 0.001
            or delta_vs_coatnet < 0.001
            or bootstrap_coatnet < 0.65
            or bootstrap_single < 0.60
            or min_target_delta < -0.03
        ):
            failures.append("blend numeric thresholds failed")
    elif decision == "PASS_E031_DUALHEAD_COMPLEMENT_NO_SUBMISSION":
        if not complement_pass:
            failures.append(f"complement_gates={complement_gates!r}")
        if (
            not 0 < alpha < 1
            or validation_gain < 0.001
            or validation_blend_auc < 0.75
            or blend_gold_auc < 0.88
            or int(gold.get("blend_target_improve_count_vs_coatnet", -1)) < 3
            or validation_correlation > 0.995
            or gold_head_correlation > 0.995
            or min_target_delta < -0.15
        ):
            failures.append("complement numeric thresholds failed")

    xai = summary.get("xai", {})
    if (
        xai.get("target_count") != 12
        or xai.get("mil_nondegenerate_count") != 12
        or int(xai.get("mil_positive_deletion_count", -1)) < 7
        or float(xai.get("mil_median_deletion_advantage", math.nan)) <= 0
        or xai.get("transformer_nondegenerate_count") != 12
        or int(xai.get("transformer_positive_deletion_count", -1)) < 7
        or float(xai.get("transformer_median_deletion_advantage", math.nan)) <= 0
    ):
        failures.append(f"xai={xai!r}")

    for key in ("submission_created", "checkpoint_exported", "feature_cache_exported"):
        if summary.get(key) is not False:
            failures.append(f"{key}={summary.get(key)!r}")

    required_files = {
        "e031_validation_blend_grid.csv",
        "e031_validation_predictions.npz",
        "e031_gold58_predictions.npz",
        "e031_gold58_target_metrics.csv",
        "e031_head_history.csv",
        "e031_xai_deletion_metrics.csv",
        "e031_gates.csv",
    }
    missing_files = sorted(name for name in required_files if not (output_dir / name).is_file())
    if missing_files:
        failures.append(f"missing_files={missing_files}")

    if failures:
        raise SystemExit("E031 STOP: " + "; ".join(failures))

    return {
        "decision": decision,
        "selected_blend_method": method,
        "selected_alpha_transformer": alpha,
        "validation_mil_macro_auc": mil_validation_auc,
        "validation_transformer_macro_auc": transformer_validation_auc,
        "validation_blend_macro_auc": validation_blend_auc,
        "validation_blend_gain": validation_gain,
        "gold58_mil_macro_auc": mil_gold_auc,
        "gold58_transformer_macro_auc": transformer_gold_auc,
        "gold58_blend_macro_auc": blend_gold_auc,
        "gold58_blend_delta_vs_coatnet": delta_vs_coatnet,
        "bootstrap_vs_coatnet_win_rate": bootstrap_coatnet,
        "all_required_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
