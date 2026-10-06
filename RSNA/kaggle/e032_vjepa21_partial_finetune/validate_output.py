#!/usr/bin/env python3
"""Validate E032 output integrity and optionally require a promotion signal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_WEIGHT_BYTES = 1_664_223_428
EXPECTED_WEIGHT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
EXPECTED_BASELINE_PAYLOAD_SHA256 = (
    "7b356156eca446527d89b61cc444ab4622609a63a75eef95d0a3c521fefb99ee"
)
EXPECTED_REFERENCE_PAYLOAD_SHA256 = (
    "d5b0f6f48661bd6c9c2693859593df6c8e5c5297835cf4b387fccf5d5d95657f"
)
EXPECTED_TRAIN_IDS_SHA256 = "0bdc475565a2ff7ea1afc5be9aec69aab0aa615050df796aeac6af4fd2fb051c"
EXPECTED_VALIDATION_IDS_SHA256 = "223ed533e435180acf9f086fff74a12de527977f9bcff7641db7b06d11739565"
EXPECTED_GOLD_IDS_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
PASS_DECISION = "PASS_E032_FINETUNE_SIGNAL_NO_SUBMISSION"
ALLOWED_DECISIONS = {
    "STOP_E032_REQUIRED_GATE_FAILED",
    "STOP_E032_PRE_GOLD_FAILED",
    "STOP_E032_GOLD_SIGNAL_FAILED",
    PASS_DECISION,
}


def finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def read_gate_csv(path: Path) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["passed"] not in {"True", "False"}:
                raise ValueError(f"invalid gate boolean: {row}")
            result.setdefault(row["group"], {})[row["gate"]] = row["passed"] == "True"
    return result


def validate(output_dir: Path, require_signal: bool = False) -> dict:
    failures: list[str] = []
    summary_path = output_dir / "e032_summary.json"
    if not summary_path.is_file():
        raise SystemExit("E032 INVALID: e032_summary.json missing")
    summary = json.loads(summary_path.read_text())

    decision = summary.get("decision")
    if decision not in ALLOWED_DECISIONS:
        failures.append(f"decision={decision!r}")

    required = summary.get("required_gates", {})
    pre_gold = summary.get("pre_gold_gates", {})
    gold_signal = summary.get("gold_signal_gates", {})
    if not required:
        failures.append("required_gates missing")
    failed_required = sorted(name for name, passed in required.items() if passed is not True)
    if failed_required:
        failures.append(f"failed_required={failed_required}")
    if not pre_gold or not gold_signal:
        failures.append("pre_gold_gates or gold_signal_gates missing")

    identity = summary.get("identity", {})
    if identity.get("counts") != {"train": 512, "validation": 128, "gold": 58}:
        failures.append(f"counts={identity.get('counts')!r}")
    expected_identity = {
        "train_ids_sha256": EXPECTED_TRAIN_IDS_SHA256,
        "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
        "gold_ids_sha256": EXPECTED_GOLD_IDS_SHA256,
        "baseline_payload_sha256": EXPECTED_BASELINE_PAYLOAD_SHA256,
        "reference_payload_sha256": EXPECTED_REFERENCE_PAYLOAD_SHA256,
        "payload_identity_exact": True,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            failures.append(f"identity.{key}={identity.get(key)!r}")

    model = summary.get("model", {})
    expected_model = {
        "repo_commit": EXPECTED_REPO_COMMIT,
        "checkpoint_bytes": EXPECTED_WEIGHT_BYTES,
        "checkpoint_sha256": EXPECTED_WEIGHT_SHA256,
        "unfrozen_blocks": [10, 11],
        "unfrozen_output_norm": 3,
        "trainable_prefixes": ["blocks.10.", "blocks.11.", "norms_block.3."],
        "frozen_encoder_unchanged": True,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            failures.append(f"model.{key}={model.get(key)!r}")
    trainable_count = int(model.get("trainable_encoder_parameter_count", -1))
    trainable_ratio = float(model.get("trainable_encoder_ratio", math.nan))
    drift = float(model.get("trainable_relative_l2_drift", math.nan))
    if not 10_000_000 <= trainable_count <= 20_000_000:
        failures.append(f"trainable_count={trainable_count}")
    if not 0 < trainable_ratio <= 0.25:
        failures.append(f"trainable_ratio={trainable_ratio}")
    if not 0 < drift <= 0.05:
        failures.append(f"trainable_drift={drift}")
    if model.get("encoder_initial_sha256") == model.get("encoder_final_sha256"):
        failures.append("encoder full hash did not change")

    input_contract = summary.get("input", {})
    expected_input = {
        "shape": [1, 3, 16, 384, 384],
        "feature_shape_per_slot": [8, 3072],
        "grayscale_repeated_rgb": True,
        "per_slot_independent": True,
        "cross_series_clip_count": 0,
    }
    for key, expected in expected_input.items():
        if input_contract.get(key) != expected:
            failures.append(f"input.{key}={input_contract.get(key)!r}")

    training = summary.get("training", {})
    if training.get("mode") != "frozen_mil_warmstart_then_last2_encoder_blocks_plus_output_norm":
        failures.append(f"training.mode={training.get('mode')!r}")
    if training.get("gradient_accumulation") != 4:
        failures.append(f"gradient_accumulation={training.get('gradient_accumulation')!r}")
    if training.get("optimizer_parameter_set_exact") is not True:
        failures.append("optimizer parameter set is not exact")
    if training.get("optimizer_excludes_frozen_encoder") is not True:
        failures.append("optimizer includes frozen encoder")
    epochs_ran = int(training.get("finetune_epochs_ran", -1))
    if not 1 <= epochs_ran <= 4:
        failures.append(f"finetune_epochs_ran={epochs_ran}")

    validation = summary.get("validation", {})
    if validation.get("selected_on") != "fixed_non_gold_validation":
        failures.append(f"selected_on={validation.get('selected_on')!r}")
    e031_auc = float(validation.get("e031_reference_macro_auc", math.nan))
    warmup_auc = float(validation.get("warmup_macro_auc", math.nan))
    finetuned_auc = float(validation.get("finetuned_macro_auc", math.nan))
    gain_warmup = float(validation.get("gain_vs_warmup", math.nan))
    gain_e031 = float(validation.get("gain_vs_e031", math.nan))
    min_target_delta = float(validation.get("min_target_delta_vs_warmup", math.nan))
    correlation = float(validation.get("prediction_correlation_vs_warmup", math.nan))
    if not finite(
        e031_auc, warmup_auc, finetuned_auc, gain_warmup,
        gain_e031, min_target_delta, correlation,
    ):
        failures.append("non-finite validation metric")
    if abs(gain_warmup - (finetuned_auc - warmup_auc)) > 1e-12:
        failures.append("inconsistent validation gain vs warmup")
    if abs(gain_e031 - (finetuned_auc - e031_auc)) > 1e-12:
        failures.append("inconsistent validation gain vs E031")

    xai = summary.get("xai", {})
    if (
        xai.get("target_count") != 12
        or xai.get("nondegenerate_count") != 12
        or not finite(xai.get("median_deletion_advantage", math.nan))
    ):
        failures.append(f"xai={xai!r}")

    pre_gold_pass = bool(pre_gold) and all(value is True for value in pre_gold.values())
    gold = summary.get("gold58", {})
    gold_gate_passed = gold.get("gate_passed")
    evaluation_count = gold.get("evaluation_count")
    if gold_gate_passed != pre_gold_pass:
        failures.append("inconsistent Gold gate flag")
    if evaluation_count not in {0, 1}:
        failures.append(f"gold_evaluation_count={evaluation_count!r}")
    if pre_gold_pass and evaluation_count != 1:
        failures.append("pre-Gold passed without exactly one Gold evaluation")
    if not pre_gold_pass and evaluation_count != 0:
        failures.append("Gold was evaluated after a failed pre-Gold gate")

    gold_prediction_path = output_dir / "e032_gold58_predictions.npz"
    gold_target_path = output_dir / "e032_gold58_target_metrics.csv"
    if evaluation_count == 0:
        if gold.get("metrics") is not None:
            failures.append("Gold metrics present with evaluation_count=0")
        if gold_prediction_path.exists() or gold_target_path.exists():
            failures.append("Gold artifacts present after skipped evaluation")
    else:
        metrics = gold.get("metrics") or {}
        if metrics.get("evaluation_count") != 1:
            failures.append("Gold metrics evaluation_count mismatch")
        gold_values = [
            metrics.get("coatnet_macro_auc", math.nan),
            metrics.get("e031_frozen_mil_macro_auc", math.nan),
            metrics.get("e032_finetuned_macro_auc", math.nan),
            metrics.get("delta_vs_e031", math.nan),
            metrics.get("delta_vs_coatnet", math.nan),
            metrics.get("min_target_delta_vs_e031", math.nan),
            (metrics.get("paired_bootstrap_vs_e031") or {}).get("win_rate", math.nan),
        ]
        if not finite(*gold_values):
            failures.append("non-finite Gold metric")
        if not gold_prediction_path.is_file() or not gold_target_path.is_file():
            failures.append("Gold artifacts missing after evaluation")

    all_gold_signal = bool(gold_signal) and all(value is True for value in gold_signal.values())
    expected_decision = (
        "STOP_E032_PRE_GOLD_FAILED"
        if not pre_gold_pass
        else PASS_DECISION
        if all_gold_signal
        else "STOP_E032_GOLD_SIGNAL_FAILED"
    )
    if decision != expected_decision:
        failures.append(f"decision mismatch: expected {expected_decision!r}")
    if require_signal and decision != PASS_DECISION:
        failures.append("promotion signal was required")

    for key in ("submission_created", "checkpoint_exported", "feature_cache_exported"):
        if summary.get(key) is not False:
            failures.append(f"{key}={summary.get(key)!r}")

    required_files = {
        "e032_cache_metrics.csv",
        "e032_clip_metrics.csv",
        "e032_finetune_history.csv",
        "e032_gates.csv",
        "e032_identity.json",
        "e032_validation_predictions.npz",
        "e032_validation_target_metrics.csv",
        "e032_warmup_history.csv",
        "e032_xai_deletion_metrics.csv",
    }
    missing_files = sorted(name for name in required_files if not (output_dir / name).is_file())
    if missing_files:
        failures.append(f"missing_files={missing_files}")

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    forbidden_files = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and (path.suffix.lower() in forbidden_suffixes or "feature_cache" in path.name.lower())
    )
    if forbidden_files:
        failures.append(f"forbidden_files={forbidden_files}")
    if any(path.name == "submission.csv" for path in output_dir.iterdir()):
        failures.append("submission.csv exists")

    validation_path = output_dir / "e032_validation_predictions.npz"
    if validation_path.is_file():
        with np.load(validation_path, allow_pickle=True) as payload:
            expected_keys = ["ids", "y_true", "e031_reference", "warmup", "finetuned"]
            if payload.files != expected_keys:
                failures.append(f"validation prediction keys={payload.files!r}")
            else:
                if payload["y_true"].shape != (128, 12):
                    failures.append(f"validation y shape={payload['y_true'].shape}")
                for key in expected_keys[2:]:
                    if payload[key].shape != (128, 12) or not np.isfinite(payload[key]).all():
                        failures.append(f"validation {key} invalid")
                if ids_sha256([str(value) for value in payload["ids"].tolist()]) != EXPECTED_VALIDATION_IDS_SHA256:
                    failures.append("validation prediction IDs mismatch")

    if gold_prediction_path.is_file():
        with np.load(gold_prediction_path, allow_pickle=True) as payload:
            expected_keys = ["ids", "y_true", "coatnet", "e031_frozen_mil", "e032_finetuned"]
            if payload.files != expected_keys:
                failures.append(f"Gold prediction keys={payload.files!r}")
            elif any(payload[key].shape != (58, 12) for key in expected_keys[1:]):
                failures.append("Gold prediction shape mismatch")
            if ids_sha256([str(value) for value in payload["ids"].tolist()]) != EXPECTED_GOLD_IDS_SHA256:
                failures.append("Gold prediction IDs mismatch")

    gates_path = output_dir / "e032_gates.csv"
    if gates_path.is_file():
        try:
            csv_gates = read_gate_csv(gates_path)
        except (KeyError, ValueError) as error:
            failures.append(f"invalid gate CSV: {error}")
        else:
            expected_groups = {
                "required": required,
                "pre_gold": pre_gold,
                "gold_signal": gold_signal,
            }
            if csv_gates != expected_groups:
                failures.append("gate CSV does not match summary")

    if failures:
        raise SystemExit("E032 INVALID: " + "; ".join(failures))

    return {
        "decision": decision,
        "validation_e031_reference_macro_auc": e031_auc,
        "validation_warmup_macro_auc": warmup_auc,
        "validation_finetuned_macro_auc": finetuned_auc,
        "validation_gain_vs_warmup": gain_warmup,
        "pre_gold_pass": pre_gold_pass,
        "gold_evaluation_count": evaluation_count,
        "promotion_signal": decision == PASS_DECISION,
        "all_required_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir, require_signal=args.require_signal), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
