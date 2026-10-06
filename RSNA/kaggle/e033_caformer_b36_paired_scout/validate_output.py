#!/usr/bin/env python3
"""Validate E033 Kaggle output integrity and its fail-closed decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_TRAIN_IDS_SHA256 = "0bdc475565a2ff7ea1afc5be9aec69aab0aa615050df796aeac6af4fd2fb051c"
EXPECTED_VALIDATION_IDS_SHA256 = "223ed533e435180acf9f086fff74a12de527977f9bcff7641db7b06d11739565"
EXPECTED_GOLD_IDS_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_BASELINE_PAYLOAD_SHA256 = (
    "7b356156eca446527d89b61cc444ab4622609a63a75eef95d0a3c521fefb99ee"
)
EXPECTED_MODELS = {
    "coatnet": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "caformer": "caformer_b36.sail_in22k_ft_in1k_384",
}
EXPECTED_FEATURE_DIMS = {"coatnet": 1024, "caformer": 768}
HEAD_SEEDS = [33001, 33002, 33003]
PASS_DECISION = "PASS_E033_BACKBONE_SIGNAL_NO_SUBMISSION"
ALLOWED_DECISIONS = {
    "STOP_E033_REQUIRED_GATE_FAILED",
    "STOP_E033_PRE_GOLD_FAILED",
    "STOP_E033_GOLD_SIGNAL_FAILED",
    PASS_DECISION,
}


def finite(*values) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


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
    summary_path = output_dir / "e033_summary.json"
    if not summary_path.is_file():
        raise SystemExit("E033 INVALID: e033_summary.json missing")
    summary = json.loads(summary_path.read_text())

    decision = summary.get("decision")
    if decision not in ALLOWED_DECISIONS:
        failures.append(f"decision={decision!r}")

    required = summary.get("required_gates", {})
    pre_gold = summary.get("pre_gold_gates", {})
    gold_signal = summary.get("gold_signal_gates", {})
    if not required or not pre_gold or not gold_signal:
        failures.append("one or more gate groups missing")

    identity = summary.get("identity", {})
    if identity.get("counts") != {"train": 512, "validation": 128, "gold": 58}:
        failures.append(f"counts={identity.get('counts')!r}")
    expected_identity = {
        "train_ids_sha256": EXPECTED_TRAIN_IDS_SHA256,
        "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
        "gold_ids_sha256": EXPECTED_GOLD_IDS_SHA256,
        "baseline_payload_sha256": EXPECTED_BASELINE_PAYLOAD_SHA256,
        "identity_exact": True,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            failures.append(f"identity.{key}={identity.get(key)!r}")

    input_contract = summary.get("input", {})
    if input_contract.get("shape") != [1, 24, 3, 384, 384]:
        failures.append(f"input.shape={input_contract.get('shape')!r}")
    if input_contract.get("safe_same_series_windows") is not True:
        failures.append("safe same-series contract failed")
    if input_contract.get("cross_series_window_count") != 0:
        failures.append("cross-series windows present")
    if input_contract.get("zero_slice_window_count") != 0:
        failures.append("zero-slice windows present")

    model_rows = summary.get("models", [])
    model_map = {row.get("model_key"): row for row in model_rows if isinstance(row, dict)}
    if set(model_map) != set(EXPECTED_MODELS):
        failures.append(f"model keys={sorted(model_map)}")
    for model_key, architecture in EXPECTED_MODELS.items():
        row = model_map.get(model_key, {})
        if row.get("architecture") != architecture:
            failures.append(f"{model_key}.architecture={row.get('architecture')!r}")
        if row.get("input_size") != [3, 384, 384]:
            failures.append(f"{model_key}.input_size={row.get('input_size')!r}")
        expected_feature_dim = EXPECTED_FEATURE_DIMS[model_key]
        if row.get("num_features") != expected_feature_dim:
            failures.append(f"{model_key}.num_features={row.get('num_features')!r}")
        if row.get("feature_shape") != [640, 24, expected_feature_dim]:
            failures.append(f"{model_key}.feature_shape={row.get('feature_shape')!r}")
        if row.get("feature_finite") is not True:
            failures.append(f"{model_key}.feature_finite={row.get('feature_finite')!r}")
        if not finite(row.get("feature_std"), row.get("windows_per_second"), row.get("peak_gpu_gib")):
            failures.append(f"{model_key} non-finite feature/runtime metric")
        if float(row.get("feature_std", 0.0)) <= 0.01:
            failures.append(f"{model_key}.feature_std={row.get('feature_std')!r}")
        state_hash = row.get("state_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            failures.append(f"{model_key}.state_sha256 invalid")

    training = summary.get("training", {})
    if training.get("mode") != "paired_frozen_backbones_fresh_target_window_mil":
        failures.append(f"training.mode={training.get('mode')!r}")
    if training.get("head_seeds") != HEAD_SEEDS:
        failures.append(f"head_seeds={training.get('head_seeds')!r}")
    if training.get("shared_hidden_dim") != 512:
        failures.append(f"shared_hidden_dim={training.get('shared_hidden_dim')!r}")
    if training.get("selection_split") != "fixed_non_gold_validation":
        failures.append(f"selection_split={training.get('selection_split')!r}")
    head_records = training.get("head_records", [])
    if len(head_records) != 6:
        failures.append(f"head_record_count={len(head_records)}")
    expected_pairs = {(model, seed) for model in EXPECTED_MODELS for seed in HEAD_SEEDS}
    observed_pairs = {
        (row.get("model_key"), row.get("seed"))
        for row in head_records if isinstance(row, dict)
    }
    if observed_pairs != expected_pairs:
        failures.append(f"head_pairs={sorted(observed_pairs)!r}")
    for row in head_records:
        if row.get("updated") is not True:
            failures.append(f"head not updated: {row}")
        if row.get("input_dim") != EXPECTED_FEATURE_DIMS.get(row.get("model_key")):
            failures.append(f"head input_dim invalid: {row}")
        if row.get("hidden_dim") != 512:
            failures.append(f"head hidden_dim invalid: {row}")
        if not 0 <= int(row.get("best_epoch", -1)) < 80:
            failures.append(f"head epoch invalid: {row}")
        if not finite(row.get("validation_macro_auc")):
            failures.append(f"head AUC non-finite: {row}")

    validation = summary.get("validation", {})
    validation_values = [
        validation.get("coatnet_macro_auc"),
        validation.get("caformer_macro_auc"),
        validation.get("delta_caformer_minus_coatnet"),
        validation.get("min_target_delta"),
        validation.get("prediction_correlation"),
    ]
    if not finite(*validation_values):
        failures.append("non-finite validation metric")
    else:
        expected_delta = (
            float(validation["caformer_macro_auc"]) - float(validation["coatnet_macro_auc"])
        )
        if abs(float(validation["delta_caformer_minus_coatnet"]) - expected_delta) > 1e-10:
            failures.append("inconsistent validation delta")
    if not 0 <= int(validation.get("target_wins", -1)) <= 12:
        failures.append(f"validation target_wins={validation.get('target_wins')!r}")

    all_required = bool(required) and all(value is True for value in required.values())
    all_pre_gold = bool(pre_gold) and all(value is True for value in pre_gold.values())
    open_gold = all_required and all_pre_gold
    gold = summary.get("gold58", {})
    if gold.get("gate_passed") != open_gold:
        failures.append("gold.gate_passed inconsistent with required/pre-Gold gates")
    evaluation_count = gold.get("evaluation_count")
    if evaluation_count not in {0, 1}:
        failures.append(f"gold evaluation_count={evaluation_count!r}")
    if open_gold and evaluation_count != 1:
        failures.append("Gold gate passed without one evaluation")
    if not open_gold and evaluation_count != 0:
        failures.append("Gold evaluated after gate failure")

    gold_predictions_path = output_dir / "e033_gold58_predictions.npz"
    gold_targets_path = output_dir / "e033_gold58_target_metrics.csv"
    if evaluation_count == 0:
        if gold.get("metrics") is not None:
            failures.append("Gold metrics present when evaluation was skipped")
        if gold_predictions_path.exists() or gold_targets_path.exists():
            failures.append("Gold artifacts present when evaluation was skipped")
    else:
        metrics = gold.get("metrics") or {}
        metric_values = [
            metrics.get("e029_finetuned_coatnet_macro_auc"),
            metrics.get("paired_frozen_coatnet_macro_auc"),
            metrics.get("paired_frozen_caformer_macro_auc"),
            metrics.get("delta_caformer_minus_frozen_coatnet"),
            metrics.get("min_target_delta"),
            (metrics.get("paired_bootstrap") or {}).get("win_rate"),
        ]
        if metrics.get("evaluation_count") != 1 or not finite(*metric_values):
            failures.append(f"invalid Gold metrics={metrics!r}")
        else:
            expected_delta = (
                float(metrics["paired_frozen_caformer_macro_auc"])
                - float(metrics["paired_frozen_coatnet_macro_auc"])
            )
            if abs(float(metrics["delta_caformer_minus_frozen_coatnet"]) - expected_delta) > 1e-10:
                failures.append("inconsistent Gold delta")
            if abs(float(metrics["e029_finetuned_coatnet_macro_auc"]) - 0.9174298724699895) > 1e-12:
                failures.append("E029 Gold baseline drift")
        if not gold_predictions_path.is_file() or not gold_targets_path.is_file():
            failures.append("Gold artifacts missing after evaluation")

    all_gold_signal = bool(gold_signal) and all(value is True for value in gold_signal.values())
    expected_decision = (
        "STOP_E033_REQUIRED_GATE_FAILED"
        if not all_required
        else "STOP_E033_PRE_GOLD_FAILED"
        if not all_pre_gold
        else PASS_DECISION
        if all_gold_signal
        else "STOP_E033_GOLD_SIGNAL_FAILED"
    )
    if decision != expected_decision:
        failures.append(f"decision mismatch: expected {expected_decision!r}")
    if require_signal and decision != PASS_DECISION:
        failures.append("promotion signal was required")

    for key in ("submission_created", "checkpoint_exported", "feature_cache_exported"):
        if summary.get(key) is not False:
            failures.append(f"{key}={summary.get(key)!r}")

    required_files = {
        "e033_attention_metrics.csv",
        "e033_cache_metrics.csv",
        "e033_feature_metrics.csv",
        "e033_gates.csv",
        "e033_head_history.csv",
        "e033_identity.json",
        "e033_validation_predictions.npz",
        "e033_validation_target_metrics.csv",
    }
    missing = sorted(name for name in required_files if not (output_dir / name).is_file())
    if missing:
        failures.append(f"missing_files={missing}")

    gate_path = output_dir / "e033_gates.csv"
    if gate_path.is_file():
        csv_gates = read_gate_csv(gate_path)
        expected_csv_gates = {
            "required": required,
            "pre_gold": pre_gold,
            "gold_signal": gold_signal,
        }
        if csv_gates != expected_csv_gates:
            failures.append("gate CSV disagrees with summary")

    validation_path = output_dir / "e033_validation_predictions.npz"
    if validation_path.is_file():
        with np.load(validation_path, allow_pickle=True) as payload:
            if payload.files != ["ids", "y_true", "coatnet", "caformer"]:
                failures.append(f"validation keys={payload.files!r}")
            else:
                if ids_sha256(payload["ids"].astype(str).tolist()) != EXPECTED_VALIDATION_IDS_SHA256:
                    failures.append("validation prediction IDs drifted")
                for key in ("y_true", "coatnet", "caformer"):
                    values = payload[key]
                    if values.shape != (128, 12) or not np.isfinite(values).all():
                        failures.append(f"validation {key} invalid shape/finite")
                for key in ("coatnet", "caformer"):
                    values = payload[key]
                    if values.min() < 0 or values.max() > 1:
                        failures.append(f"validation {key} outside [0,1]")

    if gold_predictions_path.is_file():
        with np.load(gold_predictions_path, allow_pickle=True) as payload:
            expected_keys = [
                "ids", "y_true", "e029_finetuned_coatnet",
                "paired_frozen_coatnet", "paired_frozen_caformer",
            ]
            if payload.files != expected_keys:
                failures.append(f"Gold keys={payload.files!r}")
            else:
                if ids_sha256(payload["ids"].astype(str).tolist()) != EXPECTED_GOLD_IDS_SHA256:
                    failures.append("Gold prediction IDs drifted")
                for key in expected_keys[1:]:
                    values = payload[key]
                    if values.shape != (58, 12) or not np.isfinite(values).all():
                        failures.append(f"Gold {key} invalid shape/finite")

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    forbidden = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in forbidden_suffixes
            or "feature_cache" in path.name.lower()
            or path.name == "submission.csv"
        )
    )
    if forbidden:
        failures.append(f"forbidden_files={forbidden}")

    if failures:
        raise SystemExit("E033 INVALID: " + "; ".join(failures))
    print("E033 VALID", decision)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()
    validate(args.output_dir, require_signal=args.require_signal)


if __name__ == "__main__":
    main()
