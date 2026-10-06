#!/usr/bin/env python3
"""Fail closed unless a downloaded E029 run satisfies every P2 gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_DECISION = "PASS_P2_PROMOTE_P3"
EXPECTED_SAFE_AUC = 0.9174298724699895
MIN_GAIN = 0.0010
MIN_BOOTSTRAP_WIN_RATE = 0.65
MAX_TARGET_DROP = 0.030

REQUIRED_GATES = {
    "checkpoint_name_exact",
    "checkpoint_sha256_recorded",
    "fixed_train_count_40",
    "fixed_train_ids_exact",
    "gold_ids_exact",
    "gold_train_overlap_zero",
    "zero_step_reproduced",
    "cache_pixel_max_abs_zero",
    "cross_series_window_count_zero",
    "zero_slice_window_count_zero",
    "sampler_covers_all_available_slots",
    "optimizer_parameter_match",
    "optimizer_head_only",
    "optimizer_steps_exact",
    "gradient_norms_finite",
    "losses_finite",
    "loss_decreased",
    "backbone_unchanged",
    "head_updated",
    "best_gain_ge_001",
    "paired_bootstrap_win_ge_065",
    "max_target_drop_le_003",
    "peak_gpu_le_15_gib",
    "throughput_positive",
}


def validate(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "e029_p2_summary.json").read_text())
    failures: list[str] = []

    if summary.get("decision") != EXPECTED_DECISION:
        failures.append(f"decision={summary.get('decision')!r}")

    gates = summary.get("gates", {})
    failed_gates = sorted(name for name, passed in gates.items() if passed is not True)
    missing_gates = sorted(REQUIRED_GATES - set(gates))
    if not gates:
        failures.append("gates missing")
    elif failed_gates:
        failures.append(f"failed_gates={failed_gates}")
    if missing_gates:
        failures.append(f"missing_gates={missing_gates}")

    counts = summary.get("counts", {})
    if (counts.get("train"), counts.get("gold"), counts.get("optimizer_steps")) != (40, 58, 40):
        failures.append(f"counts={counts}")

    window_policy = summary.get("window_policy", {})
    if window_policy.get("train_windows") != 12:
        failures.append(f"train_windows={window_policy.get('train_windows')!r}")
    if window_policy.get("train_sampler") != "slot_balanced_all_available":
        failures.append(f"train_sampler={window_policy.get('train_sampler')!r}")
    if summary.get("training_mode") != "head_only_frozen_backbone":
        failures.append(f"training_mode={summary.get('training_mode')!r}")

    learning_rates = summary.get("learning_rates", {})
    if learning_rates.get("backbone") != 0.0 or learning_rates.get("head") != 1e-4:
        failures.append(f"learning_rates={learning_rates}")

    zero_auc = float(summary.get("zero_step", {}).get("macro_auc", math.nan))
    best_auc = float(summary.get("best", {}).get("macro_auc", math.nan))
    best_delta = float(summary.get("best", {}).get("delta_vs_zero", math.nan))
    min_target_delta = float(summary.get("best", {}).get("min_target_delta", math.nan))
    win_rate = float(summary.get("paired_bootstrap", {}).get("win_rate", math.nan))
    if not all(
        math.isfinite(value)
        for value in (zero_auc, best_auc, best_delta, min_target_delta, win_rate)
    ):
        failures.append("non-finite core metric")
    if abs(zero_auc - EXPECTED_SAFE_AUC) > 0.0005:
        failures.append(f"zero_step_macro_auc={zero_auc}")
    if abs(best_delta - (best_auc - zero_auc)) > 1e-12:
        failures.append(f"inconsistent_best_delta={best_delta}")
    if best_delta < MIN_GAIN:
        failures.append(f"best_delta={best_delta}")
    if win_rate < MIN_BOOTSTRAP_WIN_RATE:
        failures.append(f"paired_bootstrap_win_rate={win_rate}")
    if min_target_delta < -MAX_TARGET_DROP:
        failures.append(f"min_target_delta={min_target_delta}")

    updates = summary.get("updates", {})
    if updates.get("backbone_changed_tensors") != 0:
        failures.append(f"backbone_changed_tensors={updates.get('backbone_changed_tensors')!r}")
    if not isinstance(updates.get("head_changed_tensors"), int) or updates["head_changed_tensors"] <= 0:
        failures.append(f"head_changed_tensors={updates.get('head_changed_tensors')!r}")
    if updates.get("optimizer_head_only") is not True:
        failures.append(f"optimizer_head_only={updates.get('optimizer_head_only')!r}")

    if summary.get("optimizer_created") is not True:
        failures.append("optimizer_created is not true")
    if summary.get("submission_created") is not False:
        failures.append("submission_created is not false")
    if summary.get("checkpoint_exported") is not False:
        failures.append("checkpoint_exported is not false")

    if failures:
        raise SystemExit("E029 STOP: " + "; ".join(failures))

    return {
        "decision": summary["decision"],
        "zero_step_macro_auc": zero_auc,
        "best_macro_auc": best_auc,
        "best_delta": best_delta,
        "paired_bootstrap_win_rate": win_rate,
        "all_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
