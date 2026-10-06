#!/usr/bin/env python3
"""Fail-closed validation of a downloaded E036 output directory.

Usage: python validate_output.py kaggle/e036_p3_512_scout/output_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
MODEL_SEEDS = [2701, 2702]
VALID_DECISIONS = {
    "PASS_P3_PROMOTE_P4",
    "STOP_P3_EFFICACY_FAILED",
    "STOP_P2_MECHANICAL_FAILED",
}


def main(directory: Path) -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"  ok   {message}")
        else:
            print(f"  FAIL {message}")
            failures.append(message)

    summary_path = directory / "e036_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        return 1
    summary = json.loads(summary_path.read_text())

    print("== artifacts ==")
    for name in ["e036_summary.json", "e036_gates.csv", "e036_identity.json",
                 "e036_cache_metrics.csv", "e036_zero_step_predictions.npz"]:
        check((directory / name).exists(), f"{name} present")
    for seed in MODEL_SEEDS:
        for name in [f"e036_seed{seed}_summary.json", f"e036_predictions_seed{seed}.npz",
                     f"e036_target_metrics_seed{seed}.csv",
                     f"e036_epoch_metrics_seed{seed}.csv"]:
            check((directory / name).exists(), f"{name} present")

    print("== no banned exports ==")
    exported = [path.name for path in directory.iterdir()
                if path.suffix in {".pt", ".pth", ".safetensors"}
                or path.name == "submission.csv"]
    check(not exported, f"no checkpoint or submission exported (found {exported})")

    print("== decision and safety ==")
    decision = summary.get("decision")
    check(decision in VALID_DECISIONS, f"decision {decision!r} is one of the declared outcomes")
    check(summary.get("submission_created") is False, "submission_created is False")
    check(summary.get("checkpoint_exported") is False, "checkpoint_exported is False")
    check(summary.get("feature_cache_exported") is False, "feature_cache_exported is False")

    print("== identity ==")
    identity = summary.get("identity", {})
    check(identity.get("gold_ids_sha256") == EXPECTED_GOLD58_SHA256, "Gold58 id set unchanged")
    check(identity.get("split_seed") == 3601, "split seed 3601")
    check(len(identity.get("train_ids_sha256", "")) == 64, "train id hash recorded")
    check(len(identity.get("valid_ids_sha256", "")) == 64, "validation id hash recorded")

    print("== zero-step reproduction ==")
    zero = summary.get("zero_step", {})
    gap = abs(float(zero.get("gold58_macro_auc", 0.0)) - EXPECTED_SAFE_AUC)
    check(gap <= 0.0005, f"Gold58 zero-step reproduces E028 baseline (gap {gap:.2e})")

    print("== window safety ==")
    windows = summary.get("window_policy", {})
    check(windows.get("cross_series") == 0, "zero cross-series windows")
    check(windows.get("zero_slice") == 0, "zero zero-slice windows")

    probe = summary.get("memory_probe", {})
    check(0.0 < probe.get("peak_gib", 0.0) <= probe.get("limit_gib", 0.0),
          f"training-step probe fits the T4 budget ({probe.get('peak_gib')} GiB)")
    check(probe.get("bag_windows") == 24, "probe measured the 2x12 training bag")
    check(summary.get("configuration", {}).get("amp_dtype") == "torch.float16",
          "training ran in float16, not bfloat16")
    check(summary.get("configuration", {}).get("grad_checkpointing") is True,
          "gradient checkpointing was on")

    print("== gate consistency ==")
    mechanical = summary.get("mechanical_gates", {})
    efficacy = summary.get("efficacy_gates", {})
    check(bool(mechanical), "mechanical gates recorded")
    check(bool(efficacy), "efficacy gates recorded")
    check(summary.get("mechanical_passed") == all(mechanical.values()),
          "mechanical_passed matches its gates")
    check(summary.get("efficacy_passed") == all(efficacy.values()),
          "efficacy_passed matches its gates")
    if decision == "PASS_P3_PROMOTE_P4":
        check(summary.get("mechanical_passed") is True, "pass requires mechanical gates")
        check(summary.get("efficacy_passed") is True, "pass requires efficacy gates")

    print("== per-seed results ==")
    for record in summary.get("seeds", []):
        seed = record["seed"]
        valid = record["valid"]
        veto = record["gold58_veto"]
        updates = record["updates"]
        prefix = f"seed {seed}"
        check(record["best_weights_restored_from_epoch"] == record["best_epoch"],
              f"{prefix}: Gold58 scored the validation-selected epoch")
        check(updates["optimizer_steps"] == updates["expected_optimizer_steps"],
              f"{prefix}: optimizer step count exact")
        check(updates["backbone_changed_tensors"] > 0, f"{prefix}: backbone trained")
        check(updates["head_changed_tensors"] > 0, f"{prefix}: head trained")
        check(np.isfinite(record["losses"]).all(), f"{prefix}: losses finite")
        check(record["resources"]["peak_gpu_gib"] <= 15.0, f"{prefix}: peak GPU within T4")
        check(updates["applied_optimizer_steps"] > 0, f"{prefix}: scaler applied steps")
        check(updates["scaler_skip_rate"] <= 0.02,
              f"{prefix}: scaler skip rate {updates['scaler_skip_rate']:.4f} <= 0.02")
        print(f"    {prefix}: valid delta {valid['delta']:+.6f} "
              f"win {valid['bootstrap']['win_rate']:.3f} "
              f"ci95 [{valid['bootstrap']['ci95'][0]:+.6f}, {valid['bootstrap']['ci95'][1]:+.6f}] "
              f"min-target {valid['min_target_delta']:+.6f} "
              f"| gold delta {veto['delta']:+.6f}")

        predictions = np.load(directory / f"e036_predictions_seed{seed}.npz", allow_pickle=True)
        for key in ["valid_zero", "valid_best", "gold_zero", "gold_best"]:
            check(np.isfinite(predictions[key]).all(), f"{prefix}: {key} finite")
        check(not np.allclose(predictions["valid_zero"], predictions["valid_best"]),
              f"{prefix}: training changed validation predictions")

    print()
    print(f"DECISION {decision}")
    if failures:
        print(f"VALIDATION FAILED ({len(failures)} checks)")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
