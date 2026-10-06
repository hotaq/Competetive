#!/usr/bin/env python3
"""Fail-closed validation of a downloaded E037 output directory.

Usage: python validate_output.py kaggle/e037_tta_scout/output_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
BASELINE_ARM = "A0_baseline"
ARM_NAMES = [
    "A0_baseline", "A1_windows_all", "A2_windows_jitter3", "A3_hflip", "A4_hflip_avg",
    "A5_zoom", "A6_zoom_avg", "A7_full_avg", "A8_full_avg_logit", "A9_all_plus_zoom_avg",
]
VALID_DECISIONS = {
    "PASS_TTA_PROMOTE",
    "STOP_TTA_NO_GAIN",
    "STOP_TTA_UNDERPOWERED",
    "STOP_MECHANICAL_FAILED",
}


def main(directory: Path) -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"  ok   {message}")
        else:
            print(f"  FAIL {message}")
            failures.append(message)

    summary_path = directory / "e037_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        return 1
    summary = json.loads(summary_path.read_text())

    print("== artifacts ==")
    for name in ["e037_summary.json", "e037_gates.csv", "e037_identity.json",
                 "e037_cache_metrics.csv", "e037_arm_metrics.csv",
                 "e037_target_metrics.csv", "e037_predictions.npz"]:
        check((directory / name).exists(), f"{name} present")

    print("== no banned exports ==")
    exported = [path.name for path in directory.iterdir()
                if path.suffix in {".pt", ".pth", ".safetensors"}
                or path.name == "submission.csv"]
    check(not exported, f"no checkpoint or submission exported (found {exported})")
    check(summary.get("submission_created") is False, "submission_created is False")
    check(summary.get("checkpoint_exported") is False, "checkpoint_exported is False")
    check(summary.get("feature_cache_exported") is False, "feature_cache_exported is False")
    check(summary.get("configuration", {}).get("trained") is False, "nothing was trained")

    print("== decision ==")
    decision = summary.get("decision")
    check(decision in VALID_DECISIONS, f"decision {decision!r} is one of the declared outcomes")

    print("== identity ==")
    identity = summary.get("identity", {})
    check(identity.get("gold_ids_sha256") == EXPECTED_GOLD58_SHA256, "Gold58 id set unchanged")
    check(identity.get("eval_seed") == 3701, "evaluation split seed 3701")
    check(identity.get("bootstrap_seed") == 3702, "bootstrap seed 3702")
    check(len(identity.get("eval_ids_sha256", "")) == 64, "evaluation id hash recorded")
    counts = summary.get("counts", {})
    check(counts.get("eval") == 2048, f"2,048 evaluation studies (got {counts.get('eval')})")
    check(counts.get("gold") == 58, f"58 Gold studies (got {counts.get('gold')})")

    print("== baseline reproduction ==")
    reproduction = summary.get("reproduction", {})
    gap = abs(float(reproduction.get("gold58_a0_gap", 1.0)))
    check(gap <= 0.0005,
          f"A0 reproduces the E028 baseline (gap {gap:.2e}, "
          f"bit exact {reproduction.get('bit_exact')})")

    print("== window safety ==")
    windows = summary.get("window_policy", {})
    check(windows.get("cross_series") == 0, "zero cross-series windows")
    check(windows.get("zero_slice") == 0, "zero zero-slice windows")
    # Studies below K_EVAL are served repeated windows by `resample_centers`,
    # which is what the production path does too. They are reported, and the
    # share is checked so arm deltas are read against a known dilution.
    short_studies = windows.get("short_pool_studies")
    short_share = windows.get("short_pool_share", 1.0)
    check(short_studies is not None, "the short-candidate-pool count is reported")
    check(short_share <= 0.05,
          f"short-candidate-pool studies stay a small minority "
          f"({short_studies} studies, {short_share:.3%}, "
          f"min {windows.get('min_safe_candidates')})")

    print("== resources ==")
    resources = summary.get("resources", {})
    check(resources.get("projected_total_minutes", 1e9)
          <= resources.get("runtime_budget_minutes", 0.0),
          f"projected runtime {resources.get('projected_total_minutes')} min within budget")
    check(0.0 < resources.get("peak_gpu_gib", 0.0) <= 15.0,
          f"peak GPU within the T4 ({resources.get('peak_gpu_gib')} GiB)")

    print("== gate consistency ==")
    mechanical = summary.get("mechanical_gates", {})
    check(bool(mechanical), "mechanical gates recorded")
    check(summary.get("mechanical_passed") == all(mechanical.values()),
          "mechanical_passed matches its gates")
    for name, value in mechanical.items():
        if not value:
            check(False, f"mechanical gate {name} failed")

    print("== arms ==")
    arms = {record["arm"]: record for record in summary.get("arms", [])}
    check(list(arms) == ARM_NAMES, f"all {len(ARM_NAMES)} arms reported in order")
    check(summary.get("family_size") == len(ARM_NAMES) - 1,
          f"family size {summary.get('family_size')} excludes the baseline")
    baseline = arms.get(BASELINE_ARM, {})
    check(baseline.get("eval_macro_delta") == 0.0, "the baseline has zero delta against itself")
    check(baseline.get("passed") is False, "the baseline never counts as a winner")

    predictions = np.load(directory / "e037_predictions.npz", allow_pickle=True)
    for name in ARM_NAMES:
        record = arms[name]
        bootstrap = record["bootstrap"]
        for split in ["eval", "gold"]:
            values = predictions[f"{split}__{name}"]
            check(np.isfinite(values).all(), f"{name}: {split} predictions finite")
            check(bool(((values >= 0.0) & (values <= 1.0)).all()),
                  f"{name}: {split} predictions are probabilities")
        if name != BASELINE_ARM:
            check(not np.allclose(predictions[f"eval__{name}"],
                                  predictions[f"eval__{BASELINE_ARM}"]),
                  f"{name}: the arm actually changes the predictions")
        print(f"    {name:24s} eval {record['eval_macro_delta']:+.6f} "
              f"win {bootstrap['win_rate']:.3f} "
              f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
              f"half {bootstrap['ci95_halfwidth']:.6f} "
              f"gold {record['gold58_macro_delta']:+.6f} "
              f"{'PASS' if record['passed'] else ''}")

    print("== estimator power ==")
    estimator = summary.get("estimator", {})
    reference = estimator.get("e036_reference_halfwidth_n512", 0.0046)
    median_half = estimator.get("median_ci95_halfwidth")
    check(median_half is not None, "the interval half-width is reported")
    if median_half is not None:
        print(f"    median half-width {median_half:.6f} at n=2048 "
              f"vs {reference:.4f} measured at n=512 in E036")
    check(estimator.get("any_arm_resolves_threshold") is not None,
          "whether the estimator can resolve the threshold is recorded")
    if decision == "STOP_TTA_UNDERPOWERED":
        check(estimator.get("any_arm_resolves_threshold") is False,
              "the underpowered verdict matches the measured intervals")

    print("== decision consistency ==")
    passing = [name for name, record in arms.items() if record.get("passed")]
    if decision == "PASS_TTA_PROMOTE":
        check(bool(passing), "a pass names at least one winning arm")
        check(summary.get("best_arm") in passing, "best_arm is one of the winners")
        check(summary.get("mechanical_passed") is True, "a pass requires the mechanical gates")
        winner = arms.get(summary.get("best_arm"))
        check(winner is not None and all(winner["gates"].values()),
              "the winner cleared every one of its gates")
    else:
        check(not passing, f"a stop means no arm passed (found {passing})")
        check(summary.get("best_arm") is None, "a stop names no winner")

    print("== laterality probe ==")
    probe = summary.get("laterality_probe", {})
    check("mean_eval_delta_laterality_targets" in probe
          and "mean_eval_delta_other_targets" in probe, "both halves of the probe recorded")
    if probe:
        print(f"    horizontal flip: laterality targets "
              f"{probe['mean_eval_delta_laterality_targets']:+.6f}, "
              f"other targets {probe['mean_eval_delta_other_targets']:+.6f}")
        for target, value in probe.get("per_target_eval_delta", {}).items():
            print(f"      {target:20s} {value:+.6f}")

    print()
    print(f"DECISION {decision} best_arm {summary.get('best_arm')}")
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
