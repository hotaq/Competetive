#!/usr/bin/env python3
"""Fail-closed validation of a downloaded E038 output directory.

Usage: python validate_output.py kaggle/e038_swa_ensemble_scout/output_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_BASE_SHA256 = "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded"
BASELINE_ARM = "B0_base"
ARM_NAMES = ["B0_base", "B1_swa", "B2_ens_prob", "B3_ens_logit",
             "B4_base_all", "B5_ens_prob_all"]
# The arms that only differ if the two checkpoints differ. Their collapsing onto
# the baseline is E038's null result, so it is printed, never failed.
SWA_ARMS = ["B1_swa", "B2_ens_prob", "B3_ens_logit", "B5_ens_prob_all"]
E037_A1_DELTA = 0.0010437443710005212
E037_A1_CI95 = [0.0006198022190293764, 0.0015000035609342985]
VALID_DECISIONS = {
    "PASS_ENSEMBLE_PROMOTE",
    "STOP_ENSEMBLE_NO_GAIN",
    "STOP_ENSEMBLE_UNDERPOWERED",
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

    summary_path = directory / "e038_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        return 1
    summary = json.loads(summary_path.read_text())

    print("== artifacts ==")
    for name in ["e038_summary.json", "e038_gates.csv", "e038_identity.json",
                 "e038_cache_metrics.csv", "e038_arm_metrics.csv",
                 "e038_target_metrics.csv", "e038_predictions.npz"]:
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
    check(identity.get("base_checkpoint_sha256") == EXPECTED_BASE_SHA256,
          "the base checkpoint is the 0.927 file")
    check(identity.get("swa_checkpoint_sha256") != identity.get("base_checkpoint_sha256"),
          "the SWA checkpoint is a different file")
    check(identity.get("eval_seed") == 3701, "evaluation split seed 3701")
    check(identity.get("bootstrap_seed") == 3802, "bootstrap seed 3802")
    check(len(identity.get("eval_ids_sha256", "")) == 64, "evaluation id hash recorded")
    check(len(identity.get("e037_split_sha256", "")) == 64, "E037's split hash recorded")
    counts = summary.get("counts", {})
    check(counts.get("eval") == 2301, f"2,301 evaluation studies (got {counts.get('eval')})")
    check(counts.get("gold") == 58, f"58 Gold studies (got {counts.get('gold')})")
    check(counts.get("e037_eval") == 2048, "E037's 2,048-study draw was reconstructed")
    check(counts.get("labelled_non_gold_pool") == 4349,
          f"the pool is 4,349 studies (got {counts.get('labelled_non_gold_pool')})")

    print("== the split really is E037's untouched complement ==")
    mechanical = summary.get("mechanical_gates", {})
    check(mechanical.get("e037_split_reconstructed") is True,
          "E037's draw reconstructed to its recorded hash")
    check(mechanical.get("eval_disjoint_from_e037") is True,
          "no study E037 scored appears in E038's split")
    check(mechanical.get("gold_overlap_zero") is True, "Gold58 stays out of the split")

    print("== baseline reproduction ==")
    reproduction = summary.get("reproduction", {})
    gap = abs(float(reproduction.get("gold58_b0_gap", 1.0)))
    check(gap <= 0.0005,
          f"B0 reproduces the E028 baseline (gap {gap:.2e}, "
          f"bit exact {reproduction.get('bit_exact')})")

    print("== the second checkpoint ==")
    swa_load = summary.get("swa_load", {})
    divergence = summary.get("parameter_divergence", {})
    check(swa_load.get("arch") is not None, "the SWA architecture is recorded")
    print(f"    arch {swa_load.get('arch')} res {swa_load.get('res')} "
          f"entries {swa_load.get('state_entries')} "
          f"normalisations {swa_load.get('normalisations')}")
    check(divergence.get("max_abs_param_diff", 0.0) > 0.0,
          f"the SWA weights differ from base "
          f"(max |dw| {divergence.get('max_abs_param_diff')})")
    print(f"    mean |dw| {divergence.get('mean_abs_param_diff')} over "
          f"{divergence.get('parameters_compared')} parameters, "
          f"{divergence.get('identical_tensors')}/{divergence.get('float_tensors')} "
          f"tensors identical")

    print("== how much room the ensemble had ==")
    ensemble = summary.get("ensemble_divergence", {})
    check("mean_spearman" in ensemble, "base-vs-SWA rank agreement recorded")
    print(f"    mean spearman {ensemble.get('mean_spearman')}, "
          f"min {ensemble.get('min_spearman')}, "
          f"mean |dp| {ensemble.get('mean_abs_prob_diff')}, "
          f"max |dp| {ensemble.get('max_abs_prob_diff')}")
    separable = ensemble.get("swa_arms_differ_from_baseline")
    check(separable is not None, "whether the SWA arms are separable at all is recorded")
    if separable is False:
        print("    NOTE the SWA arms match the baseline: the two members predict alike, "
              "which is a result about the checkpoints, not a broken run")

    print("== window safety ==")
    windows = summary.get("window_policy", {})
    check(windows.get("cross_series") == 0, "zero cross-series windows")
    check(windows.get("zero_slice") == 0, "zero zero-slice windows")
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

    predictions = np.load(directory / "e038_predictions.npz", allow_pickle=True)
    for name in ARM_NAMES:
        record = arms[name]
        bootstrap = record["bootstrap"]
        for split in ["eval", "gold"]:
            values = predictions[f"{split}__{name}"]
            check(np.isfinite(values).all(), f"{name}: {split} predictions finite")
            check(bool(((values >= 0.0) & (values <= 1.0)).all()),
                  f"{name}: {split} predictions are probabilities")
        if name == "B4_base_all":
            check(not np.allclose(predictions[f"eval__{name}"],
                                  predictions[f"eval__{BASELINE_ARM}"]),
                  "B4 changes the predictions - it must, it pools more windows")
        elif name in SWA_ARMS:
            identical = np.allclose(predictions[f"eval__{name}"],
                                    predictions[f"eval__{BASELINE_ARM}"])
            if identical:
                print(f"    NOTE {name} is indistinguishable from the baseline")
        print(f"    {name:20s} eval {record['eval_macro_delta']:+.6f} "
              f"win {bootstrap['win_rate']:.3f} "
              f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
              f"half {bootstrap['ci95_halfwidth']:.6f} "
              f"gold {record['gold58_macro_delta']:+.6f} "
              f"{'PASS' if record['passed'] else ''}")

    print("== estimator power ==")
    estimator = summary.get("estimator", {})
    reference = estimator.get("e037_reference_halfwidth_n2048", 0.000849)
    median_half = estimator.get("median_ci95_halfwidth")
    check(median_half is not None, "the interval half-width is reported")
    if median_half is not None:
        print(f"    median half-width {median_half:.6f} at n=2301 "
              f"vs {reference:.6f} measured at n=2048 in E037")
    check(estimator.get("any_arm_resolves_threshold") is not None,
          "whether the estimator can resolve the threshold is recorded")
    if decision == "STOP_ENSEMBLE_UNDERPOWERED":
        check(estimator.get("any_arm_resolves_threshold") is False,
              "the underpowered verdict matches the measured intervals")

    print("== decision consistency ==")
    passing = [name for name, record in arms.items() if record.get("passed")]
    if decision == "PASS_ENSEMBLE_PROMOTE":
        check(bool(passing), "a pass names at least one winning arm")
        check(summary.get("best_arm") in passing, "best_arm is one of the winners")
        check(summary.get("mechanical_passed") is True, "a pass requires the mechanical gates")
        winner = arms.get(summary.get("best_arm"))
        check(winner is not None and all(winner["gates"].values()),
              "the winner cleared every one of its gates")
    else:
        check(not passing, f"a stop means no arm passed (found {passing})")
        check(summary.get("best_arm") is None, "a stop names no winner")

    print("== E037's A1 out of sample (reported, not gated) ==")
    replication = summary.get("e037_a1_replication", {})
    check(bool(replication), "the replication block is present")
    if replication:
        check(replication.get("e037_delta") == E037_A1_DELTA,
              "the E037 figure it is compared against is the recorded one")
        check(replication.get("splits_disjoint") is True,
              "the comparison is between disjoint splits")
        print(f"    E037 n=2048 {replication.get('e037_delta'):+.6f} "
              f"ci95 [{E037_A1_CI95[0]:+.6f}, {E037_A1_CI95[1]:+.6f}]")
        e038_ci = replication.get("e038_ci95", [float("nan"), float("nan")])
        print(f"    E038 n=2301 {replication.get('e038_delta'):+.6f} "
              f"ci95 [{e038_ci[0]:+.6f}, {e038_ci[1]:+.6f}]")
        print(f"    sign agrees {replication.get('sign_agrees')}, "
              f"intervals overlap {replication.get('intervals_overlap')}, "
              f"inside E037's interval {replication.get('e038_delta_inside_e037_ci')}")

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
