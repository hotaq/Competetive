#!/usr/bin/env python3
"""Fail-closed validation of a downloaded E039 output directory.

Usage: python validate_output.py kaggle/e039_checkpoint_family_scout/output_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_SHA256 = {
    "v4_base": "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded",
    "v4_swa": "d8bb0f8751b4bb65750257869ddc4c7a3c0cdfc6e62596fc68193919406c53eb",
    "v5_maxspan": "20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a",
    "v9_finespacing": "0850149a98cf04c7bacc1ea99b3af048102a256de14766b129b1d7255c02de7b",
}
MODEL_LABELS = ["v4_base", "v4_swa", "v5_maxspan", "v9_finespacing"]
BASELINE_ARM = "C0_v4_base"
INCUMBENT_ARM = "C3_v4_swa_all"
ARM_NAMES = ["C0_v4_base", "C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all",
             "C4_v5", "C5_v5_all", "C6_v9", "C7_v9_all",
             "C8_ens_v4swa_v5_all", "C9_ens_v4swa_v9_all", "C10_ens_four_all"]
# The three arms E038 already measured on this exact split. They are the run's
# self-test, not candidates, so their deltas are checked against E038's and they
# are excluded from promotion.
REPRODUCTION_ARMS = ["C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all"]
PROMOTABLE_ARMS = [name for name in ARM_NAMES
                   if name != BASELINE_ARM and name not in REPRODUCTION_ARMS]
# Arms that pool more windows must move the predictions whatever the weights
# hold. Arms that only swap the checkpoint may collapse onto the baseline; that
# would be a result about the checkpoints, so it is printed, never failed.
ALL_WINDOW_ARMS = ["C1_v4_base_all", "C3_v4_swa_all", "C5_v5_all", "C7_v9_all"]
E038_EXPECTED_DELTA = {
    "C1_v4_base_all": 0.0010661988349757179,
    "C2_v4_swa": 0.0020829288947438185,
    "C3_v4_swa_all": 0.0030813625662755584,
}
E038_B0_EVAL_MACRO = 0.9079593249126852
E038_EVAL_IDS_SHA256 = "70b80686cecebf82817cc2dea96ba06f67bb9f93924b543315efd116a6f3f225"
E037_A1_DELTA = 0.0010437443710005212
E037_A1_CI95 = [0.0006198022190293764, 0.0015000035609342985]
MIN_VALID_GAIN = 0.0020
VALID_DECISIONS = {
    "PASS_PROMOTE_NEW_CHECKPOINT",
    "STOP_INCUMBENT_HOLDS",
    "STOP_FAMILY_UNDERPOWERED",
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

    summary_path = directory / "e039_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        return 1
    summary = json.loads(summary_path.read_text())

    print("== artifacts ==")
    for name in ["e039_summary.json", "e039_gates.csv", "e039_identity.json",
                 "e039_cache_metrics.csv", "e039_arm_metrics.csv",
                 "e039_target_metrics.csv", "e039_predictions.npz"]:
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
    decision_relaxed = summary.get("decision_relaxed")
    check(decision in VALID_DECISIONS, f"decision {decision!r} is one of the declared outcomes")
    check(decision_relaxed in VALID_DECISIONS,
          f"relaxed decision {decision_relaxed!r} is one of the declared outcomes")
    check(summary.get("verdicts_agree")
          == (decision == decision_relaxed
              and summary.get("best_arm") == summary.get("best_arm_relaxed")),
          "verdicts_agree matches the two verdicts it summarises")
    if not summary.get("verdicts_agree"):
        print(f"    NOTE strict {decision} ({summary.get('best_arm')}) and "
              f"relaxed {decision_relaxed} ({summary.get('best_arm_relaxed')}) differ: "
              f"Gold58's 58 studies are the only thing between them")

    print("== identity ==")
    identity = summary.get("identity", {})
    check(identity.get("gold_ids_sha256") == EXPECTED_GOLD58_SHA256, "Gold58 id set unchanged")
    checkpoint_sha256 = identity.get("checkpoint_sha256", {})
    check(checkpoint_sha256 == EXPECTED_SHA256,
          "all four checkpoints are the files E039 declared")
    check(len(set(checkpoint_sha256.values())) == len(MODEL_LABELS),
          "the four checkpoints are four distinct files")
    check(identity.get("eval_seed") == 3701, "evaluation split seed 3701")
    check(identity.get("bootstrap_seed") == 3802, "bootstrap seed 3802")
    check(identity.get("eval_ids_sha256") == E038_EVAL_IDS_SHA256,
          "the split is byte-for-byte the one E038 measured on")
    check(identity.get("e038_split_matched") is True, "and the notebook agrees that it is")
    check(len(identity.get("e037_split_sha256", "")) == 64, "E037's split hash recorded")
    counts = summary.get("counts", {})
    check(counts.get("eval") == 2301, f"2,301 evaluation studies (got {counts.get('eval')})")
    check(counts.get("gold") == 58, f"58 Gold studies (got {counts.get('gold')})")
    check(counts.get("e037_eval") == 2048, "E037's 2,048-study draw was reconstructed")
    check(counts.get("checkpoints") == 4, f"four checkpoints (got {counts.get('checkpoints')})")
    check(counts.get("labelled_non_gold_pool") == 4349,
          f"the pool is 4,349 studies (got {counts.get('labelled_non_gold_pool')})")

    print("== the split really is E037's untouched complement ==")
    mechanical = summary.get("mechanical_gates", {})
    check(mechanical.get("e037_split_reconstructed") is True,
          "E037's draw reconstructed to its recorded hash")
    check(mechanical.get("eval_disjoint_from_e037") is True,
          "no study E037 scored appears in E039's split")
    check(mechanical.get("gold_overlap_zero") is True, "Gold58 stays out of the split")

    print("== baseline reproduction ==")
    reproduction = summary.get("reproduction", {})
    gap = abs(float(reproduction.get("gold58_b0_gap", 1.0)))
    check(gap <= 0.0005,
          f"C0 reproduces the E028 Gold58 baseline (gap {gap:.2e}, "
          f"bit exact {reproduction.get('bit_exact')})")

    print("== the three arms E038 already answered ==")
    e038 = summary.get("e038_reproduction", {})
    check(bool(e038), "the reproduction block is present")
    baseline_gap = float(e038.get("baseline_gap", 1.0))
    check(e038.get("baseline_reproduced") is True,
          f"the eval baseline is E038's {E038_B0_EVAL_MACRO:.6f} "
          f"(measured {e038.get('baseline_macro')}, gap {baseline_gap:+.2e})")
    check(e038.get("baseline_expected") == E038_B0_EVAL_MACRO,
          "and the figure it was compared against is E038's published one")
    reproduced_arms = e038.get("arms", {})
    check(set(reproduced_arms) == set(REPRODUCTION_ARMS),
          "all three reproduction arms are reported")
    for name in REPRODUCTION_ARMS:
        record = reproduced_arms.get(name, {})
        check(record.get("e038_delta") == E038_EXPECTED_DELTA[name],
              f"{name}: compared against E038's published delta")
        check(record.get("reproduced") is True,
              f"{name}: E038 {record.get('e038_delta'):+.6f} "
              f"E039 {record.get('e039_delta'):+.6f} "
              f"gap {record.get('gap'):+.2e}")
    check(e038.get("arms_reproduced") is True,
          "every reproduction arm came back, so the rest of the run is comparable")

    print("== the four checkpoints ==")
    loaded = summary.get("checkpoint_load", {})
    check(set(loaded) == set(MODEL_LABELS), "every checkpoint's load is recorded")
    for label in MODEL_LABELS:
        info = loaded.get(label, {})
        print(f"    {label:16s} arch {info.get('arch')} res {info.get('res')} "
              f"entries {info.get('state_entries')} "
              f"normalisations {info.get('normalisations')} "
              f"loader {info.get('loader')}")
    check(mechanical.get("all_checkpoints_share_arch") is True,
          "all four share one architecture")
    check(mechanical.get("all_checkpoints_share_res") is True, "all four share one resolution")
    check(mechanical.get("no_two_checkpoints_identical") is True,
          "no two of the four hold the same weights")
    matrix = summary.get("parameter_divergence", {})
    check(len(matrix) == 6, f"six pairwise weight comparisons reported (got {len(matrix)})")
    check(all(values.get("max_abs_param_diff", 0.0) > 0.0 for values in matrix.values()),
          "every pair of checkpoints differs in the weights themselves")
    for pair, values in sorted(matrix.items()):
        print(f"    {pair:32s} max |dw| {values.get('max_abs_param_diff'):.6f} "
              f"mean |dw| {values.get('mean_abs_param_diff'):.3e} "
              f"identical {values.get('identical_tensors')}/{values.get('float_tensors')}")

    print("== how much room the ensembles had ==")
    divergence = summary.get("checkpoint_divergence", {})
    pairs = summary.get("arm_construction", {})
    check(len(divergence) == 6,
          f"six pairwise prediction comparisons over the all-window arms (got {len(divergence)})")
    for pair, values in sorted(divergence.items()):
        print(f"    {pair:44s} spearman mean {values.get('mean_spearman'):.6f} "
              f"min {values.get('min_spearman'):.6f} "
              f"mean |dp| {values.get('mean_abs_prob_diff'):.6f}")
    separable = summary.get("arms_distinct_from_baseline", {})
    check(bool(separable), "whether each arm is separable from the baseline is recorded")
    collapsed = [name for name, distinct in separable.items() if not distinct]
    check(all(separable.get(name) for name in ALL_WINDOW_ARMS),
          "every all-window arm moves the predictions, as it must")
    if collapsed:
        print(f"    NOTE indistinguishable from the baseline: {collapsed} - a result about "
              f"those checkpoints, not a broken run")
    check(bool(pairs.get("arm_checkpoints")),
          "each arm records how many checkpoints it would cost at submission time")

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
    check(summary.get("promotable_arms") == PROMOTABLE_ARMS,
          "the promotable set excludes the baseline and the three reproduction arms")
    check(summary.get("family_size") == len(PROMOTABLE_ARMS),
          f"family size {summary.get('family_size')} counts only the arms that can win")
    check(summary.get("incumbent_arm") == INCUMBENT_ARM,
          "the incumbent is the v4 SWA checkpoint over the whole bank")
    baseline = arms.get(BASELINE_ARM, {})
    check(baseline.get("eval_macro_delta") == 0.0, "the baseline has zero delta against itself")
    check(baseline.get("passed_strict") is False and baseline.get("passed_relaxed") is False,
          "the baseline never counts as a winner under either verdict")
    for name in REPRODUCTION_ARMS:
        check(arms[name].get("promotable") is False,
              f"{name} is a reproduction check, not a candidate")

    predictions = np.load(directory / "e039_predictions.npz", allow_pickle=True)
    for name in ARM_NAMES:
        record = arms[name]
        bootstrap = record["bootstrap"]
        for split in ["eval", "gold"]:
            values = predictions[f"{split}__{name}"]
            check(np.isfinite(values).all(), f"{name}: {split} predictions finite")
            check(bool(((values >= 0.0) & (values <= 1.0)).all()),
                  f"{name}: {split} predictions are probabilities")
        incumbent = record.get("bootstrap_vs_incumbent")
        check((incumbent is not None) == record["promotable"],
              f"{name}: the incumbent margin is measured for candidates only")
        print(f"    {name:20s} eval {record['eval_macro_delta']:+.6f} "
              f"win {bootstrap['win_rate']:.3f} "
              f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
              f"half {bootstrap['ci95_halfwidth']:.6f} "
              f"vs inc {record['eval_delta_vs_incumbent']:+.6f} "
              f"gold {record['gold58_macro_delta']:+.6f} "
              f"{'strict' if record['passed_strict'] else ''}"
              f"{'+relaxed' if record['passed_strict'] and record['passed_relaxed'] else ''}"
              f"{'relaxed' if record['passed_relaxed'] and not record['passed_strict'] else ''}")
        check(not (record["passed_strict"] and not record["passed_relaxed"]),
              f"{name}: nothing can pass the strict verdict without passing the relaxed one")

    print("== estimator power ==")
    estimator = summary.get("estimator", {})
    reference = estimator.get("e038_reference_halfwidth_n2301", 0.000545)
    median_half = estimator.get("median_ci95_halfwidth")
    check(median_half is not None, "the interval half-width is reported")
    if median_half is not None:
        print(f"    median half-width {median_half:.6f} at n=2301 "
              f"vs {reference:.6f} measured in E038 on the same studies")
    check(estimator.get("any_arm_resolves_threshold") is not None,
          "whether the estimator can resolve the threshold is recorded")
    if decision == "STOP_FAMILY_UNDERPOWERED":
        check(estimator.get("any_arm_resolves_threshold") is False,
              "the underpowered verdict matches the measured intervals")

    print("== decision consistency ==")
    for label, outcome, winner_name, passing in [
        ("strict", decision, summary.get("best_arm"),
         [name for name, record in arms.items() if record.get("passed_strict")]),
        ("relaxed", decision_relaxed, summary.get("best_arm_relaxed"),
         [name for name, record in arms.items() if record.get("passed_relaxed")]),
    ]:
        if outcome == "PASS_PROMOTE_NEW_CHECKPOINT":
            check(bool(passing), f"{label}: a pass names at least one winning arm")
            check(winner_name in passing, f"{label}: the winner is one of the passing arms")
            check(summary.get("mechanical_passed") is True,
                  f"{label}: a pass requires the mechanical gates")
            winner = arms.get(winner_name, {})
            check(winner.get("promotable") is True,
                  f"{label}: the winner is an arm that was allowed to win")
            check(all(winner["gates"].values()),
                  f"{label}: the winner cleared every gate against the baseline")
            check(winner["eval_macro_delta"] >= MIN_VALID_GAIN,
                  f"{label}: the winner's gain clears the floor that has not moved since E037")
            if label == "strict":
                check(all(winner["gold58_gates"].values()),
                      "strict: the winner also cleared both Gold58 gates")
        else:
            check(not passing, f"{label}: a stop means no arm passed (found {passing})")
            check(winner_name is None, f"{label}: a stop names no winner")

    candidate = summary.get("submission_candidate")
    if summary.get("mechanical_passed"):
        check(candidate is not None,
              "a completed run always names a submission candidate, incumbent included")
        check(candidate == (summary.get("best_arm") or INCUMBENT_ARM),
              "the candidate is the strict winner, or the incumbent when nothing beat it")
        required = summary.get("submission_checkpoints", [])
        check(required == arms[candidate]["checkpoints_required"],
              f"the candidate needs {required} at submission time")

    print("== E037's A1, third measurement (reported, not gated) ==")
    replication = summary.get("e037_a1_replication", {})
    check(bool(replication), "the replication block is present")
    if replication:
        check(replication.get("e037_delta") == E037_A1_DELTA,
              "the E037 figure it is compared against is the recorded one")
        check(replication.get("disjoint_from_e037_split") is True,
              "the comparison is against studies E037 never scored")
        check(replication.get("same_split_as_e038") is True,
              "and it is the same split E038 used, so this is not an independent third sample")
        print(f"    E037 n=2048 {replication.get('e037_delta'):+.6f} "
              f"ci95 [{E037_A1_CI95[0]:+.6f}, {E037_A1_CI95[1]:+.6f}]")
        e039_ci = replication.get("e039_ci95", [float("nan"), float("nan")])
        print(f"    E039 n=2301 {replication.get('e039_delta'):+.6f} "
              f"ci95 [{e039_ci[0]:+.6f}, {e039_ci[1]:+.6f}]")
        print(f"    sign agrees {replication.get('sign_agrees')}, "
              f"intervals overlap {replication.get('intervals_overlap')}, "
              f"inside E037's interval {replication.get('e039_delta_inside_e037_ci')}")

    print()
    print(f"DECISION strict  {decision} best_arm {summary.get('best_arm')}")
    print(f"DECISION relaxed {decision_relaxed} best_arm {summary.get('best_arm_relaxed')}")
    print(f"SUBMISSION CANDIDATE {candidate} {summary.get('submission_checkpoints')}")
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
