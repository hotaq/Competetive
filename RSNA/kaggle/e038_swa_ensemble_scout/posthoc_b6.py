#!/usr/bin/env python3
"""Recover the arm E038 never ran: the SWA checkpoint over every safe window.

POST HOC. This arm was not in E038's pre-registered family, so nothing here can
promote anything. It exists to answer one question the E038 result raises: the
gates named `B5_ens_prob_all` the winner, but `B1_swa` beat both 42-window
ensembles, which says averaging the base checkpoint in costs accuracy rather
than adding it. If that holds at the all-window setting too, the arm that should
have won was never measured.

It is recoverable exactly, with no GPU, because E038 saved the assembled arms:

    B4_base_all     = sigmoid(base/all)
    B5_ens_prob_all = (sigmoid(base/all) + sigmoid(swa/all)) / 2
    => sigmoid(swa/all) = 2 * B5 - B4

Usage: python posthoc_b6.py output_v1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BOOTSTRAP_SEED = 3802
BOOTSTRAP_REPEATS = 2000
MIN_VALID_GAIN = 0.0020
MIN_BOOTSTRAP_WIN = 0.98
MAX_TARGET_DROP = 0.010
GOLD58_MIN_TARGET_DELTA = -0.010


def target_aucs(y, predictions, targets):
    values = [roc_auc_score(y[:, index], predictions[:, index])
              for index in range(len(targets))]
    return np.asarray(values), float(np.mean(values))


def paired_bootstrap(y, base_predictions, arm_predictions, seed,
                     repeats=BOOTSTRAP_REPEATS):
    """E038's estimator, copied so the numbers are comparable to its summary."""
    rng = np.random.default_rng(seed)
    deltas = []
    wins = 0
    n = len(y)
    for _ in range(repeats):
        index = rng.integers(0, n, n)
        y_sample = y[index]
        if not ((y_sample.sum(0) > 0).all() and (y_sample.sum(0) < n).all()):
            continue
        _base_target, base_macro = target_aucs(y_sample, base_predictions[index],
                                               range(y.shape[1]))
        _arm_target, arm_macro = target_aucs(y_sample, arm_predictions[index],
                                             range(y.shape[1]))
        delta = arm_macro - base_macro
        deltas.append(delta)
        wins += int(delta > 0.0)
    deltas = np.asarray(deltas)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "win_rate": float(wins / len(deltas)),
        "ci95": [float(low), float(high)],
        "ci95_halfwidth": float((high - low) / 2.0),
        "valid_repeats": int(len(deltas)),
    }


def main(directory: Path) -> int:
    summary = json.loads((directory / "e038_summary.json").read_text())
    targets = list(summary["window_policy"] and
                   summary["estimator"]["eval_positives_per_target"].keys())
    predictions = np.load(directory / "e038_predictions.npz", allow_pickle=True)
    y_eval = predictions["y_eval"]
    y_gold = predictions["y_gold"]

    print("== the reconstruction algebra, checked on an arm we can verify ==")
    # B2_ens_prob is the probability-space mean of B0 and B1, both of which are
    # saved. If that identity holds on these float32 arrays, the same algebra
    # applied to B4 and B5 is trustworthy.
    for split, y in [("eval", y_eval), ("gold", y_gold)]:
        b0 = predictions[f"{split}__B0_base"].astype(np.float64)
        b1 = predictions[f"{split}__B1_swa"].astype(np.float64)
        b2 = predictions[f"{split}__B2_ens_prob"].astype(np.float64)
        residual = float(np.abs((b0 + b1) / 2.0 - b2).max())
        print(f"  {split}: max |(B0+B1)/2 - B2| = {residual:.3e}")
        if residual > 1e-6:
            print("  FAIL the saved arms do not satisfy their own definition")
            return 1

    print("== recovering sigmoid(swa/all) ==")
    recovered = {}
    for split in ["eval", "gold"]:
        b4 = predictions[f"{split}__B4_base_all"].astype(np.float64)
        b5 = predictions[f"{split}__B5_ens_prob_all"].astype(np.float64)
        swa_all = 2.0 * b5 - b4
        outside = int(((swa_all < 0.0) | (swa_all > 1.0)).sum())
        print(f"  {split}: range [{swa_all.min():.6f}, {swa_all.max():.6f}], "
              f"{outside} of {swa_all.size} values outside [0, 1] "
              f"(float32 round-trip, magnitude {max(0.0, -swa_all.min()):.2e})")
        recovered[split] = swa_all

    print()
    print("== B6_swa_all against the E038 baseline (POST HOC) ==")
    results = {}
    for split, y in [("eval", y_eval), ("gold", y_gold)]:
        baseline = predictions[f"{split}__B0_base"].astype(np.float64)
        base_target, base_macro = target_aucs(y, baseline, targets)
        arm_target, arm_macro = target_aucs(y, recovered[split], targets)
        results[split] = {
            "macro_auc": arm_macro,
            "macro_delta": arm_macro - base_macro,
            "target_delta": arm_target - base_target,
            "min_target_delta": float((arm_target - base_target).min()),
        }
        print(f"  {split:5s} macro {arm_macro:.6f}  delta {arm_macro - base_macro:+.6f}  "
              f"worst target {(arm_target - base_target).min():+.6f}")

    bootstrap = paired_bootstrap(y_eval, predictions["eval__B0_base"].astype(np.float64),
                                 recovered["eval"], BOOTSTRAP_SEED)
    print(f"  bootstrap win {bootstrap['win_rate']:.3f} "
          f"ci95 [{bootstrap['ci95'][0]:+.6f}, {bootstrap['ci95'][1]:+.6f}] "
          f"half {bootstrap['ci95_halfwidth']:.6f} "
          f"({bootstrap['valid_repeats']} valid repeats)")

    gates = {
        "gain": results["eval"]["macro_delta"] >= MIN_VALID_GAIN,
        "bootstrap_win": bootstrap["win_rate"] >= MIN_BOOTSTRAP_WIN,
        "ci_low_positive": bootstrap["ci95"][0] > 0.0,
        "no_target_collapse": results["eval"]["min_target_delta"] >= -MAX_TARGET_DROP,
        "gold58_sign_agrees": results["gold"]["macro_delta"] > 0.0,
        "gold58_no_target_collapse":
            results["gold"]["min_target_delta"] >= GOLD58_MIN_TARGET_DELTA,
    }
    print("  gates " + "  ".join(f"{name}={'Y' if value else 'N'}"
                                 for name, value in gates.items()))
    print(f"  would have cleared all six: {all(gates.values())}")

    print()
    print("== where the ensemble weight actually wants to be (DESCRIPTIVE) ==")
    # w = 1 is pure SWA, w = 0.5 is B5, w = 0 is B4. The shape of this curve is
    # the point; its argmax is fitted on this split and is not a choice anyone
    # gets to make from here.
    baseline = predictions["eval__B0_base"].astype(np.float64)
    _base_target, base_macro = target_aucs(y_eval, baseline, targets)
    base_all = predictions["eval__B4_base_all"].astype(np.float64)
    for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = (1.0 - weight) * base_all + weight * recovered["eval"]
        _blend_target, blend_macro = target_aucs(y_eval, blended, targets)
        marker = {0.0: "  (= B4)", 0.5: "  (= B5, the promoted arm)",
                  1.0: "  (= B6, never run)"}.get(weight, "")
        print(f"  w_swa {weight:.2f}  macro {blend_macro:.6f}  "
              f"delta {blend_macro - base_macro:+.6f}{marker}")

    print()
    print("== the same question at 42 windows, from arms E038 did run ==")
    for name in ["B0_base", "B2_ens_prob", "B1_swa"]:
        record = next(r for r in summary["arms"] if r["arm"] == name)
        label = {"B0_base": "w_swa 0.00", "B2_ens_prob": "w_swa 0.50",
                 "B1_swa": "w_swa 1.00"}[name]
        print(f"  {label}  {name:14s} delta {record['eval_macro_delta']:+.6f}")

    print()
    print("POST HOC - this arm was not pre-registered and promotes nothing.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
