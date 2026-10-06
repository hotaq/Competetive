#!/usr/bin/env python
"""Fail-closed validator for E034 nested-CV artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
CV_N = 640
LAB = len(TARGETS)
N_FOLDS = 5
FOLD_N = 128
ALLOWED_DECISIONS = {
    "STOP_E034_REQUIRED_GATE_FAILED",
    "STOP_E034_CAFO_CV_FAILED",
    "STOP_E034_BLEND_GUARD_FAILED",
    "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD",
}
REQUIRED_FILES = {
    "e034_alpha_metrics.csv",
    "e034_attention_metrics.csv",
    "e034_cache_metrics.csv",
    "e034_feature_metrics.csv",
    "e034_fold_metrics.csv",
    "e034_gates.csv",
    "e034_head_history.csv",
    "e034_head_metrics.csv",
    "e034_identity.json",
    "e034_inner_fold_metrics.json",
    "e034_oof_predictions.npz",
    "e034_oof_target_metrics.csv",
    "e034_summary.json",
}


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def binary_auc(labels, scores) -> float:
    labels = np.asarray(labels, np.int8)
    scores = np.asarray(scores, np.float64)
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def target_aucs(labels, predictions):
    values = np.asarray(
        [
            binary_auc(labels[:, target], predictions[:, target])
            for target in range(LAB)
        ],
        np.float64,
    )
    return values, float(np.nanmean(values))


def paired_bootstrap(labels, baseline, challenger, seed, repeats=2000):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(labels), size=len(labels))
        per_target = []
        for target in range(LAB):
            selected = labels[indexes, target]
            if np.unique(selected).size < 2:
                continue
            per_target.append(
                binary_auc(selected, challenger[indexes, target])
                - binary_auc(selected, baseline[indexes, target])
            )
        if per_target:
            differences.append(float(np.mean(per_target)))
    differences = np.asarray(differences, np.float64)
    return {
        "repeats_valid": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "win_rate": float(np.mean(differences > 0)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    raise ValueError(f"not a boolean: {value!r}")


def close(actual, expected, tolerance=1e-8) -> bool:
    return bool(
        np.isfinite(actual)
        and np.isfinite(expected)
        and abs(float(actual) - float(expected)) <= tolerance
    )


def validate(output_dir: Path, require_signal: bool = False):
    failures = []
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - existing)
    if missing:
        failures.append(f"missing files: {missing}")
        return failures, None

    summary = json.loads((output_dir / "e034_summary.json").read_text())
    identity_file = json.loads((output_dir / "e034_identity.json").read_text())
    decision = summary.get("decision")
    if decision not in ALLOWED_DECISIONS:
        failures.append(f"invalid decision: {decision!r}")
    if summary.get("experiment") != (
        "E034 nested five-fold CAFormer CV and protected blend"
    ):
        failures.append("experiment identity mismatch")
    if summary.get("identity") != identity_file:
        failures.append("summary/identity JSON mismatch")

    identity = summary.get("identity", {})
    if identity.get("counts") != {
        "cv": CV_N,
        "outer_folds": N_FOLDS,
        "gold_excluded": 58,
    }:
        failures.append("identity counts mismatch")
    if not identity.get("identity_exact"):
        failures.append("identity_exact is false")
    if identity.get("gold_label_evaluation_count") != 0:
        failures.append("Gold labels were evaluated")
    if identity.get("gold_image_encoding_count") != 0:
        failures.append("Gold images were encoded")

    with np.load(output_dir / "e034_oof_predictions.npz", allow_pickle=True) as data:
        if set(data.files) != {
            "ids",
            "folds",
            "y_binary",
            "coatnet",
            "caformer",
            "protected_blend",
        }:
            failures.append(f"OOF payload schema mismatch: {data.files}")
            return failures, summary
        ids = [str(value) for value in data["ids"].tolist()]
        folds = data["folds"].astype(np.int64)
        labels = data["y_binary"].astype(np.float64)
        predictions = {
            "coatnet": data["coatnet"].astype(np.float64),
            "caformer": data["caformer"].astype(np.float64),
            "protected_blend": data["protected_blend"].astype(np.float64),
        }

    if len(ids) != CV_N or len(set(ids)) != CV_N:
        failures.append("OOF IDs are not 640 unique studies")
    if ids_sha256(ids) != identity.get("cv_ids_sha256"):
        failures.append("OOF ID hash mismatch")
    if folds.shape != (CV_N,):
        failures.append(f"fold shape mismatch: {folds.shape}")
    if not np.array_equal(
        np.bincount(folds, minlength=N_FOLDS), np.full(N_FOLDS, FOLD_N)
    ):
        failures.append("outer fold coverage mismatch")
    if labels.shape != (CV_N, LAB):
        failures.append(f"label shape mismatch: {labels.shape}")
    if not np.isin(labels, [0.0, 1.0]).all():
        failures.append("OOF labels are not binary")
    for model_key, values in predictions.items():
        if values.shape != (CV_N, LAB):
            failures.append(f"{model_key} shape mismatch: {values.shape}")
        if not np.isfinite(values).all():
            failures.append(f"{model_key} contains non-finite values")

    computed_target = {}
    computed_macro = {}
    for model_key, values in predictions.items():
        computed_target[model_key], computed_macro[model_key] = target_aucs(
            labels, values
        )
    oof = summary.get("oof", {})
    summary_macro_keys = {
        "coatnet": "coatnet_macro_auc",
        "caformer": "caformer_macro_auc",
        "protected_blend": "protected_blend_macro_auc",
    }
    for model_key, summary_key in summary_macro_keys.items():
        if not close(computed_macro[model_key], oof.get(summary_key, np.nan)):
            failures.append(f"{summary_key} mismatch")
    if not close(
        computed_macro["caformer"] - computed_macro["coatnet"],
        oof.get("caformer_gain", np.nan),
    ):
        failures.append("caformer_gain mismatch")
    if not close(
        computed_macro["protected_blend"] - computed_macro["coatnet"],
        oof.get("protected_blend_gain", np.nan),
    ):
        failures.append("protected_blend_gain mismatch")

    target_rows = read_csv(output_dir / "e034_oof_target_metrics.csv")
    if [row.get("target") for row in target_rows] != TARGETS:
        failures.append("target metric row order mismatch")
    elif len(target_rows) == LAB:
        for target, row in enumerate(target_rows):
            expected = {
                "coatnet_auc": computed_target["coatnet"][target],
                "caformer_auc": computed_target["caformer"][target],
                "protected_blend_auc": computed_target["protected_blend"][target],
                "delta_caformer_minus_coatnet": (
                    computed_target["caformer"][target]
                    - computed_target["coatnet"][target]
                ),
                "delta_blend_minus_coatnet": (
                    computed_target["protected_blend"][target]
                    - computed_target["coatnet"][target]
                ),
            }
            for key, value in expected.items():
                if not close(float(row[key]), value):
                    failures.append(f"{TARGETS[target]} {key} mismatch")

    fold_rows = read_csv(output_dir / "e034_fold_metrics.csv")
    if len(fold_rows) != N_FOLDS * 3:
        failures.append("fold metric row count mismatch")
    else:
        indexed = {
            (int(row["outer_fold"]), row["model_key"]): float(row["macro_auc"])
            for row in fold_rows
        }
        for fold in range(N_FOLDS):
            indexes = folds == fold
            for model_key, values in predictions.items():
                _, macro = target_aucs(labels[indexes], values[indexes])
                if not close(indexed.get((fold, model_key), np.nan), macro):
                    failures.append(f"fold {fold}/{model_key} metric mismatch")

    alpha_rows = read_csv(output_dir / "e034_alpha_metrics.csv")
    if len(alpha_rows) != N_FOLDS * LAB:
        failures.append("alpha metric row count mismatch")
    else:
        allowed_alpha = {0.0, 0.25, 0.5, 0.75, 1.0}
        for row in alpha_rows:
            if float(row["selected_alpha"]) not in allowed_alpha:
                failures.append("unexpected selected alpha")
                break

    head_rows = read_csv(output_dir / "e034_head_metrics.csv")
    if len(head_rows) != 60:
        failures.append(f"head metric row count mismatch: {len(head_rows)}")
    else:
        try:
            if not all(as_bool(row["head_updated"]) for row in head_rows):
                failures.append("at least one head did not update")
        except ValueError as error:
            failures.append(str(error))

    attention_rows = read_csv(output_dir / "e034_attention_metrics.csv")
    if len(attention_rows) != 2 * LAB:
        failures.append("attention metric row count mismatch")
    else:
        try:
            if not all(as_bool(row["nondegenerate"]) for row in attention_rows):
                failures.append("degenerate attention detected")
        except ValueError as error:
            failures.append(str(error))

    cache_rows = read_csv(output_dir / "e034_cache_metrics.csv")
    if len(cache_rows) != CV_N:
        failures.append(f"cache row count mismatch: {len(cache_rows)}")
    else:
        for row in cache_rows:
            if (
                int(row["selected_windows"]) != 24
                or int(row["cross_series_windows"]) != 0
                or int(row["zero_slice_windows"]) != 0
                or int(row["cache_pixel_diff"]) != 0
            ):
                failures.append("cache contract mismatch")
                break

    inner_rows = json.loads(
        (output_dir / "e034_inner_fold_metrics.json").read_text()
    )
    if len(inner_rows) != N_FOLDS:
        failures.append("inner fold metric count mismatch")
    elif any(row.get("fold_sizes") != [128, 128, 128, 128] for row in inner_rows):
        failures.append("inner fold sizes mismatch")

    expected_bootstrap = {
        "caformer_vs_coatnet": paired_bootstrap(
            labels,
            predictions["coatnet"],
            predictions["caformer"],
            31021 + 3401,
        ),
        "protected_blend_vs_coatnet": paired_bootstrap(
            labels,
            predictions["coatnet"],
            predictions["protected_blend"],
            31021 + 3402,
        ),
    }
    reported_bootstrap = summary.get("bootstrap", {})
    for comparison, expected in expected_bootstrap.items():
        reported = reported_bootstrap.get(comparison, {})
        for key, value in expected.items():
            if key == "repeats_valid":
                if reported.get(key) != value:
                    failures.append(f"{comparison} {key} mismatch")
            elif not close(reported.get(key, np.nan), value):
                failures.append(f"{comparison} {key} mismatch")

    gate_rows = read_csv(output_dir / "e034_gates.csv")
    reported_gates = {
        (row["group"], row["gate"]): as_bool(row["passed"]) for row in gate_rows
    }
    gate_groups = (
        ("required", "required_gates"),
        ("caformer_signal", "caformer_signal_gates"),
        ("blend_signal", "blend_signal_gates"),
    )
    for group, key in gate_groups:
        gates = summary.get(key, {})
        for gate, passed in gates.items():
            if reported_gates.get((group, gate)) is not bool(passed):
                failures.append(f"gate CSV mismatch: {group}/{gate}")
    required = summary.get("required_gates", {})
    caformer_signal = summary.get("caformer_signal_gates", {})
    blend_signal = summary.get("blend_signal_gates", {})
    if not required or not all(required.values()):
        failures.append("required gates did not all pass")

    if not all(required.values()):
        expected_decision = "STOP_E034_REQUIRED_GATE_FAILED"
    elif not all(caformer_signal.values()):
        expected_decision = "STOP_E034_CAFO_CV_FAILED"
    elif not all(blend_signal.values()):
        expected_decision = "STOP_E034_BLEND_GUARD_FAILED"
    else:
        expected_decision = "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD"
    if decision != expected_decision:
        failures.append(
            f"decision mismatch: expected {expected_decision}, got {decision}"
        )
    if require_signal and decision != "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD":
        failures.append("promotion signal required")

    gold = summary.get("gold", {})
    if gold != {
        "ids_used_for_exclusion_only": True,
        "label_evaluation_count": 0,
        "image_encoding_count": 0,
        "prediction_artifact_created": False,
    }:
        failures.append("Gold fail-closed summary mismatch")
    if summary.get("checkpoint_exported") is not False:
        failures.append("checkpoint_exported is not false")
    if summary.get("feature_cache_exported") is not False:
        failures.append("feature_cache_exported is not false")
    if summary.get("submission_created") is not False:
        failures.append("submission_created is not false")

    for path in output_dir.iterdir():
        lower = path.name.lower()
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            failures.append(f"forbidden checkpoint: {path.name}")
        if path.name == "submission.csv":
            failures.append("forbidden submission.csv")
        if "gold58_prediction" in lower or "gold58_target_metric" in lower:
            failures.append(f"forbidden Gold artifact: {path.name}")

    return failures, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-signal", action="store_true")
    args = parser.parse_args()
    failures, summary = validate(args.output_dir, args.require_signal)
    if failures:
        print("E034 INVALID:")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print("E034 VALID", summary["decision"])


if __name__ == "__main__":
    main()
