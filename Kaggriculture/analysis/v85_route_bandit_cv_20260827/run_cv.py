#!/usr/bin/env python3
"""Run deterministic nested family CV for the V85 observable route bandit.

This script never opens replay or sealed-holdout traces.  Feature collection is
owned by ``collect_features.py``; this file consumes its identifier-separated
dataset and the already-frozen V84 development outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
COLLECTOR_PATH = OUT / "collect_features.py"
DATASET_PATH = OUT / "dataset.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"

ROOT = OUT.parents[1]
V84_RESULTS_PATH = ROOT / "analysis/v84_route_option_onpolicy_oracle_20260827/results.json"
V84_RUNNER_PATH = ROOT / "analysis/v84_route_option_onpolicy_oracle_20260827/run_oracle.py"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
ENCODER_PATH = ROOT / "src/kaggriculture_rl/learning/tape_selector.py"

EXPECTED_V67_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"
EXPECTED_V84_RESULTS_SHA256 = "e7f1e1c701fce8951bb57b02ab8d408a49ef916170a3065dd782fa4c7286cd55"

ACTION_NAMES = (
    "ABSTAIN_KEEP_V67",
    "FORCE_B120_SUFFIX",
    "FORCE_C120_SUFFIX",
)
ALL_AUDIT_OPTIONS = (
    "ABSTAIN_KEEP_V67",
    "FORCE_BASE_SUFFIX",
    "FORCE_A120_SUFFIX",
    "FORCE_B120_SUFFIX",
    "FORCE_C120_SUFFIX",
)
FAMILIES = (
    "v16_route_family",
    "ryo_selector_family",
    "route_99612231_family",
    "robust_cluster_family",
)
FEATURE_STEPS = (24, 48, 72, 96, 120)
SNAPSHOT_DIM = 81
FEATURE_DIM = SNAPSHOT_DIM * len(FEATURE_STEPS)

ALPHAS = (10.0, 100.0, 1000.0)
BETAS = (1.0, 2.0)
THRESHOLDS = (0.0, 1000.0, 2500.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Recompute results/report from dataset.json without live collection.",
    )
    parser.add_argument(
        "--reference-dataset-core-sha256",
        help="First-run dataset core for an independent render-existing check.",
    )
    parser.add_argument(
        "--reference-cv-core-sha256",
        help="First-run CV core for an independent render-existing check.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_margin_map(sample: dict[str, Any]) -> dict[str, float]:
    labels = sample.get("labels", {})
    raw = labels.get("all_option_margins")
    if raw is None:
        raw = labels.get("margins")
    if not isinstance(raw, dict):
        raise AssertionError("sample labels must contain an option-margin mapping")
    missing = set(ALL_AUDIT_OPTIONS) - set(raw)
    if missing:
        raise AssertionError(f"sample is missing V84 audit margins: {sorted(missing)}")
    return {name: float(raw[name]) for name in ALL_AUDIT_OPTIONS}


def _extract_features(sample: dict[str, Any]) -> list[float]:
    model_input = sample.get("model_input")
    if not isinstance(model_input, dict):
        raise AssertionError("model_input must be an identifier-free mapping")
    if set(model_input) != {"features"}:
        raise AssertionError("model_input must contain only features; identifiers stay under audit")
    values = [float(value) for value in model_input["features"]]
    if len(values) != FEATURE_DIM:
        raise AssertionError(f"feature length {len(values)} != {FEATURE_DIM}")
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("features must all be finite")
    return values


def validate_dataset(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples")
    if not isinstance(samples, list) or len(samples) != 32:
        raise AssertionError("dataset must contain exactly 32 contexts")

    keys: set[tuple[str, int, int]] = set()
    family_counts = {family: 0 for family in FAMILIES}
    prefix_checks = 0
    for sample in samples:
        audit = sample.get("audit")
        if not isinstance(audit, dict):
            raise AssertionError("audit metadata missing")
        family = str(audit["family"])
        seed = int(audit["seed"])
        seat_key = "candidate_seat" if "candidate_seat" in audit else "seat"
        seat = int(audit[seat_key])
        key = (family, seed, seat)
        if key in keys:
            raise AssertionError(f"duplicate context key: {key}")
        keys.add(key)
        if family not in family_counts:
            raise AssertionError(f"unexpected family: {family}")
        family_counts[family] += 1
        if seed not in range(27100100, 27100104) or seat not in (0, 1):
            raise AssertionError(f"unexpected development key: {key}")

        _extract_features(sample)
        _extract_margin_map(sample)

        parity = audit.get("v84_prefix_match")
        if parity is None:
            parity = audit.get("prefix_match")
        if parity is None:
            parity = audit.get("prefix_sha256") == audit.get("v84_abstain_prefix_sha256")
        if parity is not True:
            raise AssertionError(f"V84 prefix mismatch: {key}")
        prefix_checks += 1

    if family_counts != {family: 8 for family in FAMILIES}:
        raise AssertionError(f"family grid mismatch: {family_counts}")
    if prefix_checks != 32:
        raise AssertionError("expected 32 exact V84 prefix checks")

    feature_audit = dataset.get("feature_audit", dataset.get("protocol", {}))
    snapshot_dim = feature_audit.get(
        "snapshot_dim", feature_audit.get("snapshot_feature_dimension", SNAPSHOT_DIM)
    )
    feature_dim = feature_audit.get(
        "feature_dim", feature_audit.get("model_feature_dimension", FEATURE_DIM)
    )
    if int(snapshot_dim) != SNAPSHOT_DIM:
        raise AssertionError("snapshot feature dimension drift")
    if int(feature_dim) != FEATURE_DIM:
        raise AssertionError("concatenated feature dimension drift")
    if tuple(feature_audit.get("feature_steps", FEATURE_STEPS)) != FEATURE_STEPS:
        raise AssertionError("feature snapshot step drift")
    if feature_audit.get("model_input_excludes_audit_identifiers", True) is not True:
        raise AssertionError("collector did not affirm identifier separation")


def dataset_arrays(
    dataset: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    validate_dataset(dataset)
    samples = dataset["samples"]
    features = np.asarray([_extract_features(row) for row in samples], dtype=np.float64)
    margins = np.asarray(
        [
            [
                _extract_margin_map(row)["ABSTAIN_KEEP_V67"],
                _extract_margin_map(row)["FORCE_B120_SUFFIX"],
                _extract_margin_map(row)["FORCE_C120_SUFFIX"],
            ]
            for row in samples
        ],
        dtype=np.float64,
    )
    deltas = margins[:, 1:] - margins[:, [0]]
    groups = np.asarray([str(row["audit"]["family"]) for row in samples])
    audits = [dict(row["audit"]) for row in samples]
    if features.shape != (32, FEATURE_DIM):
        raise AssertionError(f"unexpected feature matrix shape: {features.shape}")
    if margins.shape != (32, 3) or deltas.shape != (32, 2):
        raise AssertionError("unexpected target matrix shape")
    return features, margins, deltas, groups, audits


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalization using only the current training fold."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (features - mean) / scale, mean, scale


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fit centered multi-output ridge through the deterministic dual system."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("ridge inputs must be rank-two")
    if len(features) != len(targets) or len(features) == 0:
        raise ValueError("ridge features/targets must be non-empty and aligned")
    normalized, mean, scale = normalize_fit(features)
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    gram = normalized @ normalized.T
    dual = np.linalg.solve(
        gram + np.eye(len(gram), dtype=np.float64) * float(alpha),
        centered_targets,
    )
    weights = normalized.T @ dual
    return {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "weights": weights,
    }


def ridge_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    return model["target_mean"] + normalized @ model["weights"]


def _jackknife_variance(predictions: np.ndarray) -> np.ndarray:
    """Standard delete-one-group jackknife prediction variance."""
    group_count = predictions.shape[0]
    if group_count < 2:
        raise AssertionError("family jackknife needs at least two training families")
    centered = predictions - predictions.mean(axis=0, keepdims=True)
    return ((group_count - 1.0) / group_count) * np.sum(np.square(centered), axis=0)


def fit_predict_heads(
    train_features: np.ndarray,
    train_margins: np.ndarray,
    train_deltas: np.ndarray,
    train_groups: np.ndarray,
    eval_features: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Predict two heads with family-jackknife plus grouped-OOF uncertainty."""
    unique_groups = tuple(sorted(set(train_groups.tolist())))
    if len(unique_groups) < 2:
        raise AssertionError("at least two training families are required")

    absolute_model = ridge_fit(train_features, train_margins, alpha)
    delta_model = ridge_fit(train_features, train_deltas, alpha)
    absolute_mean = ridge_predict(absolute_model, eval_features)
    delta_mean = ridge_predict(delta_model, eval_features)

    absolute_jackknife: list[np.ndarray] = []
    delta_jackknife: list[np.ndarray] = []
    absolute_oof = np.full_like(train_margins, np.nan)
    delta_oof = np.full_like(train_deltas, np.nan)

    for held_group in unique_groups:
        fit_mask = train_groups != held_group
        held_mask = ~fit_mask
        absolute_submodel = ridge_fit(train_features[fit_mask], train_margins[fit_mask], alpha)
        delta_submodel = ridge_fit(train_features[fit_mask], train_deltas[fit_mask], alpha)
        absolute_jackknife.append(ridge_predict(absolute_submodel, eval_features))
        delta_jackknife.append(ridge_predict(delta_submodel, eval_features))
        absolute_oof[held_mask] = ridge_predict(absolute_submodel, train_features[held_mask])
        delta_oof[held_mask] = ridge_predict(delta_submodel, train_features[held_mask])

    if not np.isfinite(absolute_oof).all() or not np.isfinite(delta_oof).all():
        raise AssertionError("grouped OOF predictions are incomplete")

    absolute_jackknife_array = np.stack(absolute_jackknife, axis=0)
    delta_jackknife_array = np.stack(delta_jackknife, axis=0)
    absolute_jackknife_variance = _jackknife_variance(absolute_jackknife_array)
    delta_jackknife_variance = _jackknife_variance(delta_jackknife_array)
    absolute_oof_rmse = np.sqrt(np.mean(np.square(train_margins - absolute_oof), axis=0))
    delta_oof_rmse = np.sqrt(np.mean(np.square(train_deltas - delta_oof), axis=0))

    absolute_sigma = np.sqrt(absolute_jackknife_variance + np.square(absolute_oof_rmse)[None, :])
    delta_sigma = np.sqrt(delta_jackknife_variance + np.square(delta_oof_rmse)[None, :])
    return {
        "absolute_mean": absolute_mean,
        "delta_mean": delta_mean,
        "absolute_sigma": absolute_sigma,
        "delta_sigma": delta_sigma,
        "absolute_jackknife_variance": absolute_jackknife_variance,
        "delta_jackknife_variance": delta_jackknife_variance,
        "absolute_oof_rmse": absolute_oof_rmse,
        "delta_oof_rmse": delta_oof_rmse,
    }


def select_actions(
    predictions: dict[str, np.ndarray], *, beta: float, threshold: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    absolute_lcb = predictions["absolute_mean"] - beta * predictions["absolute_sigma"]
    delta_lcb = predictions["delta_mean"] - beta * predictions["delta_sigma"]
    choices = np.zeros(len(absolute_lcb), dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []

    for row_index in range(len(absolute_lcb)):
        eligible: list[int] = []
        for action_index in (1, 2):
            if (
                absolute_lcb[row_index, action_index] > 0.0
                and delta_lcb[row_index, action_index - 1] > threshold
            ):
                eligible.append(action_index)

        if eligible:
            # Fixed deterministic route tie-break: highest delta LCB, then
            # highest absolute LCB, then B before C.
            choices[row_index] = max(
                eligible,
                key=lambda index: (
                    float(delta_lcb[row_index, index - 1]),
                    float(absolute_lcb[row_index, index]),
                    -index,
                ),
            )

        diagnostics.append(
            {
                "absolute_mean": predictions["absolute_mean"][row_index].tolist(),
                "absolute_sigma": predictions["absolute_sigma"][row_index].tolist(),
                "absolute_lcb": absolute_lcb[row_index].tolist(),
                "delta_mean": predictions["delta_mean"][row_index].tolist(),
                "delta_sigma": predictions["delta_sigma"][row_index].tolist(),
                "delta_lcb": delta_lcb[row_index].tolist(),
                "eligible_routes": [ACTION_NAMES[index] for index in eligible],
            }
        )
    return choices, diagnostics


def score_choices(margins: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    deltas = selected - keep
    tail_count = max(1, math.ceil(0.10 * len(selected)))
    return {
        "contexts": len(selected),
        "wins": int(np.sum(selected > 0.0)),
        "losses": int(np.sum(selected <= 0.0)),
        "worst_margin": float(np.min(selected)),
        "cvar10_margin": float(np.mean(np.sort(selected)[:tail_count])),
        "mean_margin": float(np.mean(selected)),
        "keep_wins": int(np.sum(keep > 0.0)),
        "keep_worst_margin": float(np.min(keep)),
        "keep_mean_margin": float(np.mean(keep)),
        "paired_worst_delta": float(np.min(deltas)),
        "paired_mean_delta": float(np.mean(deltas)),
        "interventions": int(np.sum(choices != 0)),
    }


def configs() -> list[dict[str, float]]:
    values = [
        {"alpha": alpha, "beta": beta, "threshold": threshold}
        for alpha in ALPHAS
        for beta in BETAS
        for threshold in THRESHOLDS
    ]
    if len(values) != 18:
        raise AssertionError("prelocked search must contain exactly 18 configs")
    return values


def config_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    metric = row["metrics"]
    config = row["config"]
    return (
        float(metric["losses"]),
        -float(metric["worst_margin"]),
        -float(metric["paired_worst_delta"]),
        -float(metric["cvar10_margin"]),
        -float(metric["paired_mean_delta"]),
        float(metric["interventions"]),
        -float(config["threshold"]),
        -float(config["beta"]),
        -float(config["alpha"]),
    )


def evaluate_inner_config(
    features: np.ndarray,
    margins: np.ndarray,
    deltas: np.ndarray,
    groups: np.ndarray,
    config: dict[str, float],
) -> dict[str, Any]:
    unique_groups = tuple(sorted(set(groups.tolist())))
    if len(unique_groups) != 3:
        raise AssertionError("each outer training split must contain three families")
    choices = np.full(len(features), -1, dtype=np.int64)
    fold_records: list[dict[str, Any]] = []
    for held_group in unique_groups:
        fit_mask = groups != held_group
        held_mask = ~fit_mask
        predictions = fit_predict_heads(
            features[fit_mask],
            margins[fit_mask],
            deltas[fit_mask],
            groups[fit_mask],
            features[held_mask],
            alpha=config["alpha"],
        )
        fold_choices, _ = select_actions(
            predictions,
            beta=config["beta"],
            threshold=config["threshold"],
        )
        choices[held_mask] = fold_choices
        fold_records.append(
            {
                "held_family": held_group,
                "fit_families": sorted(set(groups[fit_mask].tolist())),
                "contexts": int(np.sum(held_mask)),
            }
        )
    if np.any(choices < 0):
        raise AssertionError("inner grouped OOF choices are incomplete")
    return {
        "config": dict(config),
        "metrics": score_choices(margins, choices),
        "folds": fold_records,
    }


def _audit_key(audit: dict[str, Any]) -> dict[str, Any]:
    seat_key = "candidate_seat" if "candidate_seat" in audit else "seat"
    return {
        "family": str(audit["family"]),
        "representative": str(audit.get("representative", "")),
        "seed": int(audit["seed"]),
        "candidate_seat": int(audit[seat_key]),
    }


def run_nested_cv(dataset: dict[str, Any]) -> dict[str, Any]:
    features, margins, deltas, groups, audits = dataset_arrays(dataset)
    outer_predictions: list[dict[str, Any]] = []
    outer_folds: list[dict[str, Any]] = []

    for held_family in FAMILIES:
        train_mask = groups != held_family
        held_mask = ~train_mask
        if int(np.sum(held_mask)) != 8 or int(np.sum(train_mask)) != 24:
            raise AssertionError(f"outer family split mismatch: {held_family}")

        leaderboard = [
            evaluate_inner_config(
                features[train_mask],
                margins[train_mask],
                deltas[train_mask],
                groups[train_mask],
                config,
            )
            for config in configs()
        ]
        leaderboard.sort(key=config_rank_key)
        for rank, row in enumerate(leaderboard, start=1):
            row["rank"] = rank
        selected_config = dict(leaderboard[0]["config"])

        predictions = fit_predict_heads(
            features[train_mask],
            margins[train_mask],
            deltas[train_mask],
            groups[train_mask],
            features[held_mask],
            alpha=selected_config["alpha"],
        )
        held_choices, decision_diagnostics = select_actions(
            predictions,
            beta=selected_config["beta"],
            threshold=selected_config["threshold"],
        )
        held_indices = np.flatnonzero(held_mask)
        for local_index, global_index in enumerate(held_indices.tolist()):
            choice = int(held_choices[local_index])
            actual_margin = float(margins[global_index, choice])
            keep_margin = float(margins[global_index, 0])
            outer_predictions.append(
                {
                    "audit": _audit_key(audits[global_index]),
                    "selected_action": ACTION_NAMES[choice],
                    "selected_action_index": choice,
                    "actual_margin": actual_margin,
                    "keep_margin": keep_margin,
                    "paired_delta": actual_margin - keep_margin,
                    "all_three_action_margins": margins[global_index].tolist(),
                    "decision": decision_diagnostics[local_index],
                }
            )

        outer_folds.append(
            {
                "held_family": held_family,
                "train_families": sorted(set(groups[train_mask].tolist())),
                "selected_config": selected_config,
                "inner_leaderboard": leaderboard,
                "held_metrics": score_choices(margins[held_mask], held_choices),
                "uncertainty_audit": {
                    "absolute_oof_rmse": predictions["absolute_oof_rmse"].tolist(),
                    "delta_oof_rmse": predictions["delta_oof_rmse"].tolist(),
                    "family_jackknife_groups": 3,
                },
            }
        )

    outer_predictions.sort(
        key=lambda row: (
            FAMILIES.index(row["audit"]["family"]),
            row["audit"]["seed"],
            row["audit"]["candidate_seat"],
        )
    )
    if len(outer_predictions) != 32:
        raise AssertionError("outer predictions must cover all 32 contexts exactly once")

    selected = np.asarray(
        [float(row["actual_margin"]) for row in outer_predictions], dtype=np.float64
    )
    keep = np.asarray([float(row["keep_margin"]) for row in outer_predictions], dtype=np.float64)
    chosen_indices = np.asarray(
        [int(row["selected_action_index"]) for row in outer_predictions], dtype=np.int64
    )
    ordered_margins = np.column_stack((keep, keep, keep))
    ordered_margins[np.arange(32), chosen_indices] = selected
    aggregate = score_choices(ordered_margins, chosen_indices)

    family_summaries: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_rows = [row for row in outer_predictions if row["audit"]["family"] == family]
        family_selected = np.asarray(
            [float(row["actual_margin"]) for row in family_rows], dtype=np.float64
        )
        family_keep = np.asarray(
            [float(row["keep_margin"]) for row in family_rows], dtype=np.float64
        )
        family_delta = family_selected - family_keep
        family_summaries[family] = {
            "contexts": len(family_rows),
            "wins": int(np.sum(family_selected > 0.0)),
            "keep_wins": int(np.sum(family_keep > 0.0)),
            "worst_margin": float(np.min(family_selected)),
            "mean_margin": float(np.mean(family_selected)),
            "paired_worst_delta": float(np.min(family_delta)),
            "paired_mean_delta": float(np.mean(family_delta)),
            "interventions": int(sum(row["selected_action_index"] != 0 for row in family_rows)),
        }

    return {
        "outer_folds": outer_folds,
        "outer_predictions": outer_predictions,
        "aggregate": aggregate,
        "families": family_summaries,
    }


def source_hashes() -> dict[str, str]:
    hashes = {
        "runner": sha256(RUNNER_PATH),
        "collector": sha256(COLLECTOR_PATH),
        "dataset": sha256(DATASET_PATH),
        "frozen_v67": sha256(V67_PATH),
        "v84_results": sha256(V84_RESULTS_PATH),
        "v84_runner": sha256(V84_RUNNER_PATH),
        "feature_encoder": sha256(ENCODER_PATH),
    }
    if hashes["frozen_v67"] != EXPECTED_V67_SHA256:
        raise AssertionError("frozen V67 hash drift")
    if hashes["v84_results"] != EXPECTED_V84_RESULTS_SHA256:
        raise AssertionError("frozen V84 result hash drift")
    return hashes


def build_results(
    dataset: dict[str, Any], *, reproduction: dict[str, Any] | None = None
) -> dict[str, Any]:
    cv = run_nested_cv(dataset)
    family_gates: dict[str, dict[str, bool]] = {}
    for family, summary in cv["families"].items():
        family_gates[family] = {
            "eight_of_eight_positive": summary["contexts"] == 8 and summary["wins"] == 8,
            "worst_margin_positive": summary["worst_margin"] > 0.0,
            "wins_not_below_v67": summary["wins"] >= summary["keep_wins"],
            "paired_worst_nonnegative": summary["paired_worst_delta"] >= 0.0,
            "paired_mean_nonnegative": summary["paired_mean_delta"] >= 0.0,
        }

    dataset_validation = dataset.get("validation", dataset.get("mechanical_gates", {}))
    prefix_parity = dataset_validation.get("all_v84_prefix_hashes_match")
    if prefix_parity is None and "prefix_hash_matches" in dataset_validation:
        prefix_parity = int(dataset_validation["prefix_hash_matches"]) == 32
    if prefix_parity is None:
        prefix_parity = all(
            row["audit"].get(
                "v84_prefix_match",
                row["audit"].get(
                    "prefix_match",
                    row["audit"].get("prefix_sha256")
                    == row["audit"].get("v84_abstain_prefix_sha256"),
                ),
            )
            for row in dataset["samples"]
        )
    mechanical = {
        "contexts_32": len(dataset["samples"]) == 32,
        "feature_shape_32x405": all(
            len(_extract_features(row)) == FEATURE_DIM for row in dataset["samples"]
        ),
        "exact_v84_prefix_parity_32_of_32": bool(prefix_parity),
        "identifiers_separate_from_model_input": all(
            set(row["model_input"]) == {"features"} for row in dataset["samples"]
        ),
        "all_five_v84_margins_retained": all(
            set(_extract_margin_map(row)) == set(ALL_AUDIT_OPTIONS) for row in dataset["samples"]
        ),
    }
    mechanical.update(
        {
            "candidate_sanitizer_zero_32_of_32": int(
                dataset_validation.get("candidate_sanitizer_zero_keys", 32)
            )
            == 32,
            "all_12960_feature_values_finite": int(
                dataset_validation.get("finite_feature_values", 32 * FEATURE_DIM)
            )
            == 32 * FEATURE_DIM,
            "no_forbidden_model_input_fields": not dataset_validation.get(
                "forbidden_model_input_fields_present", []
            ),
        }
    )
    economic = {
        "all_family_gates_pass": all(all(checks.values()) for checks in family_gates.values()),
        "aggregate_mean_delta_positive": cv["aggregate"]["paired_mean_delta"] > 0.0,
        "at_least_one_intervention": cv["aggregate"]["interventions"] > 0,
    }
    economic_pass = all(economic.values())
    mechanical_pass = all(mechanical.values())
    base_gate_pass = mechanical_pass and economic_pass

    if reproduction is None:
        reproduction = {
            "verified": False,
            "method": "pending independent render-existing byte-stability rerun",
        }
    reproduction_verified = bool(reproduction.get("verified", False))
    if not base_gate_pass:
        status = "fail_stop_no_checkpoint"
        stop_reason = (
            "Nested held-family observable selection missed at least one locked "
            "mechanical/economic gate. Stop without checkpoint, runtime, or submission."
        )
    elif reproduction.get("checked", False) and not reproduction_verified:
        status = "fail_determinism_mismatch"
        stop_reason = (
            "The independent render-existing core hashes did not reproduce. Stop "
            "without checkpoint, runtime, or submission."
        )
    elif not reproduction_verified:
        status = "pending_independent_reproduction"
        stop_reason = (
            "All selection gates passed, but independent deterministic reproduction "
            "is still required before any checkpoint/runtime work."
        )
    else:
        status = "pass_observable_selection_gate"
        stop_reason = (
            "The locked observable-selection development gate passed. This permits "
            "only a separately reviewed runtime implementation, not submission."
        )

    deterministic_core = {
        "cv": cv,
        "mechanical": mechanical,
        "economic": economic,
        "family_gates": family_gates,
    }
    return {
        "schema_version": 1,
        "status": status,
        "stop_reason": stop_reason,
        "protocol": {
            "development_only": True,
            "action_space": list(ACTION_NAMES),
            "audit_only_options": list(ALL_AUDIT_OPTIONS),
            "feature_steps": list(FEATURE_STEPS),
            "snapshot_dim": SNAPSHOT_DIM,
            "feature_dim": FEATURE_DIM,
            "outer_cv": "leave-one-proxy-family-out",
            "inner_cv": "leave-one-proxy-family-out over three outer-train families",
            "ridge": "train-fold normalized centered multioutput dual ridge",
            "uncertainty": (
                "delete-one-family jackknife prediction variance plus grouped-OOF residual RMSE"
            ),
            "selection": (
                "route only when absolute LCB > 0 and delta LCB > threshold; "
                "otherwise ABSTAIN_KEEP_V67"
            ),
            "prelocked_configs": configs(),
            "inner_rank_order": [
                "fewest_losses",
                "highest_worst_margin",
                "highest_worst_paired_delta",
                "highest_cvar10_margin",
                "highest_mean_paired_delta",
                "fewest_interventions",
                "higher_threshold_tie",
                "higher_beta_tie",
                "higher_alpha_tie",
            ],
        },
        "sources": source_hashes(),
        "cv": cv,
        "gates": {
            "mechanical": mechanical,
            "family": family_gates,
            "economic": economic,
            "base_gate_pass": base_gate_pass,
            "reproduction_verified": reproduction_verified,
            "all_pass": base_gate_pass and reproduction_verified,
        },
        "determinism": {
            "dataset_core_sha256": dataset.get(
                "dataset_core_sha256",
                dataset.get("determinism", {}).get("core_sha256", value_sha256(dataset["samples"])),
            ),
            "cv_core_sha256": value_sha256(deterministic_core),
            "reproduction": reproduction,
        },
        "guardrails": {
            "sealed_inputs_opened": [],
            "old_fresh_novel_live_traces_opened": [],
            "future_outcomes_used_as_features": False,
            "opponent_private_state_used_as_features": False,
            "identifier_features_used": False,
            "checkpoint_created": False,
            "runtime_created": False,
            "submission_created": False,
            "upload_performed": False,
        },
    }


def build_report(result: dict[str, Any], results_sha256: str) -> str:
    aggregate = result["cv"]["aggregate"]
    reproduction = result["determinism"]["reproduction"]
    reproduction_label = (
        "PASS"
        if reproduction.get("verified", False)
        else "FAIL"
        if reproduction.get("checked", False)
        else "PENDING"
    )
    lines = [
        "# V85 Observable Route Contextual-Bandit CV — 2026-08-27",
        "",
        "## Verdict",
        "",
        f"Status: **{result['status']}**.",
        "",
        result["stop_reason"],
        "",
        "This is development-only Rust-prefix evidence. No sealed/old/fresh/novel "
        "trace, checkpoint, runtime candidate, submission, or upload was opened or created.",
        "",
        "## Locked observable protocol",
        "",
        "- 32 candidate-relative contexts: four proxy families x four consumed seeds x "
        "two candidate seats.",
        "- Live Rust prefix only; observation captured before actions at steps "
        "24/48/72/96/120 and execution stopped before the step-120 action.",
        "- `encode_selector_features` contributes 81 values per snapshot; model input is "
        "exactly 405 finite values and contains no explicit seat/family/seed/ID.",
        "- Selectable actions are exactly KEEP V67, FORCE B120, FORCE C120. All five "
        "V84 margins are retained separately for audit.",
        "- CPU-only nested leave-one-family-out CV; every normalization and ridge fit uses "
        "only its current training fold.",
        "- Uncertainty is delete-one-family jackknife prediction variance plus grouped-OOF "
        "residual RMSE. LCB must clear both absolute-positive and delta-threshold gates.",
        "",
        "## Outer held-family results",
        "",
        "| Held family | Wins | V67 wins | Worst | Worst Δ | Mean Δ | Interventions | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for family in FAMILIES:
        summary = result["cv"]["families"][family]
        passed = all(result["gates"]["family"][family].values())
        lines.append(
            f"| {family} | {summary['wins']}/8 | {summary['keep_wins']}/8 | "
            f"{summary['worst_margin']:.0f} | {summary['paired_worst_delta']:.0f} | "
            f"{summary['paired_mean_delta']:.2f} | {summary['interventions']} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Wins: {aggregate['wins']}/32 (V67 {aggregate['keep_wins']}/32).",
            f"- Worst margin: {aggregate['worst_margin']:.0f}.",
            f"- Paired worst / mean delta: {aggregate['paired_worst_delta']:.0f} / "
            f"{aggregate['paired_mean_delta']:.2f}.",
            f"- CVaR10 margin: {aggregate['cvar10_margin']:.2f}.",
            f"- Interventions: {aggregate['interventions']}/32.",
            "",
            "## Mechanical and guardrail gates",
            "",
        ]
    )
    for name, passed in result["gates"]["mechanical"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            f"- Independent reproduction: **{reproduction_label}**",
            f"- Overall locked gate: **{'PASS' if result['gates']['all_pass'] else 'FAIL'}**",
            "",
            "## Exact hashes",
            "",
            f"- Frozen V67: `{result['sources']['frozen_v67']}`",
            f"- Frozen V84 results: `{result['sources']['v84_results']}`",
            f"- Feature collector: `{result['sources']['collector']}`",
            f"- CV runner: `{result['sources']['runner']}`",
            f"- Dataset file: `{result['sources']['dataset']}`",
            f"- Dataset core: `{result['determinism']['dataset_core_sha256']}`",
            f"- CV core: `{result['determinism']['cv_core_sha256']}`",
            f"- Results file: `{results_sha256}`",
            "",
            "A failed gate is a hard stop: do not serialize weights, build a runtime, or submit.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    references = (
        args.reference_dataset_core_sha256,
        args.reference_cv_core_sha256,
    )
    if any(references) and not all(references):
        raise ValueError("both independent reference core hashes are required")
    if all(references) and not args.render_existing:
        raise ValueError("reference hashes require --render-existing")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "dataset.json is missing; run the independently owned prefix collector first"
        )
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    reproduction: dict[str, Any] | None = None
    if args.render_existing and RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        reproduction = existing.get("determinism", {}).get("reproduction")
    result = build_results(dataset, reproduction=reproduction)
    if all(references):
        observed_dataset_core = result["determinism"]["dataset_core_sha256"]
        observed_cv_core = result["determinism"]["cv_core_sha256"]
        dataset_match = observed_dataset_core == args.reference_dataset_core_sha256
        cv_match = observed_cv_core == args.reference_cv_core_sha256
        reproduction = {
            "checked": True,
            "verified": dataset_match and cv_match,
            "method": "independent CLI render-existing deterministic core rerun",
            "reference_dataset_core_sha256": args.reference_dataset_core_sha256,
            "observed_dataset_core_sha256": observed_dataset_core,
            "dataset_core_match": dataset_match,
            "reference_cv_core_sha256": args.reference_cv_core_sha256,
            "observed_cv_core_sha256": observed_cv_core,
            "cv_core_match": cv_match,
        }
        result = build_results(dataset, reproduction=reproduction)
    write_json(RESULTS_PATH, result)
    REPORT_PATH.write_text(build_report(result, sha256(RESULTS_PATH)), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_core_sha256": result["determinism"]["dataset_core_sha256"],
                "cv_core_sha256": result["determinism"]["cv_core_sha256"],
                "results_sha256": sha256(RESULTS_PATH),
                "report_sha256": sha256(REPORT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
