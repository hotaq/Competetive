#!/usr/bin/env python3
"""Evaluate the frozen V86-A route bandit with clustered out-of-fold CV.

The collector owns all Rust-engine execution.  This evaluator consumes only a
complete, source-pinned ``dataset.json`` and the frozen ``protocol.json``.  It
never opens the sealed holdout, creates a runtime agent, or performs model or
threshold search.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = OUT / "protocol.json"
DATASET_PATH = OUT / "dataset.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"
V85_RUNNER_PATH = ROOT / "analysis/v85_route_bandit_cv_20260827/run_cv.py"

EXPECTED_PROTOCOL_SHA256 = "007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30"
EXPECTED_PROTOCOL_SCHEMA = "v86-family-scale-prereg-v1"
EXPECTED_DATASET_SCHEMA = "v86-family-scale-collection-v1"
EXPECTED_V85_RUNNER_SHA256 = "1e3b69b4a92eb8e6760ba2726eee377592ecea722284e27ec8e64195aa902ae9"

KEEP = "KEEP_V67"
ROUTE_B = "FORCE_B120_SUFFIX"
ROUTE_C = "FORCE_C120_SUFFIX"
ACTION_NAMES = (KEEP, ROUTE_B, ROUTE_C)
MODEL_LABELS = ("KEEP", "B", "C")
DELTA_LABELS = ("B", "C")
FEATURE_STEPS = (24, 48, 72, 96, 120)
FEATURE_DIM = 405


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help=(
            "Recompute from the frozen dataset and compare deterministic core "
            "hashes with the existing results file."
        ),
    )
    parser.add_argument(
        "--reference-dataset-core-sha256",
        help="Explicit dataset-core reference for an independent rerun.",
    )
    parser.add_argument(
        "--reference-evaluation-core-sha256",
        help="Explicit evaluation-core reference for an independent rerun.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic in-memory V85 math parity tests only.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"{label} keys differ; missing={missing}, extra={extra}")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AssertionError(f"{label} must be >= {minimum}")
    return value


def _resolve_repo_path(relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AssertionError(f"{label} must be a non-empty relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise AssertionError(f"{label} escapes the repository")
    if not path.is_file():
        raise AssertionError(f"{label} does not exist: {relative}")
    return path


def _verify_source_record(
    record: dict[str, Any],
    label: str,
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> None:
    path = _resolve_repo_path(record.get(path_key), f"{label}.{path_key}")
    claimed = record.get(hash_key)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise AssertionError(f"{label}.{hash_key} must be a SHA-256 string")
    observed = sha256(path)
    if observed != claimed:
        raise AssertionError(f"{label} source drift: claimed {claimed}, observed {observed}")


def validate_protocol(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    observed_hash = sha256(path)
    if observed_hash != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError(
            "frozen protocol hash mismatch: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, observed {observed_hash}"
        )
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise AssertionError("unexpected protocol schema")
    if protocol.get("status") != "frozen_before_economic_outcomes":
        raise AssertionError("protocol is not frozen-before-outcomes")

    required = {
        "candidate",
        "actions",
        "families",
        "fresh_development_seeds",
        "expected_contexts",
        "model",
        "mechanical_gates",
        "adequacy_gates",
        "selector_go_gates",
        "promotion_grade_extra_gates",
        "stop_rule",
    }
    missing = required - set(protocol)
    if missing:
        raise AssertionError(f"protocol missing frozen fields: {sorted(missing)}")

    candidate = protocol["candidate"]
    if not isinstance(candidate, dict):
        raise AssertionError("protocol candidate must be an object")
    _verify_source_record(candidate, "protocol.candidate")

    actions = protocol["actions"]
    if not isinstance(actions, list):
        raise AssertionError("protocol actions must be a list")
    action_ids = tuple(row.get("id") for row in actions if isinstance(row, dict))
    if action_ids != ACTION_NAMES:
        raise AssertionError(f"unexpected frozen action order: {action_ids}")
    for row in actions[1:]:
        _verify_source_record(
            row,
            f"protocol.actions.{row['id']}",
            path_key="source_path",
            hash_key="source_sha256",
        )

    families = protocol["families"]
    if not isinstance(families, list) or len(families) != 8:
        raise AssertionError("protocol must contain exactly eight families")
    family_ids: list[str] = []
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise AssertionError(f"protocol family {index} must be an object")
        family_id = family.get("id")
        if not isinstance(family_id, str) or not family_id:
            raise AssertionError(f"protocol family {index} has invalid id")
        family_ids.append(family_id)
        _verify_source_record(family, f"protocol.families.{family_id}")
    if len(set(family_ids)) != len(family_ids):
        raise AssertionError("protocol family ids must be unique")

    seeds = protocol["fresh_development_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 8:
        raise AssertionError("protocol must contain exactly eight fresh seeds")
    for index, seed in enumerate(seeds):
        _integer(seed, f"protocol seed {index}", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise AssertionError("protocol seeds must be unique")
    if protocol["expected_contexts"] != 128:
        raise AssertionError("protocol expected_contexts must remain 128")

    model = protocol["model"]
    expected_model = {
        "type": "dual_ridge_absolute_and_delta_heads",
        "alpha": 1000.0,
        "beta": 2.0,
        "switch_threshold": 2500.0,
        "actions": list(ACTION_NAMES),
        "intervention_rule": ("absolute_lcb > 0 and delta_lcb > 2500; equality abstains"),
        "tie_break": [
            "higher_delta_lcb",
            "higher_absolute_lcb",
            "FORCE_B120_SUFFIX_before_FORCE_C120_SUFFIX",
        ],
        "hyperparameter_search": False,
    }
    if model != expected_model:
        raise AssertionError("frozen model contract changed")

    feature_shape = protocol["mechanical_gates"].get("finite_feature_shape")
    if feature_shape != [128, FEATURE_DIM]:
        raise AssertionError("unexpected frozen feature shape")
    return {
        "sha256": observed_hash,
        "family_ids": tuple(family_ids),
        "seeds": tuple(seeds),
        "family_by_id": {row["id"]: row for row in families},
    }


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalization using only the current training fold."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (features - mean) / scale, mean, scale


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fit centered multi-output ridge through the V85 dual system."""
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
    """Return standard delete-one-group jackknife prediction variance."""
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
    """Predict V85 heads with family jackknife and grouped-OOF RMSE."""
    unique_groups = tuple(sorted(set(train_groups.tolist())))
    if len(unique_groups) < 2:
        raise AssertionError("need at least two training families")

    absolute_model = ridge_fit(train_features, train_margins, alpha)
    delta_model = ridge_fit(train_features, train_deltas, alpha)
    absolute_mean = ridge_predict(absolute_model, eval_features)
    delta_mean = ridge_predict(delta_model, eval_features)

    absolute_jackknife: list[np.ndarray] = []
    delta_jackknife: list[np.ndarray] = []
    absolute_oof = np.full_like(train_margins, np.nan)
    delta_oof = np.full_like(train_deltas, np.nan)
    for group in unique_groups:
        kept = train_groups != group
        held = ~kept
        absolute_submodel = ridge_fit(train_features[kept], train_margins[kept], alpha)
        delta_submodel = ridge_fit(train_features[kept], train_deltas[kept], alpha)
        absolute_jackknife.append(ridge_predict(absolute_submodel, eval_features))
        delta_jackknife.append(ridge_predict(delta_submodel, eval_features))
        absolute_oof[held] = ridge_predict(absolute_submodel, train_features[held])
        delta_oof[held] = ridge_predict(delta_submodel, train_features[held])

    if not np.isfinite(absolute_oof).all() or not np.isfinite(delta_oof).all():
        raise AssertionError("grouped OOF predictions are incomplete or non-finite")
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


def _validate_hash_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AssertionError(f"{label} must be a SHA-256 string")
    if any(character not in "0123456789abcdef" for character in value):
        raise AssertionError(f"{label} is not lowercase hexadecimal")
    return value


def _validate_sanitizer(value: Any, label: str) -> int:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    required = {
        "changed_steps",
        "farmer_changed_steps",
        "hands_changed_steps",
        "market_changed_steps",
        "step_ids",
    }
    _require_exact_keys(value, required, label)
    changed = _integer(value["changed_steps"], f"{label}.changed_steps", minimum=0)
    component_changes = 0
    for key in (
        "farmer_changed_steps",
        "hands_changed_steps",
        "market_changed_steps",
    ):
        component_changes += _integer(value[key], f"{label}.{key}", minimum=0)
    step_ids = value["step_ids"]
    if not isinstance(step_ids, list):
        raise AssertionError(f"{label}.step_ids must be a list")
    for index, step in enumerate(step_ids):
        _integer(step, f"{label}.step_ids[{index}]", minimum=0)
    return changed + component_changes + len(step_ids)


def _validate_repair(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    _require_exact_keys(value, {"changed_steps", "step_ids", "changed_paths"}, label)
    _integer(value["changed_steps"], f"{label}.changed_steps", minimum=0)
    step_ids = value["step_ids"]
    changed_paths = value["changed_paths"]
    if not isinstance(step_ids, list):
        raise AssertionError(f"{label}.step_ids must be a list")
    if not isinstance(changed_paths, list) or not all(
        isinstance(item, str) for item in changed_paths
    ):
        raise AssertionError(f"{label}.changed_paths must be a string list")
    for index, step in enumerate(step_ids):
        _integer(step, f"{label}.step_ids[{index}]", minimum=0)


def _validate_route_audit(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    _require_exact_keys(value, {"starts", "completions", "selected_route"}, label)
    _integer(value["starts"], f"{label}.starts", minimum=0)
    _integer(value["completions"], f"{label}.completions", minimum=0)
    selected_route = value["selected_route"]
    if selected_route is not None and not isinstance(selected_route, str):
        raise AssertionError(f"{label}.selected_route must be a string or null")


def _validate_candidate_audit(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    required = {
        "raw_sha256",
        "proposed_sha256",
        "executed_sha256",
        "prefix_0_119_proposed_sha256",
        "repair",
        "sanitizer",
        "route",
        "fallback_events",
        "exception_events",
        "partial_action_events",
        "exception",
    }
    missing = required - set(value)
    if missing:
        raise AssertionError(f"{label} missing fields: {sorted(missing)}")
    for key in (
        "raw_sha256",
        "proposed_sha256",
        "executed_sha256",
        "prefix_0_119_proposed_sha256",
    ):
        _validate_hash_string(value[key], f"{label}.{key}")
    _validate_repair(value["repair"], f"{label}.repair")
    _validate_route_audit(value["route"], f"{label}.route")
    sanitizer_changes = _validate_sanitizer(value["sanitizer"], f"{label}.sanitizer")
    counters = {
        "sanitizer_changes": sanitizer_changes,
        "fallback_events": _integer(
            value["fallback_events"], f"{label}.fallback_events", minimum=0
        ),
        "exception_events": _integer(
            value["exception_events"], f"{label}.exception_events", minimum=0
        ),
        "partial_action_events": _integer(
            value["partial_action_events"],
            f"{label}.partial_action_events",
            minimum=0,
        ),
    }
    if counters["exception_events"] == 0 and value["exception"] is not None:
        raise AssertionError(f"{label}.exception must be null when count is zero")
    return counters


def _validate_prefix_parity(value: Any, label: str) -> bool:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    if set(value) != set(ACTION_NAMES):
        raise AssertionError(f"{label} must contain exactly the three actions")
    passed = True
    for action in ACTION_NAMES:
        row = value[action]
        if not isinstance(row, dict):
            raise AssertionError(f"{label}.{action} must be an object")
        _require_exact_keys(
            row,
            {"reference_prefix_sha256", "candidate_prefix_sha256", "equal"},
            f"{label}.{action}",
        )
        for key in ("reference_prefix_sha256", "candidate_prefix_sha256"):
            _validate_hash_string(row.get(key), f"{label}.{action}.{key}")
        equal = row.get("equal")
        if not isinstance(equal, bool):
            raise AssertionError(f"{label}.{action}.equal must be boolean")
        passed = passed and equal
        passed = passed and (row["reference_prefix_sha256"] == row["candidate_prefix_sha256"])
    return passed


def _validate_difference_contract(value: Any, label: str) -> bool:
    if not isinstance(value, dict) or set(value) != {ROUTE_B, ROUTE_C}:
        raise AssertionError(f"{label} must contain exactly B and C route contracts")
    passed = True
    for action in (ROUTE_B, ROUTE_C):
        row = value[action]
        if not isinstance(row, dict):
            raise AssertionError(f"{label}.{action} must be an object")
        _require_exact_keys(
            row,
            {
                "different_steps",
                "first_difference_step",
                "last_difference_step",
                "first_difference_paths",
                "no_difference_before_120",
            },
            f"{label}.{action}",
        )
        different_steps = _integer(
            row.get("different_steps"), f"{label}.{action}.different_steps", minimum=0
        )
        first = row.get("first_difference_step")
        if first is not None:
            _integer(first, f"{label}.{action}.first_difference_step", minimum=0)
        last = row.get("last_difference_step")
        if last is not None:
            _integer(last, f"{label}.{action}.last_difference_step", minimum=0)
        if first is not None and last is not None and last < first:
            raise AssertionError(f"{label}.{action} last difference precedes first")
        no_early = row.get("no_difference_before_120")
        if not isinstance(no_early, bool):
            raise AssertionError(f"{label}.{action}.no_difference_before_120 must be boolean")
        paths = row.get("first_difference_paths")
        if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
            raise AssertionError(f"{label}.{action}.first_difference_paths must be a string list")
        passed = passed and different_steps > 0
        passed = passed and first is not None and first >= 120
        passed = passed and no_early
    return passed


def _validate_dataset_sources(dataset: dict[str, Any], protocol: dict[str, Any]) -> None:
    sources = dataset.get("sources")
    if not isinstance(sources, dict):
        raise AssertionError("dataset.sources must be an object")
    candidate = sources.get("candidate")
    if not isinstance(candidate, dict):
        raise AssertionError("dataset.sources.candidate must be an object")
    for key in ("path", "sha256"):
        if candidate.get(key) != protocol["candidate"].get(key):
            raise AssertionError(f"dataset candidate {key} differs from protocol")

    opponents = sources.get("opponents")
    if not isinstance(opponents, list) or len(opponents) != len(protocol["families"]):
        raise AssertionError("dataset must pin exactly eight opponent sources")
    observed: dict[str, dict[str, Any]] = {}
    for row in opponents:
        if not isinstance(row, dict) or not isinstance(row.get("family"), str):
            raise AssertionError("dataset opponent source has invalid family")
        if row["family"] in observed:
            raise AssertionError("dataset opponent families must be unique")
        observed[row["family"]] = row
    for expected in protocol["families"]:
        row = observed.get(expected["id"])
        if row is None:
            raise AssertionError(f"dataset missing opponent {expected['id']}")
        for key in ("path", "sha256"):
            if row.get(key) != expected.get(key):
                raise AssertionError(
                    f"dataset opponent {expected['id']} {key} differs from protocol"
                )


def _dataset_core_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    protocol_ref = dataset.get("protocol")
    if not isinstance(protocol_ref, dict):
        raise AssertionError("dataset.protocol must be an object")
    return {
        "schema_version": dataset.get("schema_version"),
        "protocol_sha256": protocol_ref.get("sha256"),
        "samples": dataset.get("samples"),
        "validation": dataset.get("validation"),
    }


def validate_and_extract_dataset(
    dataset: dict[str, Any],
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
) -> dict[str, Any]:
    if dataset.get("schema_version") != EXPECTED_DATASET_SCHEMA:
        raise AssertionError("unexpected dataset schema")
    if dataset.get("status") != "collection_complete":
        raise AssertionError("refusing partial or smoke collection")
    protocol_ref = dataset.get("protocol")
    if not isinstance(protocol_ref, dict):
        raise AssertionError("dataset.protocol must be an object")
    if protocol_ref.get("sha256") != protocol_info["sha256"]:
        raise AssertionError("dataset was not collected under the frozen protocol")
    if protocol_ref.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise AssertionError("dataset protocol schema mismatch")
    if protocol_ref.get("status") != "frozen_before_economic_outcomes":
        raise AssertionError("dataset protocol status mismatch")
    _validate_dataset_sources(dataset, protocol)

    claimed_core = _validate_hash_string(dataset.get("core_sha256"), "dataset.core_sha256")
    observed_core = value_sha256(_dataset_core_payload(dataset))
    if observed_core != claimed_core:
        raise AssertionError(
            f"dataset core mismatch: claimed {claimed_core}, observed {observed_core}"
        )

    samples = dataset.get("samples")
    expected_contexts = protocol["expected_contexts"]
    if not isinstance(samples, list) or len(samples) != expected_contexts:
        raise AssertionError(f"dataset must contain exactly {expected_contexts} samples")

    family_order = {name: index for index, name in enumerate(protocol_info["family_ids"])}
    seed_order = {seed: index for index, seed in enumerate(protocol_info["seeds"])}
    expected_keys = {
        (family, seed, seat)
        for family in protocol_info["family_ids"]
        for seed in protocol_info["seeds"]
        for seat in (0, 1)
    }
    records: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, int, int]] = set()
    counters = {
        "games": 0,
        "final_step_matches": 0,
        "prefix_parity_rows": 0,
        "difference_contract_rows": 0,
        "candidate_sanitizer_changes": 0,
        "candidate_fallbacks": 0,
        "candidate_exceptions": 0,
        "candidate_partial_actions": 0,
    }

    for sample_index, sample in enumerate(samples):
        label = f"samples[{sample_index}]"
        if not isinstance(sample, dict):
            raise AssertionError(f"{label} must be an object")
        for key in ("audit", "model_input", "labels", "games"):
            if key not in sample:
                raise AssertionError(f"{label} missing {key}")
        audit = sample["audit"]
        if not isinstance(audit, dict):
            raise AssertionError(f"{label}.audit must be an object")
        family = audit.get("family")
        seed = audit.get("seed")
        seat = audit.get("candidate_seat")
        if family not in family_order:
            raise AssertionError(f"{label} has an unknown family")
        if seed not in seed_order:
            raise AssertionError(f"{label} has an unknown seed")
        _integer(seat, f"{label}.audit.candidate_seat")
        if seat not in (0, 1):
            raise AssertionError(f"{label} candidate seat must be 0 or 1")
        if audit.get("opponent_seat") != 1 - seat:
            raise AssertionError(f"{label} opponent seat mismatch")
        context_key = (family, seed, seat)
        if context_key in observed_keys:
            raise AssertionError(f"duplicate context key: {context_key}")
        observed_keys.add(context_key)
        key_object = audit.get("key")
        if key_object != {
            "family": family,
            "seed": seed,
            "candidate_seat": seat,
        }:
            raise AssertionError(f"{label}.audit.key is not canonical")
        family_contract = protocol_info["family_by_id"][family]
        if audit.get("representative") != family_contract.get("representative"):
            raise AssertionError(f"{label} representative differs from protocol")
        if audit.get("source_group") != family_contract.get("source_group"):
            raise AssertionError(f"{label} source_group differs from protocol")
        if audit.get("observed_steps") != list(FEATURE_STEPS):
            raise AssertionError(f"{label} observation steps differ from protocol")
        if audit.get("feature_count") != FEATURE_DIM:
            raise AssertionError(f"{label} feature_count must be {FEATURE_DIM}")
        prefix_ok = _validate_prefix_parity(
            audit.get("prefix_parity"), f"{label}.audit.prefix_parity"
        )
        difference_ok = _validate_difference_contract(
            audit.get("difference_contract"),
            f"{label}.audit.difference_contract",
        )
        counters["prefix_parity_rows"] += int(prefix_ok)
        counters["difference_contract_rows"] += int(difference_ok)

        model_input = sample["model_input"]
        if not isinstance(model_input, dict):
            raise AssertionError(f"{label}.model_input must be an object")
        _require_exact_keys(model_input, {"features"}, f"{label}.model_input")
        raw_features = model_input["features"]
        if not isinstance(raw_features, list) or len(raw_features) != FEATURE_DIM:
            raise AssertionError(f"{label} must contain exactly {FEATURE_DIM} features")
        features = [
            _finite_number(value, f"{label}.model_input.features[{index}]")
            for index, value in enumerate(raw_features)
        ]

        labels = sample["labels"]
        if not isinstance(labels, dict):
            raise AssertionError(f"{label}.labels must be an object")
        _require_exact_keys(
            labels,
            {"model_margins", "action_margins", "paired_deltas"},
            f"{label}.labels",
        )
        model_margins = labels["model_margins"]
        action_margins = labels["action_margins"]
        paired_deltas = labels["paired_deltas"]
        if not isinstance(model_margins, dict):
            raise AssertionError(f"{label}.labels.model_margins must be an object")
        if not isinstance(action_margins, dict):
            raise AssertionError(f"{label}.labels.action_margins must be an object")
        if not isinstance(paired_deltas, dict):
            raise AssertionError(f"{label}.labels.paired_deltas must be an object")
        _require_exact_keys(model_margins, set(MODEL_LABELS), f"{label}.labels.model_margins")
        _require_exact_keys(action_margins, set(ACTION_NAMES), f"{label}.labels.action_margins")
        _require_exact_keys(paired_deltas, set(DELTA_LABELS), f"{label}.labels.paired_deltas")
        margins = np.array(
            [
                _finite_number(model_margins[name], f"{label}.margin.{name}")
                for name in MODEL_LABELS
            ],
            dtype=np.float64,
        )
        for index, action in enumerate(ACTION_NAMES):
            observed = _finite_number(action_margins[action], f"{label}.action_margin.{action}")
            if observed != margins[index]:
                raise AssertionError(f"{label} model/action margin mismatch for {action}")
        deltas = np.array(
            [_finite_number(paired_deltas[name], f"{label}.delta.{name}") for name in DELTA_LABELS],
            dtype=np.float64,
        )
        if not np.array_equal(deltas, margins[1:] - margins[0]):
            raise AssertionError(f"{label} paired deltas are inconsistent")

        games = sample["games"]
        if not isinstance(games, dict) or set(games) != set(ACTION_NAMES):
            raise AssertionError(f"{label}.games must contain exactly three actions")
        for action_index, action in enumerate(ACTION_NAMES):
            game = games[action]
            game_label = f"{label}.games.{action}"
            if not isinstance(game, dict):
                raise AssertionError(f"{game_label} must be an object")
            if game.get("action") != action:
                raise AssertionError(f"{game_label}.action mismatch")
            if game.get("candidate_seat") != seat:
                raise AssertionError(f"{game_label}.candidate_seat mismatch")
            if game.get("opponent_seat") != 1 - seat:
                raise AssertionError(f"{game_label}.opponent_seat mismatch")
            game_margin = _finite_number(game.get("margin"), f"{game_label}.margin")
            if game_margin != margins[action_index]:
                raise AssertionError(f"{game_label} margin differs from labels")
            candidate_money = _finite_number(
                game.get("candidate_money"), f"{game_label}.candidate_money"
            )
            opponent_money = _finite_number(
                game.get("opponent_money"), f"{game_label}.opponent_money"
            )
            if candidate_money - opponent_money != game_margin:
                raise AssertionError(f"{game_label} money/margin values are inconsistent")
            final_step = _integer(game.get("final_step"), f"{game_label}.final_step")
            completed = _integer(
                game.get("completed_transitions"),
                f"{game_label}.completed_transitions",
                minimum=0,
            )
            if completed != final_step + 1:
                raise AssertionError(f"{game_label} transition/final-step mismatch")
            counters["games"] += 1
            counters["final_step_matches"] += int(
                final_step == protocol["mechanical_gates"]["all_games_final_step"]
            )
            candidate_audit = _validate_candidate_audit(
                game.get("candidate"), f"{game_label}.candidate"
            )
            parity_row = audit["prefix_parity"][action]
            if (
                game["candidate"]["prefix_0_119_proposed_sha256"]
                != parity_row["candidate_prefix_sha256"]
            ):
                raise AssertionError(f"{game_label} prefix hash differs from audit")
            counters["candidate_sanitizer_changes"] += candidate_audit["sanitizer_changes"]
            counters["candidate_fallbacks"] += candidate_audit["fallback_events"]
            counters["candidate_exceptions"] += candidate_audit["exception_events"]
            counters["candidate_partial_actions"] += candidate_audit["partial_action_events"]

        records.append(
            {
                "family": family,
                "seed": seed,
                "seat": seat,
                "features": features,
                "margins": margins.tolist(),
                "deltas": deltas.tolist(),
            }
        )

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AssertionError(
            f"dataset context cartesian product differs; missing={missing}, extra={extra}"
        )
    records.sort(
        key=lambda row: (
            family_order[row["family"]],
            seed_order[row["seed"]],
            row["seat"],
        )
    )
    return {
        "records": records,
        "dataset_core_sha256": observed_core,
        "counters": counters,
    }


def _score_choices(margins: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
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


def run_cluster_cv(
    extracted: dict[str, Any], protocol: dict[str, Any], protocol_info: dict[str, Any]
) -> dict[str, Any]:
    records = extracted["records"]
    features = np.asarray([row["features"] for row in records], dtype=np.float64)
    margins = np.asarray([row["margins"] for row in records], dtype=np.float64)
    deltas = np.asarray([row["deltas"] for row in records], dtype=np.float64)
    families = np.asarray([row["family"] for row in records], dtype=object)
    seeds = np.asarray([row["seed"] for row in records], dtype=np.int64)
    seats = np.asarray([row["seat"] for row in records], dtype=np.int64)

    choices = np.full(len(records), -1, dtype=np.int64)
    diagnostics: list[dict[str, Any] | None] = [None] * len(records)
    folds: list[dict[str, Any]] = []
    alpha = float(protocol["model"]["alpha"])
    beta = float(protocol["model"]["beta"])
    threshold = float(protocol["model"]["switch_threshold"])

    for held_family in protocol_info["family_ids"]:
        for held_seed in protocol_info["seeds"]:
            held = (families == held_family) & (seeds == held_seed)
            train = (families != held_family) & (seeds != held_seed)
            held_indices = np.flatnonzero(held)
            train_indices = np.flatnonzero(train)
            if len(held_indices) != 2 or set(seats[held_indices].tolist()) != {0, 1}:
                raise AssertionError("each family-seed fold must predict both seats")
            if len(train_indices) != 98:
                raise AssertionError(
                    f"cluster fold must contain 98 training rows, got {len(train_indices)}"
                )
            train_family_names = tuple(sorted(set(families[train_indices].tolist())))
            if len(train_family_names) != 7 or held_family in train_family_names:
                raise AssertionError("outer fold must train on exactly seven other families")
            if held_seed in set(seeds[train_indices].tolist()):
                raise AssertionError("outer fold leaked the held seed")
            training_rows_by_family = {
                family: int(np.sum(families[train_indices] == family))
                for family in train_family_names
            }
            if set(training_rows_by_family.values()) != {14}:
                raise AssertionError("each of the seven training families must contribute 14 rows")

            prediction = fit_predict_heads(
                features[train_indices],
                margins[train_indices],
                deltas[train_indices],
                families[train_indices],
                features[held_indices],
                alpha=alpha,
            )
            fold_choices, fold_diagnostics = select_actions(
                prediction, beta=beta, threshold=threshold
            )
            choices[held_indices] = fold_choices
            for local_index, record_index in enumerate(held_indices):
                diagnostics[record_index] = fold_diagnostics[local_index]
            folds.append(
                {
                    "held_family": held_family,
                    "held_seed": int(held_seed),
                    "held_rows": len(held_indices),
                    "held_seats": seats[held_indices].tolist(),
                    "train_rows": len(train_indices),
                    "train_family_count": len(train_family_names),
                    "train_families": list(train_family_names),
                    "train_seed_count": len(set(seeds[train_indices].tolist())),
                    "train_rows_by_family": training_rows_by_family,
                    "delete_one_family_training_rows": 84,
                    "selected_actions": [ACTION_NAMES[index] for index in fold_choices.tolist()],
                    "absolute_oof_rmse": prediction["absolute_oof_rmse"].tolist(),
                    "delta_oof_rmse": prediction["delta_oof_rmse"].tolist(),
                }
            )

    if np.any(choices < 0) or any(row is None for row in diagnostics):
        raise AssertionError("cluster CV did not predict every context exactly once")

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        choice = int(choices[index])
        selected_margin = float(margins[index, choice])
        keep_margin = float(margins[index, 0])
        predictions.append(
            {
                "key": {
                    "family": record["family"],
                    "seed": record["seed"],
                    "candidate_seat": record["seat"],
                },
                "selected_action": ACTION_NAMES[choice],
                "selected_action_index": choice,
                "selected_margin": selected_margin,
                "keep_margin": keep_margin,
                "actual_paired_delta": selected_margin - keep_margin,
                "diagnostic": diagnostics[index],
            }
        )

    return {
        "choices": choices,
        "features": features,
        "margins": margins,
        "families": families,
        "seeds": seeds,
        "folds": folds,
        "predictions": predictions,
    }


def _family_metrics(
    margins: np.ndarray,
    choices: np.ndarray,
    families: np.ndarray,
    family_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    rows: list[dict[str, Any]] = []
    for family in family_order:
        mask = families == family
        family_selected = selected[mask]
        family_keep = keep[mask]
        selected_wins = int(np.sum(family_selected > 0.0))
        keep_wins = int(np.sum(family_keep > 0.0))
        selected_worst = float(np.min(family_selected))
        keep_worst = float(np.min(family_keep))
        selected_mean = float(np.mean(family_selected))
        keep_mean = float(np.mean(family_keep))
        rows.append(
            {
                "family": family,
                "contexts": int(np.sum(mask)),
                "selected_wins": selected_wins,
                "keep_wins": keep_wins,
                "selected_worst_margin": selected_worst,
                "keep_worst_margin": keep_worst,
                "selected_mean_margin": selected_mean,
                "keep_mean_margin": keep_mean,
                "wins_not_worse": selected_wins >= keep_wins,
                "worst_not_worse": selected_worst >= keep_worst,
                "mean_not_worse": selected_mean >= keep_mean,
            }
        )
    return rows


def compute_gates(
    cv: dict[str, Any],
    extracted: dict[str, Any],
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    margins = cv["margins"]
    choices = cv["choices"]
    families = cv["families"]
    seeds = cv["seeds"]
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    selected_delta = selected - keep
    intervention = choices != 0
    keep_loss = keep <= 0.0
    selected_rescue = keep_loss & (selected > 0.0)
    oracle_rescue = keep_loss & (np.max(margins[:, 1:], axis=1) > 0.0)
    keep_loss_count = int(np.sum(keep_loss))

    counters = extracted["counters"]
    expected_contexts = protocol["expected_contexts"]
    expected_games = expected_contexts * len(ACTION_NAMES)
    mechanical_contract = protocol["mechanical_gates"]
    mechanical = {
        "dataset_core_matches": True,
        "contexts_match": len(margins) == expected_contexts,
        "games_match": counters["games"] == expected_games,
        "all_games_final_step": counters["final_step_matches"] == expected_games,
        "candidate_prefix_parity_through_step_119": (
            counters["prefix_parity_rows"] == expected_contexts
        ),
        "non_keep_first_difference_minimum_step_120": (
            counters["difference_contract_rows"] == expected_contexts
        ),
        "candidate_sanitizer_changes_zero": (
            counters["candidate_sanitizer_changes"]
            == mechanical_contract["candidate_sanitizer_changes"]
        ),
        "candidate_fallbacks_zero": (
            counters["candidate_fallbacks"] == mechanical_contract["candidate_fallbacks"]
        ),
        "candidate_exceptions_zero": (
            counters["candidate_exceptions"] == mechanical_contract["candidate_exceptions"]
        ),
        "candidate_partial_actions_zero": (
            counters["candidate_partial_actions"]
            == mechanical_contract["candidate_partial_actions"]
        ),
        "finite_feature_shape_128x405": (
            cv["features"].shape == tuple(mechanical_contract["finite_feature_shape"])
            and bool(np.isfinite(cv["features"]).all())
        ),
        "cluster_folds_64": len(cv["folds"]) == 64,
        "each_fold_train_rows_98": all(fold["train_rows"] == 98 for fold in cv["folds"]),
        "seven_group_jackknife": all(fold["train_family_count"] == 7 for fold in cv["folds"]),
        "deterministic_core_hash_rerun": bool(reproduction.get("verified")),
    }
    mechanical["all_pass"] = all(mechanical.values())

    adequacy_contract = protocol["adequacy_gates"]
    loss_clusters = {
        (str(families[index]), int(seeds[index])) for index in np.flatnonzero(keep_loss)
    }
    loss_families = {str(families[index]) for index in np.flatnonzero(keep_loss)}
    oracle_rescue_families = {str(families[index]) for index in np.flatnonzero(oracle_rescue)}
    oracle_rescue_fraction = (
        float(np.sum(oracle_rescue)) / keep_loss_count if keep_loss_count else 0.0
    )
    adequacy = {
        "observed": {
            "keep_loss_rows": keep_loss_count,
            "keep_loss_family_seed_clusters": len(loss_clusters),
            "keep_loss_families": len(loss_families),
            "oracle_rescued_loss_rows": int(np.sum(oracle_rescue)),
            "oracle_rescue_fraction": oracle_rescue_fraction,
            "oracle_rescue_families": len(oracle_rescue_families),
        },
        "checks": {
            "minimum_keep_loss_rows": (
                keep_loss_count >= adequacy_contract["minimum_keep_loss_rows"]
            ),
            "minimum_keep_loss_family_seed_clusters": (
                len(loss_clusters) >= adequacy_contract["minimum_keep_loss_family_seed_clusters"]
            ),
            "minimum_keep_loss_families": (
                len(loss_families) >= adequacy_contract["minimum_keep_loss_families"]
            ),
            "minimum_oracle_rescue_fraction": (
                oracle_rescue_fraction >= adequacy_contract["minimum_oracle_rescue_fraction"]
            ),
            "minimum_oracle_rescue_families": (
                len(oracle_rescue_families) >= adequacy_contract["minimum_oracle_rescue_families"]
            ),
        },
    }
    adequacy["all_pass"] = all(adequacy["checks"].values())

    family_rows = _family_metrics(margins, choices, families, protocol_info["family_ids"])
    family_no_regression = all(
        row["wins_not_worse"] and row["worst_not_worse"] and row["mean_not_worse"]
        for row in family_rows
    )
    selector_contract = protocol["selector_go_gates"]
    interventions = int(np.sum(intervention))
    selected_wins = int(np.sum(selected > 0.0))
    selected_rescue_count = int(np.sum(selected_rescue))
    selected_rescue_fraction = selected_rescue_count / keep_loss_count if keep_loss_count else 0.0
    rescued_clusters = {
        (str(families[index]), int(seeds[index])) for index in np.flatnonzero(selected_rescue)
    }
    selector = {
        "observed": {
            "interventions": interventions,
            "selected_wins": selected_wins,
            "selected_loss_rescues": selected_rescue_count,
            "keep_loss_rescue_fraction": selected_rescue_fraction,
            "rescued_family_seed_clusters": len(rescued_clusters),
            "minimum_family_wins": min(row["selected_wins"] for row in family_rows),
            "intervention_actual_paired_deltas": selected_delta[intervention].tolist(),
            "intervention_selected_margins": selected[intervention].tolist(),
        },
        "checks": {
            "minimum_interventions": (interventions >= selector_contract["minimum_interventions"]),
            "maximum_interventions": (interventions <= selector_contract["maximum_interventions"]),
            "every_intervention_actual_paired_delta_strictly_positive": (
                bool(np.all(selected_delta[intervention] > 0.0)) if interventions else False
            ),
            "every_intervention_selected_margin_strictly_positive": (
                bool(np.all(selected[intervention] > 0.0)) if interventions else False
            ),
            "minimum_keep_loss_rescue_fraction": (
                selected_rescue_fraction >= selector_contract["minimum_keep_loss_rescue_fraction"]
            ),
            "minimum_rescued_family_seed_clusters": (
                len(rescued_clusters) >= selector_contract["minimum_rescued_family_seed_clusters"]
            ),
            "minimum_selected_wins": (selected_wins >= selector_contract["minimum_selected_wins"]),
            "minimum_wins_per_family": all(
                row["selected_wins"] >= selector_contract["minimum_wins_per_family"]
                for row in family_rows
            ),
            "family_wins_worst_and_mean_never_worse_than_keep": (family_no_regression),
        },
    }
    selector["all_pass"] = all(selector["checks"].values())

    promotion_contract = protocol["promotion_grade_extra_gates"]
    promotion = {
        "checks": {
            "wins": selected_wins == promotion_contract["wins"],
            "positive_worst_margin_in_every_family": all(
                row["selected_worst_margin"] > 0.0 for row in family_rows
            ),
        }
    }
    promotion["all_pass"] = all(promotion["checks"].values())
    return {
        "mechanical": mechanical,
        "adequacy": adequacy,
        "selector": selector,
        "per_family": family_rows,
        "family_no_regression": family_no_regression,
        "promotion_grade_extra": promotion,
        "development_go_pass": (
            mechanical["all_pass"] and adequacy["all_pass"] and selector["all_pass"]
        ),
        "promotion_grade_pass": (
            mechanical["all_pass"]
            and adequacy["all_pass"]
            and selector["all_pass"]
            and promotion["all_pass"]
        ),
    }


def _evaluation_core_payload(
    cv_public: dict[str, Any], gates_without_reproduction: dict[str, Any], protocol_sha: str
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha,
        "model": {
            "alpha": 1000.0,
            "beta": 2.0,
            "switch_threshold": 2500.0,
            "actions": list(ACTION_NAMES),
        },
        "cv": cv_public,
        "gates": gates_without_reproduction,
    }


def _reproduction_reference(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str]:
    dataset_reference = args.reference_dataset_core_sha256
    evaluation_reference = args.reference_evaluation_core_sha256
    method = "explicit CLI deterministic core references"
    if args.render_existing:
        if not args.results.is_file():
            raise AssertionError("--render-existing requires an existing results file")
        existing = read_json(args.results)
        determinism = existing.get("determinism")
        if not isinstance(determinism, dict):
            raise AssertionError("existing results lack determinism references")
        existing_dataset = determinism.get("dataset_core_sha256")
        existing_evaluation = determinism.get("evaluation_core_sha256")
        if dataset_reference is not None and dataset_reference != existing_dataset:
            raise AssertionError("explicit dataset reference differs from existing result")
        if evaluation_reference is not None and evaluation_reference != existing_evaluation:
            raise AssertionError("explicit evaluation reference differs from existing result")
        dataset_reference = existing_dataset
        evaluation_reference = existing_evaluation
        method = "independent CLI render-existing deterministic core rerun"
    return dataset_reference, evaluation_reference, method


def build_results(
    dataset: dict[str, Any],
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    extracted = validate_and_extract_dataset(dataset, protocol, protocol_info)
    cv = run_cluster_cv(extracted, protocol, protocol_info)
    aggregate = _score_choices(cv["margins"], cv["choices"])
    action_counts = {
        action: int(np.sum(cv["choices"] == index)) for index, action in enumerate(ACTION_NAMES)
    }
    cv_public = {
        "outer_protocol": (
            "64 family-seed folds; exclude full held family and every row "
            "sharing held seed; predict both held seats"
        ),
        "fold_count": len(cv["folds"]),
        "folds": cv["folds"],
        "predictions": cv["predictions"],
        "aggregate": aggregate,
        "action_counts": action_counts,
    }

    unchecked_reproduction = {"verified": False}
    gates_for_core = compute_gates(cv, extracted, protocol, protocol_info, unchecked_reproduction)
    gates_for_core["mechanical"].pop("deterministic_core_hash_rerun")
    gates_for_core["mechanical"]["all_pass"] = all(gates_for_core["mechanical"].values())
    gates_for_core.pop("development_go_pass")
    gates_for_core.pop("promotion_grade_pass")
    evaluation_core = value_sha256(
        _evaluation_core_payload(cv_public, gates_for_core, protocol_info["sha256"])
    )

    dataset_reference, evaluation_reference, reproduction_method = _reproduction_reference(args)
    checked = dataset_reference is not None and evaluation_reference is not None
    reproduction = {
        "checked": checked,
        "method": reproduction_method,
        "reference_dataset_core_sha256": dataset_reference,
        "observed_dataset_core_sha256": extracted["dataset_core_sha256"],
        "dataset_core_match": (
            dataset_reference == extracted["dataset_core_sha256"] if checked else False
        ),
        "reference_evaluation_core_sha256": evaluation_reference,
        "observed_evaluation_core_sha256": evaluation_core,
        "evaluation_core_match": (evaluation_reference == evaluation_core if checked else False),
    }
    reproduction["verified"] = bool(
        checked and reproduction["dataset_core_match"] and reproduction["evaluation_core_match"]
    )
    gates = compute_gates(cv, extracted, protocol, protocol_info, reproduction)

    mechanical_without_reproduction = all(
        value
        for key, value in gates["mechanical"].items()
        if key not in {"all_pass", "deterministic_core_hash_rerun"}
    )
    if not mechanical_without_reproduction:
        status = "invalid_stop_no_checkpoint"
        stop_reason = "A locked mechanical gate failed; stop without tuning or promotion."
    elif not reproduction["checked"]:
        status = "awaiting_deterministic_rerun"
        stop_reason = (
            "Run --render-existing in a fresh process to satisfy the frozen "
            "deterministic-core gate. No downstream artifact is authorized."
        )
    elif not gates["mechanical"]["all_pass"]:
        status = "invalid_stop_no_checkpoint"
        stop_reason = "A locked mechanical gate failed; stop without tuning or promotion."
    elif not gates["adequacy"]["all_pass"]:
        status = "fail_stop_inadequate_no_checkpoint"
        stop_reason = "The frozen dataset adequacy gates failed; stop the ridge branch."
    elif not gates["selector"]["all_pass"]:
        status = "fail_stop_selector_no_checkpoint"
        stop_reason = "The frozen selector gates failed; stop the ridge branch."
    elif gates["promotion_grade_extra"]["all_pass"]:
        status = "pass_development_only_promotion_grade"
        stop_reason = (
            "V86-A passes its development gates and promotion-grade extras, but "
            "authorizes only a separately pre-registered blind validation."
        )
    else:
        status = "pass_development_only_not_promotion_grade"
        stop_reason = (
            "V86-A passes its development gates but not promotion-grade extras; "
            "it authorizes only a separately pre-registered blind validation."
        )

    return {
        "schema_version": "v86-family-scale-evaluation-v1",
        "status": status,
        "stop_reason": stop_reason,
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": protocol_info["sha256"],
            "schema_version": protocol["schema_version"],
            "status": protocol["status"],
        },
        "model": protocol["model"],
        "cv": cv_public,
        "gates": gates,
        "determinism": {
            "dataset_core_sha256": extracted["dataset_core_sha256"],
            "evaluation_core_sha256": evaluation_core,
            "reproduction": reproduction,
        },
        "sources": {
            "evaluator": {
                "path": str(RUNNER_PATH.relative_to(ROOT)),
                "sha256": sha256(RUNNER_PATH),
            },
            "dataset": {
                "path": str(args.dataset.resolve()),
                "sha256": sha256(args.dataset),
            },
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": protocol_info["sha256"],
            },
            "v85_math_reference": {
                "path": str(V85_RUNNER_PATH.relative_to(ROOT)),
                "sha256": sha256(V85_RUNNER_PATH),
            },
        },
        "guardrails": {
            "old_v84_outcomes_used_for_fit": False,
            "hyperparameter_search_performed": False,
            "sealed_holdout_opened": False,
            "checkpoint_created": False,
            "runtime_agent_created": False,
            "submission_created": False,
            "kaggle_upload_performed": False,
        },
    }


def build_report(result: dict[str, Any], results_sha256: str) -> str:
    gates = result["gates"]
    aggregate = result["cv"]["aggregate"]
    adequacy = gates["adequacy"]
    selector = gates["selector"]
    lines = [
        "# V86-A family-scale route-bandit evaluation",
        "",
        f"Status: **{result['status']}**",
        "",
        result["stop_reason"],
        "",
        "## Frozen evaluation",
        "",
        "- Fixed dual ridge: `alpha=1000`, `beta=2`, strict threshold `2500`.",
        "- Actions: `KEEP_V67`, `FORCE_B120_SUFFIX`, `FORCE_C120_SUFFIX`.",
        "- 64 `(family, seed)` folds; each fit excludes the full held family and "
        "every row with the held seed, then predicts both seats.",
        "- Each fit uses 98 rows, seven-family delete-one-group jackknife, and "
        "family-grouped OOF RMSE computed only from training rows.",
        "- No hyperparameter search and no V84 economic outcomes enter the fit.",
        "",
        "## Outcome",
        "",
        f"- Selected wins: `{aggregate['wins']}/128`.",
        f"- Interventions: `{aggregate['interventions']}`.",
        f"- Worst selected margin: `{aggregate['worst_margin']:.6f}`.",
        f"- Mean selected margin: `{aggregate['mean_margin']:.6f}`.",
        f"- KEEP wins: `{aggregate['keep_wins']}/128`.",
        f"- KEEP worst margin: `{aggregate['keep_worst_margin']:.6f}`.",
        "",
        "## Locked gates",
        "",
        "| Gate group | Pass |",
        "|---|---:|",
        f"| Mechanical | `{gates['mechanical']['all_pass']}` |",
        f"| Dataset adequacy | `{adequacy['all_pass']}` |",
        f"| Selector | `{selector['all_pass']}` |",
        f"| Family no-regression | `{gates['family_no_regression']}` |",
        f"| Promotion-grade extras | `{gates['promotion_grade_extra']['all_pass']}` |",
        "",
        "Adequacy observations: "
        f"`{adequacy['observed']['keep_loss_rows']}` KEEP-loss rows, "
        f"`{adequacy['observed']['keep_loss_family_seed_clusters']}` loss clusters, "
        f"oracle rescue fraction `{adequacy['observed']['oracle_rescue_fraction']:.6f}`.",
        "",
        "Selector observations: "
        f"rescue fraction `{selector['observed']['keep_loss_rescue_fraction']:.6f}`, "
        f"`{selector['observed']['rescued_family_seed_clusters']}` rescued clusters.",
        "",
        "## Per-family no-regression",
        "",
        "| Family | Wins selected / KEEP | Worst selected / KEEP | "
        "Mean selected / KEEP | No regression |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in gates["per_family"]:
        no_regression = row["wins_not_worse"] and row["worst_not_worse"] and row["mean_not_worse"]
        lines.append(
            f"| `{row['family']}` | {row['selected_wins']} / {row['keep_wins']} | "
            f"{row['selected_worst_margin']:.3f} / {row['keep_worst_margin']:.3f} | "
            f"{row['selected_mean_margin']:.3f} / {row['keep_mean_margin']:.3f} | "
            f"`{no_regression}` |"
        )
    lines.extend(
        [
            "",
            "## Determinism and provenance",
            "",
            f"- Protocol SHA-256: `{result['protocol']['sha256']}`.",
            f"- Dataset core SHA-256: `{result['determinism']['dataset_core_sha256']}`.",
            f"- Evaluation core SHA-256: `{result['determinism']['evaluation_core_sha256']}`.",
            f"- Independent rerun verified: `{result['determinism']['reproduction']['verified']}`.",
            f"- Results file SHA-256: `{results_sha256}`.",
            "",
            "## Guardrail",
            "",
            "This development-only evaluator cannot create a checkpoint, runtime "
            "agent, submission, sealed-holdout result, or Kaggle upload. Even a "
            "pass authorizes only a separately pre-registered blind validation.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    if sha256(V85_RUNNER_PATH) != EXPECTED_V85_RUNNER_SHA256:
        raise AssertionError("audited V85 math reference hash changed")
    spec = importlib.util.spec_from_file_location("v85_math_reference", V85_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load V85 math reference")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    rng = np.random.default_rng(86027)
    train_features = rng.normal(size=(49, 11))
    train_features[:, 0] = 7.0
    train_margins = rng.normal(loc=100.0, scale=4000.0, size=(49, 3))
    train_deltas = train_margins[:, 1:] - train_margins[:, :1]
    train_groups = np.repeat(np.asarray(list("ABCDEFG"), dtype=object), 7)
    eval_features = rng.normal(size=(6, 11))
    eval_features[:, 0] = 7.0

    observed = fit_predict_heads(
        train_features,
        train_margins,
        train_deltas,
        train_groups,
        eval_features,
        alpha=1000.0,
    )
    expected = reference.fit_predict_heads(
        train_features,
        train_margins,
        train_deltas,
        train_groups,
        eval_features,
        alpha=1000.0,
    )
    if set(observed) != set(expected):
        raise AssertionError("V85 head output keys differ")
    for key in observed:
        if not np.array_equal(observed[key], expected[key]):
            raise AssertionError(f"V85 math parity failed for {key}")
    observed_choices, observed_diagnostics = select_actions(observed, beta=2.0, threshold=2500.0)
    expected_choices, expected_diagnostics = reference.select_actions(
        expected, beta=2.0, threshold=2500.0
    )
    if not np.array_equal(observed_choices, expected_choices):
        raise AssertionError("V85 selection parity failed")
    numeric_fields = (
        "absolute_mean",
        "absolute_sigma",
        "absolute_lcb",
        "delta_mean",
        "delta_sigma",
        "delta_lcb",
    )
    for row_index in range(len(observed_diagnostics)):
        for field in numeric_fields:
            if observed_diagnostics[row_index][field] != expected_diagnostics[row_index][field]:
                raise AssertionError(f"V85 diagnostic parity failed for {field}")
    print("V86 evaluator self-test: PASS (exact audited V85 numerical parity)")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    protocol = read_json(args.protocol)
    protocol_info = validate_protocol(protocol, args.protocol)
    dataset = read_json(args.dataset)
    result = build_results(dataset, protocol, protocol_info, args)
    write_json(args.results, result)
    results_hash = sha256(args.results)
    args.report.write_text(build_report(result, results_hash), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_core_sha256": result["determinism"]["dataset_core_sha256"],
                "evaluation_core_sha256": result["determinism"]["evaluation_core_sha256"],
                "results_sha256": results_hash,
                "report_sha256": sha256(args.report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
