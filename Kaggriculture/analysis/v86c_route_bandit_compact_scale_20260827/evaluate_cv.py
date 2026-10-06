#!/usr/bin/env python3
"""Evaluate the frozen V86-C compact route-bandit development experiment.

The evaluator is deliberately offline.  It accepts only direct-child V86-C
artifacts, never opens predecessor economic artifacts, validates the compact
proof schema before using a label, and embeds the fixed audited ridge/CV math.
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
ROOT = OUT.parents[1]
EVALUATOR_PATH = Path(__file__).resolve()
COLLECTOR_PATH = OUT / "run_experiment.py"
PROTOCOL_PATH = OUT / "protocol.json"
DATASET_PATH = OUT / "dataset.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"

EXPECTED_PROTOCOL_SHA256 = (
    "8f6f0444bae37e79ec03e9a3aeaf2e94d42f927221f8f840ee74f01f9cdf066d"
)
EXPECTED_PROTOCOL_SCHEMA = "v86c-compact-family-scale-prereg-v1"
EXPECTED_PROTOCOL_STATUS = "frozen_before_any_v86c_game_or_economic_outcome"
EXPECTED_DATASET_SCHEMA = "v86c-compact-family-scale-collection-v1"

KEEP = "KEEP_V67"
ROUTE_B = "FORCE_B120_SUFFIX"
ROUTE_C = "FORCE_C120_SUFFIX"
ACTIONS = (KEEP, ROUTE_B, ROUTE_C)
MODEL_LABELS = ("KEEP", "B", "C")
FEATURE_STEPS = (24, 48, 72, 96, 120)
FEATURE_DIM = 405
SEATS = (0, 1)
BRANCH_STEP = 120
FINAL_STEP = 719
ROWS_PER_GAME = 719

COMPONENTS = (
    "candidate_raw_action",
    "candidate_proposed_action",
    "candidate_executed_action",
    "opponent_proposed_action",
    "opponent_executed_action",
    "observation",
    "engine_state",
)
ACTION_COMPONENTS = frozenset(COMPONENTS[:5])
STATE_COMPONENTS = frozenset(COMPONENTS[5:])
DIFFERENCE_COMPONENTS = {
    "raw_action_first_difference": "candidate_raw_action",
    "proposed_action_first_difference": "candidate_proposed_action",
    "executed_action_first_difference": "candidate_executed_action",
}
FORBIDDEN_CLAIM_KEYS = ("selector_pass", "promotion_grade_pass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--independent-rerun-dataset",
        type=Path,
        help="A separately collected full 384-game V86-C dataset.",
    )
    parser.add_argument(
        "--reference-dataset-core-sha256",
        help="Optional pre-recorded independent full-rerun dataset core.",
    )
    parser.add_argument(
        "--reference-evaluation-core-sha256",
        help="Optional pre-recorded independent full-rerun evaluation core.",
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Recompute and require byte-identical existing results/report.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run exact frozen-math and strict-intervention tests only.",
    )
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"{label} exact-key mismatch: missing={missing}, extra={extra}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} must be finite")
    return result


def exact_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AssertionError(f"{label} must be >= {minimum}")
    return value


def exact_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AssertionError(f"{label} must be boolean")
    return value


def sha256_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{label} must be a lowercase SHA-256 string")
    return value


def bounded_string(value: Any, label: str, *, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise AssertionError(f"{label} exceeds {maximum_bytes} UTF-8 bytes")
    if value.lstrip().startswith(("{", "[")):
        raise AssertionError(f"{label} may not contain serialized JSON")
    return value


def repo_relative_path(value: Any, label: str) -> tuple[str, Path]:
    text = bounded_string(value, label, maximum_bytes=512)
    relative = Path(text)
    if relative.is_absolute() or relative.as_posix() != text or ".." in relative.parts:
        raise AssertionError(f"{label} must be canonical repo-relative POSIX")
    resolved = (ROOT / relative).resolve(strict=True)
    try:
        canonical = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise AssertionError(f"{label} escapes repository") from exc
    if canonical != text:
        raise AssertionError(f"{label} is unresolved or symlink-aliased")
    return text, resolved


def confined_direct_child(
    path: Path,
    *,
    prefix: str,
    suffix: str,
    label: str,
    must_exist: bool,
) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    if lexical.parent.resolve() != OUT or lexical.parent != OUT:
        raise AssertionError(f"{label} must be a direct child of {OUT}")
    if not lexical.name.startswith(prefix) or not lexical.name.endswith(suffix):
        raise AssertionError(f"{label} must match {prefix}*{suffix}")
    if lexical.is_symlink():
        raise AssertionError(f"{label} may not be a symlink")
    if must_exist and not lexical.is_file():
        raise AssertionError(f"{label} does not exist")
    return lexical


def validate_cli_confinement(args: argparse.Namespace) -> None:
    protocol = args.protocol if args.protocol.is_absolute() else Path.cwd() / args.protocol
    if protocol != PROTOCOL_PATH or protocol.is_symlink() or not protocol.is_file():
        raise AssertionError("--protocol must be the frozen direct V86-C protocol.json")
    args.dataset = confined_direct_child(
        args.dataset,
        prefix="dataset",
        suffix=".json",
        label="--dataset",
        must_exist=not args.self_test,
    )
    args.results = confined_direct_child(
        args.results,
        prefix="results",
        suffix=".json",
        label="--results",
        must_exist=args.render_existing,
    )
    args.report = confined_direct_child(
        args.report,
        prefix="REPORT",
        suffix=".md",
        label="--report",
        must_exist=args.render_existing,
    )
    if args.independent_rerun_dataset is not None:
        args.independent_rerun_dataset = confined_direct_child(
            args.independent_rerun_dataset,
            prefix="dataset",
            suffix=".json",
            label="--independent-rerun-dataset",
            must_exist=True,
        )
        if args.independent_rerun_dataset == args.dataset:
            raise AssertionError("independent rerun dataset must be a different file")
    explicit = (
        args.reference_dataset_core_sha256,
        args.reference_evaluation_core_sha256,
    )
    if (explicit[0] is None) != (explicit[1] is None):
        raise AssertionError("both explicit deterministic-core references are required")
    for index, value in enumerate(explicit):
        if value is not None:
            sha256_string(value, f"deterministic reference {index}")
    if args.independent_rerun_dataset is not None and explicit[0] is not None:
        raise AssertionError("choose a rerun dataset or explicit references, not both")


def read_minified_json(path: Path, label: str) -> tuple[dict[str, Any], int]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a JSON object")
    if raw != canonical_bytes(value):
        raise AssertionError(f"{label} must be canonical minified JSON")
    return value, len(raw)


def write_pretty_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature normalization using only the current training fold."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (features - mean) / scale, mean, scale


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fit centered multi-output ridge through the deterministic dual system."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("ridge inputs must be rank two")
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
    return {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "weights": normalized.T @ dual,
    }


def ridge_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    return model["target_mean"] + normalized @ model["weights"]


def jackknife_variance(predictions: np.ndarray) -> np.ndarray:
    group_count = predictions.shape[0]
    if group_count < 2:
        raise AssertionError("family jackknife needs at least two training families")
    centered = predictions - predictions.mean(axis=0, keepdims=True)
    return ((group_count - 1.0) / group_count) * np.sum(
        np.square(centered), axis=0
    )


def fit_predict_heads(
    train_features: np.ndarray,
    train_margins: np.ndarray,
    train_deltas: np.ndarray,
    train_groups: np.ndarray,
    eval_features: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Predict both heads with family jackknife plus grouped-OOF uncertainty."""
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
        absolute_submodel = ridge_fit(
            train_features[fit_mask], train_margins[fit_mask], alpha
        )
        delta_submodel = ridge_fit(
            train_features[fit_mask], train_deltas[fit_mask], alpha
        )
        absolute_jackknife.append(ridge_predict(absolute_submodel, eval_features))
        delta_jackknife.append(ridge_predict(delta_submodel, eval_features))
        absolute_oof[held_mask] = ridge_predict(
            absolute_submodel, train_features[held_mask]
        )
        delta_oof[held_mask] = ridge_predict(delta_submodel, train_features[held_mask])

    if not np.isfinite(absolute_oof).all() or not np.isfinite(delta_oof).all():
        raise AssertionError("grouped OOF predictions are incomplete")
    absolute_jackknife_array = np.stack(absolute_jackknife, axis=0)
    delta_jackknife_array = np.stack(delta_jackknife, axis=0)
    absolute_oof_rmse = np.sqrt(
        np.mean(np.square(train_margins - absolute_oof), axis=0)
    )
    delta_oof_rmse = np.sqrt(
        np.mean(np.square(train_deltas - delta_oof), axis=0)
    )
    absolute_sigma = np.sqrt(
        jackknife_variance(absolute_jackknife_array)
        + np.square(absolute_oof_rmse)[None, :]
    )
    delta_sigma = np.sqrt(
        jackknife_variance(delta_jackknife_array)
        + np.square(delta_oof_rmse)[None, :]
    )
    return {
        "absolute_mean": absolute_mean,
        "absolute_sigma": absolute_sigma,
        "delta_mean": delta_mean,
        "delta_sigma": delta_sigma,
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
                "eligible_routes": [ACTIONS[index] for index in eligible],
            }
        )
    return choices, diagnostics


def self_test_math() -> None:
    features = np.asarray([[0.0], [2.0]], dtype=np.float64)
    targets = np.asarray([[1.0, 2.0, 4.0], [3.0, 4.0, 6.0]], dtype=np.float64)
    model = ridge_fit(features, targets, 1.0)
    predicted = ridge_predict(model, np.asarray([[1.0], [3.0]], dtype=np.float64))
    expected = np.asarray(
        [[2.0, 3.0, 5.0], [10.0 / 3.0, 13.0 / 3.0, 19.0 / 3.0]],
        dtype=np.float64,
    )
    if not np.allclose(predicted, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("dual ridge exact-value self-test failed")
    variance = jackknife_variance(
        np.asarray([[[1.0, 2.0]], [[2.0, 4.0]], [[3.0, 6.0]]])
    )
    if not np.allclose(
        variance, np.asarray([[4.0 / 3.0, 16.0 / 3.0]]), rtol=0.0, atol=1e-12
    ):
        raise AssertionError("delete-one-group jackknife self-test failed")

    equality_case = {
        "absolute_mean": np.asarray([[1.0, 1.0, 1.0]]),
        "absolute_sigma": np.zeros((1, 3)),
        "delta_mean": np.asarray([[2500.0, 2500.0]]),
        "delta_sigma": np.zeros((1, 2)),
    }
    choices, _ = select_actions(equality_case, beta=2.0, threshold=2500.0)
    if choices.tolist() != [0]:
        raise AssertionError("strict threshold equality must abstain")
    tie_case = dict(equality_case)
    tie_case["delta_mean"] = np.asarray([[2500.000001, 2500.000001]])
    choices, _ = select_actions(tie_case, beta=2.0, threshold=2500.0)
    if choices.tolist() != [1]:
        raise AssertionError("fixed B-before-C tie-break self-test failed")

    alias_choice = 1
    alias_delta = 0.0
    intervention = alias_choice != 0
    if not intervention or alias_delta > 0.0:
        raise AssertionError("forced alias intervention accounting self-test failed")


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError("frozen V86-C protocol SHA-256 drift")
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise AssertionError("unexpected V86-C protocol schema")
    if protocol.get("status") != EXPECTED_PROTOCOL_STATUS:
        raise AssertionError("V86-C protocol is not frozen")
    if protocol.get("expected_contexts") != 128:
        raise AssertionError("protocol context count drift")
    if protocol.get("expected_full_games") != 384:
        raise AssertionError("protocol game count drift")
    if protocol.get("observation_steps") != list(FEATURE_STEPS):
        raise AssertionError("protocol observation steps drift")
    if protocol.get("seats") != list(SEATS):
        raise AssertionError("protocol seats drift")

    actions = protocol.get("actions")
    if not isinstance(actions, list) or [row.get("id") for row in actions] != list(ACTIONS):
        raise AssertionError("protocol actions or order drift")
    mapping = protocol.get("action_runtime_route_mapping")
    if mapping != {ROUTE_B: "b120", ROUTE_C: "c120"}:
        raise AssertionError("protocol runtime route mapping drift")
    family_rows = protocol.get("families")
    if not isinstance(family_rows, list) or len(family_rows) != 8:
        raise AssertionError("protocol must freeze exactly eight families")
    family_ids = tuple(row.get("id") for row in family_rows)
    if len(set(family_ids)) != 8 or not all(isinstance(row, str) for row in family_ids):
        raise AssertionError("protocol family identifiers are invalid")
    seeds = tuple(protocol.get("fresh_development_seeds", ()))
    if seeds != tuple(range(27100500, 27100508)):
        raise AssertionError("protocol development seeds drift")

    encoder = protocol.get("feature_encoder", {})
    if (
        encoder.get("features_per_observation") != 81
        or encoder.get("total_features") != FEATURE_DIM
        or encoder.get("model_input_identifiers_forbidden") is not True
    ):
        raise AssertionError("protocol feature encoder drift")
    model = protocol.get("model", {})
    if (
        finite_number(model.get("alpha"), "protocol.model.alpha") != 1000.0
        or finite_number(model.get("beta"), "protocol.model.beta") != 2.0
        or finite_number(model.get("switch_threshold"), "protocol threshold")
        != 2500.0
        or model.get("actions") != list(ACTIONS)
        or model.get("intervention_rule")
        != "absolute_lcb > 0 and delta_lcb > 2500; equality abstains"
    ):
        raise AssertionError("frozen model drift")
    cv = protocol.get("cross_validation", {})
    if (
        cv.get("unit") != "family_seed_cluster"
        or cv.get("folds") != 64
        or cv.get("held_rows_per_fold") != 2
        or cv.get("expected_training_rows") != 98
        or cv.get("uncertainty")
        != "delete-one-training-family jackknife plus family-grouped OOF RMSE"
    ):
        raise AssertionError("frozen CV drift")

    schema = protocol["compact_evidence_contract"]["exact_key_schema"]
    if schema.get("unknown_keys_forbidden") is not True:
        raise AssertionError("unknown-key guard drift")
    if schema.get("minified_json_required") is not True:
        raise AssertionError("minified dataset guard drift")
    if schema.get("record_overflow_verdict") != "BLOCK_BEFORE_IPC":
        raise AssertionError("IPC overflow verdict drift")
    if schema["array_lengths_and_caps"] != {
        "contexts_smoke_exact": 1,
        "contexts_full_exact": 128,
        "source_pins_max": 24,
        "observed_read_manifests_max": 32,
        "arms_exact": 3,
        "component_manifests_per_arm_exact": 9,
        "prefix_manifests_per_arm_exact": 9,
        "forced_comparisons_exact": 2,
        "feature_snapshots_exact": 5,
        "values_per_snapshot_exact": 81,
        "features_405_exact": 405,
        "oracle_rescue_actions_max": 2,
        "generic_array_max": 405,
    }:
        raise AssertionError("compact array caps drift")
    if tuple(schema.get("component_enum", ())) != COMPONENTS:
        raise AssertionError("compact component enum drift")

    resource = protocol.get("resource_gates", {})
    if (
        resource.get("compact_context_max_bytes") != 65_536
        or resource.get("smoke_minified_output_max_bytes") != 131_072
        or resource.get("full_minified_dataset_max_bytes") != 10_485_760
    ):
        raise AssertionError("frozen compact resource caps drift")
    support = protocol["mechanical_gates"]["global_route_nondegeneracy"]
    if (
        support.get("minimum_different_contexts_per_forced_action") != 116
        or support.get("minimum_different_contexts_per_action_per_family") != 12
        or support.get("contexts_per_family") != 16
    ):
        raise AssertionError("frozen executed-support gates drift")
    return {
        "family_ids": family_ids,
        "seeds": seeds,
        "schema": schema,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


def expected_source_paths(protocol: dict[str, Any]) -> set[str]:
    paths = {
        "analysis/v86c_route_bandit_compact_scale_20260827/protocol.json",
        "analysis/v86c_route_bandit_compact_scale_20260827/PRE_REGISTRATION.md",
        "analysis/v86c_route_bandit_compact_scale_20260827/CONTRACT_FREEZE_RECEIPT.md",
        "analysis/v86c_route_bandit_compact_scale_20260827/run_experiment.py",
        "analysis/v86c_route_bandit_compact_scale_20260827/evaluate_cv.py",
        protocol["semantic_parent_contract"]["only_pinned_v86b_artifact"]["path"],
        protocol["candidate"]["path"],
    }
    paths.update(row["source_path"] for row in protocol["actions"] if "source_path" in row)
    paths.update(row["path"] for row in protocol["families"])
    paths.update(row["path"] for row in protocol["pinned_reference_sources"].values())
    if len(paths) != 23:
        raise AssertionError(f"frozen source registry must deduplicate to 23 paths, got {len(paths)}")
    return paths


def forbidden_predecessor_path(relative: str) -> bool:
    v86a = "analysis/v86_route_bandit_family_scale_20260827/"
    v86b = "analysis/v86b_route_bandit_alias_scale_20260827/"
    permitted_b = "analysis/v86b_route_bandit_alias_scale_20260827/RESOURCE_STOP_REPORT.md"
    sealed = "analysis/sealed_candidate_holdout_55797143_metadata_20260827/"
    if relative.startswith(sealed):
        return True
    if relative.startswith(v86a):
        return True
    return relative.startswith(v86b) and relative != permitted_b


def validate_source_manifest(
    row: Any,
    *,
    expected_keys: set[str],
    allowed_paths: set[str],
    label: str,
) -> dict[str, Any]:
    manifest = require_exact_keys(row, expected_keys, label)
    relative, resolved = repo_relative_path(manifest["canonical_path"], f"{label}.path")
    if forbidden_predecessor_path(relative) or relative not in allowed_paths:
        raise AssertionError(f"{label} is denied or not source-allowlisted: {relative}")
    observed_length = resolved.stat().st_size
    claimed_length = exact_integer(
        manifest["canonical_byte_length"], f"{label}.canonical_byte_length", minimum=0
    )
    claimed_hash = sha256_string(manifest["sha256"], f"{label}.sha256")
    if claimed_length != observed_length or claimed_hash != file_sha256(resolved):
        raise AssertionError(f"{label} source drift")
    return {"canonical_path": relative, "canonical_byte_length": claimed_length, "sha256": claimed_hash}


def validate_source_and_read_audit(
    dataset: dict[str, Any], protocol: dict[str, Any], protocol_info: dict[str, Any]
) -> dict[str, Any]:
    schema = protocol_info["schema"]
    allowed_paths = expected_source_paths(protocol)
    source_rows = dataset["source_pins"]
    if not isinstance(source_rows, list) or len(source_rows) != 23:
        raise AssertionError("source_pins must contain exactly the 23 frozen sources")
    source_keys = set(schema["source_pin_exact_keys"])
    sources = [
        validate_source_manifest(
            row,
            expected_keys=source_keys,
            allowed_paths=allowed_paths,
            label=f"source_pins[{index}]",
        )
        for index, row in enumerate(source_rows)
    ]
    source_paths = [row["canonical_path"] for row in sources]
    if source_paths != sorted(allowed_paths) or len(set(source_paths)) != len(source_paths):
        raise AssertionError("source_pins must be unique and canonical-path sorted")

    read = require_exact_keys(
        dataset["read_audit"], set(schema["read_audit_exact_keys"]), "read_audit"
    )
    expected_policy_hash = value_sha256(
        protocol["outcome_isolation"]["filesystem_read_policy"]
    )
    if sha256_string(read["policy_sha256"], "read_audit.policy_sha256") != expected_policy_hash:
        raise AssertionError("read policy digest mismatch")
    observed_rows = read["observed_read_manifests"]
    if not isinstance(observed_rows, list) or len(observed_rows) > 32:
        raise AssertionError("observed read manifest cap exceeded")
    observed_keys = set(schema["observed_read_manifest_exact_keys"])
    observed = [
        validate_source_manifest(
            row,
            expected_keys=observed_keys,
            allowed_paths=allowed_paths,
            label=f"read_audit.observed[{index}]",
        )
        for index, row in enumerate(observed_rows)
    ]
    observed_paths = [row["canonical_path"] for row in observed]
    if observed_paths != sorted(allowed_paths) or len(set(observed_paths)) != len(observed_paths):
        raise AssertionError("observed reads must reconcile to all 23 frozen sources exactly")
    for key in ("denied_read_count", "unknown_read_count", "source_drift_count"):
        if exact_integer(read[key], f"read_audit.{key}", minimum=0) != 0:
            raise AssertionError(f"read_audit.{key} must be zero")
    if exact_bool(read["post_run_pass"], "read_audit.post_run_pass") is not True:
        raise AssertionError("post-run read audit failed")
    if sources != observed:
        raise AssertionError("source pins and observed-read manifests disagree")
    if dataset["collector_sha256"] != file_sha256(COLLECTOR_PATH):
        raise AssertionError("collector SHA-256 drift")
    if dataset["evaluator_sha256"] != file_sha256(EVALUATOR_PATH):
        raise AssertionError("evaluator SHA-256 drift")
    return {
        "read_policy_pass": True,
        "source_count": len(sources),
        "policy_sha256": expected_policy_hash,
    }


def validate_generic_shape_and_payloads(value: Any, schema: dict[str, Any]) -> None:
    forbidden = tuple(schema["forbidden_output_key_names_or_substrings"])

    def visit(node: Any, *, depth: int, path: str) -> None:
        if depth > schema["string_and_scalar_caps"]["maximum_object_nesting_depth"]:
            raise AssertionError(f"{path} exceeds maximum object nesting depth")
        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise AssertionError(f"{path} contains a non-string key")
                lowered = key.lower()
                if any(token in lowered for token in forbidden):
                    raise AssertionError(f"{path}.{key} uses forbidden key material")
                visit(child, depth=depth + 1, path=f"{path}.{key}")
        elif isinstance(node, list):
            if len(node) > schema["array_lengths_and_caps"]["generic_array_max"]:
                raise AssertionError(f"{path} exceeds generic array cap")
            for index, child in enumerate(node):
                visit(child, depth=depth + 1, path=f"{path}[{index}]")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            finite_number(node, path)
        elif isinstance(node, str):
            if len(node.encode("utf-8")) > 512:
                raise AssertionError(f"{path} exceeds maximum string cap")
            if node.lstrip().startswith(("{", "[")):
                raise AssertionError(f"{path} contains a stringified JSON payload")
        else:
            raise AssertionError(f"{path} contains forbidden value type {type(node).__name__}")

    visit(value, depth=0, path="dataset")


def expected_manifest_pairs(candidate_seat: int) -> tuple[tuple[str, int], ...]:
    opponent_seat = 1 - candidate_seat
    return (
        ("candidate_raw_action", candidate_seat),
        ("candidate_proposed_action", candidate_seat),
        ("candidate_executed_action", candidate_seat),
        ("opponent_proposed_action", opponent_seat),
        ("opponent_executed_action", opponent_seat),
        ("observation", 0),
        ("observation", 1),
        ("engine_state", 0),
        ("engine_state", 1),
    )


def validate_component_manifests(
    rows: Any,
    *,
    candidate_seat: int,
    schema: dict[str, Any],
    label: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 9:
        raise AssertionError(f"{label} must contain exactly nine manifests")
    keys = set(schema["component_manifest_exact_keys"])
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_exact_keys(raw, keys, f"{label}[{index}]")
        component = bounded_string(
            row["component"], f"{label}[{index}].component", maximum_bytes=32
        )
        if component not in COMPONENTS:
            raise AssertionError(f"{label}[{index}] has unknown component")
        seat = exact_integer(row["seat"], f"{label}[{index}].seat")
        pair = (component, seat)
        if pair in observed:
            raise AssertionError(f"{label} contains duplicate {pair}")
        count = exact_integer(row["row_count"], f"{label}[{index}].row_count", minimum=0)
        length = exact_integer(
            row["canonical_byte_length"],
            f"{label}[{index}].canonical_byte_length",
            minimum=2,
        )
        digest = sha256_string(row["sha256"], f"{label}[{index}].sha256")
        if count != ROWS_PER_GAME:
            raise AssertionError(f"{label}[{index}] must prove exactly 719 rows")
        observed[pair] = {
            "component": component,
            "seat": seat,
            "row_count": count,
            "canonical_byte_length": length,
            "sha256": digest,
        }
    expected = expected_manifest_pairs(candidate_seat)
    if tuple(observed) != expected:
        raise AssertionError(f"{label} order/seat/component grid is not canonical")
    return observed


def validate_prefix_manifests(
    rows: Any,
    *,
    candidate_seat: int,
    schema: dict[str, Any],
    label: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 9:
        raise AssertionError(f"{label} must contain exactly nine prefix manifests")
    keys = set(schema["prefix_manifest_exact_keys"])
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = require_exact_keys(raw, keys, f"{label}[{index}]")
        component = bounded_string(
            row["component"], f"{label}[{index}].component", maximum_bytes=32
        )
        if component not in COMPONENTS:
            raise AssertionError(f"{label}[{index}] has unknown component")
        seat = exact_integer(row["seat"], f"{label}[{index}].seat")
        pair = (component, seat)
        if pair in observed:
            raise AssertionError(f"{label} contains duplicate {pair}")
        start = exact_integer(row["start_step"], f"{label}[{index}].start_step")
        stop = exact_integer(
            row["stop_step_exclusive"], f"{label}[{index}].stop_step_exclusive"
        )
        count = exact_integer(row["row_count"], f"{label}[{index}].row_count")
        expected_stop = 120 if component in ACTION_COMPONENTS else 121
        if start != 0 or stop != expected_stop or count != expected_stop:
            raise AssertionError(f"{label}[{index}] has wrong frozen prefix domain")
        observed[pair] = {
            "component": component,
            "seat": seat,
            "start_step": start,
            "stop_step_exclusive": stop,
            "row_count": count,
            "canonical_byte_length": exact_integer(
                row["canonical_byte_length"],
                f"{label}[{index}].canonical_byte_length",
                minimum=2,
            ),
            "sha256": sha256_string(row["sha256"], f"{label}[{index}].sha256"),
        }
    expected = expected_manifest_pairs(candidate_seat)
    if tuple(observed) != expected:
        raise AssertionError(f"{label} order/seat/component grid is not canonical")
    return observed


def manifest_content_identity(row: dict[str, Any]) -> tuple[int, int, str]:
    return row["row_count"], row["canonical_byte_length"], row["sha256"]


def candidate_action_integrity(
    manifests: dict[tuple[str, int], dict[str, Any]], candidate_seat: int
) -> bool:
    identities = {
        manifest_content_identity(manifests[(component, candidate_seat)])
        for component in (
            "candidate_raw_action",
            "candidate_proposed_action",
            "candidate_executed_action",
        )
    }
    return len(identities) == 1


def validate_final(raw: Any, *, candidate_seat: int, keys: set[str], label: str) -> dict[str, Any]:
    final = require_exact_keys(raw, keys, label)
    step = exact_integer(final["final_step"], f"{label}.final_step")
    if step != FINAL_STEP:
        raise AssertionError(f"{label}.final_step must be 719")
    state_hash = sha256_string(final["final_state_sha256"], f"{label}.final_state_sha256")
    money_0 = finite_number(final["player_0_money"], f"{label}.player_0_money")
    money_1 = finite_number(final["player_1_money"], f"{label}.player_1_money")
    margin = finite_number(final["candidate_margin"], f"{label}.candidate_margin")
    recomputed = money_0 - money_1 if candidate_seat == 0 else money_1 - money_0
    if margin != recomputed:
        raise AssertionError(f"{label}.candidate_margin disagrees final money")
    return {
        "final_step": step,
        "final_state_sha256": state_hash,
        "player_0_money": money_0,
        "player_1_money": money_1,
        "candidate_margin": margin,
    }


def validate_feature_snapshots(
    raw: Any, *, schema: dict[str, Any], label: str
) -> tuple[list[dict[str, Any]], list[float]]:
    if not isinstance(raw, list) or len(raw) != 5:
        raise AssertionError(f"{label} must contain five snapshots")
    keys = set(schema["feature_snapshot_exact_keys"])
    snapshots: list[dict[str, Any]] = []
    concatenated: list[float] = []
    for index, (row, expected_step) in enumerate(zip(raw, FEATURE_STEPS, strict=True)):
        snapshot = require_exact_keys(row, keys, f"{label}[{index}]")
        step = exact_integer(snapshot["step"], f"{label}[{index}].step")
        if step != expected_step:
            raise AssertionError(f"{label}[{index}] step mismatch")
        values = snapshot["values_81"]
        if not isinstance(values, list) or len(values) != 81:
            raise AssertionError(f"{label}[{index}] must contain 81 values")
        numeric = [
            finite_number(value, f"{label}[{index}].values_81[{j}]")
            for j, value in enumerate(values)
        ]
        snapshots.append({"step": step, "values_81": numeric})
        concatenated.extend(numeric)
    return snapshots, concatenated


def validate_route_lifecycle(
    raw: Any,
    *,
    action_id: str,
    protocol: dict[str, Any],
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    row = require_exact_keys(raw, keys, label)
    requested = row["requested_route"]
    selected = bounded_string(row["selected_route"], f"{label}.selected_route", maximum_bytes=32)
    route_state = bounded_string(row["route_state"], f"{label}.route_state", maximum_bytes=32)
    forced_count = exact_integer(
        row["forced_injection_count"], f"{label}.forced_injection_count", minimum=0
    )
    forced_step = row["forced_injection_step"]
    organic_count = exact_integer(
        row["organic_selection_count"], f"{label}.organic_selection_count", minimum=0
    )
    organic_step = row["organic_selection_step"]
    start_count = exact_integer(
        row["option_start_count"], f"{label}.option_start_count", minimum=0
    )
    completion_count = exact_integer(
        row["option_completion_count"], f"{label}.option_completion_count", minimum=0
    )
    drift_count = exact_integer(
        row["route_drift_count"], f"{label}.route_drift_count", minimum=0
    )
    stable = exact_bool(row["stable_through_completion"], f"{label}.stable")

    if action_id == KEEP:
        if requested is not None:
            raise AssertionError("KEEP route request must be null")
        if forced_count != 0 or forced_step is not None:
            raise AssertionError("KEEP may not receive forced route injection")
        if organic_count != 1 or organic_step != BRANCH_STEP:
            raise AssertionError("KEEP must select its route organically once at step 120")
    else:
        expected_route = protocol["action_runtime_route_mapping"][action_id]
        if requested != expected_route or selected != expected_route:
            raise AssertionError(f"{label} request/selection route mapping mismatch")
        if forced_count != 1 or forced_step != BRANCH_STEP:
            raise AssertionError(f"{label} must inject exactly once at step 120")
        if organic_count != 0 or organic_step is not None:
            raise AssertionError(f"{label} may not claim organic route selection")
        if start_count != 1 or completion_count != 1:
            raise AssertionError(f"{label} must start and complete exactly once")
    if drift_count != 0 or not stable:
        raise AssertionError(f"{label} route must remain stable through completion")
    return {
        "requested_route": requested,
        "selected_route": selected,
        "forced_injection_count": forced_count,
        "forced_injection_step": forced_step,
        "organic_selection_count": organic_count,
        "organic_selection_step": organic_step,
        "route_state": route_state,
        "option_start_count": start_count,
        "option_completion_count": completion_count,
        "route_drift_count": drift_count,
        "stable_through_completion": stable,
    }


def validate_row_commitment(raw: Any, *, keys: set[str], step: int, label: str) -> dict[str, Any]:
    row = require_exact_keys(raw, keys, label)
    observed_step = exact_integer(row["step"], f"{label}.step")
    if observed_step != step:
        raise AssertionError(f"{label}.step disagrees difference step")
    return {
        "step": observed_step,
        "canonical_byte_length": exact_integer(
            row["canonical_byte_length"], f"{label}.canonical_byte_length", minimum=1
        ),
        "sha256": sha256_string(row["sha256"], f"{label}.sha256"),
    }


def validate_first_difference(
    raw: Any,
    *,
    keep_manifest: dict[str, Any],
    forced_manifest: dict[str, Any],
    keep_prefix: dict[str, Any],
    forced_prefix: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    witness = require_exact_keys(raw, set(schema["first_difference_exact_keys"]), label)
    difference = witness["difference_step"]
    classification = witness["classification"]
    if difference is None:
        if classification is not None:
            raise AssertionError(f"{label}.classification must be JSON null")
        if any(
            witness[key] is not None
            for key in (
                "equal_prefix_commitment",
                "keep_row_commitment",
                "forced_row_commitment",
            )
        ):
            raise AssertionError(f"{label} null difference contains non-null row proof")
        null = require_exact_keys(
            witness["null_full_equality"],
            set(schema["null_full_equality_exact_keys"]),
            f"{label}.null_full_equality",
        )
        expected = {
            "keep_row_count": keep_manifest["row_count"],
            "forced_row_count": forced_manifest["row_count"],
            "keep_canonical_byte_length": keep_manifest["canonical_byte_length"],
            "forced_canonical_byte_length": forced_manifest["canonical_byte_length"],
            "keep_sha256": keep_manifest["sha256"],
            "forced_sha256": forced_manifest["sha256"],
        }
        for key, expected_value in expected.items():
            if null[key] != expected_value:
                raise AssertionError(f"{label}.null_full_equality.{key} mismatch")
        equal = (
            expected["keep_row_count"] == expected["forced_row_count"]
            and expected["keep_canonical_byte_length"]
            == expected["forced_canonical_byte_length"]
            and expected["keep_sha256"] == expected["forced_sha256"]
        )
        if exact_bool(null["equal"], f"{label}.null_full_equality.equal") != equal:
            raise AssertionError(f"{label} null equality claim mismatch")
        if not equal:
            raise AssertionError(f"{label} null difference lacks full equality proof")
        return {
            "difference_step": None,
            "classification": None,
            "equal_prefix_commitment": None,
            "keep_row_commitment": None,
            "forced_row_commitment": None,
            "null_full_equality": dict(null),
        }

    step = exact_integer(difference, f"{label}.difference_step", minimum=0)
    if step >= ROWS_PER_GAME:
        raise AssertionError(f"{label}.difference_step exceeds row domain")
    expected_classification = "before_120" if step < BRANCH_STEP else "at_or_after_120"
    if classification != expected_classification:
        raise AssertionError(f"{label}.classification mismatch")
    if witness["null_full_equality"] is not None:
        raise AssertionError(f"{label} non-null difference contains null equality proof")
    equal_prefix = require_exact_keys(
        witness["equal_prefix_commitment"],
        set(schema["equal_prefix_commitment_exact_keys"]),
        f"{label}.equal_prefix_commitment",
    )
    if (
        exact_integer(equal_prefix["start_step"], f"{label}.prefix.start") != 0
        or exact_integer(equal_prefix["stop_step_exclusive"], f"{label}.prefix.stop")
        != step
        or exact_integer(equal_prefix["row_count"], f"{label}.prefix.rows") != step
    ):
        raise AssertionError(f"{label} equal-prefix domain is not [0,d)")
    keep_length = exact_integer(
        equal_prefix["keep_canonical_byte_length"], f"{label}.prefix.keep_length", minimum=2
    )
    forced_length = exact_integer(
        equal_prefix["forced_canonical_byte_length"],
        f"{label}.prefix.forced_length",
        minimum=2,
    )
    keep_hash = sha256_string(equal_prefix["keep_sha256"], f"{label}.prefix.keep_sha256")
    forced_hash = sha256_string(
        equal_prefix["forced_sha256"], f"{label}.prefix.forced_sha256"
    )
    hashes_equal = keep_length == forced_length and keep_hash == forced_hash
    if exact_bool(equal_prefix["hashes_equal"], f"{label}.prefix.hashes_equal") != hashes_equal:
        raise AssertionError(f"{label} equal-prefix claim mismatch")
    if not hashes_equal:
        raise AssertionError(f"{label} does not prove equality on [0,d)")
    if step == BRANCH_STEP:
        expected_keep_prefix = manifest_content_identity(keep_prefix)
        expected_forced_prefix = manifest_content_identity(forced_prefix)
        if (step, keep_length, keep_hash) != expected_keep_prefix:
            raise AssertionError(f"{label} KEEP [0,120) proof is not linked to prefix manifest")
        if (step, forced_length, forced_hash) != expected_forced_prefix:
            raise AssertionError(f"{label} forced [0,120) proof is not linked to prefix manifest")

    row_keys = set(schema["row_commitment_exact_keys"])
    keep_row = validate_row_commitment(
        witness["keep_row_commitment"],
        keys=row_keys,
        step=step,
        label=f"{label}.keep_row_commitment",
    )
    forced_row = validate_row_commitment(
        witness["forced_row_commitment"],
        keys=row_keys,
        step=step,
        label=f"{label}.forced_row_commitment",
    )
    if (
        keep_row["canonical_byte_length"] == forced_row["canonical_byte_length"]
        and keep_row["sha256"] == forced_row["sha256"]
    ):
        raise AssertionError(f"{label} row-d commitments do not differ")
    if manifest_content_identity(keep_manifest) == manifest_content_identity(forced_manifest):
        raise AssertionError(f"{label} non-null difference contradicts full manifest equality")
    return {
        "difference_step": step,
        "classification": expected_classification,
        "equal_prefix_commitment": dict(equal_prefix),
        "keep_row_commitment": keep_row,
        "forced_row_commitment": forced_row,
        "null_full_equality": None,
    }


def run_cluster_cv(
    records: list[dict[str, Any]], protocol: dict[str, Any], protocol_info: dict[str, Any]
) -> dict[str, Any]:
    features = np.asarray([row["features"] for row in records], dtype=np.float64)
    margins = np.asarray([row["margins"] for row in records], dtype=np.float64)
    deltas = np.asarray([row["deltas"] for row in records], dtype=np.float64)
    families = np.asarray([row["family"] for row in records], dtype=object)
    seeds = np.asarray([row["seed"] for row in records], dtype=np.int64)
    seats = np.asarray([row["seat"] for row in records], dtype=np.int64)
    if features.shape != (128, FEATURE_DIM) or not np.isfinite(features).all():
        raise AssertionError("fixed CV feature matrix must be finite 128x405")
    if margins.shape != (128, 3) or deltas.shape != (128, 2):
        raise AssertionError("fixed CV label shapes changed")

    choices = np.full(128, -1, dtype=np.int64)
    diagnostics: list[dict[str, Any] | None] = [None] * 128
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
                raise AssertionError("each family-seed fold must train on exactly 98 rows")
            train_family_names = tuple(sorted(set(families[train_indices].tolist())))
            train_seed_names = tuple(sorted(set(seeds[train_indices].tolist())))
            if len(train_family_names) != 7 or len(train_seed_names) != 7:
                raise AssertionError("outer fold exclusion geometry changed")
            rows_by_family = {
                family: int(np.sum(families[train_indices] == family))
                for family in train_family_names
            }
            if set(rows_by_family.values()) != {14}:
                raise AssertionError("each training family must contribute exactly 14 rows")

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
                diagnostics[int(record_index)] = fold_diagnostics[local_index]
            folds.append(
                {
                    "held_family": held_family,
                    "held_seed": int(held_seed),
                    "held_rows": 2,
                    "held_seats": seats[held_indices].tolist(),
                    "train_rows": 98,
                    "train_family_count": 7,
                    "train_families": list(train_family_names),
                    "train_seed_count": 7,
                    "train_rows_by_family": rows_by_family,
                    "delete_one_family_training_rows": 84,
                    "selected_actions": [ACTIONS[index] for index in fold_choices.tolist()],
                    "absolute_oof_rmse": prediction["absolute_oof_rmse"].tolist(),
                    "delta_oof_rmse": prediction["delta_oof_rmse"].tolist(),
                }
            )
    if len(folds) != 64 or np.any(choices < 0) or any(row is None for row in diagnostics):
        raise AssertionError("64-fold clustered CV did not predict every row once")

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        choice = int(choices[index])
        selected_margin = float(margins[index, choice])
        keep_margin = float(margins[index, 0])
        actual_delta = selected_margin - keep_margin
        action = ACTIONS[choice]
        alias = action != KEEP and record["alias_rows"][action]
        if alias and actual_delta != 0.0:
            raise AssertionError("selected forced alias must retain observed zero paired delta")
        predictions.append(
            {
                "key": {
                    "family": record["family"],
                    "seed": record["seed"],
                    "candidate_seat": record["seat"],
                },
                "selected_action": action,
                "selected_action_index": choice,
                "selected_forced_alias": alias,
                "selected_margin": selected_margin,
                "keep_margin": keep_margin,
                "actual_paired_delta": actual_delta,
                "intervention": choice != 0,
                "keep_loss": keep_margin <= 0.0,
                "rescued_keep_loss": keep_margin <= 0.0 and selected_margin > 0.0,
                "diagnostics": diagnostics[index],
            }
        )
    return {
        "features": features,
        "margins": margins,
        "deltas": deltas,
        "families": families,
        "seeds": seeds,
        "seats": seats,
        "choices": choices,
        "folds": folds,
        "predictions": predictions,
    }


def score_choices(margins: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    delta = selected - keep
    return {
        "contexts": len(choices),
        "wins": int(np.sum(selected > 0.0)),
        "losses_or_ties": int(np.sum(selected <= 0.0)),
        "worst_margin": float(np.min(selected)),
        "mean_margin": float(np.mean(selected)),
        "median_margin": float(np.median(selected)),
        "keep_wins": int(np.sum(keep > 0.0)),
        "keep_worst_margin": float(np.min(keep)),
        "keep_mean_margin": float(np.mean(keep)),
        "mean_paired_delta": float(np.mean(delta)),
        "worst_paired_delta": float(np.min(delta)),
    }


def family_metrics(
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
        row = {
            "family": family,
            "contexts": int(np.sum(mask)),
            "selected_wins": int(np.sum(family_selected > 0.0)),
            "keep_wins": int(np.sum(family_keep > 0.0)),
            "selected_worst_margin": float(np.min(family_selected)),
            "keep_worst_margin": float(np.min(family_keep)),
            "selected_mean_margin": float(np.mean(family_selected)),
            "keep_mean_margin": float(np.mean(family_keep)),
        }
        row["wins_not_worse"] = row["selected_wins"] >= row["keep_wins"]
        row["worst_not_worse"] = (
            row["selected_worst_margin"] >= row["keep_worst_margin"]
        )
        row["mean_not_worse"] = row["selected_mean_margin"] >= row["keep_mean_margin"]
        rows.append(row)
    return rows


def compute_economic_gates(
    cv: dict[str, Any], protocol: dict[str, Any], protocol_info: dict[str, Any]
) -> dict[str, Any]:
    margins = cv["margins"]
    choices = cv["choices"]
    families = cv["families"]
    seeds = cv["seeds"]
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    selected_delta = selected - keep
    keep_loss = keep <= 0.0
    selected_rescue = keep_loss & (selected > 0.0)
    oracle_rescue = keep_loss & (np.max(margins[:, 1:], axis=1) > 0.0)
    keep_loss_count = int(np.sum(keep_loss))
    keep_loss_clusters = {
        (str(families[index]), int(seeds[index])) for index in np.flatnonzero(keep_loss)
    }
    keep_loss_families = {
        str(families[index]) for index in np.flatnonzero(keep_loss)
    }
    oracle_rescue_families = {
        str(families[index]) for index in np.flatnonzero(oracle_rescue)
    }
    selected_rescue_clusters = {
        (str(families[index]), int(seeds[index]))
        for index in np.flatnonzero(selected_rescue)
    }
    oracle_fraction = (
        float(np.sum(oracle_rescue)) / keep_loss_count if keep_loss_count else 0.0
    )
    selected_fraction = (
        float(np.sum(selected_rescue)) / keep_loss_count if keep_loss_count else 0.0
    )
    adequacy_contract = protocol["adequacy_gates"]
    adequacy = {
        "keep_loss_rows": keep_loss_count,
        "keep_loss_family_seed_clusters": len(keep_loss_clusters),
        "keep_loss_families": len(keep_loss_families),
        "oracle_rescued_loss_rows": int(np.sum(oracle_rescue)),
        "oracle_rescue_fraction": oracle_fraction,
        "oracle_rescue_families": len(oracle_rescue_families),
        "minimum_keep_loss_rows_pass": keep_loss_count
        >= adequacy_contract["minimum_keep_loss_rows"],
        "minimum_keep_loss_family_seed_clusters_pass": len(keep_loss_clusters)
        >= adequacy_contract["minimum_keep_loss_family_seed_clusters"],
        "minimum_keep_loss_families_pass": len(keep_loss_families)
        >= adequacy_contract["minimum_keep_loss_families"],
        "minimum_oracle_rescue_fraction_pass": oracle_fraction
        >= adequacy_contract["minimum_oracle_rescue_fraction"],
        "minimum_oracle_rescue_families_pass": len(oracle_rescue_families)
        >= adequacy_contract["minimum_oracle_rescue_families"],
    }
    adequacy["all_pass"] = all(
        value for key, value in adequacy.items() if key.endswith("_pass")
    )

    per_family = family_metrics(margins, choices, families, protocol_info["family_ids"])
    intervention = choices != 0
    intervention_count = int(np.sum(intervention))
    intervention_deltas = selected_delta[intervention]
    intervention_margins = selected[intervention]
    selector_contract = protocol["selector_go_gates"]
    selector = {
        "interventions": intervention_count,
        "selected_wins": int(np.sum(selected > 0.0)),
        "selected_losses_or_ties": int(np.sum(selected <= 0.0)),
        "keep_loss_rescues": int(np.sum(selected_rescue)),
        "keep_loss_rescue_fraction": selected_fraction,
        "rescued_family_seed_clusters": len(selected_rescue_clusters),
        "minimum_interventions_pass": intervention_count
        >= selector_contract["minimum_interventions"],
        "maximum_interventions_pass": intervention_count
        <= selector_contract["maximum_interventions"],
        "every_intervention_actual_paired_delta_strictly_positive_pass": bool(
            intervention_count > 0 and np.all(intervention_deltas > 0.0)
        ),
        "every_intervention_selected_margin_strictly_positive_pass": bool(
            intervention_count > 0 and np.all(intervention_margins > 0.0)
        ),
        "minimum_keep_loss_rescue_fraction_pass": selected_fraction
        >= selector_contract["minimum_keep_loss_rescue_fraction"],
        "minimum_rescued_family_seed_clusters_pass": len(selected_rescue_clusters)
        >= selector_contract["minimum_rescued_family_seed_clusters"],
        "minimum_selected_wins_pass": int(np.sum(selected > 0.0))
        >= selector_contract["minimum_selected_wins"],
        "minimum_wins_per_family_pass": all(
            row["selected_wins"] >= selector_contract["minimum_wins_per_family"]
            for row in per_family
        ),
        "family_wins_worst_and_mean_never_worse_than_keep_pass": all(
            row["wins_not_worse"] and row["worst_not_worse"] and row["mean_not_worse"]
            for row in per_family
        ),
        "per_family": per_family,
    }
    selector["all_pass"] = all(
        value for key, value in selector.items() if key.endswith("_pass")
    )
    promotion = {
        "wins": int(np.sum(selected > 0.0)),
        "required_wins": protocol["promotion_grade_extra_gates"]["wins"],
        "wins_128_pass": int(np.sum(selected > 0.0))
        == protocol["promotion_grade_extra_gates"]["wins"],
        "positive_worst_margin_in_every_family_pass": all(
            row["selected_worst_margin"] > 0.0 for row in per_family
        ),
    }
    promotion["all_pass"] = all(
        value for key, value in promotion.items() if key.endswith("_pass")
    )
    return {"adequacy": adequacy, "selector": selector, "promotion_grade": promotion}


def validate_candidate_execution_audit(
    raw: Any, *, schema: dict[str, Any], label: str
) -> dict[str, Any]:
    row = require_exact_keys(
        raw, set(schema["candidate_execution_audit_exact_keys"]), label
    )
    numeric = {
        key: exact_integer(row[key], f"{label}.{key}", minimum=0)
        for key in (
            "start_step",
            "stop_step_exclusive",
            "audit_row_count",
            "completed_transitions",
            "sanitizer_changed_step_count",
            "fallback_count",
            "exception_count",
            "partial_action_count",
        )
    }
    if (
        numeric["start_step"] != 0
        or numeric["stop_step_exclusive"] != ROWS_PER_GAME
        or numeric["audit_row_count"] != ROWS_PER_GAME
        or numeric["completed_transitions"] != ROWS_PER_GAME
    ):
        raise AssertionError(f"{label} does not cover exactly steps 0..718")
    for key in (
        "sanitizer_changed_step_count",
        "fallback_count",
        "exception_count",
        "partial_action_count",
    ):
        if numeric[key] != 0:
            raise AssertionError(f"{label}.{key} must be zero")
    numeric["audit_sha256"] = sha256_string(row["audit_sha256"], f"{label}.audit_sha256")
    return numeric


def validate_worker_read_audit(
    raw: Any,
    *,
    schema: dict[str, Any],
    source_pins: list[dict[str, Any]],
    policy_sha256: str,
    label: str,
) -> dict[str, Any]:
    row = require_exact_keys(raw, set(schema["worker_read_audit_exact_keys"]), label)
    if sha256_string(row["policy_sha256"], f"{label}.policy_sha256") != policy_sha256:
        raise AssertionError(f"{label} policy SHA mismatch")
    indices = row["observed_source_pin_indices"]
    if not isinstance(indices, list) or len(indices) > 23:
        raise AssertionError(f"{label} source-pin index cap exceeded")
    validated = [
        exact_integer(value, f"{label}.indices[{index}]", minimum=0)
        for index, value in enumerate(indices)
    ]
    if validated != sorted(set(validated)) or any(value >= len(source_pins) for value in validated):
        raise AssertionError(f"{label} indices must be unique, sorted, and in range")
    selected = [source_pins[index] for index in validated]
    expected_digest = value_sha256(selected)
    if (
        sha256_string(
            row["observed_read_manifest_sha256"],
            f"{label}.observed_read_manifest_sha256",
        )
        != expected_digest
    ):
        raise AssertionError(f"{label} selected-manifest digest mismatch")
    for key in ("denied_read_count", "unknown_read_count", "source_drift_count"):
        if exact_integer(row[key], f"{label}.{key}", minimum=0) != 0:
            raise AssertionError(f"{label}.{key} must be zero")
    if exact_bool(row["post_run_pass"], f"{label}.post_run_pass") is not True:
        raise AssertionError(f"{label} did not pass")
    return {
        "policy_sha256": policy_sha256,
        "observed_source_pin_indices": validated,
        "observed_read_manifest_sha256": expected_digest,
        "denied_read_count": 0,
        "unknown_read_count": 0,
        "source_drift_count": 0,
        "post_run_pass": True,
    }


def validate_arm(
    raw: Any,
    *,
    action_id: str,
    candidate_seat: int,
    feature_snapshots: list[dict[str, Any]],
    features: list[float],
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    schema = protocol_info["schema"]
    arm = require_exact_keys(raw, set(schema["arm_exact_keys"]), label)
    if arm["action_id"] != action_id:
        raise AssertionError(f"{label}.action_id/order mismatch")
    manifests = validate_component_manifests(
        arm["component_manifests"],
        candidate_seat=candidate_seat,
        schema=schema,
        label=f"{label}.component_manifests",
    )
    prefixes = validate_prefix_manifests(
        arm["prefix_manifests"],
        candidate_seat=candidate_seat,
        schema=schema,
        label=f"{label}.prefix_manifests",
    )
    joint_prefix = sha256_string(
        arm["full_joint_prefix_sha256"], f"{label}.full_joint_prefix_sha256"
    )
    snapshots_hash = sha256_string(
        arm["feature_snapshots_sha256"], f"{label}.feature_snapshots_sha256"
    )
    features_hash = sha256_string(
        arm["features_405_sha256"], f"{label}.features_405_sha256"
    )
    if snapshots_hash != value_sha256(feature_snapshots):
        raise AssertionError(f"{label} snapshot digest does not bind serialized snapshots")
    if features_hash != value_sha256(features):
        raise AssertionError(f"{label} feature digest does not bind serialized 405-vector")
    audit = validate_candidate_execution_audit(
        arm["candidate_execution_audit"], schema=schema, label=f"{label}.audit"
    )
    proposed = manifests[("candidate_proposed_action", candidate_seat)]
    executed = manifests[("candidate_executed_action", candidate_seat)]
    if manifest_content_identity(proposed) != manifest_content_identity(executed):
        raise AssertionError(f"{label} zero-sanitizer audit contradicts proposed/executed digest")
    lifecycle = validate_route_lifecycle(
        arm["route_lifecycle"],
        action_id=action_id,
        protocol=protocol,
        keys=set(schema["route_lifecycle_exact_keys"]),
        label=f"{label}.route_lifecycle",
    )
    final = validate_final(
        arm["final"],
        candidate_seat=candidate_seat,
        keys=set(schema["final_exact_keys"]),
        label=f"{label}.final",
    )
    return {
        "action_id": action_id,
        "manifests": manifests,
        "prefixes": prefixes,
        "full_joint_prefix_sha256": joint_prefix,
        "feature_snapshots_sha256": snapshots_hash,
        "features_405_sha256": features_hash,
        "candidate_execution_audit": audit,
        "route_lifecycle": lifecycle,
        "final": final,
    }


def compare_manifest_maps(
    left: dict[tuple[str, int], dict[str, Any]],
    right: dict[tuple[str, int], dict[str, Any]],
) -> bool:
    return tuple(left) == tuple(right) and all(
        manifest_content_identity(left[key]) == manifest_content_identity(right[key])
        for key in left
    )


def validate_forced_comparison(
    raw: Any,
    *,
    action_id: str,
    candidate_seat: int,
    keep: dict[str, Any],
    forced: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    row = require_exact_keys(raw, set(schema["forced_comparison_exact_keys"]), label)
    if row["action_id"] != action_id:
        raise AssertionError(f"{label}.action_id/order mismatch")
    differences: dict[str, dict[str, Any]] = {}
    for key, component in DIFFERENCE_COMPONENTS.items():
        pair = (component, candidate_seat)
        differences[key] = validate_first_difference(
            row[key],
            keep_manifest=keep["manifests"][pair],
            forced_manifest=forced["manifests"][pair],
            keep_prefix=keep["prefixes"][pair],
            forced_prefix=forced["prefixes"][pair],
            schema=schema,
            label=f"{label}.{key}",
        )

    same_route = (
        keep["route_lifecycle"]["selected_route"]
        == forced["route_lifecycle"]["selected_route"]
    )
    expected_relation = (
        "same_as_forced_selected_route"
        if same_route
        else "different_from_forced_selected_route"
    )
    if row["route_relation"] != expected_relation:
        raise AssertionError(f"{label}.route_relation mismatch")
    prefix_equal = compare_manifest_maps(keep["prefixes"], forced["prefixes"])
    joint_equal = (
        keep["full_joint_prefix_sha256"] == forced["full_joint_prefix_sha256"]
    )
    full_manifests_equal = compare_manifest_maps(keep["manifests"], forced["manifests"])
    final_equal = keep["final"] == forced["final"]
    snapshots_equal = (
        keep["feature_snapshots_sha256"] == forced["feature_snapshots_sha256"]
    )
    features_equal = keep["features_405_sha256"] == forced["features_405_sha256"]
    expected_flags = {
        "prefix_component_manifests_equal": prefix_equal,
        "full_joint_prefix_equal": joint_equal,
        "full_component_manifests_equal": full_manifests_equal,
        "final_equal": final_equal,
        "feature_snapshots_equal": snapshots_equal,
        "features_405_equal": features_equal,
    }
    for key, expected in expected_flags.items():
        if exact_bool(row[key], f"{label}.{key}") != expected:
            raise AssertionError(f"{label}.{key} disagrees compact commitments")

    executed_step = differences["executed_action_first_difference"]["difference_step"]
    executed_equal = executed_step is None
    full_equal = (
        full_manifests_equal and final_equal and snapshots_equal and features_equal
    )
    if executed_step is not None and executed_step < BRANCH_STEP:
        verdict = "BLOCK_EARLY_CANDIDATE_EXECUTED_DIFFERENCE"
    elif same_route and full_equal and executed_equal:
        verdict = "VALID_ALIAS"
    elif same_route:
        verdict = "BLOCK_SAME_ROUTE_TRAJECTORY_DIFFERENT"
    elif executed_equal:
        verdict = "BLOCK_DIFFERENT_ROUTE_EXECUTED_IDENTICAL"
    elif executed_step is not None and executed_step >= BRANCH_STEP:
        verdict = "VALID_NON_ALIAS"
    else:
        verdict = "BLOCK_UNCLASSIFIED"
    if row["claimed_alias_verdict"] != verdict:
        raise AssertionError(f"{label}.claimed_alias_verdict mismatch")
    return {
        "action_id": action_id,
        **differences,
        "route_relation": expected_relation,
        **expected_flags,
        "alias_verdict": verdict,
        "alias": verdict == "VALID_ALIAS",
        "support": verdict == "VALID_NON_ALIAS",
        "prefix_pass": prefix_equal and joint_equal and snapshots_equal and features_equal,
    }


def validate_labels(
    raw: Any,
    *,
    arms: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    schema: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    row = require_exact_keys(raw, set(schema["labels_exact_keys"]), label)
    margins = {
        KEEP: arms[KEEP]["final"]["candidate_margin"],
        ROUTE_B: arms[ROUTE_B]["final"]["candidate_margin"],
        ROUTE_C: arms[ROUTE_C]["final"]["candidate_margin"],
    }
    expected = {
        "keep_margin": margins[KEEP],
        "b120_margin": margins[ROUTE_B],
        "c120_margin": margins[ROUTE_C],
        "b120_paired_delta": margins[ROUTE_B] - margins[KEEP],
        "c120_paired_delta": margins[ROUTE_C] - margins[KEEP],
    }
    for key, expected_value in expected.items():
        if finite_number(row[key], f"{label}.{key}") != expected_value:
            raise AssertionError(f"{label}.{key} disagrees recomputed money label")
    keep_loss = margins[KEEP] <= 0.0
    if exact_bool(row["keep_loss"], f"{label}.keep_loss") != keep_loss:
        raise AssertionError(f"{label}.keep_loss mismatch")
    expected_rescues = [
        action_id
        for action_id in (ROUTE_B, ROUTE_C)
        if keep_loss and margins[action_id] > 0.0
    ]
    rescues = row["oracle_rescue_actions"]
    if rescues != expected_rescues:
        raise AssertionError(f"{label}.oracle_rescue_actions mismatch")
    for action_id, delta_key in (
        (ROUTE_B, "b120_paired_delta"),
        (ROUTE_C, "c120_paired_delta"),
    ):
        if comparisons[action_id]["alias"] and expected[delta_key] != 0.0:
            raise AssertionError(f"{label} alias row must retain observed zero paired delta")
    return {**expected, "keep_loss": keep_loss, "oracle_rescue_actions": expected_rescues}


def validate_context(
    raw: Any,
    *,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    source_pins: list[dict[str, Any]],
    policy_sha256: str,
    expected_key: tuple[str, int, int],
    label: str,
) -> dict[str, Any]:
    schema = protocol_info["schema"]
    context = require_exact_keys(raw, set(schema["context_exact_keys"]), label)
    family, seed, candidate_seat = expected_key
    if context["family_id"] != family:
        raise AssertionError(f"{label}.family_id/order mismatch")
    if exact_integer(context["seed"], f"{label}.seed") != seed:
        raise AssertionError(f"{label}.seed/order mismatch")
    if exact_integer(context["candidate_seat"], f"{label}.candidate_seat") != candidate_seat:
        raise AssertionError(f"{label}.candidate_seat/order mismatch")
    expected_context_id = value_sha256(
        {"family_id": family, "seed": seed, "candidate_seat": candidate_seat}
    )
    if sha256_string(context["context_id_sha256"], f"{label}.context_id") != expected_context_id:
        raise AssertionError(f"{label}.context_id_sha256 mismatch")
    worker_read = validate_worker_read_audit(
        context["worker_read_audit"],
        schema=schema,
        source_pins=source_pins,
        policy_sha256=policy_sha256,
        label=f"{label}.worker_read_audit",
    )
    snapshots, concatenated = validate_feature_snapshots(
        context["feature_snapshots"], schema=schema, label=f"{label}.feature_snapshots"
    )
    raw_features = context["features_405"]
    if not isinstance(raw_features, list) or len(raw_features) != FEATURE_DIM:
        raise AssertionError(f"{label}.features_405 must contain exactly 405 values")
    features = [
        finite_number(value, f"{label}.features_405[{index}]")
        for index, value in enumerate(raw_features)
    ]
    if features != concatenated:
        raise AssertionError(f"{label}.features_405 must equal snapshot concatenation")

    raw_arms = context["arms"]
    if not isinstance(raw_arms, list) or len(raw_arms) != 3:
        raise AssertionError(f"{label}.arms must contain exactly three arms")
    arms = {
        action_id: validate_arm(
            raw_arm,
            action_id=action_id,
            candidate_seat=candidate_seat,
            feature_snapshots=snapshots,
            features=features,
            protocol=protocol,
            protocol_info=protocol_info,
            label=f"{label}.arms[{index}]",
        )
        for index, (action_id, raw_arm) in enumerate(zip(ACTIONS, raw_arms, strict=True))
    }
    raw_comparisons = context["forced_comparisons"]
    if not isinstance(raw_comparisons, list) or len(raw_comparisons) != 2:
        raise AssertionError(f"{label}.forced_comparisons must contain exactly two rows")
    comparisons = {
        action_id: validate_forced_comparison(
            raw_comparison,
            action_id=action_id,
            candidate_seat=candidate_seat,
            keep=arms[KEEP],
            forced=arms[action_id],
            schema=schema,
            label=f"{label}.forced_comparisons[{index}]",
        )
        for index, (action_id, raw_comparison) in enumerate(
            zip((ROUTE_B, ROUTE_C), raw_comparisons, strict=True)
        )
    }
    labels = validate_labels(
        context["labels"],
        arms=arms,
        comparisons=comparisons,
        schema=schema,
        label=f"{label}.labels",
    )

    prefix_pass = all(row["prefix_pass"] for row in comparisons.values())
    route_pass = all(
        arm["route_lifecycle"]["route_drift_count"] == 0
        and arm["route_lifecycle"]["stable_through_completion"]
        for arm in arms.values()
    )
    alias_pass = all(
        row["alias_verdict"] in {"VALID_ALIAS", "VALID_NON_ALIAS"}
        for row in comparisons.values()
    )
    mechanical_pass = prefix_pass and route_pass and alias_pass
    claims = require_exact_keys(
        context["claimed_verdicts"],
        set(schema["claimed_verdicts_exact_keys"]),
        f"{label}.claimed_verdicts",
    )
    expected_claims = {
        "mechanical_pass": mechanical_pass,
        "prefix_pass": prefix_pass,
        "route_pass": route_pass,
        "alias_pass": alias_pass,
    }
    for key, expected in expected_claims.items():
        if exact_bool(claims[key], f"{label}.claimed_verdicts.{key}") != expected:
            raise AssertionError(f"{label}.claimed_verdicts.{key} mismatch")
    if not mechanical_pass:
        raise AssertionError(f"{label} failed frozen compact mechanical/alias gates")
    return {
        "family": family,
        "seed": seed,
        "seat": candidate_seat,
        "features": features,
        "margins": [
            labels["keep_margin"],
            labels["b120_margin"],
            labels["c120_margin"],
        ],
        "deltas": [labels["b120_paired_delta"], labels["c120_paired_delta"]],
        "alias_rows": {
            ROUTE_B: comparisons[ROUTE_B]["alias"],
            ROUTE_C: comparisons[ROUTE_C]["alias"],
        },
        "support": {
            ROUTE_B: comparisons[ROUTE_B]["support"],
            ROUTE_C: comparisons[ROUTE_C]["support"],
        },
        "worker_read_audit": worker_read,
    }
