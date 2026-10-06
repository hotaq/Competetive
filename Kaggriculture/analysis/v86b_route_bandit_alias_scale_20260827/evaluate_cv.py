#!/usr/bin/env python3
"""Evaluate the frozen V86-B alias-permissive route-bandit experiment.

This file is an offline evaluator.  It never executes a game, opens the sealed
holdout, or reads a V86-A economic dataset/result.  The numerical ridge/CV
implementation is imported only from the source-hashed V86-A evaluator; all
V86-B dataset, alias, support, adequacy, selector, and promotion gates are
validated here against the frozen V86-B protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = OUT / "protocol.json"
DATASET_PATH = OUT / "dataset.json"
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"

EXPECTED_PROTOCOL_SHA256 = "511152132c5fa6f245ee8eabb10f29d60007ddd79a7c3b3d523ba7dfa9ee5731"
EXPECTED_PROTOCOL_SCHEMA = "v86b-alias-family-scale-prereg-v4"
EXPECTED_PROTOCOL_STATUS = "frozen_before_v86b_economic_outcomes"
EXPECTED_DATASET_SCHEMA = "v86b-alias-family-scale-collection-v1"
EXPECTED_PRE_REGISTRATION_SHA256 = (
    "a1e442116f1c19aacd50647b0df4d9d3ffb08cac11b8e408ba16498601076e3b"
)
EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256 = (
    "d0c1462ac6234590df6416f5eb2355e2f4585e535ebefa00ac246f8a655bafe3"
)
EXPECTED_MATH_REFERENCE_SHA256 = "4a99d63638e862572f4afb5ba1296ba5a218d4b09e85b663b2f7522ce6a5ccf4"
EXPECTED_COLLECTOR_SHA256 = "f4cb36b06c2fc59b34fa12995172d725d28c40eeed6bfcf460a2f342d7d59add"

KEEP = "KEEP_V67"
ROUTE_B = "FORCE_B120_SUFFIX"
ROUTE_C = "FORCE_C120_SUFFIX"
ACTIONS = (KEEP, ROUTE_B, ROUTE_C)
MODEL_LABELS = ("KEEP", "B", "C")
DELTA_LABELS = ("B", "C")
FEATURE_STEPS = (24, 48, 72, 96, 120)
FEATURE_DIM = 405
SEATS = (0, 1)
FINAL_STEP = 719
BRANCH_STEP = 120
COMPONENT_NAMES = (
    "candidate_raw_action_trace",
    "candidate_proposed_action_trace",
    "candidate_executed_action_trace",
    "opponent_proposed_action_trace",
    "opponent_executed_action_trace",
    "observation_trajectory_by_seat",
    "state_trajectory_by_seat",
    "canonical_final_state",
    "final_economics",
    "feature_snapshots",
)
PREFIX_COMPONENT_NAMES = (*COMPONENT_NAMES[:7], "feature_snapshots")


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
            "Recompute results/report using deterministic-core references from the "
            "existing results file."
        ),
    )
    parser.add_argument(
        "--reference-dataset-core-sha256",
        help="Require this already-known dataset deterministic-core hash.",
    )
    parser.add_argument(
        "--reference-evaluation-core-sha256",
        help="Require this already-known evaluation deterministic-core hash.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run exact source-pinned numerical parity tests without reading a dataset.",
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


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} must be finite")
    return result


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AssertionError(f"{label} must be >= {minimum}")
    return value


def hash_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{label} must be a lowercase SHA-256 string")
    return value


def resolve_repo_path(relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AssertionError(f"{label} must be a non-empty relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise AssertionError(f"{label} escapes repository")
    if not path.is_file():
        raise AssertionError(f"{label} does not exist: {relative}")
    return path


def reject_forbidden_v86a_economic_path(path: Path, label: str) -> None:
    resolved = path.resolve()
    forbidden_dir = (ROOT / "analysis/v86_route_bandit_family_scale_20260827").resolve()
    if resolved.is_relative_to(forbidden_dir):
        raise AssertionError(f"{label} resolves anywhere under the forbidden V86-A tree")


def confined_direct_child(
    path: Path,
    *,
    name_prefix: str,
    suffix: str,
    label: str,
) -> Path:
    resolved = path.resolve()
    reject_forbidden_v86a_economic_path(resolved, label)
    if resolved.parent != OUT.resolve():
        raise AssertionError(f"{label} must resolve to a direct child of {OUT}")
    if not resolved.name.startswith(name_prefix) or not resolved.name.endswith(suffix):
        raise AssertionError(
            f"{label} must be named {name_prefix}*{suffix} inside the V86-B directory"
        )
    return resolved


def validate_cli_confinement(args: argparse.Namespace) -> None:
    protocol = args.protocol.resolve()
    reject_forbidden_v86a_economic_path(protocol, "--protocol")
    if protocol != PROTOCOL_PATH.resolve():
        raise AssertionError("--protocol must be the exact frozen V86-B protocol.json")
    args.protocol = protocol
    args.dataset = confined_direct_child(
        args.dataset,
        name_prefix="dataset",
        suffix=".json",
        label="--dataset",
    )
    args.results = confined_direct_child(
        args.results,
        name_prefix="results",
        suffix=".json",
        label="--results",
    )
    args.report = confined_direct_child(
        args.report,
        name_prefix="REPORT",
        suffix=".md",
        label="--report",
    )


def verify_source_record(
    record: dict[str, Any],
    label: str,
    *,
    path_key: str = "path",
    hash_key: str = "sha256",
) -> Path:
    if not isinstance(record, dict):
        raise AssertionError(f"{label} must be an object")
    path = resolve_repo_path(record.get(path_key), f"{label}.{path_key}")
    claimed = hash_string(record.get(hash_key), f"{label}.{hash_key}")
    observed = sha256(path)
    if observed != claimed:
        raise AssertionError(f"{label} source drift: claimed {claimed}, observed {observed}")
    return path


def validate_protocol(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    reject_forbidden_v86a_economic_path(path, "protocol")
    observed_hash = sha256(path)
    if observed_hash != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError(
            "frozen protocol hash mismatch: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, observed {observed_hash}"
        )
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise AssertionError("unexpected protocol schema")
    if protocol.get("status") != EXPECTED_PROTOCOL_STATUS:
        raise AssertionError("protocol is not the frozen V86-B contract")

    required = {
        "candidate",
        "actions",
        "families",
        "fresh_development_seeds",
        "seats",
        "expected_contexts",
        "expected_full_games",
        "observation_steps",
        "feature_encoder",
        "model",
        "cross_validation",
        "mechanical_gates",
        "adequacy_gates",
        "selector_go_gates",
        "selector_accounting_semantics",
        "promotion_grade_extra_gates",
        "pinned_reference_sources",
        "v86a_exclusion_evidence",
        "stop_rule",
    }
    missing = required - set(protocol)
    if missing:
        raise AssertionError(f"protocol missing frozen fields: {sorted(missing)}")

    candidate = protocol["candidate"]
    verify_source_record(candidate, "protocol.candidate")
    actions = protocol["actions"]
    if not isinstance(actions, list):
        raise AssertionError("protocol.actions must be a list")
    action_ids = tuple(row.get("id") for row in actions if isinstance(row, dict))
    if action_ids != ACTIONS:
        raise AssertionError(f"unexpected frozen action order: {action_ids}")
    for row in actions[1:]:
        verify_source_record(
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
        verify_source_record(family, f"protocol.families.{family_id}")
    if len(set(family_ids)) != 8:
        raise AssertionError("protocol family ids must be unique")

    seeds = protocol["fresh_development_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 8:
        raise AssertionError("protocol must contain exactly eight seeds")
    checked_seeds = tuple(integer(seed, "protocol seed", minimum=0) for seed in seeds)
    if len(set(checked_seeds)) != 8:
        raise AssertionError("protocol seeds must be unique")
    if protocol["seats"] != list(SEATS):
        raise AssertionError("protocol seats changed")
    if protocol["expected_contexts"] != 128:
        raise AssertionError("protocol expected_contexts changed")
    if protocol["expected_full_games"] != 384:
        raise AssertionError("protocol expected_full_games changed")
    if protocol["observation_steps"] != list(FEATURE_STEPS):
        raise AssertionError("protocol observation steps changed")

    expected_model = {
        "type": "dual_ridge_absolute_and_delta_heads",
        "alpha": 1000.0,
        "beta": 2.0,
        "switch_threshold": 2500.0,
        "actions": list(ACTIONS),
        "intervention_rule": ("absolute_lcb > 0 and delta_lcb > 2500; equality abstains"),
        "tie_break": [
            "higher_delta_lcb",
            "higher_absolute_lcb",
            "FORCE_B120_SUFFIX_before_FORCE_C120_SUFFIX",
        ],
        "hyperparameter_search": False,
    }
    if protocol["model"] != expected_model:
        raise AssertionError("frozen model contract changed")
    expected_cv = {
        "unit": "family_seed_cluster",
        "folds": 64,
        "held_rows_per_fold": 2,
        "training_exclusion": ("exclude every row in held family and every row with held seed"),
        "expected_training_rows": 98,
        "uncertainty": ("delete-one-training-family jackknife plus family-grouped OOF RMSE"),
    }
    if protocol["cross_validation"] != expected_cv:
        raise AssertionError("frozen CV contract changed")

    mechanical = protocol["mechanical_gates"]
    if mechanical.get("finite_feature_shape") != [128, FEATURE_DIM]:
        raise AssertionError("frozen feature shape changed")
    if mechanical.get("candidate_prefix_parity_basis") != (
        "full_joint_prefix_gate_for_every_KEEP_B_C_arm"
    ):
        raise AssertionError("full-joint prefix basis missing")
    prefix_gate = mechanical.get("full_joint_prefix_gate")
    if not isinstance(prefix_gate, dict) or prefix_gate.get("arms") != list(ACTIONS):
        raise AssertionError("full-joint prefix arm contract changed")
    if prefix_gate.get("candidate_and_opponent_action_steps_inclusive") != [0, 119]:
        raise AssertionError("full-joint action prefix range changed")
    alias_rule = mechanical.get("forced_option_alias_rule")
    if not isinstance(alias_rule, dict):
        raise AssertionError("forced alias rule missing")
    if alias_rule.get("applies_to") != [ROUTE_B, ROUTE_C]:
        raise AssertionError("forced alias actions changed")
    if not alias_rule.get("alias_rows_are_retained"):
        raise AssertionError("alias retention contract missing")
    support = mechanical.get("global_route_nondegeneracy")
    if not isinstance(support, dict):
        raise AssertionError("global nondegeneracy contract missing")
    if support.get("minimum_different_contexts_per_forced_action") != 116:
        raise AssertionError("global support threshold changed")
    if support.get("minimum_different_contexts_per_action_per_family") != 12:
        raise AssertionError("family support threshold changed")

    pinned = protocol["pinned_reference_sources"]
    if not isinstance(pinned, dict):
        raise AssertionError("pinned_reference_sources must be an object")
    for name, record in pinned.items():
        verify_source_record(record, f"protocol.pinned_reference_sources.{name}")
    math_record = pinned.get("v86a_cv_reference")
    if not isinstance(math_record, dict):
        raise AssertionError("V86-A evaluator math reference missing")
    if math_record.get("sha256") != EXPECTED_MATH_REFERENCE_SHA256:
        raise AssertionError("audited V86-A evaluator hash changed in protocol")

    allowed_stop = protocol["v86a_exclusion_evidence"].get("allowed_mechanical_stop_report")
    verify_source_record(allowed_stop, "protocol.allowed_mechanical_stop_report")

    return {
        "sha256": observed_hash,
        "family_ids": tuple(family_ids),
        "family_by_id": {row["id"]: row for row in families},
        "seeds": checked_seeds,
        "math_reference_path": resolve_repo_path(
            math_record["path"], "protocol.v86a_cv_reference.path"
        ),
    }


def load_math_reference(protocol_info: dict[str, Any]) -> ModuleType:
    path: Path = protocol_info["math_reference_path"]
    expected_path = (
        ROOT / "analysis/v86_route_bandit_family_scale_20260827/evaluate_cv.py"
    ).resolve()
    if path.resolve() != expected_path:
        raise AssertionError("math import must be the exact pinned V86-A evaluator source")
    if sha256(path) != EXPECTED_MATH_REFERENCE_SHA256:
        raise AssertionError("audited V86-A math source drift")
    spec = importlib.util.spec_from_file_location("v86b_audited_math", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load audited V86-A math source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {"run_cluster_cv", "fit_predict_heads", "select_actions", "self_test"}
    missing = sorted(name for name in required if not callable(getattr(module, name, None)))
    if missing:
        raise AssertionError(f"audited math reference missing functions: {missing}")
    return module


def component_length(value: Any) -> int:
    if isinstance(value, dict) and value and all(isinstance(rows, list) for rows in value.values()):
        return sum(len(rows) for rows in value.values())
    return len(value) if isinstance(value, (dict, list, tuple)) else 1


def component_record(value: Any) -> dict[str, Any]:
    length = component_length(value)
    return {"length": length, "sha256": value_sha256(value)}


def build_component_values(game: dict[str, Any]) -> dict[str, Any]:
    traces = game["traces"]
    trajectories = game["trajectories"]
    return {
        "candidate_raw_action_trace": traces["candidate"]["raw"],
        "candidate_proposed_action_trace": traces["candidate"]["proposed"],
        "candidate_executed_action_trace": traces["candidate"]["executed"],
        "opponent_proposed_action_trace": traces["opponent"]["proposed"],
        "opponent_executed_action_trace": traces["opponent"]["executed"],
        "observation_trajectory_by_seat": trajectories["observations_by_seat"],
        "state_trajectory_by_seat": trajectories["states_by_seat"],
        "canonical_final_state": game["final"]["state"],
        "final_economics": {
            "money_by_seat": game["final"]["money_by_seat"],
            "candidate_margin": game["final"]["candidate_margin"],
        },
        "feature_snapshots": game["features"]["snapshots"],
    }


def filter_step_rows(rows: list[dict[str, Any]], *, before_step: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["step"]) < before_step]


def joint_prefix_values(game: dict[str, Any]) -> dict[str, Any]:
    traces = game["traces"]
    trajectories = game["trajectories"]
    return {
        "candidate_raw_action_trace": filter_step_rows(
            traces["candidate"]["raw"], before_step=BRANCH_STEP
        ),
        "candidate_proposed_action_trace": filter_step_rows(
            traces["candidate"]["proposed"], before_step=BRANCH_STEP
        ),
        "candidate_executed_action_trace": filter_step_rows(
            traces["candidate"]["executed"], before_step=BRANCH_STEP
        ),
        "opponent_proposed_action_trace": filter_step_rows(
            traces["opponent"]["proposed"], before_step=BRANCH_STEP
        ),
        "opponent_executed_action_trace": filter_step_rows(
            traces["opponent"]["executed"], before_step=BRANCH_STEP
        ),
        "observation_trajectory_by_seat": {
            seat: filter_step_rows(rows, before_step=BRANCH_STEP + 1)
            for seat, rows in trajectories["observations_by_seat"].items()
        },
        "state_trajectory_by_seat": {
            seat: filter_step_rows(rows, before_step=BRANCH_STEP + 1)
            for seat, rows in trajectories["states_by_seat"].items()
        },
        "feature_snapshots": filter_step_rows(
            game["features"]["snapshots"], before_step=BRANCH_STEP + 1
        ),
    }


def first_step_difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int | None:
    for left_row, right_row in zip(left, right, strict=False):
        if left_row != right_row:
            return min(int(left_row["step"]), int(right_row["step"]))
    if len(left) == len(right):
        return None
    tail = left[len(right) :] or right[len(left) :]
    return int(tail[0]["step"])


def earliest_component_difference(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    steps: list[int] = []
    for name in (
        "candidate_raw_action_trace",
        "candidate_proposed_action_trace",
        "candidate_executed_action_trace",
        "opponent_proposed_action_trace",
        "opponent_executed_action_trace",
        "feature_snapshots",
    ):
        step = first_step_difference(left[name], right[name])
        if step is not None:
            steps.append(step)
    for name in ("observation_trajectory_by_seat", "state_trajectory_by_seat"):
        for seat in ("0", "1"):
            step = first_step_difference(left[name][seat], right[name][seat])
            if step is not None:
                steps.append(step)
    if left["canonical_final_state"] != right["canonical_final_state"]:
        steps.append(FINAL_STEP)
    if left["final_economics"] != right["final_economics"]:
        steps.append(FINAL_STEP)
    return min(steps) if steps else None


def expected_joint_prefix(keep: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keep_prefix = joint_prefix_values(keep)
    candidate_prefix = joint_prefix_values(candidate)
    exact = {name: keep_prefix[name] == candidate_prefix[name] for name in PREFIX_COMPONENT_NAMES}
    hashes = {
        name: {
            "keep_sha256": value_sha256(keep_prefix[name]),
            "candidate_sha256": value_sha256(candidate_prefix[name]),
            "keep_length": component_length(keep_prefix[name]),
            "candidate_length": component_length(candidate_prefix[name]),
        }
        for name in PREFIX_COMPONENT_NAMES
    }
    hash_equal = {
        name: pair["keep_sha256"] == pair["candidate_sha256"] for name, pair in hashes.items()
    }
    return {
        "through_step": BRANCH_STEP - 1,
        "exact_equal_by_component": exact,
        "hash_equal_by_component": hash_equal,
        "component_hashes": hashes,
        "all_exact_equal": all(exact.values()),
        "all_hash_equal": all(hash_equal.values()),
    }


def expected_alias_contract(keep: dict[str, Any], forced: dict[str, Any]) -> dict[str, Any]:
    keep_components = build_component_values(keep)
    forced_components = build_component_values(forced)
    component_exact = {
        name: keep_components[name] == forced_components[name] for name in COMPONENT_NAMES
    }
    component_hash = {
        name: keep["component_manifest"][name] == forced["component_manifest"][name]
        for name in COMPONENT_NAMES
    }
    if component_exact != component_hash:
        raise AssertionError("component exact/hash equality disagreement")
    keep_executed = keep["traces"]["candidate"]["executed"]
    forced_executed = forced["traces"]["candidate"]["executed"]
    executed_first = first_step_difference(keep_executed, forced_executed)
    executed_equal = executed_first is None
    full_equal = all(component_exact.values())
    full_first = earliest_component_difference(keep_components, forced_components)
    keep_route = keep["route_lifecycle"]["selected_route"]
    forced_route = forced["route_lifecycle"]["selected_route"]
    same_route = keep_route == forced_route
    if executed_first is not None and executed_first < BRANCH_STEP:
        verdict = "BLOCK_EARLY_CANDIDATE_EXECUTED_DIFFERENCE"
    elif same_route and full_equal and executed_equal:
        verdict = "VALID_ALIAS"
    elif same_route:
        verdict = "BLOCK_SAME_ROUTE_TRAJECTORY_DIFFERENT"
    elif executed_equal:
        verdict = "BLOCK_DIFFERENT_ROUTE_EXECUTED_IDENTICAL"
    elif executed_first is not None and executed_first >= BRANCH_STEP:
        verdict = "VALID_NON_ALIAS"
    else:
        verdict = "BLOCK_UNCLASSIFIED"
    return {
        "keep_selected_route": keep_route,
        "forced_selected_route": forced_route,
        "same_selected_route": same_route,
        "full_trajectory_exact_equal": full_equal,
        "full_trajectory_first_difference_step": full_first,
        "candidate_executed_exact_equal": executed_equal,
        "candidate_executed_first_difference_step": executed_first,
        "candidate_executed_support": verdict == "VALID_NON_ALIAS",
        "component_exact_equal": component_exact,
        "component_hash_equal": component_hash,
        "exact_hash_equality_consistent": True,
        "verdict": verdict,
    }


def validate_sanitizer(value: Any, label: str) -> int:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    require_exact_keys(value, {"changed_steps", "step_ids", "changed_paths"}, label)
    changed = integer(value["changed_steps"], f"{label}.changed_steps", minimum=0)
    step_ids = value["step_ids"]
    paths = value["changed_paths"]
    if not isinstance(step_ids, list):
        raise AssertionError(f"{label}.step_ids must be a list")
    for index, step in enumerate(step_ids):
        integer(step, f"{label}.step_ids[{index}]", minimum=0)
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise AssertionError(f"{label}.changed_paths must be a string list")
    if changed == 0 and (step_ids or paths):
        raise AssertionError(f"{label} reports zero changes with non-empty evidence")
    return changed


def validate_step_rows(
    rows: Any,
    label: str,
    *,
    value_key: str,
    expected_steps: range,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise AssertionError(f"{label} must be a list")
    if len(rows) != len(expected_steps):
        raise AssertionError(f"{label} length must be {len(expected_steps)}, observed {len(rows)}")
    for expected_step, row in zip(expected_steps, rows, strict=True):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}[{expected_step}] must be an object")
        require_exact_keys(row, {"step", value_key}, f"{label}[{expected_step}]")
        if row["step"] != expected_step:
            raise AssertionError(f"{label} step order changed at {expected_step}")
    return rows


def validate_route_lifecycle(
    value: Any,
    action_id: str,
    protocol: dict[str, Any],
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be an object")
    expected_keys = {
        "action_id",
        "requested_runtime_route",
        "selected_route",
        "selection_step",
        "route_absent_through_step_119",
        "forced_injection_count",
        "forced_injection_steps",
        "forced_option_start_count",
        "forced_option_completion_count",
        "organic_route_selection_count",
        "organic_route_selection_steps",
        "forced_choice_before_step_120_count",
        "route_choice_drift_count",
        "selected_route_stable_through_completion",
        "selected_route_matches_request",
        "lifecycle_complete",
        "selection_events",
        "route_history",
    }
    require_exact_keys(value, expected_keys, label)
    if value["action_id"] != action_id:
        raise AssertionError(f"{label}.action_id mismatch")
    if not isinstance(value["selected_route"], str) or not value["selected_route"]:
        raise AssertionError(f"{label}.selected_route must be non-empty")
    if value["selection_step"] != BRANCH_STEP:
        raise AssertionError(f"{label}.selection_step must be 120")
    if value["route_absent_through_step_119"] is not True:
        raise AssertionError(f"{label} selected a route before step 120")
    if value["forced_choice_before_step_120_count"] != 0:
        raise AssertionError(f"{label} has an early forced route choice")
    if value["route_choice_drift_count"] != 0:
        raise AssertionError(f"{label} has route drift")
    if value["selected_route_stable_through_completion"] is not True:
        raise AssertionError(f"{label} route was not stable through completion")
    if value["lifecycle_complete"] is not True:
        raise AssertionError(f"{label} lifecycle is incomplete")
    if not isinstance(value["selection_events"], list):
        raise AssertionError(f"{label}.selection_events must be a list")
    if not isinstance(value["route_history"], list):
        raise AssertionError(f"{label}.route_history must be a list")

    if action_id == KEEP:
        if value["requested_runtime_route"] is not None:
            raise AssertionError(f"{label} KEEP must not request a forced route")
        expected = {
            "forced_injection_count": 0,
            "forced_injection_steps": [],
            "forced_option_start_count": 0,
            "forced_option_completion_count": 0,
            "organic_route_selection_count": 1,
            "organic_route_selection_steps": [BRANCH_STEP],
            "selected_route_matches_request": True,
        }
    else:
        action = next(row for row in protocol["actions"] if row["id"] == action_id)
        runtime_route = action["runtime_route_key"]
        if value["requested_runtime_route"] != runtime_route:
            raise AssertionError(f"{label} runtime request does not match protocol")
        if value["selected_route"] != runtime_route:
            raise AssertionError(f"{label} selected route does not match request")
        expected = {
            "forced_injection_count": 1,
            "forced_injection_steps": [BRANCH_STEP],
            "forced_option_start_count": 1,
            "forced_option_completion_count": 1,
            "organic_route_selection_count": 0,
            "organic_route_selection_steps": [],
            "selected_route_matches_request": True,
        }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise AssertionError(
                f"{label}.{key}: expected {expected_value!r}, observed {value[key]!r}"
            )

    expected_event = {
        "step": BRANCH_STEP,
        "before": None,
        "after": value["selected_route"],
        "forced": action_id != KEEP,
    }
    if value["selection_events"] != [expected_event]:
        raise AssertionError(f"{label}.selection_events do not prove one step-120 choice")
    history = value["route_history"]
    if len(history) != FINAL_STEP:
        raise AssertionError(f"{label}.route_history must contain 719 calls")
    for step, row in enumerate(history):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}.route_history[{step}] must be an object")
        require_exact_keys(
            row,
            {
                "step",
                "route_before_injection",
                "injected",
                "route_before_agent",
                "route_after_agent",
            },
            f"{label}.route_history[{step}]",
        )
        if row["step"] != step:
            raise AssertionError(f"{label}.route_history step order changed")
        if step < BRANCH_STEP:
            expected_history = (None, False, None, None)
        elif step == BRANCH_STEP and action_id == KEEP:
            expected_history = (None, False, None, value["selected_route"])
        elif step == BRANCH_STEP:
            expected_history = (
                None,
                True,
                value["selected_route"],
                value["selected_route"],
            )
        else:
            expected_history = (
                value["selected_route"],
                False,
                value["selected_route"],
                value["selected_route"],
            )
        observed_history = (
            row["route_before_injection"],
            row["injected"],
            row["route_before_agent"],
            row["route_after_agent"],
        )
        if observed_history != expected_history:
            raise AssertionError(f"{label}.route_history evidence drift at step {step}")


def validate_game(
    game: Any,
    action_id: str,
    candidate_seat: int,
    protocol: dict[str, Any],
    label: str,
) -> dict[str, int]:
    if not isinstance(game, dict):
        raise AssertionError(f"{label} must be an object")
    require_exact_keys(
        game,
        {
            "action_id",
            "final_step",
            "completed_transitions",
            "completed",
            "candidate_seat",
            "opponent_seat",
            "candidate_audit",
            "opponent_audit",
            "route_lifecycle",
            "traces",
            "trajectories",
            "final",
            "features",
            "component_manifest",
            "game_core_sha256",
        },
        label,
    )
    if game["action_id"] != action_id:
        raise AssertionError(f"{label}.action_id mismatch")
    if game["final_step"] != FINAL_STEP or game["completed_transitions"] != FINAL_STEP:
        raise AssertionError(f"{label} did not complete all 719 transitions")
    if game["completed"] is not True:
        raise AssertionError(f"{label}.completed must be true")
    if game["candidate_seat"] != candidate_seat:
        raise AssertionError(f"{label}.candidate_seat mismatch")
    if game["opponent_seat"] != 1 - candidate_seat:
        raise AssertionError(f"{label}.opponent_seat mismatch")

    candidate_audit = game["candidate_audit"]
    if not isinstance(candidate_audit, dict):
        raise AssertionError(f"{label}.candidate_audit must be an object")
    require_exact_keys(
        candidate_audit,
        {
            "sanitizer",
            "fallback_events",
            "exception_events",
            "partial_action_events",
            "exception",
        },
        f"{label}.candidate_audit",
    )
    candidate_changes = validate_sanitizer(
        candidate_audit["sanitizer"], f"{label}.candidate_audit.sanitizer"
    )
    counters = {
        "candidate_sanitizer_changes": candidate_changes,
        "candidate_fallbacks": integer(
            candidate_audit["fallback_events"],
            f"{label}.candidate_audit.fallback_events",
            minimum=0,
        ),
        "candidate_exceptions": integer(
            candidate_audit["exception_events"],
            f"{label}.candidate_audit.exception_events",
            minimum=0,
        ),
        "candidate_partial_actions": integer(
            candidate_audit["partial_action_events"],
            f"{label}.candidate_audit.partial_action_events",
            minimum=0,
        ),
    }
    if counters["candidate_exceptions"] == 0 and candidate_audit["exception"] is not None:
        raise AssertionError(f"{label} has exception text with zero exception events")
    if any(counters.values()):
        raise AssertionError(f"{label} violates zero candidate-repair/error contract")

    opponent_audit = game["opponent_audit"]
    if not isinstance(opponent_audit, dict):
        raise AssertionError(f"{label}.opponent_audit must be an object")
    require_exact_keys(opponent_audit, {"sanitizer"}, f"{label}.opponent_audit")
    validate_sanitizer(opponent_audit["sanitizer"], f"{label}.opponent_audit.sanitizer")
    validate_route_lifecycle(game["route_lifecycle"], action_id, protocol, f"{label}.route")

    traces = game["traces"]
    if not isinstance(traces, dict):
        raise AssertionError(f"{label}.traces must be an object")
    require_exact_keys(traces, {"candidate", "opponent"}, f"{label}.traces")
    for actor in ("candidate", "opponent"):
        actor_traces = traces[actor]
        if not isinstance(actor_traces, dict):
            raise AssertionError(f"{label}.traces.{actor} must be an object")
        require_exact_keys(actor_traces, {"raw", "proposed", "executed"}, f"{label}.traces.{actor}")
        for trace_name in ("raw", "proposed", "executed"):
            validate_step_rows(
                actor_traces[trace_name],
                f"{label}.traces.{actor}.{trace_name}",
                value_key="action",
                expected_steps=range(FINAL_STEP),
            )

    trajectories = game["trajectories"]
    if not isinstance(trajectories, dict):
        raise AssertionError(f"{label}.trajectories must be an object")
    require_exact_keys(
        trajectories,
        {"observations_by_seat", "states_by_seat"},
        f"{label}.trajectories",
    )
    for trajectory_name, value_key in (
        ("observations_by_seat", "observation"),
        ("states_by_seat", "state"),
    ):
        by_seat = trajectories[trajectory_name]
        if not isinstance(by_seat, dict):
            raise AssertionError(f"{label}.{trajectory_name} must be an object")
        require_exact_keys(by_seat, {"0", "1"}, f"{label}.{trajectory_name}")
        for seat in ("0", "1"):
            validate_step_rows(
                by_seat[seat],
                f"{label}.{trajectory_name}.{seat}",
                value_key=value_key,
                expected_steps=range(FINAL_STEP),
            )

    final = game["final"]
    if not isinstance(final, dict):
        raise AssertionError(f"{label}.final must be an object")
    require_exact_keys(
        final,
        {
            "state",
            "money_by_seat",
            "candidate_money",
            "opponent_money",
            "candidate_margin",
        },
        f"{label}.final",
    )
    money = final["money_by_seat"]
    if not isinstance(money, dict):
        raise AssertionError(f"{label}.final.money_by_seat must be an object")
    require_exact_keys(money, {"0", "1"}, f"{label}.final.money_by_seat")
    state = final["state"]
    if not isinstance(state, dict) or "farms" not in state:
        raise AssertionError(f"{label}.final.state must contain farms")
    farms = state["farms"]
    if not isinstance(farms, list) or len(farms) != 2:
        raise AssertionError(f"{label}.final.state.farms must contain two farms")
    if any(not isinstance(farm, dict) or "money" not in farm for farm in farms):
        raise AssertionError(f"{label}.final.state farm money missing")
    seat_money = {
        seat: finite_number(value, f"{label}.money.{seat}") for seat, value in money.items()
    }
    if any(
        seat_money[str(seat)] != finite_number(farms[seat]["money"], f"{label}.farm.{seat}.money")
        for seat in SEATS
    ):
        raise AssertionError(f"{label}.money_by_seat disagrees with final farms")
    candidate_money = finite_number(final["candidate_money"], f"{label}.candidate_money")
    opponent_money = finite_number(final["opponent_money"], f"{label}.opponent_money")
    margin = finite_number(final["candidate_margin"], f"{label}.candidate_margin")
    if candidate_money != seat_money[str(candidate_seat)]:
        raise AssertionError(f"{label}.candidate_money disagrees with seat money")
    if opponent_money != seat_money[str(1 - candidate_seat)]:
        raise AssertionError(f"{label}.opponent_money disagrees with seat money")
    if margin != candidate_money - opponent_money:
        raise AssertionError(f"{label}.candidate_margin arithmetic mismatch")

    features = game["features"]
    if not isinstance(features, dict):
        raise AssertionError(f"{label}.features must be an object")
    require_exact_keys(
        features,
        {"observation_steps", "snapshots", "vector", "finite"},
        f"{label}.features",
    )
    if features["observation_steps"] != list(FEATURE_STEPS):
        raise AssertionError(f"{label} observation-step schedule changed")
    snapshots = features["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != len(FEATURE_STEPS):
        raise AssertionError(f"{label}.features.snapshots must contain five rows")
    flattened: list[float] = []
    for expected_step, row in zip(FEATURE_STEPS, snapshots, strict=True):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}.snapshot.{expected_step} must be an object")
        require_exact_keys(row, {"step", "values"}, f"{label}.snapshot.{expected_step}")
        if row["step"] != expected_step:
            raise AssertionError(f"{label}.snapshot step mismatch")
        values = row["values"]
        if not isinstance(values, list) or len(values) != 81:
            raise AssertionError(f"{label}.snapshot must have 81 values")
        flattened.extend(finite_number(value, f"{label}.snapshot.value") for value in values)
    vector = features["vector"]
    if not isinstance(vector, list) or len(vector) != FEATURE_DIM:
        raise AssertionError(f"{label}.features.vector must have 405 values")
    checked_vector = [finite_number(value, f"{label}.features.vector") for value in vector]
    if checked_vector != flattened:
        raise AssertionError(f"{label}.features.vector does not concatenate snapshots")
    if features["finite"] is not True:
        raise AssertionError(f"{label}.features.finite must be true")

    component_values = build_component_values(game)
    manifest = game["component_manifest"]
    if not isinstance(manifest, dict):
        raise AssertionError(f"{label}.component_manifest must be an object")
    require_exact_keys(manifest, set(COMPONENT_NAMES), f"{label}.component_manifest")
    expected_manifest = {name: component_record(value) for name, value in component_values.items()}
    if manifest != expected_manifest:
        raise AssertionError(f"{label}.component_manifest does not match full values")

    claimed_core = hash_string(game["game_core_sha256"], f"{label}.game_core_sha256")
    expected_core = value_sha256(
        {
            "action_id": action_id,
            "candidate_audit": candidate_audit,
            "opponent_audit": opponent_audit,
            "route_lifecycle": game["route_lifecycle"],
            "components": component_values,
            "feature_vector": vector,
            "final_step": FINAL_STEP,
            "completed_transitions": FINAL_STEP,
        }
    )
    if claimed_core != expected_core:
        raise AssertionError(f"{label}.game_core_sha256 mismatch")
    return counters


def expected_source_record_keys(protocol: dict[str, Any]) -> set[str]:
    keys = {
        "protocol.json",
        "PRE_REGISTRATION.md",
        "CONTRACT_FREEZE_RECEIPT.md",
        "run_experiment.py",
        "candidate",
        "permitted_v86a_mechanical_stop_report",
    }
    keys.update(f"action:{action}" for action in ACTIONS[1:])
    keys.update(f"family:{family['id']}" for family in protocol["families"])
    keys.update(f"reference:{name}" for name in protocol["pinned_reference_sources"])
    return keys


def validate_source_records(
    records: Any, protocol: dict[str, Any], protocol_info: dict[str, Any]
) -> None:
    if not isinstance(records, dict):
        raise AssertionError("dataset.source_records must be an object")
    require_exact_keys(records, expected_source_record_keys(protocol), "source_records")
    for name, record in records.items():
        expected_keys = (
            {"path", "sha256", "use"}
            if name == ("permitted_v86a_mechanical_stop_report")
            else {"path", "sha256"}
        )
        if not isinstance(record, dict):
            raise AssertionError(f"source_records.{name} must be an object")
        require_exact_keys(record, expected_keys, f"source_records.{name}")
        verify_source_record(record, f"source_records.{name}")

    expected_fixed_hashes = {
        "protocol.json": protocol_info["sha256"],
        "PRE_REGISTRATION.md": EXPECTED_PRE_REGISTRATION_SHA256,
        "CONTRACT_FREEZE_RECEIPT.md": EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256,
        "run_experiment.py": EXPECTED_COLLECTOR_SHA256,
        "candidate": protocol["candidate"]["sha256"],
    }
    for action in protocol["actions"][1:]:
        expected_fixed_hashes[f"action:{action['id']}"] = action["source_sha256"]
    for family in protocol["families"]:
        expected_fixed_hashes[f"family:{family['id']}"] = family["sha256"]
    for name, record in protocol["pinned_reference_sources"].items():
        expected_fixed_hashes[f"reference:{name}"] = record["sha256"]
    stop_record = protocol["v86a_exclusion_evidence"]["allowed_mechanical_stop_report"]
    expected_fixed_hashes["permitted_v86a_mechanical_stop_report"] = stop_record["sha256"]
    for name, expected_hash in expected_fixed_hashes.items():
        if records[name]["sha256"] != expected_hash:
            raise AssertionError(f"source_records.{name} does not match frozen source")
    if records["permitted_v86a_mechanical_stop_report"]["use"] != (
        "mechanical-stop evidence only; no economic values parsed"
    ):
        raise AssertionError("permitted V86-A stop-report use changed")


def validate_sample(
    sample: Any,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    label = f"samples[{index}]"
    if not isinstance(sample, dict):
        raise AssertionError(f"{label} must be an object")
    require_exact_keys(
        sample,
        {"audit", "model_input", "labels", "games", "sample_core_sha256"},
        label,
    )
    audit = sample["audit"]
    if not isinstance(audit, dict):
        raise AssertionError(f"{label}.audit must be an object")
    require_exact_keys(
        audit,
        {
            "key",
            "family",
            "representative",
            "source_group",
            "seed",
            "candidate_seat",
            "opponent_seat",
            "observed_steps",
            "feature_count",
            "joint_prefix_parity",
            "alias_contract",
        },
        f"{label}.audit",
    )
    key = audit["key"]
    if not isinstance(key, dict):
        raise AssertionError(f"{label}.audit.key must be an object")
    require_exact_keys(key, {"family", "seed", "candidate_seat"}, f"{label}.key")
    family = audit["family"]
    if family not in protocol_info["family_by_id"]:
        raise AssertionError(f"{label} contains an unknown family")
    seed = integer(audit["seed"], f"{label}.seed", minimum=0)
    if seed not in protocol_info["seeds"]:
        raise AssertionError(f"{label} contains an unknown seed")
    seat = integer(audit["candidate_seat"], f"{label}.candidate_seat")
    if seat not in SEATS:
        raise AssertionError(f"{label}.candidate_seat must be 0 or 1")
    if key != {"family": family, "seed": seed, "candidate_seat": seat}:
        raise AssertionError(f"{label}.audit.key disagrees with audit fields")
    if audit["opponent_seat"] != 1 - seat:
        raise AssertionError(f"{label}.opponent_seat mismatch")
    family_record = protocol_info["family_by_id"][family]
    if audit["representative"] != family_record["representative"]:
        raise AssertionError(f"{label}.representative mismatch")
    if audit["source_group"] != family_record["source_group"]:
        raise AssertionError(f"{label}.source_group mismatch")
    if audit["observed_steps"] != list(FEATURE_STEPS):
        raise AssertionError(f"{label}.observed_steps mismatch")
    if audit["feature_count"] != FEATURE_DIM:
        raise AssertionError(f"{label}.feature_count mismatch")

    games = sample["games"]
    if not isinstance(games, dict):
        raise AssertionError(f"{label}.games must be an object")
    require_exact_keys(games, set(ACTIONS), f"{label}.games")
    counters = {
        "candidate_sanitizer_changes": 0,
        "candidate_fallbacks": 0,
        "candidate_exceptions": 0,
        "candidate_partial_actions": 0,
    }
    opponent_changes = 0
    for action_id in ACTIONS:
        game_counters = validate_game(
            games[action_id], action_id, seat, protocol, f"{label}.games.{action_id}"
        )
        for name, value in game_counters.items():
            counters[name] += value
        opponent_changes += games[action_id]["opponent_audit"]["sanitizer"]["changed_steps"]

    keep = games[KEEP]
    joint = audit["joint_prefix_parity"]
    alias = audit["alias_contract"]
    if not isinstance(joint, dict) or not isinstance(alias, dict):
        raise AssertionError(f"{label} prefix/alias evidence must be objects")
    require_exact_keys(joint, {ROUTE_B, ROUTE_C}, f"{label}.joint_prefix_parity")
    require_exact_keys(alias, {ROUTE_B, ROUTE_C}, f"{label}.alias_contract")
    support: dict[str, bool] = {}
    alias_rows: dict[str, bool] = {}
    for action_id in ACTIONS[1:]:
        expected_prefix = expected_joint_prefix(keep, games[action_id])
        if joint[action_id] != expected_prefix:
            raise AssertionError(f"{label}.{action_id} joint-prefix evidence mismatch")
        if not expected_prefix["all_exact_equal"] or not expected_prefix["all_hash_equal"]:
            raise AssertionError(f"{label}.{action_id} violates full-joint prefix parity")
        expected_alias = expected_alias_contract(keep, games[action_id])
        if alias[action_id] != expected_alias:
            raise AssertionError(f"{label}.{action_id} alias evidence mismatch")
        if expected_alias["verdict"] not in {"VALID_ALIAS", "VALID_NON_ALIAS"}:
            raise AssertionError(f"{label}.{action_id} fails alias truth table")
        support[action_id] = expected_alias["candidate_executed_support"]
        alias_rows[action_id] = expected_alias["verdict"] == "VALID_ALIAS"

    model_input = sample["model_input"]
    if not isinstance(model_input, dict):
        raise AssertionError(f"{label}.model_input must be an object")
    require_exact_keys(model_input, {"features"}, f"{label}.model_input")
    features = model_input["features"]
    if not isinstance(features, list) or len(features) != FEATURE_DIM:
        raise AssertionError(f"{label}.model_input.features must have length 405")
    checked_features = [finite_number(value, f"{label}.features") for value in features]
    if checked_features != keep["features"]["vector"]:
        raise AssertionError(f"{label}.model_input differs from KEEP observations")
    for action_id in ACTIONS[1:]:
        if games[action_id]["features"]["vector"] != checked_features:
            raise AssertionError(f"{label}.{action_id} feature prefix differs from KEEP")

    labels = sample["labels"]
    if not isinstance(labels, dict):
        raise AssertionError(f"{label}.labels must be an object")
    require_exact_keys(
        labels,
        {"model_margins", "action_margins", "paired_deltas"},
        f"{label}.labels",
    )
    model_margins = labels["model_margins"]
    action_margins = labels["action_margins"]
    paired_deltas = labels["paired_deltas"]
    for value, keys, name in (
        (model_margins, set(MODEL_LABELS), "model_margins"),
        (action_margins, set(ACTIONS), "action_margins"),
        (paired_deltas, set(DELTA_LABELS), "paired_deltas"),
    ):
        if not isinstance(value, dict):
            raise AssertionError(f"{label}.{name} must be an object")
        require_exact_keys(value, keys, f"{label}.{name}")
    expected_action_margins = {
        action_id: games[action_id]["final"]["candidate_margin"] for action_id in ACTIONS
    }
    checked_action_margins = {
        action_id: finite_number(action_margins[action_id], f"{label}.{action_id}.margin")
        for action_id in ACTIONS
    }
    if checked_action_margins != expected_action_margins:
        raise AssertionError(f"{label}.action_margins disagree with games")
    expected_model_margins = {
        "KEEP": checked_action_margins[KEEP],
        "B": checked_action_margins[ROUTE_B],
        "C": checked_action_margins[ROUTE_C],
    }
    checked_model_margins = {
        name: finite_number(model_margins[name], f"{label}.model_margins.{name}")
        for name in MODEL_LABELS
    }
    if checked_model_margins != expected_model_margins:
        raise AssertionError(f"{label}.model_margins mapping mismatch")
    expected_deltas = {
        "B": expected_model_margins["B"] - expected_model_margins["KEEP"],
        "C": expected_model_margins["C"] - expected_model_margins["KEEP"],
    }
    checked_deltas = {
        name: finite_number(paired_deltas[name], f"{label}.paired_deltas.{name}")
        for name in DELTA_LABELS
    }
    if checked_deltas != expected_deltas:
        raise AssertionError(f"{label}.paired_deltas arithmetic mismatch")
    for action_id, model_name in ((ROUTE_B, "B"), (ROUTE_C, "C")):
        if alias_rows[action_id] and checked_deltas[model_name] != 0.0:
            raise AssertionError(f"{label}.{action_id} alias delta must be exactly zero")

    claimed_sample_core = hash_string(sample["sample_core_sha256"], f"{label}.sample_core_sha256")
    expected_sample_core = value_sha256(
        {name: sample[name] for name in ("audit", "model_input", "labels", "games")}
    )
    if claimed_sample_core != expected_sample_core:
        raise AssertionError(f"{label}.sample_core_sha256 mismatch")
    return {
        "key": (family, seed, seat),
        "family": family,
        "seed": seed,
        "seat": seat,
        "features": checked_features,
        "margins": [checked_model_margins[name] for name in MODEL_LABELS],
        "deltas": [checked_deltas[name] for name in DELTA_LABELS],
        "support": support,
        "alias_rows": alias_rows,
        "counters": counters,
        "opponent_sanitizer_changes": opponent_changes,
    }


def dataset_core_payload(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        name: dataset[name]
        for name in (
            "schema_version",
            "protocol_sha256",
            "source_records",
            "post_run_source_drift_audit",
            "samples",
            "collection_validation",
            "selector_accounting_semantics",
        )
    }


def validate_and_extract_dataset(
    dataset: dict[str, Any],
    dataset_path: Path,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
) -> dict[str, Any]:
    reject_forbidden_v86a_economic_path(dataset_path, "dataset")
    require_exact_keys(
        dataset,
        {
            "schema_version",
            "protocol_sha256",
            "source_records",
            "post_run_source_drift_audit",
            "samples",
            "collection_validation",
            "selector_accounting_semantics",
            "status",
            "full_run",
            "deterministic_core_components",
            "deterministic_core_sha256",
            "prohibitions",
        },
        "dataset",
    )
    if dataset["schema_version"] != EXPECTED_DATASET_SCHEMA:
        raise AssertionError("unexpected V86-B dataset schema")
    if dataset["protocol_sha256"] != protocol_info["sha256"]:
        raise AssertionError("dataset was not collected under frozen protocol")
    if dataset["status"] != "mechanical_pass_ready_for_frozen_cv":
        raise AssertionError("dataset is not a full mechanical-pass collection")
    if dataset["full_run"] is not True:
        raise AssertionError("partial V86-B collections are forbidden for CV")
    if dataset["selector_accounting_semantics"] != protocol["selector_accounting_semantics"]:
        raise AssertionError("dataset alias selector-accounting semantics changed")
    expected_core_components = [
        "protocol_sha256",
        "source_records",
        "post_run_source_drift_audit",
        "samples",
        "collection_validation",
        "selector_accounting_semantics",
    ]
    if dataset["deterministic_core_components"] != expected_core_components:
        raise AssertionError("dataset deterministic-core coverage changed")
    prohibitions = dataset["prohibitions"]
    if not isinstance(prohibitions, dict):
        raise AssertionError("dataset.prohibitions must be an object")
    require_exact_keys(
        prohibitions,
        {
            "v86a_dataset_or_results_read",
            "sealed_holdout_opened",
            "checkpoint_created",
            "runtime_agent_created",
            "submission_prepared",
            "kaggle_upload_performed",
        },
        "dataset.prohibitions",
    )
    if any(value is not False for value in prohibitions.values()):
        raise AssertionError("dataset reports a forbidden side effect")
    validate_source_records(dataset["source_records"], protocol, protocol_info)
    drift_audit = dataset["post_run_source_drift_audit"]
    if not isinstance(drift_audit, dict):
        raise AssertionError("post_run_source_drift_audit must be an object")
    require_exact_keys(
        drift_audit,
        {
            "checked_record_count",
            "expected_record_count",
            "startup_manifest_sha256",
            "post_run_manifest_sha256",
            "source_records_exact_equal",
            "protocol_exact_equal",
            "pass",
        },
        "post_run_source_drift_audit",
    )
    manifest_hash = value_sha256(dataset["source_records"])
    if drift_audit != {
        "checked_record_count": 28,
        "expected_record_count": 28,
        "startup_manifest_sha256": manifest_hash,
        "post_run_manifest_sha256": manifest_hash,
        "source_records_exact_equal": True,
        "protocol_exact_equal": True,
        "pass": True,
    }:
        raise AssertionError("post-run source drift audit does not prove exact equality")

    samples = dataset["samples"]
    if not isinstance(samples, list) or len(samples) != protocol["expected_contexts"]:
        raise AssertionError("dataset must retain all 128 contexts, including aliases")
    records = [
        validate_sample(sample, protocol, protocol_info, index)
        for index, sample in enumerate(samples)
    ]
    family_order = {name: index for index, name in enumerate(protocol_info["family_ids"])}
    expected_order = sorted(
        (family, seed, seat)
        for family in protocol_info["family_ids"]
        for seed in protocol_info["seeds"]
        for seat in SEATS
    )
    expected_order.sort(key=lambda row: (family_order[row[0]], row[1], row[2]))
    observed_order = [record["key"] for record in records]
    if observed_order != expected_order:
        raise AssertionError("dataset context order/grid is incomplete or changed")
    if len(set(observed_order)) != len(observed_order):
        raise AssertionError("dataset contains duplicate family-seed-seat rows")

    support_counts = {action_id: 0 for action_id in ACTIONS[1:]}
    support_by_family = {
        action_id: {family: 0 for family in protocol_info["family_ids"]}
        for action_id in ACTIONS[1:]
    }
    counters = {
        "candidate_sanitizer_changes": 0,
        "candidate_fallbacks": 0,
        "candidate_exceptions": 0,
        "candidate_partial_actions": 0,
        "opponent_sanitizer_changes": 0,
    }
    alias_counts = {action_id: 0 for action_id in ACTIONS[1:]}
    for record in records:
        for action_id in ACTIONS[1:]:
            if record["support"][action_id]:
                support_counts[action_id] += 1
                support_by_family[action_id][record["family"]] += 1
            if record["alias_rows"][action_id]:
                alias_counts[action_id] += 1
        for name, value in record["counters"].items():
            counters[name] += value
        counters["opponent_sanitizer_changes"] += record["opponent_sanitizer_changes"]

    support_contract = protocol["mechanical_gates"]["global_route_nondegeneracy"]
    support_pass = all(
        count >= support_contract["minimum_different_contexts_per_forced_action"]
        for count in support_counts.values()
    ) and all(
        count >= support_contract["minimum_different_contexts_per_action_per_family"]
        for family_counts in support_by_family.values()
        for count in family_counts.values()
    )
    nondegeneracy = {
        "support_definition": "candidate executed-action difference step >= 120",
        "counts_by_action": support_counts,
        "counts_by_action_and_family": support_by_family,
        "full_grid_global_minimum": 116,
        "full_grid_per_family_minimum": 12,
        "evaluated": True,
        "pass": support_pass,
    }
    expected_validation = {
        "context_count": 128,
        "game_count": 384,
        "unique_key_count": 128,
        "expected_subset_games": 384,
        "full_grid_expected_contexts": 128,
        "full_grid_expected_games": 384,
        "full_grid_complete": True,
        "all_games_final_step_719": True,
        "all_games_completed_transitions_719": True,
        "candidate_sanitizer_changed_steps": counters["candidate_sanitizer_changes"],
        "candidate_fallback_events": counters["candidate_fallbacks"],
        "candidate_exception_events": counters["candidate_exceptions"],
        "candidate_partial_action_events": counters["candidate_partial_actions"],
        "opponent_sanitizer_changed_steps": counters["opponent_sanitizer_changes"],
        "model_input_fields": ["features"],
        "finite_feature_shape": [128, FEATURE_DIM],
        "all_features_finite": True,
        "route_lifecycle_gate_pass": True,
        "joint_prefix_check_count": 256,
        "joint_prefix_parity_all": True,
        "alias_truth_table_check_count": 256,
        "alias_truth_table_all_valid": True,
        "exact_hash_equality_consistent": True,
        "post_run_source_drift_pass": True,
        "nondegeneracy": nondegeneracy,
        "mechanical_pass": support_pass
        and all(
            value == 0
            for value in (
                counters["candidate_sanitizer_changes"],
                counters["candidate_fallbacks"],
                counters["candidate_exceptions"],
                counters["candidate_partial_actions"],
            )
        ),
    }
    validation = dataset["collection_validation"]
    if not isinstance(validation, dict):
        raise AssertionError("dataset.collection_validation must be an object")
    require_exact_keys(validation, set(expected_validation), "collection_validation")
    if validation != expected_validation:
        raise AssertionError("collection_validation disagrees with recomputed evidence")
    if not expected_validation["mechanical_pass"]:
        raise AssertionError("V86-B mechanical or nondegeneracy gate failed")

    claimed_core = hash_string(
        dataset["deterministic_core_sha256"], "dataset.deterministic_core_sha256"
    )
    observed_core = value_sha256(dataset_core_payload(dataset))
    if claimed_core != observed_core:
        raise AssertionError(
            f"dataset deterministic core mismatch: {claimed_core} != {observed_core}"
        )
    return {
        "records": records,
        "counters": counters,
        "support_counts": support_counts,
        "support_by_family": support_by_family,
        "alias_counts": alias_counts,
        "dataset_core_sha256": observed_core,
        "dataset_file_sha256": sha256(dataset_path),
        "mechanical_validation": expected_validation,
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
        row["worst_not_worse"] = row["selected_worst_margin"] >= row["keep_worst_margin"]
        row["mean_not_worse"] = row["selected_mean_margin"] >= row["keep_mean_margin"]
        rows.append(row)
    return rows


def score_choices(margins: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    selected = margins[np.arange(len(choices)), choices]
    keep = margins[:, 0]
    deltas = selected - keep
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
        "mean_paired_delta": float(np.mean(deltas)),
        "worst_paired_delta": float(np.min(deltas)),
    }


def compute_gates(
    cv: dict[str, Any],
    extracted: dict[str, Any],
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    reproduction_verified: bool,
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

    keep_loss_clusters = {
        (str(families[index]), int(seeds[index])) for index in np.flatnonzero(keep_loss)
    }
    keep_loss_families = {str(families[index]) for index in np.flatnonzero(keep_loss)}
    oracle_rescue_families = {str(families[index]) for index in np.flatnonzero(oracle_rescue)}
    selected_rescue_clusters = {
        (str(families[index]), int(seeds[index])) for index in np.flatnonzero(selected_rescue)
    }
    oracle_rescue_fraction = (
        float(np.sum(oracle_rescue)) / keep_loss_count if keep_loss_count else 0.0
    )
    selected_rescue_fraction = (
        float(np.sum(selected_rescue)) / keep_loss_count if keep_loss_count else 0.0
    )

    mechanical = {
        "dataset_deterministic_core_valid": True,
        "contexts_128": len(margins) == 128,
        "games_384": extracted["mechanical_validation"]["game_count"] == 384,
        "all_games_final_step_719": extracted["mechanical_validation"]["all_games_final_step_719"],
        "all_games_completed_transitions_719": extracted["mechanical_validation"][
            "all_games_completed_transitions_719"
        ],
        "full_joint_prefix_parity": extracted["mechanical_validation"]["joint_prefix_parity_all"],
        "alias_truth_table_valid": extracted["mechanical_validation"][
            "alias_truth_table_all_valid"
        ],
        "exact_hash_equality_consistent": extracted["mechanical_validation"][
            "exact_hash_equality_consistent"
        ],
        "global_and_family_executed_support": extracted["mechanical_validation"]["nondegeneracy"][
            "pass"
        ],
        "post_run_source_drift_pass": extracted["mechanical_validation"][
            "post_run_source_drift_pass"
        ],
        "candidate_sanitizer_changes_zero": extracted["counters"]["candidate_sanitizer_changes"]
        == 0,
        "candidate_fallbacks_zero": extracted["counters"]["candidate_fallbacks"] == 0,
        "candidate_exceptions_zero": extracted["counters"]["candidate_exceptions"] == 0,
        "candidate_partial_actions_zero": extracted["counters"]["candidate_partial_actions"] == 0,
        "finite_feature_shape_128x405": cv["features"].shape == (128, FEATURE_DIM)
        and bool(np.isfinite(cv["features"]).all()),
        "fold_count_64": len(cv["folds"]) == 64,
        "training_rows_98_every_fold": all(fold["train_rows"] == 98 for fold in cv["folds"]),
        "seven_training_families_every_fold": all(
            fold["train_family_count"] == 7 for fold in cv["folds"]
        ),
        "deterministic_core_hash_rerun": reproduction_verified,
    }
    mechanical["all_pass"] = all(mechanical.values())

    adequacy_contract = protocol["adequacy_gates"]
    adequacy = {
        "keep_loss_rows": keep_loss_count,
        "keep_loss_family_seed_clusters": len(keep_loss_clusters),
        "keep_loss_families": len(keep_loss_families),
        "oracle_rescue_rows": int(np.sum(oracle_rescue)),
        "oracle_rescue_fraction": oracle_rescue_fraction,
        "oracle_rescue_families": len(oracle_rescue_families),
        "minimum_keep_loss_rows_pass": keep_loss_count
        >= adequacy_contract["minimum_keep_loss_rows"],
        "minimum_keep_loss_family_seed_clusters_pass": len(keep_loss_clusters)
        >= adequacy_contract["minimum_keep_loss_family_seed_clusters"],
        "minimum_keep_loss_families_pass": len(keep_loss_families)
        >= adequacy_contract["minimum_keep_loss_families"],
        "minimum_oracle_rescue_fraction_pass": oracle_rescue_fraction
        >= adequacy_contract["minimum_oracle_rescue_fraction"],
        "minimum_oracle_rescue_families_pass": len(oracle_rescue_families)
        >= adequacy_contract["minimum_oracle_rescue_families"],
    }
    adequacy["all_pass"] = all(value for key, value in adequacy.items() if key.endswith("_pass"))

    per_family = family_metrics(margins, choices, families, protocol_info["family_ids"])
    intervention_count = int(np.sum(intervention))
    intervention_deltas = selected_delta[intervention]
    intervention_margins = selected[intervention]
    selector_contract = protocol["selector_go_gates"]
    selector = {
        "interventions": intervention_count,
        "selected_wins": int(np.sum(selected > 0.0)),
        "selected_losses_or_ties": int(np.sum(selected <= 0.0)),
        "keep_loss_rescues": int(np.sum(selected_rescue)),
        "keep_loss_rescue_fraction": selected_rescue_fraction,
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
        "minimum_keep_loss_rescue_fraction_pass": selected_rescue_fraction
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
    selector["all_pass"] = all(value for key, value in selector.items() if key.endswith("_pass"))

    promotion = {
        "wins": int(np.sum(selected > 0.0)),
        "required_wins": protocol["promotion_grade_extra_gates"]["wins"],
        "wins_128_pass": int(np.sum(selected > 0.0)) == 128,
        "positive_worst_margin_in_every_family_pass": all(
            row["selected_worst_margin"] > 0.0 for row in per_family
        ),
    }
    promotion["all_pass"] = all(value for key, value in promotion.items() if key.endswith("_pass"))
    return {
        "mechanical": mechanical,
        "adequacy": adequacy,
        "selector": selector,
        "promotion_grade_extra": promotion,
    }


def evaluation_core_payload(
    *,
    protocol_sha256: str,
    dataset_core_sha256: str,
    cv_public: dict[str, Any],
    gates_without_reproduction: dict[str, Any],
    alias_accounting: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_sha256": protocol_sha256,
        "dataset_core_sha256": dataset_core_sha256,
        "model": {
            "type": "dual_ridge_absolute_and_delta_heads",
            "alpha": 1000.0,
            "beta": 2.0,
            "switch_threshold": 2500.0,
            "actions": list(ACTIONS),
            "hyperparameter_search": False,
        },
        "alias_accounting": alias_accounting,
        "cv": cv_public,
        "gates": gates_without_reproduction,
    }


def reproduction_references(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str, dict[str, Any] | None]:
    dataset_reference = args.reference_dataset_core_sha256
    evaluation_reference = args.reference_evaluation_core_sha256
    existing: dict[str, Any] | None = None
    method = "explicit CLI deterministic-core references"
    if args.render_existing:
        reject_forbidden_v86a_economic_path(args.results, "results")
        if not args.results.is_file():
            raise AssertionError("--render-existing requires an existing results file")
        existing = read_json(args.results)
        determinism = existing.get("determinism")
        if not isinstance(determinism, dict):
            raise AssertionError("existing results lack determinism references")
        existing_dataset = determinism.get("dataset_core_sha256")
        existing_evaluation = determinism.get("evaluation_core_sha256")
        hash_string(existing_dataset, "existing dataset core")
        hash_string(existing_evaluation, "existing evaluation core")
        if dataset_reference is not None and dataset_reference != existing_dataset:
            raise AssertionError("explicit dataset reference differs from existing result")
        if evaluation_reference is not None and evaluation_reference != existing_evaluation:
            raise AssertionError("explicit evaluation reference differs from existing result")
        dataset_reference = existing_dataset
        evaluation_reference = existing_evaluation
        method = "independent render-existing deterministic-core rerun"
    for value, label in (
        (dataset_reference, "reference dataset core"),
        (evaluation_reference, "reference evaluation core"),
    ):
        if value is not None:
            hash_string(value, label)
    return dataset_reference, evaluation_reference, method, existing


def build_results(
    dataset: dict[str, Any],
    dataset_path: Path,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    math_reference: ModuleType,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    extracted = validate_and_extract_dataset(dataset, dataset_path, protocol, protocol_info)
    cv = math_reference.run_cluster_cv(extracted, protocol, protocol_info)
    if cv["features"].shape != (128, FEATURE_DIM):
        raise AssertionError("audited CV returned an unexpected feature matrix")
    action_counts = {
        action_id: int(np.sum(cv["choices"] == index)) for index, action_id in enumerate(ACTIONS)
    }
    selected_alias_counts = {action_id: 0 for action_id in ACTIONS[1:]}
    selected_alias_rows: list[dict[str, Any]] = []
    for index, choice in enumerate(cv["choices"].tolist()):
        if choice == 0:
            continue
        action_id = ACTIONS[int(choice)]
        if extracted["records"][index]["alias_rows"][action_id]:
            selected_alias_counts[action_id] += 1
            prediction = cv["predictions"][index]
            if prediction["actual_paired_delta"] != 0.0:
                raise AssertionError("selected alias must retain exact zero paired delta")
            selected_alias_rows.append(prediction["key"])
    alias_accounting = {
        "all_128_rows_retained": len(extracted["records"]) == 128,
        "observed_alias_rows_by_action": extracted["alias_counts"],
        "executed_support_rows_by_action": extracted["support_counts"],
        "executed_support_rows_by_action_and_family": extracted["support_by_family"],
        "selected_alias_interventions_by_action": selected_alias_counts,
        "selected_alias_intervention_rows": selected_alias_rows,
        "selected_alias_counts_as_intervention": True,
        "selected_alias_observed_delta_must_be_zero": True,
    }
    cv_public = {
        "outer_protocol": (
            "64 family-seed folds; exclude every row in held family and every row "
            "with held seed; predict both held seats"
        ),
        "fold_count": len(cv["folds"]),
        "folds": cv["folds"],
        "predictions": cv["predictions"],
        "aggregate": score_choices(cv["margins"], cv["choices"]),
        "action_counts": action_counts,
    }

    gates_for_core = compute_gates(
        cv, extracted, protocol, protocol_info, reproduction_verified=False
    )
    gates_for_core["mechanical"].pop("deterministic_core_hash_rerun")
    gates_for_core["mechanical"]["all_pass"] = all(gates_for_core["mechanical"].values())
    evaluation_core = value_sha256(
        evaluation_core_payload(
            protocol_sha256=protocol_info["sha256"],
            dataset_core_sha256=extracted["dataset_core_sha256"],
            cv_public=cv_public,
            gates_without_reproduction=gates_for_core,
            alias_accounting=alias_accounting,
        )
    )
    dataset_reference, evaluation_reference, method, existing = reproduction_references(args)
    checked = dataset_reference is not None and evaluation_reference is not None
    reproduction = {
        "checked": checked,
        "method": method,
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
    gates = compute_gates(
        cv,
        extracted,
        protocol,
        protocol_info,
        reproduction_verified=reproduction["verified"],
    )
    mechanical_without_reproduction = all(
        value
        for key, value in gates["mechanical"].items()
        if key not in {"all_pass", "deterministic_core_hash_rerun"}
    )
    if not mechanical_without_reproduction:
        status = "invalid_stop_no_checkpoint"
        stop_reason = "A frozen mechanical, alias, or support gate failed."
    elif not reproduction["checked"]:
        status = "awaiting_deterministic_rerun"
        stop_reason = (
            "Run --render-existing in a fresh process to satisfy the frozen "
            "deterministic-core gate."
        )
    elif not gates["mechanical"]["all_pass"]:
        status = "invalid_stop_no_checkpoint"
        stop_reason = "The deterministic-core rerun did not match."
    elif not gates["adequacy"]["all_pass"]:
        status = "fail_stop_inadequate_no_checkpoint"
        stop_reason = "The frozen dataset-adequacy gates failed; stop ridge branch."
    elif not gates["selector"]["all_pass"]:
        status = "fail_stop_selector_no_checkpoint"
        stop_reason = "The frozen selector gates failed; stop ridge branch."
    elif gates["promotion_grade_extra"]["all_pass"]:
        status = "pass_development_only_promotion_grade"
        stop_reason = (
            "Development gates passed; only separately pre-registered disjoint "
            "family-and-seed validation is authorized."
        )
    else:
        status = "pass_selector_only_not_promotion_grade"
        stop_reason = "Selector gates passed but promotion-grade 128/128 evidence did not."

    result = {
        "schema_version": "v86b-alias-family-scale-evaluation-v1",
        "status": status,
        "stop_reason": stop_reason,
        "protocol": {
            "path": str(Path(args.protocol).resolve().relative_to(ROOT)),
            "sha256": protocol_info["sha256"],
            "schema_version": EXPECTED_PROTOCOL_SCHEMA,
        },
        "dataset": {
            "path": str(dataset_path.resolve().relative_to(ROOT)),
            "file_sha256": extracted["dataset_file_sha256"],
            "deterministic_core_sha256": extracted["dataset_core_sha256"],
            "schema_version": EXPECTED_DATASET_SCHEMA,
        },
        "sources": {
            "evaluate_cv.py": {
                "path": str(RUNNER_PATH.relative_to(ROOT)),
                "sha256": sha256(RUNNER_PATH),
            },
            "run_experiment.py": {
                "path": str((OUT / "run_experiment.py").resolve().relative_to(ROOT)),
                "sha256": EXPECTED_COLLECTOR_SHA256,
            },
            "audited_v86a_math_only": {
                "path": str(protocol_info["math_reference_path"].relative_to(ROOT)),
                "sha256": EXPECTED_MATH_REFERENCE_SHA256,
                "v86a_dataset_or_results_opened": False,
            },
        },
        "frozen_model": protocol["model"],
        "alias_accounting": alias_accounting,
        "cv": cv_public,
        "gates": gates,
        "determinism": {
            "dataset_core_sha256": extracted["dataset_core_sha256"],
            "evaluation_core_sha256": evaluation_core,
            "reproduction": reproduction,
        },
        "authorization": {
            "checkpoint": False,
            "runtime_agent": False,
            "sealed_holdout": False,
            "submission": False,
            "kaggle_upload": False,
            "next_if_full_pass": (
                "one separately pre-registered disjoint family-and-seed validation only"
            ),
        },
    }
    return result, existing


def pass_mark(value: Any) -> str:
    return "PASS" if value is True else "FAIL"


def build_report(result: dict[str, Any], results_sha256: str) -> str:
    gates = result["gates"]
    determinism = result["determinism"]
    reproduction = determinism["reproduction"]
    aggregate = result["cv"]["aggregate"]
    aliases = result["alias_accounting"]
    lines = [
        "# V86-B Alias-Permissive Route-Bandit Evaluation",
        "",
        f"Status: **{result['status']}**",
        "",
        result["stop_reason"],
        "",
        "This is development evidence only. It authorizes no checkpoint, runtime "
        "agent, sealed-holdout opening, submission, or Kaggle upload.",
        "",
        "## Frozen inputs",
        "",
        f"- Protocol SHA-256: `{result['protocol']['sha256']}`",
        f"- Dataset SHA-256: `{result['dataset']['file_sha256']}`",
        f"- Dataset deterministic core: `{determinism['dataset_core_sha256']}`",
        f"- Evaluation deterministic core: `{determinism['evaluation_core_sha256']}`",
        f"- Evaluator SHA-256: `{result['sources']['evaluate_cv.py']['sha256']}`",
        f"- Results SHA-256: `{results_sha256}`",
        "- Model: fixed dual ridge, alpha=1000, beta=2, threshold=2500; no hyperparameter search.",
        "",
        "## Determinism",
        "",
        f"- Checked: `{reproduction['checked']}`",
        f"- Dataset core match: `{reproduction['dataset_core_match']}`",
        f"- Evaluation core match: `{reproduction['evaluation_core_match']}`",
        f"- Verified: **{pass_mark(reproduction['verified'])}**",
        f"- Method: {reproduction['method']}",
        "",
        "## Alias and executed-support accounting",
        "",
        "All 128 rows remain in every denominator. Choosing a forced alias remains "
        "an intervention with observed paired delta exactly zero, so it fails the "
        "strictly-positive intervention gate.",
        "",
        "| Action | Alias rows | Executed-support rows | Selected alias interventions |",
        "|---|---:|---:|---:|",
    ]
    for action_id in ACTIONS[1:]:
        lines.append(
            f"| `{action_id}` | {aliases['observed_alias_rows_by_action'][action_id]} "
            f"| {aliases['executed_support_rows_by_action'][action_id]} "
            f"| {aliases['selected_alias_interventions_by_action'][action_id]} |"
        )
    lines.extend(
        [
            "",
            "## Selected policy",
            "",
            f"- Wins: `{aggregate['wins']}/128`",
            f"- Worst margin: `{aggregate['worst_margin']:.6f}`",
            f"- Mean margin: `{aggregate['mean_margin']:.6f}`",
            f"- Mean paired delta: `{aggregate['mean_paired_delta']:.6f}`",
            "- Action counts: "
            + ", ".join(
                f"`{action}`={count}" for action, count in result["cv"]["action_counts"].items()
            ),
            "",
            "## Gate summary",
            "",
            "| Gate family | Verdict |",
            "|---|---|",
            f"| Mechanical | **{pass_mark(gates['mechanical']['all_pass'])}** |",
            f"| Adequacy | **{pass_mark(gates['adequacy']['all_pass'])}** |",
            f"| Selector | **{pass_mark(gates['selector']['all_pass'])}** |",
            "| Promotion-grade extra | "
            f"**{pass_mark(gates['promotion_grade_extra']['all_pass'])}** |",
            "",
            "### Adequacy evidence",
            "",
            f"- KEEP-loss rows: `{gates['adequacy']['keep_loss_rows']}`",
            "- KEEP-loss family-seed clusters: "
            f"`{gates['adequacy']['keep_loss_family_seed_clusters']}`",
            f"- KEEP-loss families: `{gates['adequacy']['keep_loss_families']}`",
            f"- Oracle rescue fraction: `{gates['adequacy']['oracle_rescue_fraction']:.6f}`",
            f"- Oracle rescue families: `{gates['adequacy']['oracle_rescue_families']}`",
            "",
            "### Selector evidence",
            "",
            f"- Interventions: `{gates['selector']['interventions']}`",
            f"- KEEP-loss rescue fraction: `{gates['selector']['keep_loss_rescue_fraction']:.6f}`",
            "- Rescued family-seed clusters: "
            f"`{gates['selector']['rescued_family_seed_clusters']}`",
            "- Strict positive actual delta for every intervention: "
            f"`{gates['selector']['every_intervention_actual_paired_delta_strictly_positive_pass']}`",
            "",
            "## Per-family no-regression evidence",
            "",
            "| Family | Selected wins | KEEP wins | Selected worst | KEEP worst | "
            "Selected mean | KEEP mean | No regression |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in gates["selector"]["per_family"]:
        no_regression = row["wins_not_worse"] and row["worst_not_worse"] and row["mean_not_worse"]
        lines.append(
            f"| `{row['family']}` | {row['selected_wins']} | {row['keep_wins']} "
            f"| {row['selected_worst_margin']:.3f} | {row['keep_worst_margin']:.3f} "
            f"| {row['selected_mean_margin']:.3f} | {row['keep_mean_margin']:.3f} "
            f"| {pass_mark(no_regression)} |"
        )
    lines.extend(
        [
            "",
            "## Stop condition",
            "",
            result["stop_reason"],
            "",
            "No post-hoc threshold, model, action, family, seed, alias, weighting, "
            "or gate change is authorized by this result.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test(protocol: dict[str, Any], protocol_info: dict[str, Any]) -> None:
    collector_path = OUT / "run_experiment.py"
    if sha256(collector_path) != EXPECTED_COLLECTOR_SHA256:
        raise AssertionError("frozen V86-B collector hash changed")
    reference = load_math_reference(protocol_info)
    reference.self_test()
    synthetic = {
        "absolute_mean": np.asarray(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "absolute_sigma": np.zeros((3, 3), dtype=np.float64),
        "delta_mean": np.asarray(
            [[2500.0, 2500.0], [2501.0, 2501.0], [3000.0, 3000.0]],
            dtype=np.float64,
        ),
        "delta_sigma": np.zeros((3, 2), dtype=np.float64),
    }
    choices, _ = reference.select_actions(synthetic, beta=2.0, threshold=2500.0)
    if choices.tolist() != [0, 1, 2]:
        raise AssertionError("strict threshold/absolute/tie-break math self-test failed")
    if protocol["selector_accounting_semantics"] != {
        "selected_forced_alias_counts_as_intervention": True,
        "selected_forced_alias_observed_paired_delta": 0,
        (
            "selected_forced_alias_fails_every_intervention_actual_paired_delta_strictly_positive"
        ): True,
        "alias_must_not_be_reclassified_as_keep_or_abstain": True,
    }:
        raise AssertionError("frozen alias accounting changed")
    print("V86-B evaluator self-test: PASS (exact audited V85/V86-A math parity)")


def main() -> None:
    args = parse_args()
    validate_cli_confinement(args)
    protocol = read_json(args.protocol)
    protocol_info = validate_protocol(protocol, args.protocol)
    if args.self_test:
        self_test(protocol, protocol_info)
        return

    reject_forbidden_v86a_economic_path(args.dataset, "dataset")
    dataset = read_json(args.dataset)
    math_reference = load_math_reference(protocol_info)
    result, existing = build_results(
        dataset,
        args.dataset,
        protocol,
        protocol_info,
        math_reference,
        args,
    )
    if args.render_existing and existing is not None:
        existing_reproduction = existing.get("determinism", {}).get("reproduction", {})
        if existing_reproduction.get("verified") is True and result != existing:
            raise AssertionError("render-existing result is not byte-deterministic")
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
