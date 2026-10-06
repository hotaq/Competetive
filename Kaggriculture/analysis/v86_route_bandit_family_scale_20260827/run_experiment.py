#!/usr/bin/env python3
"""Collect the frozen V86-A family-scale route-bandit development dataset.

This runner consumes ``protocol.json`` and executes only the three pre-registered
V67 route actions.  It never fits a selector, reads a sealed holdout, creates a
checkpoint, or prepares a Kaggle submission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import kaggriculture_rs  # noqa: E402
from kaggriculture_rs import Engine  # noqa: E402

from kaggriculture_rl.learning.tape_selector import (  # noqa: E402
    encode_selector_features,
)

RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = OUT / "protocol.json"
DEFAULT_OUTPUT_PATH = OUT / "dataset.json"
V84_RUNNER_PATH = ROOT / "analysis/v84_route_option_onpolicy_oracle_20260827/run_oracle.py"
V85_COLLECTOR_PATH = ROOT / "analysis/v85_route_bandit_cv_20260827/collect_features.py"
ENCODER_PATH = ROOT / "src/kaggriculture_rl/learning/tape_selector.py"
HELPER_PATH = ROOT / "scripts/evaluate_file_agent_vs_replay.py"
FAMILY_SELECTION_PATH = OUT / "family_selection.json"
FAMILY_SELECTOR_PATH = OUT / "select_families.py"
PRE_REGISTRATION_PATH = OUT / "PRE_REGISTRATION.md"
FAMILY_SELECTION_SUPPLEMENT_PATH = OUT / "FREEZE_SUPPLEMENT_FAMILY_SELECTION.md"
CONTRACT_FREEZE_RECEIPT_PATH = OUT / "CONTRACT_FREEZE_RECEIPT_V2.md"

EXPECTED_PROTOCOL_SHA256 = "007e36ed80bc2f08cb1cd2495db00ce4bd5132804f761550bae0b72326ef3b30"
EXPECTED_V84_RUNNER_SHA256 = "9fc45c32923aeba7891a04a2d4e09e4f6b159266b1c7a81ecb42761deb5081a6"
EXPECTED_V85_COLLECTOR_SHA256 = "a5d69bd1c447f7b0f5a5eebf0020a71163778e66dc5bedd9b999771e3d1749ef"
EXPECTED_ENCODER_SHA256 = "71172b8a853d2e63798e20c6e2cd7b64763d79d4c11e9ba01822669a0382c31f"
EXPECTED_HELPER_SHA256 = "03fcb73cd657e09b4c74498e71ad093813d16c8dad669aeb78f4013b2ed9966f"
EXPECTED_RUST_PACKAGE_SHA256 = "672f7aa733ccdf6146bb001b8f465ffbb9a94d5e1bfed3f4041ff0e8a5688adb"
EXPECTED_RUST_NATIVE_SHA256 = "315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd"
EXPECTED_FAMILY_SELECTION_SHA256 = (
    "9e6bd7adbcfcb92ec51e798646db4be746459ca0b006400ab7aafc848ed67b38"
)
EXPECTED_FAMILY_SELECTOR_SHA256 = "79cb474b7a266207d557c69dd2d30a74bf3ac9199fd248dc72d26acf7c0c226f"
EXPECTED_PRE_REGISTRATION_SHA256 = (
    "875791c2811ffcb1f464b8daeb8959c0626d1341cbed5a8763ccd16cbe4aea6c"
)
EXPECTED_FAMILY_SELECTION_SUPPLEMENT_SHA256 = (
    "16e1826923234a049cff74631daba878723050f4878174926a45c39502724060"
)
EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256 = (
    "4a5b62e82d0ed671d27095db417c49ae53b0ab917653206a0f2893057080d11c"
)
EXPECTED_CORE_SUBMISSION_IDS = (55714246, 55734995, 55774239, 55742738)
EXPECTED_BACKUP_SUBMISSION_IDS = (55616096, 55617399)
BRANCH_STEP = 120
ACTION_IDS = ("KEEP_V67", "FORCE_B120_SUFFIX", "FORCE_C120_SUFFIX")
V84_OPTIONS = {
    "KEEP_V67": "ABSTAIN_KEEP_V67",
    "FORCE_B120_SUFFIX": "FORCE_B120_SUFFIX",
    "FORCE_C120_SUFFIX": "FORCE_C120_SUFFIX",
}
MODEL_ACTIONS = {
    "KEEP_V67": "KEEP",
    "FORCE_B120_SUFFIX": "B",
    "FORCE_C120_SUFFIX": "C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel context workers.")
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=0,
        help="Run only the first N contexts for a mechanical smoke test (0 = all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Single dataset*.json destination inside the frozen V86-A analysis directory.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.limit_contexts < 0:
        parser.error("--limit-contexts must be non-negative")
    return args


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != OUT.resolve():
        raise ValueError(f"--output must stay directly inside {OUT}")
    if resolved.suffix != ".json" or not resolved.name.startswith("dataset"):
        raise ValueError("--output filename must match dataset*.json")
    return resolved


def load_python_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_exact_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"pinned hash mismatch for {label}: {actual} != {expected}")
    return actual


def _resolved_protocol_path(path_text: str) -> Path:
    path = (ROOT / path_text).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(f"protocol path escapes repository: {path_text}") from error
    return path


def verify_family_selection(protocol: dict[str, Any]) -> dict[str, Any]:
    selection_hash = require_exact_hash(
        FAMILY_SELECTION_PATH,
        EXPECTED_FAMILY_SELECTION_SHA256,
        "family_selection.json",
    )
    selector_hash = require_exact_hash(
        FAMILY_SELECTOR_PATH,
        EXPECTED_FAMILY_SELECTOR_SHA256,
        "select_families.py",
    )
    pre_registration_hash = require_exact_hash(
        PRE_REGISTRATION_PATH,
        EXPECTED_PRE_REGISTRATION_SHA256,
        "PRE_REGISTRATION.md",
    )
    supplement_hash = require_exact_hash(
        FAMILY_SELECTION_SUPPLEMENT_PATH,
        EXPECTED_FAMILY_SELECTION_SUPPLEMENT_SHA256,
        "family-selection freeze supplement",
    )
    contract_receipt_hash = require_exact_hash(
        CONTRACT_FREEZE_RECEIPT_PATH,
        EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256,
        "contract freeze receipt v2",
    )
    selection = json.loads(FAMILY_SELECTION_PATH.read_text(encoding="utf-8"))
    assert selection["schema_version"] == 1
    assert selection["status"] == "pre_registered_selection_only"
    assert selection["selection"]["core_submission_ids"] == list(EXPECTED_CORE_SUBMISSION_IDS)
    assert selection["selection"]["backup_submission_ids"] == list(EXPECTED_BACKUP_SUBMISSION_IDS)
    assert selection["guardrails"] == {
        "games_executed": False,
        "preexisting_snapshot_rank_read": True,
        "public_tape_actions_read": True,
        "rewards_read": False,
        "sealed_holdout_opened": False,
        "v67_or_route_margins_read": False,
        "v86_candidate_economic_outcomes_read": False,
    }
    selector_source = selection["sources"]["selector"]
    assert selector_source["path"] == str(FAMILY_SELECTOR_PATH.relative_to(ROOT))
    assert selector_source["sha256"] == selector_hash

    by_submission_id = {int(row["submission_id"]): row for row in selection["families"]}
    assert len(by_submission_id) == len(selection["families"])
    assert set(EXPECTED_CORE_SUBMISSION_IDS).issubset(by_submission_id)
    assert set(EXPECTED_BACKUP_SUBMISSION_IDS).issubset(by_submission_id)

    protocol_top20: dict[int, dict[str, Any]] = {}
    for family in protocol["families"]:
        family_id = str(family["id"])
        if not family_id.startswith("top20_"):
            continue
        submission_id = int(family_id.rsplit("_", 1)[1])
        if submission_id in protocol_top20:
            raise AssertionError(f"duplicate Top20 submission in protocol: {submission_id}")
        protocol_top20[submission_id] = family
    assert tuple(protocol_top20) == EXPECTED_CORE_SUBMISSION_IDS
    for submission_id, family in protocol_top20.items():
        selected = by_submission_id[submission_id]
        medoid = selected["medoid"]
        assert int(selected["farthest_first"]["order"]) <= len(EXPECTED_CORE_SUBMISSION_IDS)
        assert family["path"] == medoid["agent_file"]
        assert family["sha256"] == medoid["agent_sha256"]
        assert family["source_group"] == f"submission-{submission_id}"

    backups = protocol["mechanical_only_backups_in_order"]
    assert len(backups) == len(EXPECTED_BACKUP_SUBMISSION_IDS)
    for backup, submission_id in zip(backups, EXPECTED_BACKUP_SUBMISSION_IDS, strict=True):
        selected = by_submission_id[submission_id]
        medoid = selected["medoid"]
        assert str(submission_id) in backup["id"]
        assert backup["path"] == medoid["agent_file"]
        assert backup["sha256"] == medoid["agent_sha256"]

    return {
        "family_selection.json": {
            "path": str(FAMILY_SELECTION_PATH.relative_to(ROOT)),
            "sha256": selection_hash,
            "core_submission_ids": list(EXPECTED_CORE_SUBMISSION_IDS),
            "backup_submission_ids": list(EXPECTED_BACKUP_SUBMISSION_IDS),
        },
        "select_families.py": {
            "path": str(FAMILY_SELECTOR_PATH.relative_to(ROOT)),
            "sha256": selector_hash,
        },
        "PRE_REGISTRATION.md": {
            "path": str(PRE_REGISTRATION_PATH.relative_to(ROOT)),
            "sha256": pre_registration_hash,
        },
        "family_selection_freeze_supplement": {
            "path": str(FAMILY_SELECTION_SUPPLEMENT_PATH.relative_to(ROOT)),
            "sha256": supplement_hash,
        },
        "contract_freeze_receipt_v2": {
            "path": str(CONTRACT_FREEZE_RECEIPT_PATH.relative_to(ROOT)),
            "sha256": contract_receipt_hash,
        },
    }


def load_and_verify_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_hash = require_exact_hash(
        PROTOCOL_PATH,
        EXPECTED_PROTOCOL_SHA256,
        "protocol.json",
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "v86-family-scale-prereg-v1"
    assert protocol["status"] == "frozen_before_economic_outcomes"
    assert tuple(action["id"] for action in protocol["actions"]) == ACTION_IDS
    assert protocol["model"]["actions"] == list(ACTION_IDS)
    assert protocol["observation_steps"] == [24, 48, 72, 96, 120]
    assert protocol["feature_encoder"]["features_per_observation"] == 81
    assert protocol["feature_encoder"]["total_features"] == 405
    assert protocol["feature_encoder"]["model_input_identifiers_forbidden"] is True
    assert protocol["seats"] == [0, 1]
    assert protocol["expected_contexts"] == (
        len(protocol["families"])
        * len(protocol["fresh_development_seeds"])
        * len(protocol["seats"])
    )
    assert protocol["expected_full_games"] == protocol["expected_contexts"] * len(ACTION_IDS)
    assert all(protocol["guardrails"].values())

    source_records: dict[str, Any] = {
        "protocol.json": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": protocol_hash,
        },
        "run_experiment.py": {
            "path": str(RUNNER_PATH.relative_to(ROOT)),
            "sha256": sha256(RUNNER_PATH),
        },
    }
    source_records.update(verify_family_selection(protocol))

    candidate = protocol["candidate"]
    candidate_path = _resolved_protocol_path(candidate["path"])
    candidate_hash = require_exact_hash(candidate_path, candidate["sha256"], "candidate")
    source_records["candidate"] = {
        "name": candidate["name"],
        "path": candidate["path"],
        "sha256": candidate_hash,
    }

    action_records: dict[str, Any] = {}
    for action in protocol["actions"]:
        action_id = action["id"]
        if action_id == "KEEP_V67":
            action_records[action_id] = {
                "path": candidate["path"],
                "sha256": candidate_hash,
            }
            continue
        action_path = _resolved_protocol_path(action["source_path"])
        action_hash = require_exact_hash(
            action_path,
            action["source_sha256"],
            f"action {action_id}",
        )
        action_records[action_id] = {
            "path": action["source_path"],
            "sha256": action_hash,
        }
    source_records["actions"] = action_records

    opponent_records: list[dict[str, Any]] = []
    for family in protocol["families"]:
        family_path = _resolved_protocol_path(family["path"])
        family_hash = require_exact_hash(
            family_path,
            family["sha256"],
            f"family {family['id']}",
        )
        opponent_records.append(
            {
                "family": family["id"],
                "path": family["path"],
                "sha256": family_hash,
            }
        )
    source_records["opponents"] = opponent_records

    backup_records: list[dict[str, Any]] = []
    for backup in protocol["mechanical_only_backups_in_order"]:
        backup_path = _resolved_protocol_path(backup["path"])
        backup_hash = require_exact_hash(
            backup_path,
            backup["sha256"],
            f"backup {backup['id']}",
        )
        backup_records.append({"id": backup["id"], "path": backup["path"], "sha256": backup_hash})
    source_records["mechanical_only_backups"] = backup_records

    rust_package_path = Path(kaggriculture_rs.__file__).resolve()
    rust_native_path = Path(kaggriculture_rs._native.__file__).resolve()
    implementation_pins = (
        ("v84_route_runner", V84_RUNNER_PATH, EXPECTED_V84_RUNNER_SHA256),
        ("v85_feature_collector", V85_COLLECTOR_PATH, EXPECTED_V85_COLLECTOR_SHA256),
        ("feature_encoder", ENCODER_PATH, EXPECTED_ENCODER_SHA256),
        ("evaluation_helper", HELPER_PATH, EXPECTED_HELPER_SHA256),
        ("rust_python_package", rust_package_path, EXPECTED_RUST_PACKAGE_SHA256),
        ("rust_python_extension", rust_native_path, EXPECTED_RUST_NATIVE_SHA256),
    )
    for label, path, expected in implementation_pins:
        actual = require_exact_hash(path, expected, label)
        source_records[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
        }
    return protocol, source_records


def repair_stats() -> dict[str, Any]:
    return {"changed_steps": 0, "step_ids": [], "changed_paths": []}


class GameCapture:
    def __init__(
        self,
        *,
        candidate_seat: int,
        feature_steps: tuple[int, ...],
        collect_features: bool,
        v84_module: ModuleType,
    ) -> None:
        self.candidate_seat = candidate_seat
        self.feature_steps = feature_steps
        self.collect_features = collect_features
        self.v84_module = v84_module
        self.current_player: int | None = None
        self.current_step: int | None = None
        self.snapshots: dict[int, list[float]] = {}
        self.raw_actions: dict[int, list[Any]] = {0: [], 1: []}
        self.proposed_actions: dict[int, list[Any]] = {0: [], 1: []}
        self.repairs: dict[int, dict[str, Any]] = {0: repair_stats(), 1: repair_stats()}

    def observe(self, player: int, observation: dict[str, Any]) -> None:
        step = int(observation["step"])
        self.current_player = player
        self.current_step = step
        if not self.collect_features or player != self.candidate_seat:
            return
        if step not in self.feature_steps:
            return
        encoded = encode_selector_features(observation)
        if encoded.shape != (81,):
            raise AssertionError(f"unexpected encoder shape at step {step}: {encoded.shape}")
        snapshot = [float(value) for value in encoded.tolist()]
        if not all(math.isfinite(value) for value in snapshot):
            raise AssertionError(f"non-finite V85 feature at step {step}")
        previous = self.snapshots.setdefault(step, snapshot)
        if previous != snapshot:
            raise AssertionError(f"non-deterministic duplicate observation at step {step}")

    def record_normalization(self, raw: Any, normalized: Any) -> None:
        if self.current_player is None or self.current_step is None:
            raise AssertionError("action normalization occurred without a current observation")
        player = self.current_player
        raw_copy = copy.deepcopy(raw)
        normalized_copy = copy.deepcopy(normalized)
        self.raw_actions[player].append(raw_copy)
        self.proposed_actions[player].append(normalized_copy)
        if raw_copy == normalized_copy:
            return
        stats = self.repairs[player]
        stats["changed_steps"] += 1
        stats["step_ids"].append(self.current_step)
        changed_paths = self.v84_module.diff_paths(raw_copy, normalized_copy)
        stats["changed_paths"] = sorted(set(stats["changed_paths"]) | set(changed_paths))


class CapturingEngine:
    def __init__(self, configuration: dict[str, Any], capture: GameCapture) -> None:
        self._engine = Engine(configuration)
        self._capture = capture

    @property
    def state(self) -> dict[str, Any]:
        return self._engine.state

    def observation(self, player: int) -> dict[str, Any]:
        observation = self._engine.observation(player)
        self._capture.observe(player, observation)
        return observation

    def sanitize_action(self, player: int, action: dict[str, Any]) -> dict[str, Any]:
        return self._engine.sanitize_action(player, action)

    def step(self, actions: list[dict[str, Any]]) -> None:
        self._engine.step(actions)


def _action_diff(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    v84_module: ModuleType,
) -> dict[str, Any]:
    differences = [
        step
        for step, (reference_action, candidate_action) in enumerate(
            zip(reference, candidate, strict=False)
        )
        if reference_action != candidate_action
    ]
    if len(reference) != len(candidate):
        differences.extend(
            range(min(len(reference), len(candidate)), max(len(reference), len(candidate)))
        )
    first = min(differences) if differences else None
    last = max(differences) if differences else None
    first_paths: list[str] = []
    if first is not None and first < len(reference) and first < len(candidate):
        first_paths = v84_module.diff_paths(reference[first], candidate[first])
    return {
        "different_steps": len(set(differences)),
        "first_difference_step": first,
        "last_difference_step": last,
        "first_difference_paths": first_paths,
        "no_difference_before_120": first is None or first >= BRANCH_STEP,
    }


def _transform_outcome(
    *,
    action_id: str,
    outcome: dict[str, Any],
    capture: GameCapture,
) -> dict[str, Any]:
    candidate_seat = int(outcome["candidate_seat"])
    opponent_seat = int(outcome["opponent_seat"])
    trace = outcome["trace_sha256"]

    candidate_proposed_hash = value_sha256(capture.proposed_actions[candidate_seat])
    opponent_proposed_hash = value_sha256(capture.proposed_actions[opponent_seat])
    if candidate_proposed_hash != trace["candidate_proposed"]:
        raise AssertionError("candidate normalization capture hash drifted from V84 runner")
    if opponent_proposed_hash != trace["opponent_proposed"]:
        raise AssertionError("opponent normalization capture hash drifted from V84 runner")

    exception_events = int(outcome["exception_events"])
    exception_player = capture.current_player if exception_events else None
    candidate_exception_events = int(exception_player == candidate_seat) * exception_events
    opponent_exception_events = int(exception_player == opponent_seat) * exception_events

    return {
        "action": action_id,
        "candidate_seat": candidate_seat,
        "opponent_seat": opponent_seat,
        "candidate_money": float(outcome["candidate_money"]),
        "opponent_money": float(outcome["opponent_money"]),
        "margin": float(outcome["candidate_margin"]),
        "completed_transitions": int(outcome["completed_transitions"]),
        "final_step": int(outcome["final_step"]),
        "candidate": {
            "raw_sha256": value_sha256(capture.raw_actions[candidate_seat]),
            "proposed_sha256": candidate_proposed_hash,
            "executed_sha256": trace["candidate_sanitized"],
            "prefix_0_119_proposed_sha256": trace["candidate_prefix_0_119_proposed"],
            "repair": capture.repairs[candidate_seat],
            "sanitizer": outcome["candidate_sanitizer"],
            "fallback_events": int(outcome["fallback_events"]),
            "exception_events": candidate_exception_events,
            "partial_action_events": int(outcome["partial_option_events"]),
            "exception": outcome["exception"] if candidate_exception_events else None,
        },
        "opponent": {
            "raw_sha256": value_sha256(capture.raw_actions[opponent_seat]),
            "proposed_sha256": opponent_proposed_hash,
            "executed_sha256": trace["opponent_sanitized"],
            "repair": capture.repairs[opponent_seat],
            "sanitizer": outcome["opponent_sanitizer"],
            "fallback_events": 0,
            "exception_events": opponent_exception_events,
            "exception": outcome["exception"] if opponent_exception_events else None,
        },
        "route": {
            "starts": int(outcome["option_starts"]),
            "completions": int(outcome["option_completions"]),
            "selected_route": outcome["selected_route"],
        },
    }


def play_action(
    *,
    family: dict[str, Any],
    seed: int,
    candidate_seat: int,
    action_id: str,
    feature_steps: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float] | None]:
    module_name = f"_v86_v84_{family['id']}_{seed}_{candidate_seat}_{action_id}"
    v84_module = load_python_module(V84_RUNNER_PATH, module_name)
    capture = GameCapture(
        candidate_seat=candidate_seat,
        feature_steps=feature_steps,
        collect_features=action_id == "KEEP_V67",
        v84_module=v84_module,
    )
    original_engine = v84_module.Engine
    original_normalize = v84_module.normalize_legacy_action

    def engine_factory(configuration: dict[str, Any]) -> CapturingEngine:
        return CapturingEngine(configuration, capture)

    def tracked_normalize(action: Any) -> dict[str, Any]:
        raw = copy.deepcopy(action)
        normalized = original_normalize(action)
        capture.record_normalization(raw, normalized)
        return normalized

    v84_module.Engine = engine_factory
    v84_module.normalize_legacy_action = tracked_normalize
    try:
        outcome, candidate_proposed = v84_module.play_option(
            family=family["id"],
            representative=family["representative"],
            proxy_path=_resolved_protocol_path(family["path"]),
            seed=seed,
            candidate_seat=candidate_seat,
            option=V84_OPTIONS[action_id],
        )
    finally:
        v84_module.Engine = original_engine
        v84_module.normalize_legacy_action = original_normalize

    game = _transform_outcome(action_id=action_id, outcome=outcome, capture=capture)
    features: list[float] | None = None
    if action_id == "KEEP_V67":
        if tuple(sorted(capture.snapshots)) != feature_steps:
            raise AssertionError(
                f"missing KEEP feature snapshots: {sorted(capture.snapshots)} != {feature_steps}"
            )
        features = [value for step in feature_steps for value in capture.snapshots[step]]
        if len(features) != 405 or not all(math.isfinite(value) for value in features):
            raise AssertionError("KEEP feature vector must contain 405 finite values")
    return game, candidate_proposed, features


def run_context(task: dict[str, Any]) -> dict[str, Any]:
    family = task["family"]
    seed = int(task["seed"])
    candidate_seat = int(task["candidate_seat"])
    feature_steps = tuple(int(step) for step in task["feature_steps"])

    games: dict[str, dict[str, Any]] = {}
    proposed: dict[str, list[dict[str, Any]]] = {}
    features: list[float] | None = None
    for action_id in ACTION_IDS:
        game, action_trace, keep_features = play_action(
            family=family,
            seed=seed,
            candidate_seat=candidate_seat,
            action_id=action_id,
            feature_steps=feature_steps,
        )
        games[action_id] = game
        proposed[action_id] = action_trace
        if keep_features is not None:
            features = keep_features
    if features is None:
        raise AssertionError("KEEP trajectory did not produce model features")

    keep_prefix = games["KEEP_V67"]["candidate"]["prefix_0_119_proposed_sha256"]
    prefix_parity: dict[str, dict[str, Any]] = {}
    for action_id in ACTION_IDS:
        candidate_prefix = games[action_id]["candidate"]["prefix_0_119_proposed_sha256"]
        prefix_parity[action_id] = {
            "reference_prefix_sha256": keep_prefix,
            "candidate_prefix_sha256": candidate_prefix,
            "equal": (
                proposed[action_id][:BRANCH_STEP] == proposed["KEEP_V67"][:BRANCH_STEP]
                and candidate_prefix == keep_prefix
            ),
        }

    diff_module = load_python_module(
        V84_RUNNER_PATH,
        f"_v86_diff_{family['id']}_{seed}_{candidate_seat}",
    )
    difference_contract = {
        action_id: _action_diff(proposed["KEEP_V67"], proposed[action_id], diff_module)
        for action_id in ACTION_IDS[1:]
    }
    action_margins = {action_id: games[action_id]["margin"] for action_id in ACTION_IDS}
    model_margins = {
        MODEL_ACTIONS[action_id]: margin for action_id, margin in action_margins.items()
    }
    keep_margin = action_margins["KEEP_V67"]

    return {
        "audit": {
            "key": {
                "family": family["id"],
                "seed": seed,
                "candidate_seat": candidate_seat,
            },
            "family": family["id"],
            "representative": family["representative"],
            "source_group": family["source_group"],
            "seed": seed,
            "candidate_seat": candidate_seat,
            "opponent_seat": 1 - candidate_seat,
            "observed_steps": list(feature_steps),
            "feature_count": len(features),
            "keep_prefix_sha256": keep_prefix,
            "prefix_parity": prefix_parity,
            "difference_contract": difference_contract,
        },
        "model_input": {"features": features},
        "labels": {
            "model_margins": model_margins,
            "action_margins": action_margins,
            "paired_deltas": {
                "B": action_margins["FORCE_B120_SUFFIX"] - keep_margin,
                "C": action_margins["FORCE_C120_SUFFIX"] - keep_margin,
            },
        },
        "games": games,
    }


def sample_sort_key(sample: dict[str, Any], family_order: dict[str, int]) -> tuple[int, int, int]:
    key = sample["audit"]["key"]
    return family_order[key["family"]], int(key["seed"]), int(key["candidate_seat"])


def validate_collection(
    *,
    protocol: dict[str, Any],
    samples: list[dict[str, Any]],
    full_run: bool,
) -> dict[str, Any]:
    unique_keys = {
        (
            sample["audit"]["family"],
            int(sample["audit"]["seed"]),
            int(sample["audit"]["candidate_seat"]),
        )
        for sample in samples
    }
    all_games = [sample["games"][action_id] for sample in samples for action_id in ACTION_IDS]
    features = [sample["model_input"]["features"] for sample in samples]
    prefix_checks = [
        parity for sample in samples for parity in sample["audit"]["prefix_parity"].values()
    ]
    route_differences = [
        contract
        for sample in samples
        for contract in sample["audit"]["difference_contract"].values()
    ]

    candidate_sanitizer_changes = sum(
        int(game["candidate"]["sanitizer"]["changed_steps"]) for game in all_games
    )
    candidate_fallbacks = sum(int(game["candidate"]["fallback_events"]) for game in all_games)
    candidate_exceptions = sum(int(game["candidate"]["exception_events"]) for game in all_games)
    candidate_partials = sum(int(game["candidate"]["partial_action_events"]) for game in all_games)
    opponent_repairs = sum(int(game["opponent"]["repair"]["changed_steps"]) for game in all_games)
    opponent_sanitizer_changes = sum(
        int(game["opponent"]["sanitizer"]["changed_steps"]) for game in all_games
    )
    expected_subset_games = len(samples) * len(ACTION_IDS)

    result = {
        "context_count": len(samples),
        "game_count": len(all_games),
        "unique_key_count": len(unique_keys),
        "expected_subset_games": expected_subset_games,
        "full_grid_expected_contexts": int(protocol["expected_contexts"]),
        "full_grid_expected_games": int(protocol["expected_full_games"]),
        "full_grid_complete": (
            full_run
            and len(samples) == int(protocol["expected_contexts"])
            and len(all_games) == int(protocol["expected_full_games"])
        ),
        "all_games_final_step_719": all(int(game["final_step"]) == 719 for game in all_games),
        "all_games_completed_transitions_719": all(
            int(game["completed_transitions"]) == 719 for game in all_games
        ),
        "candidate_prefix_parity_checks": len(prefix_checks),
        "candidate_prefix_parity_all": all(bool(check["equal"]) for check in prefix_checks),
        "non_keep_first_difference_minimum_step_120": all(
            contract["first_difference_step"] is not None
            and int(contract["first_difference_step"]) >= BRANCH_STEP
            and bool(contract["no_difference_before_120"])
            for contract in route_differences
        ),
        "candidate_sanitizer_changed_steps": candidate_sanitizer_changes,
        "candidate_fallback_events": candidate_fallbacks,
        "candidate_exception_events": candidate_exceptions,
        "candidate_partial_action_events": candidate_partials,
        "opponent_repair_changed_steps": opponent_repairs,
        "opponent_sanitizer_changed_steps": opponent_sanitizer_changes,
        "finite_feature_shape": [
            len(features),
            len(features[0]) if features else 0,
        ],
        "all_features_finite": all(
            len(row) == 405 and all(type(value) is float and math.isfinite(value) for value in row)
            for row in features
        ),
        "model_input_fields": sorted(
            {field for sample in samples for field in sample["model_input"]}
        ),
        "identifiers_separated_from_model_input": all(
            set(sample["model_input"]) == {"features"} for sample in samples
        ),
        "labels_separated_from_model_input": all(
            "labels" in sample and "labels" not in sample["model_input"] for sample in samples
        ),
    }
    result["mechanical_pass_for_processed_subset"] = (
        len(unique_keys) == len(samples)
        and len(all_games) == expected_subset_games
        and result["all_games_final_step_719"]
        and result["all_games_completed_transitions_719"]
        and result["candidate_prefix_parity_all"]
        and result["non_keep_first_difference_minimum_step_120"]
        and candidate_sanitizer_changes == 0
        and candidate_fallbacks == 0
        and candidate_exceptions == 0
        and candidate_partials == 0
        and result["finite_feature_shape"] == [len(samples), 405]
        and result["all_features_finite"]
        and result["identifiers_separated_from_model_input"]
        and result["labels_separated_from_model_input"]
    )
    return result


def build_tasks(protocol: dict[str, Any], limit_contexts: int) -> list[dict[str, Any]]:
    tasks = [
        {
            "family": family,
            "seed": int(seed),
            "candidate_seat": int(seat),
            "feature_steps": list(protocol["observation_steps"]),
        }
        for family in protocol["families"]
        for seed in protocol["fresh_development_seeds"]
        for seat in protocol["seats"]
    ]
    if limit_contexts:
        tasks = tasks[:limit_contexts]
    return tasks


def run_tasks(tasks: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    if jobs == 1:
        samples = []
        for index, task in enumerate(tasks, start=1):
            samples.append(run_context(task))
            print(f"completed context {index}/{len(tasks)}", flush=True)
        return samples

    samples = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_context, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            samples.append(future.result())
            if index == 1 or index == len(tasks) or index % 8 == 0:
                print(f"completed context {index}/{len(tasks)}", flush=True)
    return samples


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output)
    protocol, pinned_sources = load_and_verify_protocol()
    tasks = build_tasks(protocol, args.limit_contexts)
    if not tasks:
        raise AssertionError("no contexts selected")
    full_run = args.limit_contexts == 0

    samples = run_tasks(tasks, args.jobs)
    family_order = {family["id"]: index for index, family in enumerate(protocol["families"])}
    samples.sort(key=lambda sample: sample_sort_key(sample, family_order))
    validation = validate_collection(protocol=protocol, samples=samples, full_run=full_run)

    if sha256(RUNNER_PATH) != pinned_sources["run_experiment.py"]["sha256"]:
        raise AssertionError("run_experiment.py changed while games were running")
    sources = pinned_sources
    payload: dict[str, Any] = {
        "schema_version": "v86-family-scale-collection-v1",
        "status": "collection_complete" if full_run else "smoke_complete",
        "protocol": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "schema_version": protocol["schema_version"],
            "status": protocol["status"],
        },
        "sources": sources,
        "execution": {
            "requested_jobs": args.jobs,
            "limit_contexts": args.limit_contexts,
            "selected_contexts": len(tasks),
            "selected_full_games": len(tasks) * len(ACTION_IDS),
            "protocol_expected_contexts": protocol["expected_contexts"],
            "protocol_expected_full_games": protocol["expected_full_games"],
            "fresh_load_per_game": True,
        },
        "samples": samples,
        "validation": validation,
    }
    payload["core_sha256"] = value_sha256(
        {
            "schema_version": payload["schema_version"],
            "protocol_sha256": payload["protocol"]["sha256"],
            "samples": payload["samples"],
            "validation": payload["validation"],
        }
    )
    write_json(output_path, payload)
    print(f"wrote {output_path}")
    print(f"core_sha256={payload['core_sha256']}")
    print(f"mechanical_pass={validation['mechanical_pass_for_processed_subset']}")


if __name__ == "__main__":
    main()
