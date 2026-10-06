#!/usr/bin/env python3
"""Collect the frozen V86-B alias-permissive route-bandit dataset.

The runner executes only the three actions and the fresh development grid in
``protocol.json``.  It never reads a V86-A dataset/result, fits a selector,
opens the sealed holdout, creates a checkpoint, or prepares/uploads a Kaggle
submission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import kaggriculture_rs  # noqa: E402
from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine  # noqa: E402

from kaggriculture_rl.learning.tape_selector import (  # noqa: E402
    encode_selector_features,
)

RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = OUT / "protocol.json"
PRE_REGISTRATION_PATH = OUT / "PRE_REGISTRATION.md"
CONTRACT_FREEZE_RECEIPT_PATH = OUT / "CONTRACT_FREEZE_RECEIPT.md"
DEFAULT_OUTPUT_PATH = OUT / "dataset.json"

EXPECTED_PROTOCOL_SHA256 = "511152132c5fa6f245ee8eabb10f29d60007ddd79a7c3b3d523ba7dfa9ee5731"
EXPECTED_PRE_REGISTRATION_SHA256 = (
    "a1e442116f1c19aacd50647b0df4d9d3ffb08cac11b8e408ba16498601076e3b"
)
EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256 = (
    "d0c1462ac6234590df6416f5eb2355e2f4585e535ebefa00ac246f8a655bafe3"
)

ACTION_IDS = ("KEEP_V67", "FORCE_B120_SUFFIX", "FORCE_C120_SUFFIX")
MODEL_ACTIONS = {
    "KEEP_V67": "KEEP",
    "FORCE_B120_SUFFIX": "B",
    "FORCE_C120_SUFFIX": "C",
}
BRANCH_STEP = 120
FINAL_STEP = 719
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}
FORBIDDEN_V86A_OUTCOME_PATHS = frozenset(
    {
        "analysis/v86_route_bandit_family_scale_20260827/dataset.json",
        "analysis/v86_route_bandit_family_scale_20260827/dataset_smoke.json",
        "analysis/v86_route_bandit_family_scale_20260827/results.json",
    }
)

Json = dict[str, Any]
Agent = Callable[..., dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="number of independent context workers (default: 4)",
    )
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=0,
        help="run only the first N contexts; 0 means the frozen full grid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="direct V86-B dataset*.json output path",
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


def canonical_value(value: Any) -> Any:
    """Return the exact JSON-domain representation used for hashes/output."""

    return json.loads(canonical_bytes(value))


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
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != OUT.resolve():
        raise ValueError(f"--output must stay directly inside {OUT}")
    if resolved.suffix != ".json" or not resolved.name.startswith("dataset"):
        raise ValueError("--output filename must match dataset*.json")
    return resolved


def require_exact_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"pinned hash mismatch for {label}: {actual} != {expected}")
    return actual


def resolve_protocol_path(path_text: str) -> Path:
    if path_text in FORBIDDEN_V86A_OUTCOME_PATHS:
        raise AssertionError(f"forbidden V86-A outcome path requested: {path_text}")
    path = (ROOT / path_text).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(f"protocol path escapes repository: {path_text}") from error
    return path


def verify_protocol() -> tuple[Json, Json]:
    protocol_hash = require_exact_hash(PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "protocol.json")
    pre_hash = require_exact_hash(
        PRE_REGISTRATION_PATH,
        EXPECTED_PRE_REGISTRATION_SHA256,
        "PRE_REGISTRATION.md",
    )
    receipt_hash = require_exact_hash(
        CONTRACT_FREEZE_RECEIPT_PATH,
        EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256,
        "CONTRACT_FREEZE_RECEIPT.md",
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    if protocol["schema_version"] != "v86b-alias-family-scale-prereg-v4":
        raise AssertionError("unexpected V86-B protocol schema")
    if protocol["status"] != "frozen_before_v86b_economic_outcomes":
        raise AssertionError("V86-B protocol is not frozen")
    if tuple(action["id"] for action in protocol["actions"]) != ACTION_IDS:
        raise AssertionError("frozen action order changed")
    if protocol["fresh_development_seeds"] != list(range(27100400, 27100408)):
        raise AssertionError("fresh V86-B seed grid changed")
    if protocol["seats"] != [0, 1]:
        raise AssertionError("seat grid changed")
    if protocol["observation_steps"] != [24, 48, 72, 96, 120]:
        raise AssertionError("observation schedule changed")
    if protocol["expected_contexts"] != 128:
        raise AssertionError("expected context count changed")
    if protocol["expected_full_games"] != 384:
        raise AssertionError("expected game count changed")

    action_by_id = {action["id"]: action for action in protocol["actions"]}
    if action_by_id["FORCE_B120_SUFFIX"]["runtime_route_key"] != "b120":
        raise AssertionError("B action must map to runtime route b120")
    if action_by_id["FORCE_C120_SUFFIX"]["runtime_route_key"] != "c120":
        raise AssertionError("C action must map to runtime route c120")

    source_records: Json = {
        "protocol.json": {
            "path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": protocol_hash,
        },
        "PRE_REGISTRATION.md": {
            "path": str(PRE_REGISTRATION_PATH.relative_to(ROOT)),
            "sha256": pre_hash,
        },
        "CONTRACT_FREEZE_RECEIPT.md": {
            "path": str(CONTRACT_FREEZE_RECEIPT_PATH.relative_to(ROOT)),
            "sha256": receipt_hash,
        },
        "run_experiment.py": {
            "path": str(RUNNER_PATH.relative_to(ROOT)),
            "sha256": sha256(RUNNER_PATH),
        },
    }

    candidate = protocol["candidate"]
    candidate_path = resolve_protocol_path(candidate["path"])
    source_records["candidate"] = {
        "path": candidate["path"],
        "sha256": require_exact_hash(candidate_path, candidate["sha256"], "candidate V67"),
    }

    for action in protocol["actions"]:
        if action["id"] == "KEEP_V67":
            continue
        action_path = resolve_protocol_path(action["source_path"])
        source_records[f"action:{action['id']}"] = {
            "path": action["source_path"],
            "sha256": require_exact_hash(
                action_path,
                action["source_sha256"],
                f"action {action['id']}",
            ),
        }

    if len(protocol["families"]) != 8:
        raise AssertionError("frozen family count changed")
    for family in protocol["families"]:
        family_path = resolve_protocol_path(family["path"])
        source_records[f"family:{family['id']}"] = {
            "path": family["path"],
            "sha256": require_exact_hash(family_path, family["sha256"], f"family {family['id']}"),
        }

    for label, pinned in protocol["pinned_reference_sources"].items():
        pinned_path = resolve_protocol_path(pinned["path"])
        source_records[f"reference:{label}"] = {
            "path": pinned["path"],
            "sha256": require_exact_hash(pinned_path, pinned["sha256"], f"reference {label}"),
        }

    stop_report = protocol["v86a_exclusion_evidence"]["allowed_mechanical_stop_report"]
    stop_path = resolve_protocol_path(stop_report["path"])
    source_records["permitted_v86a_mechanical_stop_report"] = {
        "path": stop_report["path"],
        "sha256": require_exact_hash(
            stop_path,
            stop_report["sha256"],
            "permitted V86-A mechanical stop report",
        ),
        "use": "mechanical-stop evidence only; no economic values parsed",
    }

    package_path = Path(kaggriculture_rs.__file__).resolve()
    native_path = Path(kaggriculture_rs._native.__file__).resolve()
    package_pin = protocol["pinned_reference_sources"]["rust_python_package"]
    native_pin = protocol["pinned_reference_sources"]["rust_python_extension"]
    if package_path != resolve_protocol_path(package_pin["path"]):
        raise AssertionError(f"unexpected Rust package path: {package_path}")
    if native_path != resolve_protocol_path(native_pin["path"]):
        raise AssertionError(f"unexpected Rust extension path: {native_path}")

    return protocol, source_records


def empty_repair_stats() -> Json:
    return {"changed_steps": 0, "step_ids": [], "changed_paths": []}


def diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)):
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            paths.extend(diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        if len(left) != len(right):
            paths.append(f"{prefix}.length")
        return paths
    return [] if left == right else [prefix]


def record_repair(
    stats: Json, step: int, proposed: dict[str, Any], executed: dict[str, Any]
) -> None:
    if proposed == executed:
        return
    stats["changed_steps"] += 1
    stats["step_ids"].append(step)
    stats["changed_paths"].extend(diff_paths(proposed, executed))


class RouteControlledV67:
    """Fresh V67 wrapper with auditable KEEP or one step-120 route request."""

    def __init__(
        self,
        *,
        agent: Agent,
        accepts_configuration: bool,
        embedded_v59: Any,
        action_id: str,
        runtime_route_key: str | None,
    ) -> None:
        self.agent = agent
        self.accepts_configuration = accepts_configuration
        self.embedded_v59 = embedded_v59
        self.action_id = action_id
        self.runtime_route_key = runtime_route_key
        self.forced_injection_steps: list[int] = []
        self.forced_injection_count = 0
        self.organic_selection_steps: list[int] = []
        self.selection_events: list[Json] = []
        self.route_history: list[Json] = []
        self.selected_route: str | None = None
        self.selection_step: int | None = None
        self.route_choice_drift_count = 0
        self.forced_choice_before_120_count = 0

        suffix_keys = set(embedded_v59._SUFFIX_AGENTS)
        if suffix_keys != {"a120", "b120", "c120"}:
            raise AssertionError(f"unexpected V59 suffix catalog: {suffix_keys}")
        if action_id == "KEEP_V67" and runtime_route_key is not None:
            raise AssertionError("KEEP must not request a runtime route")
        if action_id != "KEEP_V67" and runtime_route_key not in suffix_keys:
            raise AssertionError(
                f"forced action {action_id} maps to invalid route {runtime_route_key}"
            )

    def __call__(self, observation: Json, configuration: Json) -> Json:
        step = int(observation["step"])
        player = int(observation["player"])
        choices = self.embedded_v59._CHOICE_BY_PLAYER
        route_before_injection = choices.get(player)

        if step < BRANCH_STEP and route_before_injection is not None:
            self.forced_choice_before_120_count += 1

        injected = False
        if self.runtime_route_key is not None and step == BRANCH_STEP:
            if route_before_injection is not None:
                raise AssertionError(
                    "forced route already selected before step-120 injection: "
                    f"{route_before_injection}"
                )
            choices[player] = self.runtime_route_key
            self.forced_injection_count += 1
            self.forced_injection_steps.append(step)
            injected = True

        route_before_agent = choices.get(player)
        raw_action = run_agent(
            self.agent,
            self.accepts_configuration,
            observation,
            configuration,
        )
        route_after_agent = choices.get(player)

        if route_after_agent != route_before_agent:
            self.selection_events.append(
                {
                    "step": step,
                    "before": route_before_agent,
                    "after": route_after_agent,
                    "forced": False,
                }
            )
            if self.runtime_route_key is None and route_before_agent is None:
                self.organic_selection_steps.append(step)

        if injected:
            self.selection_events.append(
                {
                    "step": step,
                    "before": route_before_injection,
                    "after": route_before_agent,
                    "forced": True,
                }
            )

        if route_after_agent is not None and self.selected_route is None:
            self.selected_route = str(route_after_agent)
            self.selection_step = step
        if (
            self.selected_route is not None
            and self.selection_step is not None
            and step > self.selection_step
            and (
                route_before_injection != self.selected_route
                or route_after_agent != self.selected_route
            )
        ):
            self.route_choice_drift_count += 1

        self.route_history.append(
            {
                "step": step,
                "route_before_injection": route_before_injection,
                "injected": injected,
                "route_before_agent": route_before_agent,
                "route_after_agent": route_after_agent,
            }
        )
        return raw_action

    def lifecycle(self, *, completed: bool) -> Json:
        stable = self.route_choice_drift_count == 0
        route_absent_through_119 = all(
            event["route_before_injection"] is None
            and event["route_before_agent"] is None
            and event["route_after_agent"] is None
            for event in self.route_history
            if event["step"] < BRANCH_STEP
        )
        forced = self.runtime_route_key is not None
        selected_expected_route = self.selected_route == self.runtime_route_key if forced else True
        selection_at_120 = self.selection_step == BRANCH_STEP
        forced_complete = (
            completed
            and forced
            and self.forced_injection_count == 1
            and self.forced_injection_steps == [BRANCH_STEP]
            and selected_expected_route
            and selection_at_120
            and stable
        )
        keep_complete = (
            completed
            and not forced
            and self.forced_injection_count == 0
            and self.organic_selection_steps == [BRANCH_STEP]
            and self.selected_route is not None
            and selection_at_120
            and stable
        )
        return {
            "action_id": self.action_id,
            "requested_runtime_route": self.runtime_route_key,
            "selected_route": self.selected_route,
            "selection_step": self.selection_step,
            "route_absent_through_step_119": route_absent_through_119,
            "forced_injection_count": self.forced_injection_count,
            "forced_injection_steps": self.forced_injection_steps,
            "forced_option_start_count": self.forced_injection_count if forced else 0,
            "forced_option_completion_count": int(forced_complete),
            "organic_route_selection_count": len(self.organic_selection_steps),
            "organic_route_selection_steps": self.organic_selection_steps,
            "forced_choice_before_step_120_count": (self.forced_choice_before_120_count),
            "route_choice_drift_count": self.route_choice_drift_count,
            "selected_route_stable_through_completion": stable and completed,
            "selected_route_matches_request": selected_expected_route,
            "lifecycle_complete": forced_complete if forced else keep_complete,
            "selection_events": self.selection_events,
            "route_history": self.route_history,
        }


def append_step(trace: list[Json], step: int, field: str, value: Any) -> None:
    trace.append({"step": step, field: canonical_value(value)})


def component_length(value: Any) -> int:
    if isinstance(value, dict) and value and all(isinstance(rows, list) for rows in value.values()):
        return sum(len(rows) for rows in value.values())
    return len(value) if isinstance(value, (dict, list, tuple)) else 1


def component_record(value: Any) -> Json:
    length = component_length(value)
    return {"length": length, "sha256": value_sha256(value)}


def build_component_values(game: Json) -> Json:
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


def play_action(
    *,
    protocol: Json,
    family: Json,
    seed: int,
    candidate_seat: int,
    action_id: str,
    feature_steps: tuple[int, ...],
) -> Json:
    action = next(item for item in protocol["actions"] if item["id"] == action_id)
    runtime_route_key = action.get("runtime_route_key")
    candidate_path = resolve_protocol_path(protocol["candidate"]["path"])
    opponent_path = resolve_protocol_path(family["path"])
    _, candidate_agent, candidate_accepts_configuration = load_agent(candidate_path)
    _, opponent_agent, opponent_accepts_configuration = load_agent(opponent_path)
    embedded_v59 = candidate_agent.__globals__.get("v59")
    if embedded_v59 is None:
        raise AssertionError("frozen V67 overlay does not expose embedded v59")
    candidate = RouteControlledV67(
        agent=candidate_agent,
        accepts_configuration=candidate_accepts_configuration,
        embedded_v59=embedded_v59,
        action_id=action_id,
        runtime_route_key=runtime_route_key,
    )

    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    candidate_raw: list[Json] = []
    candidate_proposed: list[Json] = []
    candidate_executed: list[Json] = []
    opponent_raw: list[Json] = []
    opponent_proposed: list[Json] = []
    opponent_executed: list[Json] = []
    observations_by_seat: dict[str, list[Json]] = {"0": [], "1": []}
    states_by_seat: dict[str, list[Json]] = {"0": [], "1": []}
    snapshots: dict[int, list[float]] = {}
    candidate_repairs = empty_repair_stats()
    opponent_repairs = empty_repair_stats()
    exception: Json | None = None
    completed_transitions = 0

    try:
        for step in range(FINAL_STEP):
            actions: list[dict[str, Any]] = []
            for player in (0, 1):
                observation = canonical_value(engine.observation(player))
                state = canonical_value(engine.state)
                append_step(observations_by_seat[str(player)], step, "observation", observation)
                append_step(states_by_seat[str(player)], step, "state", state)

                if player == candidate_seat:
                    if step in feature_steps:
                        encoded = encode_selector_features(observation)
                        if encoded.shape != (81,):
                            raise AssertionError(
                                f"unexpected encoder shape at step {step}: {encoded.shape}"
                            )
                        values = [float(value) for value in encoded.tolist()]
                        if not all(math.isfinite(value) for value in values):
                            raise AssertionError(f"non-finite selector feature at step {step}")
                        previous = snapshots.setdefault(step, values)
                        if previous != values:
                            raise AssertionError(f"duplicate feature snapshot drift at step {step}")
                    raw = candidate(observation, configuration)
                else:
                    raw = run_agent(
                        opponent_agent,
                        opponent_accepts_configuration,
                        observation,
                        configuration,
                    )

                raw_capture = copy.deepcopy(raw)
                proposed = normalize_legacy_action(raw)
                proposed_capture = copy.deepcopy(proposed)
                executed = engine.sanitize_action(player, proposed)
                if player == candidate_seat:
                    append_step(candidate_raw, step, "action", raw_capture)
                    append_step(candidate_proposed, step, "action", proposed_capture)
                    append_step(candidate_executed, step, "action", executed)
                    record_repair(candidate_repairs, step, proposed_capture, executed)
                else:
                    append_step(opponent_raw, step, "action", raw_capture)
                    append_step(opponent_proposed, step, "action", proposed_capture)
                    append_step(opponent_executed, step, "action", executed)
                    record_repair(opponent_repairs, step, proposed_capture, executed)
                actions.append(executed)
            engine.step(actions)
            completed_transitions += 1
    except Exception as error:  # retain exact failing context for a mechanical stop
        exception = {
            "step": int(engine.state.get("step", completed_transitions)),
            "type": type(error).__name__,
            "message": str(error),
        }

    final_state = canonical_value(engine.state)
    final_step = int(final_state["step"])
    completed = final_step == FINAL_STEP and completed_transitions == FINAL_STEP
    farms = final_state["farms"]
    money_by_seat = {str(seat): int(farms[seat]["money"]) for seat in (0, 1)}
    candidate_money = money_by_seat[str(candidate_seat)]
    opponent_money = money_by_seat[str(1 - candidate_seat)]
    feature_snapshot_rows = [
        {"step": step, "values": snapshots[step]} for step in sorted(snapshots)
    ]
    feature_vector = [value for step in feature_steps for value in snapshots.get(step, [])]
    lifecycle = candidate.lifecycle(completed=completed)
    partial_actions = int(not completed or not lifecycle["lifecycle_complete"])

    game: Json = {
        "action_id": action_id,
        "final_step": final_step,
        "completed_transitions": completed_transitions,
        "completed": completed,
        "candidate_seat": candidate_seat,
        "opponent_seat": 1 - candidate_seat,
        "candidate_audit": {
            "sanitizer": candidate_repairs,
            "fallback_events": 0,
            "exception_events": int(exception is not None),
            "partial_action_events": partial_actions,
            "exception": exception,
        },
        "opponent_audit": {"sanitizer": opponent_repairs},
        "route_lifecycle": lifecycle,
        "traces": {
            "candidate": {
                "raw": candidate_raw,
                "proposed": candidate_proposed,
                "executed": candidate_executed,
            },
            "opponent": {
                "raw": opponent_raw,
                "proposed": opponent_proposed,
                "executed": opponent_executed,
            },
        },
        "trajectories": {
            "observations_by_seat": observations_by_seat,
            "states_by_seat": states_by_seat,
        },
        "final": {
            "state": final_state,
            "money_by_seat": money_by_seat,
            "candidate_money": candidate_money,
            "opponent_money": opponent_money,
            "candidate_margin": candidate_money - opponent_money,
        },
        "features": {
            "observation_steps": list(feature_steps),
            "snapshots": feature_snapshot_rows,
            "vector": feature_vector,
            "finite": len(feature_vector) == 405
            and all(math.isfinite(value) for value in feature_vector),
        },
    }
    component_values = build_component_values(game)
    game["component_manifest"] = {
        name: component_record(value) for name, value in component_values.items()
    }
    game["game_core_sha256"] = value_sha256(
        {
            "action_id": game["action_id"],
            "candidate_audit": game["candidate_audit"],
            "opponent_audit": game["opponent_audit"],
            "route_lifecycle": game["route_lifecycle"],
            "components": component_values,
            "feature_vector": feature_vector,
            "final_step": final_step,
            "completed_transitions": completed_transitions,
        }
    )
    return game


def filter_step_rows(rows: list[Json], *, before_step: int) -> list[Json]:
    return [row for row in rows if int(row["step"]) < before_step]


def joint_prefix_values(game: Json) -> Json:
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


def first_step_difference(left: list[Json], right: list[Json]) -> int | None:
    for left_row, right_row in zip(left, right, strict=False):
        if left_row != right_row:
            return min(int(left_row["step"]), int(right_row["step"]))
    if len(left) == len(right):
        return None
    tail = left[len(right) :] or right[len(left) :]
    return int(tail[0]["step"])


def earliest_component_difference(left: Json, right: Json) -> int | None:
    steps: list[int] = []
    list_components = (
        "candidate_raw_action_trace",
        "candidate_proposed_action_trace",
        "candidate_executed_action_trace",
        "opponent_proposed_action_trace",
        "opponent_executed_action_trace",
        "feature_snapshots",
    )
    for name in list_components:
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


def compare_to_keep(keep: Json, forced: Json) -> Json:
    keep_components = build_component_values(keep)
    forced_components = build_component_values(forced)
    component_exact_equal = {
        name: keep_components[name] == forced_components[name] for name in keep_components
    }
    component_hash_equal = {
        name: (keep["component_manifest"][name] == forced["component_manifest"][name])
        for name in keep_components
    }
    if component_exact_equal != component_hash_equal:
        raise AssertionError("exact/hash component equality disagreement")

    keep_executed = keep["traces"]["candidate"]["executed"]
    forced_executed = forced["traces"]["candidate"]["executed"]
    candidate_executed_first_difference = first_step_difference(keep_executed, forced_executed)
    candidate_executed_equal = candidate_executed_first_difference is None
    full_trajectory_equal = all(component_exact_equal.values())
    full_first_difference = earliest_component_difference(keep_components, forced_components)
    keep_route = keep["route_lifecycle"]["selected_route"]
    forced_route = forced["route_lifecycle"]["selected_route"]
    same_route = keep_route == forced_route

    if (
        candidate_executed_first_difference is not None
        and candidate_executed_first_difference < BRANCH_STEP
    ):
        verdict = "BLOCK_EARLY_CANDIDATE_EXECUTED_DIFFERENCE"
    elif same_route and full_trajectory_equal and candidate_executed_equal:
        verdict = "VALID_ALIAS"
    elif same_route:
        verdict = "BLOCK_SAME_ROUTE_TRAJECTORY_DIFFERENT"
    elif candidate_executed_equal:
        verdict = "BLOCK_DIFFERENT_ROUTE_EXECUTED_IDENTICAL"
    elif (
        candidate_executed_first_difference is not None
        and candidate_executed_first_difference >= BRANCH_STEP
    ):
        verdict = "VALID_NON_ALIAS"
    else:
        verdict = "BLOCK_UNCLASSIFIED"

    return {
        "keep_selected_route": keep_route,
        "forced_selected_route": forced_route,
        "same_selected_route": same_route,
        "full_trajectory_exact_equal": full_trajectory_equal,
        "full_trajectory_first_difference_step": full_first_difference,
        "candidate_executed_exact_equal": candidate_executed_equal,
        "candidate_executed_first_difference_step": (candidate_executed_first_difference),
        "candidate_executed_support": verdict == "VALID_NON_ALIAS",
        "component_exact_equal": component_exact_equal,
        "component_hash_equal": component_hash_equal,
        "exact_hash_equality_consistent": True,
        "verdict": verdict,
    }


def compare_joint_prefix(keep: Json, candidate: Json) -> Json:
    keep_prefix = joint_prefix_values(keep)
    candidate_prefix = joint_prefix_values(candidate)
    exact_equal_by_component = {
        name: keep_prefix[name] == candidate_prefix[name] for name in keep_prefix
    }
    hashes = {
        name: {
            "keep_sha256": value_sha256(keep_prefix[name]),
            "candidate_sha256": value_sha256(candidate_prefix[name]),
            "keep_length": component_length(keep_prefix[name]),
            "candidate_length": component_length(candidate_prefix[name]),
        }
        for name in keep_prefix
    }
    hash_equal_by_component = {
        name: pair["keep_sha256"] == pair["candidate_sha256"] for name, pair in hashes.items()
    }
    if exact_equal_by_component != hash_equal_by_component:
        raise AssertionError("joint-prefix exact/hash equality disagreement")
    return {
        "through_step": BRANCH_STEP - 1,
        "exact_equal_by_component": exact_equal_by_component,
        "hash_equal_by_component": hash_equal_by_component,
        "component_hashes": hashes,
        "all_exact_equal": all(exact_equal_by_component.values()),
        "all_hash_equal": all(hash_equal_by_component.values()),
    }


def run_context(task: Json) -> Json:
    protocol = task["protocol"]
    family = task["family"]
    seed = int(task["seed"])
    candidate_seat = int(task["candidate_seat"])
    feature_steps = tuple(int(step) for step in protocol["observation_steps"])
    games = {
        action_id: play_action(
            protocol=protocol,
            family=family,
            seed=seed,
            candidate_seat=candidate_seat,
            action_id=action_id,
            feature_steps=feature_steps,
        )
        for action_id in ACTION_IDS
    }
    keep = games["KEEP_V67"]
    if keep["features"]["observation_steps"] != list(feature_steps):
        raise AssertionError("KEEP feature schedule drift")

    joint_prefix_parity = {
        action_id: compare_joint_prefix(keep, games[action_id]) for action_id in ACTION_IDS[1:]
    }
    alias_contract = {
        action_id: compare_to_keep(keep, games[action_id]) for action_id in ACTION_IDS[1:]
    }
    margins = {
        action_id: int(game["final"]["candidate_margin"]) for action_id, game in games.items()
    }
    paired_deltas = {
        MODEL_ACTIONS[action_id]: margins[action_id] - margins["KEEP_V67"]
        for action_id in ACTION_IDS[1:]
    }
    for action_id, evidence in alias_contract.items():
        if evidence["verdict"] == "VALID_ALIAS" and paired_deltas[MODEL_ACTIONS[action_id]] != 0:
            raise AssertionError("valid alias must retain observed zero paired delta")

    key = {
        "family": family["id"],
        "seed": seed,
        "candidate_seat": candidate_seat,
    }
    sample: Json = {
        "audit": {
            "key": key,
            "family": family["id"],
            "representative": family["representative"],
            "source_group": family["source_group"],
            "seed": seed,
            "candidate_seat": candidate_seat,
            "opponent_seat": 1 - candidate_seat,
            "observed_steps": list(feature_steps),
            "feature_count": len(keep["features"]["vector"]),
            "joint_prefix_parity": joint_prefix_parity,
            "alias_contract": alias_contract,
        },
        "model_input": {"features": keep["features"]["vector"]},
        "labels": {
            "model_margins": {
                MODEL_ACTIONS[action_id]: margin for action_id, margin in margins.items()
            },
            "action_margins": margins,
            "paired_deltas": paired_deltas,
        },
        "games": games,
    }
    sample["sample_core_sha256"] = value_sha256(sample)
    return sample


def sample_sort_key(sample: Json, family_order: dict[str, int]) -> tuple[int, int, int]:
    audit = sample["audit"]
    return (
        family_order[audit["family"]],
        int(audit["seed"]),
        int(audit["candidate_seat"]),
    )


def validate_collection(*, protocol: Json, samples: list[Json], full_run: bool) -> Json:
    games = [game for sample in samples for game in sample["games"].values()]
    unique_keys = {
        (
            sample["audit"]["family"],
            sample["audit"]["seed"],
            sample["audit"]["candidate_seat"],
        )
        for sample in samples
    }
    action_by_id = {action["id"]: action for action in protocol["actions"]}

    route_gate_pass = True
    for game in games:
        route = game["route_lifecycle"]
        action_id = game["action_id"]
        if action_id == "KEEP_V67":
            route_gate_pass &= (
                route["route_absent_through_step_119"]
                and route["forced_injection_count"] == 0
                and route["organic_route_selection_count"] == 1
                and route["organic_route_selection_steps"] == [BRANCH_STEP]
                and route["selection_step"] == BRANCH_STEP
                and route["route_choice_drift_count"] == 0
                and route["lifecycle_complete"]
            )
        else:
            route_gate_pass &= (
                route["requested_runtime_route"] == action_by_id[action_id]["runtime_route_key"]
                and route["selected_route"] == action_by_id[action_id]["runtime_route_key"]
                and route["route_absent_through_step_119"]
                and route["forced_injection_count"] == 1
                and route["forced_injection_steps"] == [BRANCH_STEP]
                and route["forced_option_start_count"] == 1
                and route["forced_option_completion_count"] == 1
                and route["forced_choice_before_step_120_count"] == 0
                and route["route_choice_drift_count"] == 0
                and route["lifecycle_complete"]
            )

    joint_prefix_checks = [
        evidence
        for sample in samples
        for evidence in sample["audit"]["joint_prefix_parity"].values()
    ]
    alias_checks = [
        evidence for sample in samples for evidence in sample["audit"]["alias_contract"].values()
    ]
    support_counts = {action_id: 0 for action_id in ACTION_IDS[1:]}
    support_by_family = {
        action_id: {family["id"]: 0 for family in protocol["families"]}
        for action_id in ACTION_IDS[1:]
    }
    for sample in samples:
        family_id = sample["audit"]["family"]
        for action_id, evidence in sample["audit"]["alias_contract"].items():
            if evidence["candidate_executed_support"]:
                support_counts[action_id] += 1
                support_by_family[action_id][family_id] += 1

    nondegeneracy = {
        "support_definition": "candidate executed-action difference at step >= 120",
        "counts_by_action": support_counts,
        "counts_by_action_and_family": support_by_family,
        "full_grid_global_minimum": 116,
        "full_grid_per_family_minimum": 12,
        "evaluated": full_run,
        "pass": (
            all(count >= 116 for count in support_counts.values())
            and all(
                count >= 12
                for family_counts in support_by_family.values()
                for count in family_counts.values()
            )
            if full_run
            else None
        ),
    }

    validation: Json = {
        "context_count": len(samples),
        "game_count": len(games),
        "unique_key_count": len(unique_keys),
        "expected_subset_games": len(samples) * len(ACTION_IDS),
        "full_grid_expected_contexts": protocol["expected_contexts"],
        "full_grid_expected_games": protocol["expected_full_games"],
        "full_grid_complete": full_run
        and len(samples) == protocol["expected_contexts"]
        and len(games) == protocol["expected_full_games"],
        "all_games_final_step_719": all(game["final_step"] == FINAL_STEP for game in games),
        "all_games_completed_transitions_719": all(
            game["completed_transitions"] == FINAL_STEP for game in games
        ),
        "candidate_sanitizer_changed_steps": sum(
            game["candidate_audit"]["sanitizer"]["changed_steps"] for game in games
        ),
        "candidate_fallback_events": sum(
            game["candidate_audit"]["fallback_events"] for game in games
        ),
        "candidate_exception_events": sum(
            game["candidate_audit"]["exception_events"] for game in games
        ),
        "candidate_partial_action_events": sum(
            game["candidate_audit"]["partial_action_events"] for game in games
        ),
        "opponent_sanitizer_changed_steps": sum(
            game["opponent_audit"]["sanitizer"]["changed_steps"] for game in games
        ),
        "model_input_fields": sorted(
            {field for sample in samples for field in sample["model_input"]}
        ),
        "finite_feature_shape": [
            len(samples),
            len(samples[0]["model_input"]["features"]) if samples else 0,
        ],
        "all_features_finite": all(
            len(sample["model_input"]["features"]) == 405
            and all(math.isfinite(value) for value in sample["model_input"]["features"])
            for sample in samples
        ),
        "route_lifecycle_gate_pass": bool(route_gate_pass),
        "joint_prefix_check_count": len(joint_prefix_checks),
        "joint_prefix_parity_all": all(
            check["all_exact_equal"] and check["all_hash_equal"] for check in joint_prefix_checks
        ),
        "alias_truth_table_check_count": len(alias_checks),
        "alias_truth_table_all_valid": all(
            check["verdict"] in {"VALID_ALIAS", "VALID_NON_ALIAS"} for check in alias_checks
        ),
        "exact_hash_equality_consistent": all(
            check["exact_hash_equality_consistent"] for check in alias_checks
        ),
        "nondegeneracy": nondegeneracy,
    }
    validation["mechanical_pass"] = (
        validation["unique_key_count"] == len(samples)
        and validation["game_count"] == validation["expected_subset_games"]
        and validation["all_games_final_step_719"]
        and validation["all_games_completed_transitions_719"]
        and validation["candidate_sanitizer_changed_steps"] == 0
        and validation["candidate_fallback_events"] == 0
        and validation["candidate_exception_events"] == 0
        and validation["candidate_partial_action_events"] == 0
        and validation["model_input_fields"] == ["features"]
        and validation["finite_feature_shape"] == [len(samples), 405]
        and validation["all_features_finite"]
        and validation["route_lifecycle_gate_pass"]
        and validation["joint_prefix_parity_all"]
        and validation["alias_truth_table_all_valid"]
        and validation["exact_hash_equality_consistent"]
        and (nondegeneracy["pass"] is not False)
    )
    return validation


def build_tasks(protocol: Json, limit_contexts: int) -> list[Json]:
    tasks = [
        {
            "protocol": protocol,
            "family": family,
            "seed": seed,
            "candidate_seat": candidate_seat,
        }
        for family in protocol["families"]
        for seed in protocol["fresh_development_seeds"]
        for candidate_seat in protocol["seats"]
    ]
    if len(tasks) != protocol["expected_contexts"]:
        raise AssertionError("frozen task grid does not match expected_contexts")
    return tasks[:limit_contexts] if limit_contexts else tasks


def run_tasks(tasks: list[Json], jobs: int) -> list[Json]:
    if jobs == 1:
        return [run_context(task) for task in tasks]
    samples: list[Json] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_context, task): task for task in tasks}
        for future in as_completed(futures):
            samples.append(future.result())
    return samples


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output)
    protocol, source_records = verify_protocol()
    tasks = build_tasks(protocol, args.limit_contexts)
    samples = run_tasks(tasks, args.jobs)
    family_order = {family["id"]: index for index, family in enumerate(protocol["families"])}
    samples.sort(key=lambda sample: sample_sort_key(sample, family_order))
    full_run = len(tasks) == protocol["expected_contexts"]
    validation = validate_collection(protocol=protocol, samples=samples, full_run=full_run)
    post_run_protocol, post_run_source_records = verify_protocol()
    post_run_source_drift_audit = {
        "checked_record_count": len(post_run_source_records),
        "expected_record_count": len(source_records),
        "startup_manifest_sha256": value_sha256(source_records),
        "post_run_manifest_sha256": value_sha256(post_run_source_records),
        "source_records_exact_equal": post_run_source_records == source_records,
        "protocol_exact_equal": post_run_protocol == protocol,
    }
    post_run_source_drift_audit["pass"] = (
        post_run_source_drift_audit["checked_record_count"] == 28
        and post_run_source_drift_audit["expected_record_count"] == 28
        and post_run_source_drift_audit["source_records_exact_equal"]
        and post_run_source_drift_audit["protocol_exact_equal"]
    )
    if not post_run_source_drift_audit["pass"]:
        raise AssertionError("pinned source drift detected between collection start and write")
    validation["post_run_source_drift_pass"] = True
    validation["mechanical_pass"] = (
        validation["mechanical_pass"] and validation["post_run_source_drift_pass"]
    )
    core = {
        "schema_version": "v86b-alias-family-scale-collection-v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_records": source_records,
        "post_run_source_drift_audit": post_run_source_drift_audit,
        "samples": samples,
        "collection_validation": validation,
        "selector_accounting_semantics": protocol["selector_accounting_semantics"],
    }
    dataset = {
        **core,
        "status": (
            "mechanical_pass_ready_for_frozen_cv"
            if validation["mechanical_pass"] and full_run
            else "partial_collection_not_authorized_for_cv"
            if validation["mechanical_pass"]
            else "mechanical_stop"
        ),
        "full_run": full_run,
        "deterministic_core_components": [
            "protocol_sha256",
            "source_records",
            "post_run_source_drift_audit",
            "samples",
            "collection_validation",
            "selector_accounting_semantics",
        ],
        "deterministic_core_sha256": value_sha256(core),
        "prohibitions": {
            "v86a_dataset_or_results_read": False,
            "sealed_holdout_opened": False,
            "checkpoint_created": False,
            "runtime_agent_created": False,
            "submission_prepared": False,
            "kaggle_upload_performed": False,
        },
    }
    write_json(output_path, dataset)
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(ROOT)),
                "status": dataset["status"],
                "contexts": len(samples),
                "games": validation["game_count"],
                "deterministic_core_sha256": dataset["deterministic_core_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
