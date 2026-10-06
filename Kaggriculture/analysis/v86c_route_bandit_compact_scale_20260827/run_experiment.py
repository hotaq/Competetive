#!/usr/bin/env python3
"""Collect the frozen V86-C compact route-bandit evidence dataset.

Complete game histories live only inside one worker evaluating one context.
Each worker validates and emits one minified compact JSON record before IPC.
The parent never receives Python object graphs or full game histories.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[2]
ROOT_RESOLVED = ROOT.resolve()
OUT = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = OUT / "protocol.json"
PRE_REGISTRATION_PATH = OUT / "PRE_REGISTRATION.md"
CONTRACT_FREEZE_RECEIPT_PATH = OUT / "CONTRACT_FREEZE_RECEIPT.md"
EVALUATOR_PATH = OUT / "evaluate_cv.py"
DEFAULT_OUTPUT_PATH = OUT / "dataset.json"

# Updated only after the joint collector/evaluator contract is re-frozen.
EXPECTED_PROTOCOL_SHA256 = "PENDING_PROTOCOL_FREEZE"
EXPECTED_PRE_REGISTRATION_SHA256 = (
    "d020fa7f4b1ba8147956862cb76e35f459d94c9ac58731375b91a43b79655728"
)
EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256 = "PENDING_CONTRACT_FREEZE"

ACTION_IDS = ("KEEP_V67", "FORCE_B120_SUFFIX", "FORCE_C120_SUFFIX")
MODEL_ACTIONS = {
    "KEEP_V67": "KEEP",
    "FORCE_B120_SUFFIX": "B",
    "FORCE_C120_SUFFIX": "C",
}
BRANCH_STEP = 120
FINAL_STEP = 719
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}
MAX_CONTEXT_BYTES = 65_536
MAX_SMOKE_BYTES = 131_072
MAX_FULL_BYTES = 10_485_760
EXPECTED_SOURCE_PIN_COUNT = 23
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMPONENT_ORDER = (
    "CANDIDATE_RAW_ACTION",
    "CANDIDATE_PROPOSED_ACTION",
    "CANDIDATE_EXECUTED_ACTION",
    "OPPONENT_PROPOSED_ACTION",
    "OPPONENT_EXECUTED_ACTION",
    "OBSERVATION",
    "ENGINE_STATE",
)

Json = dict[str, Any]
Agent = Callable[..., dict[str, Any]]
Engine: Any = None
encode_selector_features: Any = None
_MODULE_COUNTER = 0


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_value(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def source_pin(path: Path) -> Json:
    canonical = path.resolve(strict=True)
    relative = canonical.relative_to(ROOT_RESOLVED).as_posix()
    return {
        "canonical_path": relative,
        "canonical_byte_length": canonical.stat().st_size,
        "sha256": sha256_file(canonical),
    }


class RepositoryReadAuditor:
    """Default-deny repository read interceptor shared by parent and workers."""

    def __init__(self) -> None:
        self._allowed: set[Path] = set()
        self._observed: set[Path] = set()
        self._denied_count = 0
        self._unknown_count = 0
        self._lock = threading.RLock()

    @staticmethod
    def _read_access(mode: Any, flags: Any) -> bool:
        if isinstance(mode, str):
            return "r" in mode or "+" in mode
        if isinstance(flags, int):
            return not bool(flags & os.O_WRONLY) or bool(flags & os.O_RDWR)
        return True

    @staticmethod
    def _lexical_absolute(raw: str | bytes | os.PathLike[str]) -> Path:
        path = Path(os.fsdecode(raw))
        return path if path.is_absolute() else Path.cwd() / path

    @staticmethod
    def _denied_relative(relative: str) -> bool:
        if relative.startswith("analysis/sealed_candidate_holdout_55797143_metadata_20260827/"):
            return True
        blocked_roots = (
            "analysis/v86_route_bandit_family_scale_20260827/",
            "analysis/v86b_route_bandit_alias_scale_20260827/",
        )
        if not relative.startswith(blocked_roots):
            return False
        if relative == (
            "analysis/v86b_route_bandit_alias_scale_20260827/"
            "RESOURCE_STOP_REPORT.md"
        ):
            return False
        name = Path(relative).name.lower()
        blocked_tokens = (
            "dataset",
            "result",
            "label",
            "margin",
            "reward",
            "score",
            "oracle",
            "rescue",
            "selector",
            "evaluate_cv",
        )
        return any(token in name for token in blocked_tokens)

    def allow_paths(self, paths: Iterable[Path]) -> None:
        allowed: set[Path] = set()
        for lexical in paths:
            absolute = lexical if lexical.is_absolute() else ROOT / lexical
            canonical = absolute.resolve(strict=True)
            try:
                absolute.absolute().relative_to(ROOT_RESOLVED)
                canonical.relative_to(ROOT_RESOLVED)
            except ValueError as error:
                raise AssertionError("repository source path escapes root") from error
            relative = canonical.relative_to(ROOT_RESOLVED).as_posix()
            if self._denied_relative(relative):
                raise AssertionError("denylisted source cannot enter allowlist")
            allowed.add(canonical)
        with self._lock:
            self._allowed.update(allowed)

    def _inspect_path(self, raw: Any) -> None:
        if isinstance(raw, int) or not isinstance(raw, (str, bytes, os.PathLike)):
            return
        lexical = self._lexical_absolute(raw)
        try:
            lexical_in_repo = lexical.absolute().is_relative_to(ROOT_RESOLVED)
            canonical = lexical.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            if lexical.absolute().is_relative_to(ROOT_RESOLVED):
                with self._lock:
                    self._unknown_count += 1
                raise PermissionError("unresolved repository read blocked")
            return
        canonical_in_repo = canonical.is_relative_to(ROOT_RESOLVED)
        if not lexical_in_repo and not canonical_in_repo:
            return
        if lexical_in_repo and not canonical_in_repo:
            with self._lock:
                self._unknown_count += 1
            raise PermissionError("symlink-escaped repository read blocked")
        relative = canonical.relative_to(ROOT_RESOLVED).as_posix()
        if self._denied_relative(relative):
            with self._lock:
                self._denied_count += 1
            raise PermissionError("denylisted repository read blocked")
        with self._lock:
            if canonical not in self._allowed:
                self._unknown_count += 1
                raise PermissionError("non-allowlisted repository read blocked")
            self._observed.add(canonical)

    def hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args:
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if self._read_access(mode, flags):
                self._inspect_path(args[0])
        elif event == "import" and len(args) > 1 and args[1]:
            self._inspect_path(args[1])

    def policy_sha256(self) -> str:
        with self._lock:
            allowed = sorted(
                path.relative_to(ROOT_RESOLVED).as_posix() for path in self._allowed
            )
        return value_sha256(
            {
                "version": "V86C_DEFAULT_DENY_V1",
                "deny_before_allow": True,
                "allowed_repository_paths": allowed,
            }
        )

    def audit_record(self, pins: list[Json], source_drift_count: int) -> Json:
        with self._lock:
            observed = sorted(self._observed)
            denied_count = self._denied_count
            unknown_count = self._unknown_count
        manifests = [source_pin(path) for path in observed]
        expected_paths = {pin["canonical_path"] for pin in pins}
        observed_paths = {manifest["canonical_path"] for manifest in manifests}
        post_run_pass = (
            expected_paths == observed_paths
            and denied_count == 0
            and unknown_count == 0
            and source_drift_count == 0
        )
        return {
            "policy_sha256": self.policy_sha256(),
            "observed_read_manifests": manifests,
            "denied_read_count": denied_count,
            "unknown_read_count": unknown_count,
            "source_drift_count": source_drift_count,
            "post_run_pass": post_run_pass,
        }


AUDITOR = RepositoryReadAuditor()
AUDITOR.allow_paths(
    path
    for path in (
        PROTOCOL_PATH,
        PRE_REGISTRATION_PATH,
        CONTRACT_FREEZE_RECEIPT_PATH,
        RUNNER_PATH,
        EVALUATOR_PATH,
    )
    if path.exists()
)
sys.addaudithook(AUDITOR.hook)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="maximum concurrent raw-JSON worker subprocesses (default: %(default)s)",
    )
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=0,
        help="0 runs the full frozen grid; 1 is the resource smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="direct V86-C dataset*.json output path",
    )
    parser.add_argument("--_worker-task-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.limit_contexts < 0:
        parser.error("--limit-contexts must be non-negative")
    return args


def resolve_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != OUT.resolve():
        raise ValueError("--output must stay directly inside the V86-C directory")
    if resolved.suffix != ".json" or not resolved.name.startswith("dataset"):
        raise ValueError("--output filename must match dataset*.json")
    return resolved


def write_minified(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)


def require_exact_hash(path: Path, expected: str, label: str) -> str:
    if not SHA256_RE.fullmatch(expected):
        raise AssertionError(f"{label} freeze hash is not finalized")
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"pinned {label} hash mismatch")
    return actual


def repository_path(path_text: str) -> Path:
    lexical = ROOT / path_text
    canonical = lexical.resolve(strict=True)
    try:
        lexical.absolute().relative_to(ROOT_RESOLVED)
        canonical.relative_to(ROOT_RESOLVED)
    except ValueError as error:
        raise AssertionError("protocol source path escapes repository") from error
    relative = canonical.relative_to(ROOT_RESOLVED).as_posix()
    if RepositoryReadAuditor._denied_relative(relative):
        raise AssertionError("protocol requested denylisted source")
    return canonical


def protocol_source_expectations(protocol: Json) -> dict[Path, str | None]:
    expected: dict[Path, str | None] = {}

    def add(path: Path, digest: str | None) -> None:
        canonical = path.resolve(strict=True)
        old = expected.get(canonical)
        if old is not None and digest is not None and old != digest:
            raise AssertionError("conflicting source hashes for one canonical path")
        expected[canonical] = old if old is not None else digest

    add(PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256)
    add(PRE_REGISTRATION_PATH, EXPECTED_PRE_REGISTRATION_SHA256)
    add(CONTRACT_FREEZE_RECEIPT_PATH, EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256)
    add(RUNNER_PATH, None)
    add(EVALUATOR_PATH, None)
    add(repository_path(protocol["candidate"]["path"]), protocol["candidate"]["sha256"])
    for action in protocol["actions"]:
        if source_path := action.get("source_path"):
            add(repository_path(source_path), action["source_sha256"])
    for family in protocol["families"]:
        add(repository_path(family["path"]), family["sha256"])
    for pin in protocol["pinned_reference_sources"].values():
        add(repository_path(pin["path"]), pin["sha256"])
    if len(expected) != EXPECTED_SOURCE_PIN_COUNT:
        raise AssertionError("unexpected deduplicated source-pin count")
    return expected


def verify_protocol_shape(protocol: Json) -> None:
    if protocol["schema_version"] != "v86c-compact-family-scale-prereg-v1":
        raise AssertionError("unexpected V86-C protocol schema")
    if tuple(action["id"] for action in protocol["actions"]) != ACTION_IDS:
        raise AssertionError("V86-C action order drift")
    if protocol["fresh_development_seeds"] != list(range(27100500, 27100508)):
        raise AssertionError("V86-C development seed drift")
    if protocol["seats"] != [0, 1]:
        raise AssertionError("V86-C seat grid drift")
    if protocol["observation_steps"] != [24, 48, 72, 96, 120]:
        raise AssertionError("V86-C observation schedule drift")
    if len(protocol["families"]) != 8:
        raise AssertionError("V86-C family count drift")
    if protocol["expected_contexts"] != 128 or protocol["expected_full_games"] != 384:
        raise AssertionError("V86-C grid size drift")
    action_by_id = {action["id"]: action for action in protocol["actions"]}
    if action_by_id["FORCE_B120_SUFFIX"]["runtime_route_key"] != "b120":
        raise AssertionError("B runtime route mapping drift")
    if action_by_id["FORCE_C120_SUFFIX"]["runtime_route_key"] != "c120":
        raise AssertionError("C runtime route mapping drift")


def load_and_pin_sources() -> tuple[Json, list[Json]]:
    require_exact_hash(PROTOCOL_PATH, EXPECTED_PROTOCOL_SHA256, "protocol.json")
    require_exact_hash(
        PRE_REGISTRATION_PATH,
        EXPECTED_PRE_REGISTRATION_SHA256,
        "PRE_REGISTRATION.md",
    )
    require_exact_hash(
        CONTRACT_FREEZE_RECEIPT_PATH,
        EXPECTED_CONTRACT_FREEZE_RECEIPT_SHA256,
        "CONTRACT_FREEZE_RECEIPT.md",
    )
    protocol = cast(Json, json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
    verify_protocol_shape(protocol)
    expectations = protocol_source_expectations(protocol)
    AUDITOR.allow_paths(expectations)
    pins: list[Json] = []
    for path, expected in expectations.items():
        pin = source_pin(path)
        if expected is not None and pin["sha256"] != expected:
            raise AssertionError("pinned source hash mismatch")
        pins.append(pin)
    pins.sort(key=lambda pin: pin["canonical_path"])
    return protocol, pins


def post_run_source_audit(protocol: Json, startup_pins: list[Json]) -> tuple[list[Json], int]:
    expectations = protocol_source_expectations(protocol)
    pins = [source_pin(path) for path in expectations]
    pins.sort(key=lambda pin: pin["canonical_path"])
    drift_count = sum(left != right for left, right in zip(startup_pins, pins, strict=True))
    return pins, drift_count


def load_source_module(path: Path, name: str, *, package: bool = False) -> ModuleType:
    """Execute one exact allowlisted source without consulting repository pyc files."""
    source = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name if package else ""
    if package:
        module.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        spec.submodule_search_locations = [str(path.parent)]
    module.__spec__ = spec
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def prepare_runtime(protocol: Json) -> None:
    global Engine, encode_selector_features
    if Engine is None:
        package_pin = protocol["pinned_reference_sources"]["rust_python_package"]
        package_path = repository_path(package_pin["path"])
        package = load_source_module(package_path, "kaggriculture_rs", package=True)
        Engine = package.Engine
    if encode_selector_features is None:
        encoder_pin = protocol["pinned_reference_sources"]["selector_encoder_source"]
        encoder_path = repository_path(encoder_pin["path"])
        selector_module = load_source_module(
            encoder_path,
            "v86c_frozen_tape_selector",
        )
        encode_selector_features = selector_module.encode_selector_features


def load_agent(path: Path) -> tuple[ModuleType, Agent, bool]:
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    module_name = (
        "v86c_file_agent_"
        + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        + f"_{_MODULE_COUNTER}"
    )
    module = load_source_module(path, module_name)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError("frozen agent source does not export callable agent")
    accepts_configuration = len(inspect.signature(agent).parameters) >= 2
    return module, agent, accepts_configuration


def run_agent(
    agent: Agent,
    accepts_configuration: bool,
    observation: Json,
    configuration: Json,
) -> Json:
    action = (
        agent(observation, configuration)
        if accepts_configuration
        else agent(observation)
    )
    if not isinstance(action, dict):
        raise TypeError("agent return is not a dictionary")
    return action


def normalize_legacy_action(action: Json) -> Json:
    """Match the audited B runner's Kaggle quantity-default normalization."""

    def normalize_unit(command: Any) -> Any:
        if (
            isinstance(command, list)
            and len(command) == 2
            and command[0] in {"PICKUP", "PLACE"}
        ):
            return [*command, 1]
        return command

    return {
        "farmer": normalize_unit(action.get("farmer", ["PASS"])),
        "hands": [normalize_unit(command) for command in action.get("hands", [])],
        "market": list(action.get("market", [])),
    }


class RouteControlledV67:
    """Fresh V67 wrapper with one auditable optional step-120 route request."""

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
        self.forced_injection_count = 0
        self.forced_injection_step: int | None = None
        self.organic_selection_count = 0
        self.organic_selection_step: int | None = None
        self.selected_route: str | None = None
        self.selection_step: int | None = None
        self.route_choice_drift_count = 0
        self.forced_choice_before_120_count = 0
        self.route_absent_through_step_119 = True

        suffix_keys = set(embedded_v59._SUFFIX_AGENTS)
        if suffix_keys != {"a120", "b120", "c120"}:
            raise AssertionError("unexpected V59 suffix catalog")
        if action_id == "KEEP_V67" and runtime_route_key is not None:
            raise AssertionError("KEEP must not request a runtime route")
        if action_id != "KEEP_V67" and runtime_route_key not in suffix_keys:
            raise AssertionError("forced action maps to an invalid runtime route")

    def __call__(self, observation: Json, configuration: Json) -> Json:
        step = int(observation["step"])
        player = int(observation["player"])
        choices = self.embedded_v59._CHOICE_BY_PLAYER
        route_before_injection = choices.get(player)
        if step < BRANCH_STEP and route_before_injection is not None:
            self.forced_choice_before_120_count += 1
            self.route_absent_through_step_119 = False

        if self.runtime_route_key is not None and step == BRANCH_STEP:
            if route_before_injection is not None:
                raise AssertionError("forced route existed before step-120 injection")
            choices[player] = self.runtime_route_key
            self.forced_injection_count += 1
            self.forced_injection_step = step

        route_before_agent = choices.get(player)
        raw_action = run_agent(
            self.agent,
            self.accepts_configuration,
            observation,
            configuration,
        )
        route_after_agent = choices.get(player)
        if step < BRANCH_STEP and any(
            route is not None
            for route in (route_before_injection, route_before_agent, route_after_agent)
        ):
            self.route_absent_through_step_119 = False
        if (
            self.runtime_route_key is None
            and route_before_agent is None
            and route_after_agent is not None
        ):
            self.organic_selection_count += 1
            self.organic_selection_step = step
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
        return raw_action

    def lifecycle(self, *, completed: bool) -> Json:
        forced = self.runtime_route_key is not None
        stable = self.route_choice_drift_count == 0
        selected_expected_route = (
            self.selected_route == self.runtime_route_key if forced else True
        )
        selection_at_120 = self.selection_step == BRANCH_STEP
        forced_complete = (
            completed
            and forced
            and self.forced_injection_count == 1
            and self.forced_injection_step == BRANCH_STEP
            and selected_expected_route
            and selection_at_120
            and stable
        )
        keep_complete = (
            completed
            and not forced
            and self.forced_injection_count == 0
            and self.organic_selection_count == 1
            and self.organic_selection_step == BRANCH_STEP
            and self.selected_route is not None
            and selection_at_120
            and stable
        )
        return {
            "route_request": self.runtime_route_key,
            "selected_route": self.selected_route,
            "selection_step": self.selection_step,
            "route_absent_through_step_119": self.route_absent_through_step_119,
            "forced_injection_count": self.forced_injection_count,
            "forced_injection_step": self.forced_injection_step,
            "forced_option_start_count": self.forced_injection_count if forced else 0,
            "forced_option_completion_count": int(forced_complete),
            "organic_selection_count": self.organic_selection_count,
            "organic_selection_step": self.organic_selection_step,
            "forced_choice_before_step_120_count": self.forced_choice_before_120_count,
            "route_drift_count": self.route_choice_drift_count,
            "stable_through_completion": stable and completed,
            "selected_route_matches_request": selected_expected_route,
            "lifecycle_complete": forced_complete if forced else keep_complete,
        }


def append_step(rows: list[Json], step: int, field: str, value: Any) -> None:
    rows.append({"step": step, field: canonical_value(value)})


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
    candidate_path = repository_path(protocol["candidate"]["path"])
    opponent_path = repository_path(family["path"])
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
    opponent_proposed: list[Json] = []
    opponent_executed: list[Json] = []
    observations_by_seat: dict[str, list[Json]] = {"0": [], "1": []}
    states_by_seat: dict[str, list[Json]] = {"0": [], "1": []}
    snapshots: dict[int, list[float]] = {}
    candidate_sanitizer_changes = 0
    opponent_sanitizer_changes = 0
    candidate_audit_rows: list[Json] = []
    completed_transitions = 0

    for step in range(FINAL_STEP):
        actions: list[Json] = []
        for player in (0, 1):
            observation = canonical_value(engine.observation(player))
            state = canonical_value(engine.state)
            append_step(
                observations_by_seat[str(player)],
                step,
                "observation",
                observation,
            )
            append_step(states_by_seat[str(player)], step, "state", state)
            if player == candidate_seat:
                if step in feature_steps:
                    encoded = encode_selector_features(observation)
                    if encoded.shape != (81,):
                        raise AssertionError("unexpected selector feature shape")
                    values = [float(value) for value in encoded.tolist()]
                    if not all(math.isfinite(value) for value in values):
                        raise AssertionError("non-finite selector feature")
                    snapshots[step] = values
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
            executed = canonical_value(engine.sanitize_action(player, proposed))
            if player == candidate_seat:
                append_step(candidate_raw, step, "action", raw_capture)
                append_step(candidate_proposed, step, "action", proposed_capture)
                append_step(candidate_executed, step, "action", executed)
                changed = proposed_capture != executed
                candidate_sanitizer_changes += int(changed)
                candidate_audit_rows.append(
                    {
                        "step": step,
                        "proposed_sha256": value_sha256(proposed_capture),
                        "executed_sha256": value_sha256(executed),
                        "sanitizer_changed": changed,
                        "fallback": False,
                        "exception": False,
                        "partial": False,
                    }
                )
            else:
                append_step(opponent_proposed, step, "action", proposed_capture)
                append_step(opponent_executed, step, "action", executed)
                opponent_sanitizer_changes += int(proposed_capture != executed)
            actions.append(executed)
        engine.step(actions)
        completed_transitions += 1

    final_state = canonical_value(engine.state)
    final_step = int(final_state["step"])
    completed = final_step == FINAL_STEP
    lifecycle = candidate.lifecycle(completed=completed)
    farms = final_state["farms"]
    money_by_seat = {str(seat): int(farms[seat]["money"]) for seat in (0, 1)}
    candidate_money = money_by_seat[str(candidate_seat)]
    opponent_money = money_by_seat[str(1 - candidate_seat)]
    feature_snapshots = [
        {"step": step, "values_81": snapshots[step]} for step in feature_steps
    ]
    feature_vector = [
        value for step in feature_steps for value in snapshots.get(step, [])
    ]
    if len(feature_vector) != 405 or not all(
        math.isfinite(value) for value in feature_vector
    ):
        raise AssertionError("incomplete or non-finite selector feature vector")
    return {
        "action_id": action_id,
        "candidate_seat": candidate_seat,
        "opponent_seat": 1 - candidate_seat,
        "candidate_execution_audit": {
            "start_step": 0,
            "stop_step_exclusive": FINAL_STEP,
            "audit_row_count": len(candidate_audit_rows),
            "completed_transitions": completed_transitions,
            "sanitizer_changed_step_count": candidate_sanitizer_changes,
            "fallback_count": 0,
            "exception_count": 0,
            "partial_action_count": int(
                not completed or not lifecycle["lifecycle_complete"]
            ),
            "audit_sha256": value_sha256(
                {
                    "domain": "v86c_candidate_execution_audit_v1",
                    "rows": candidate_audit_rows,
                }
            ),
        },
        "opponent_execution_audit": {
            "sanitizer_changes": opponent_sanitizer_changes,
        },
        "route_lifecycle": lifecycle,
        "histories": {
            "candidate_raw": candidate_raw,
            "candidate_proposed": candidate_proposed,
            "candidate_executed": candidate_executed,
            "opponent_proposed": opponent_proposed,
            "opponent_executed": opponent_executed,
            "observations_by_seat": observations_by_seat,
            "states_by_seat": states_by_seat,
        },
        "final_state": final_state,
        "money_by_seat": money_by_seat,
        "candidate_margin": candidate_money - opponent_money,
        "feature_snapshots": feature_snapshots,
        "features_405": feature_vector,
    }


def history_components(game: Json) -> list[tuple[str, int, list[Json], int]]:
    histories = game["histories"]
    candidate_seat = int(game["candidate_seat"])
    opponent_seat = int(game["opponent_seat"])
    components = [
        (
            "CANDIDATE_RAW_ACTION",
            candidate_seat,
            histories["candidate_raw"],
            BRANCH_STEP,
        ),
        (
            "CANDIDATE_PROPOSED_ACTION",
            candidate_seat,
            histories["candidate_proposed"],
            BRANCH_STEP,
        ),
        (
            "CANDIDATE_EXECUTED_ACTION",
            candidate_seat,
            histories["candidate_executed"],
            BRANCH_STEP,
        ),
        (
            "OPPONENT_PROPOSED_ACTION",
            opponent_seat,
            histories["opponent_proposed"],
            BRANCH_STEP,
        ),
        (
            "OPPONENT_EXECUTED_ACTION",
            opponent_seat,
            histories["opponent_executed"],
            BRANCH_STEP,
        ),
    ]
    for component, rows_by_seat in (
        ("OBSERVATION", histories["observations_by_seat"]),
        ("ENGINE_STATE", histories["states_by_seat"]),
    ):
        for seat in (0, 1):
            components.append(
                (component, seat, rows_by_seat[str(seat)], BRANCH_STEP + 1)
            )
    order = {component: index for index, component in enumerate(COMPONENT_ORDER)}
    components.sort(key=lambda item: (order[item[0]], item[1]))
    return components


def compact_manifest(component: str, seat: int, rows: list[Json]) -> Json:
    payload = canonical_bytes(rows)
    return {
        "component": component,
        "seat": seat,
        "row_count": len(rows),
        "canonical_byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def commitment(rows: list[Json]) -> Json:
    payload = canonical_bytes(rows)
    return {
        "row_count": len(rows),
        "canonical_byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def compact_arm(game: Json) -> Json:
    component_manifests: list[Json] = []
    prefix_manifests: list[Json] = []
    for component, seat, rows, prefix_stop in history_components(game):
        if len(rows) != FINAL_STEP:
            raise AssertionError("full component row count is not 719")
        if any(int(row["step"]) != index for index, row in enumerate(rows)):
            raise AssertionError("component step domain is not contiguous 0..718")
        prefix = [row for row in rows if int(row["step"]) < prefix_stop]
        if len(prefix) != prefix_stop:
            raise AssertionError("component prefix row count mismatch")
        component_manifests.append(compact_manifest(component, seat, rows))
        prefix_manifests.append(compact_manifest(component, seat, prefix))
    lifecycle = game["route_lifecycle"]
    audit = game["candidate_execution_audit"]
    return {
        "action_id": game["action_id"],
        "component_manifests": component_manifests,
        "prefix_manifests": prefix_manifests,
        "full_joint_prefix_sha256": value_sha256(prefix_manifests),
        "candidate_execution_audit": {
            "completed_transitions": int(audit["completed_transitions"]),
            "sanitizer_changes": int(audit["sanitizer_changes"]),
            "fallbacks": int(audit["fallbacks"]),
            "exceptions": int(audit["exceptions"]),
            "partial_actions": int(audit["partial_actions"]),
        },
        "route_lifecycle": {
            "route_request": lifecycle["route_request"],
            "selected_route": lifecycle["selected_route"],
            "selection_step": lifecycle["selection_step"],
            "route_absent_through_step_119": bool(
                lifecycle["route_absent_through_step_119"]
            ),
            "forced_injection_count": int(lifecycle["forced_injection_count"]),
            "forced_injection_step": lifecycle["forced_injection_step"],
            "forced_option_start_count": int(
                lifecycle["forced_option_start_count"]
            ),
            "forced_option_completion_count": int(
                lifecycle["forced_option_completion_count"]
            ),
            "organic_selection_count": int(lifecycle["organic_selection_count"]),
            "organic_selection_step": lifecycle["organic_selection_step"],
            "forced_choice_before_step_120_count": int(
                lifecycle["forced_choice_before_step_120_count"]
            ),
            "route_drift_count": int(lifecycle["route_drift_count"]),
            "stable_through_completion": bool(
                lifecycle["stable_through_completion"]
            ),
            "selected_route_matches_request": bool(
                lifecycle["selected_route_matches_request"]
            ),
            "lifecycle_complete": bool(lifecycle["lifecycle_complete"]),
        },
        "final": {
            "final_step": int(game["final_state"]["step"]),
            "final_state_sha256": value_sha256(game["final_state"]),
            "money_by_seat": {
                "0": int(game["money_by_seat"]["0"]),
                "1": int(game["money_by_seat"]["1"]),
            },
            "candidate_margin": int(game["candidate_margin"]),
        },
    }


def first_difference(keep_rows: list[Json], forced_rows: list[Json]) -> Json:
    if len(keep_rows) != FINAL_STEP or len(forced_rows) != FINAL_STEP:
        raise AssertionError("first-difference inputs must both have 719 rows")
    difference_index = next(
        (
            index
            for index, (keep_row, forced_row) in enumerate(
                zip(keep_rows, forced_rows, strict=True)
            )
            if keep_row != forced_row
        ),
        None,
    )
    if difference_index is None:
        keep_payload = canonical_bytes(keep_rows)
        forced_payload = canonical_bytes(forced_rows)
        keep_sha256 = hashlib.sha256(keep_payload).hexdigest()
        forced_sha256 = hashlib.sha256(forced_payload).hexdigest()
        equal = keep_payload == forced_payload
        if not equal:
            raise AssertionError("null first difference disagrees with full bytes")
        return {
            "difference_step": None,
            "classification": None,
            "equal_prefix_commitment": None,
            "keep_row_commitment": None,
            "forced_row_commitment": None,
            "null_full_equality": {
                "keep_row_count": len(keep_rows),
                "forced_row_count": len(forced_rows),
                "keep_canonical_byte_length": len(keep_payload),
                "forced_canonical_byte_length": len(forced_payload),
                "keep_sha256": keep_sha256,
                "forced_sha256": forced_sha256,
                "equal": equal,
            },
        }
    keep_row = keep_rows[difference_index]
    forced_row = forced_rows[difference_index]
    difference_step = int(keep_row["step"])
    if int(forced_row["step"]) != difference_step:
        raise AssertionError("row-d step mismatch")
    keep_row_commitment = commitment([keep_row])
    forced_row_commitment = commitment([forced_row])
    if keep_row_commitment["sha256"] == forced_row_commitment["sha256"]:
        raise AssertionError("differing row commitments unexpectedly match")
    return {
        "difference_step": difference_step,
        "classification": (
            "before_120" if difference_step < BRANCH_STEP else "at_or_after_120"
        ),
        "equal_prefix_commitment": commitment(keep_rows[:difference_index]),
        "keep_row_commitment": keep_row_commitment,
        "forced_row_commitment": forced_row_commitment,
        "null_full_equality": None,
    }


def component_prefix_equalities(keep_arm: Json, forced_arm: Json) -> list[Json]:
    keep = {
        (manifest["component"], manifest["seat"]): manifest
        for manifest in keep_arm["prefix_manifests"]
    }
    forced = {
        (manifest["component"], manifest["seat"]): manifest
        for manifest in forced_arm["prefix_manifests"]
    }
    if keep.keys() != forced.keys():
        raise AssertionError("prefix manifest key mismatch")
    return [
        {
            "component": component,
            "seat": seat,
            "equal": keep[(component, seat)] == forced[(component, seat)],
        }
        for component, seat in keep
    ]


def compare_forced(
    *,
    keep_game: Json,
    forced_game: Json,
    keep_arm: Json,
    forced_arm: Json,
) -> Json:
    keep_histories = keep_game["histories"]
    forced_histories = forced_game["histories"]
    raw_difference = first_difference(
        keep_histories["candidate_raw"],
        forced_histories["candidate_raw"],
    )
    proposed_difference = first_difference(
        keep_histories["candidate_proposed"],
        forced_histories["candidate_proposed"],
    )
    executed_difference = first_difference(
        keep_histories["candidate_executed"],
        forced_histories["candidate_executed"],
    )
    keep_route = keep_arm["route_lifecycle"]["selected_route"]
    forced_route = forced_arm["route_lifecycle"]["selected_route"]
    same_route = keep_route == forced_route
    route_relation = (
        "same_as_forced_selected_route"
        if same_route
        else "different_from_forced_selected_route"
    )
    component_equal = (
        keep_arm["component_manifests"] == forced_arm["component_manifests"]
    )
    final_equal = keep_arm["final"] == forced_arm["final"]
    features_equal = (
        keep_game["feature_snapshots"] == forced_game["feature_snapshots"]
        and keep_game["features_405"] == forced_game["features_405"]
    )
    full_evidence_equal = component_equal and final_equal and features_equal
    executed_step = executed_difference["difference_step"]
    if executed_step is not None and executed_step < BRANCH_STEP:
        verdict = "BLOCK_EARLY_CANDIDATE_EXECUTED_DIFFERENCE"
    elif same_route and full_evidence_equal and executed_step is None:
        verdict = "VALID_ALIAS"
    elif same_route:
        verdict = "BLOCK_SAME_ROUTE_TRAJECTORY_DIFFERENT"
    elif executed_step is None:
        verdict = "BLOCK_DIFFERENT_ROUTE_EXECUTED_IDENTICAL"
    elif executed_step >= BRANCH_STEP:
        verdict = "VALID_NON_ALIAS"
    else:
        verdict = "BLOCK_UNCLASSIFIED"
    prefix_equalities = component_prefix_equalities(keep_arm, forced_arm)
    return {
        "action_id": forced_game["action_id"],
        "raw_action_first_difference": raw_difference,
        "proposed_action_first_difference": proposed_difference,
        "executed_action_first_difference": executed_difference,
        "route_relation": route_relation,
        "component_prefix_equal": prefix_equalities,
        "full_joint_prefix_equal": (
            keep_arm["full_joint_prefix_sha256"]
            == forced_arm["full_joint_prefix_sha256"]
        ),
        "full_component_manifests_equal": component_equal,
        "full_final_equal": final_equal,
        "full_features_equal": features_equal,
        "full_evidence_equal": full_evidence_equal,
        "claimed_alias_verdict": verdict,
    }


def route_gate_pass(action_id: str, arm: Json, expected_route: str | None) -> bool:
    lifecycle = arm["route_lifecycle"]
    if action_id == "KEEP_V67":
        return bool(
            lifecycle["route_request"] is None
            and lifecycle["route_absent_through_step_119"]
            and lifecycle["forced_injection_count"] == 0
            and lifecycle["forced_injection_step"] is None
            and lifecycle["organic_selection_count"] == 1
            and lifecycle["organic_selection_step"] == BRANCH_STEP
            and lifecycle["selection_step"] == BRANCH_STEP
            and lifecycle["route_drift_count"] == 0
            and lifecycle["lifecycle_complete"]
        )
    return bool(
        lifecycle["route_request"] == expected_route
        and lifecycle["selected_route"] == expected_route
        and lifecycle["selection_step"] == BRANCH_STEP
        and lifecycle["forced_injection_count"] == 1
        and lifecycle["forced_injection_step"] == BRANCH_STEP
        and lifecycle["forced_option_start_count"] == 1
        and lifecycle["forced_option_completion_count"] == 1
        and lifecycle["organic_selection_count"] == 0
        and lifecycle["organic_selection_step"] is None
        and lifecycle["forced_choice_before_step_120_count"] == 0
        and lifecycle["route_drift_count"] == 0
        and lifecycle["selected_route_matches_request"]
        and lifecycle["lifecycle_complete"]
    )


def compact_context(task: Json) -> Json:
    protocol, startup_pins = load_and_pin_sources()
    family_id = str(task["family_id"])
    family = next(
        (item for item in protocol["families"] if item["id"] == family_id),
        None,
    )
    if family is None:
        raise AssertionError("worker task requested unknown family")
    seed = int(task["seed"])
    candidate_seat = int(task["candidate_seat"])
    if seed not in protocol["fresh_development_seeds"]:
        raise AssertionError("worker task requested non-frozen seed")
    if candidate_seat not in protocol["seats"]:
        raise AssertionError("worker task requested non-frozen seat")
    prepare_runtime(protocol)
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
    keep_game = games["KEEP_V67"]
    for action_id in ACTION_IDS[1:]:
        if games[action_id]["feature_snapshots"] != keep_game["feature_snapshots"]:
            raise AssertionError("feature snapshot prefix drift")
        if games[action_id]["features_405"] != keep_game["features_405"]:
            raise AssertionError("feature vector prefix drift")
    arms_by_id = {action_id: compact_arm(games[action_id]) for action_id in ACTION_IDS}
    forced_comparisons = [
        compare_forced(
            keep_game=keep_game,
            forced_game=games[action_id],
            keep_arm=arms_by_id["KEEP_V67"],
            forced_arm=arms_by_id[action_id],
        )
        for action_id in ACTION_IDS[1:]
    ]
    margins = {
        MODEL_ACTIONS[action_id]: int(games[action_id]["candidate_margin"])
        for action_id in ACTION_IDS
    }
    paired_deltas = {
        MODEL_ACTIONS[action_id]: margins[MODEL_ACTIONS[action_id]] - margins["KEEP"]
        for action_id in ACTION_IDS[1:]
    }
    for comparison in forced_comparisons:
        if comparison["claimed_alias_verdict"] == "VALID_ALIAS":
            model_action = MODEL_ACTIONS[comparison["action_id"]]
            if paired_deltas[model_action] != 0:
                raise AssertionError("valid alias paired delta is not zero")
    keep_loss = margins["KEEP"] <= 0
    oracle_rescue_actions = [
        action for action in ("B", "C") if keep_loss and margins[action] > 0
    ]
    action_by_id = {action["id"]: action for action in protocol["actions"]}
    arms = [arms_by_id[action_id] for action_id in ACTION_IDS]
    mechanical_pass = all(
        arm["final"]["final_step"] == FINAL_STEP
        and arm["candidate_execution_audit"]["completed_transitions"] == FINAL_STEP
        and arm["candidate_execution_audit"]["sanitizer_changes"] == 0
        and arm["candidate_execution_audit"]["fallbacks"] == 0
        and arm["candidate_execution_audit"]["exceptions"] == 0
        and arm["candidate_execution_audit"]["partial_actions"] == 0
        and route_gate_pass(
            arm["action_id"],
            arm,
            action_by_id[arm["action_id"]].get("runtime_route_key"),
        )
        for arm in arms
    )
    prefix_pass = all(
        comparison["full_joint_prefix_equal"]
        and all(item["equal"] for item in comparison["component_prefix_equal"])
        for comparison in forced_comparisons
    )
    alias_pass = all(
        comparison["claimed_alias_verdict"] in {"VALID_ALIAS", "VALID_NON_ALIAS"}
        for comparison in forced_comparisons
    )
    _, source_drift_count = post_run_source_audit(protocol, startup_pins)
    worker_read_audit = AUDITOR.audit_record(startup_pins, source_drift_count)
    context = {
        "context_id_sha256": value_sha256(
            {
                "family_id": family_id,
                "seed": seed,
                "candidate_seat": candidate_seat,
            }
        ),
        "family_id": family_id,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "feature_snapshots": keep_game["feature_snapshots"],
        "features_405": keep_game["features_405"],
        "arms": arms,
        "forced_comparisons": forced_comparisons,
        "labels": {
            "margins": margins,
            "paired_deltas": paired_deltas,
            "keep_loss": keep_loss,
            "oracle_rescue_actions": oracle_rescue_actions,
        },
        "claimed_verdicts": {
            "mechanical_pass": mechanical_pass,
            "prefix_pass": prefix_pass,
            "alias_pass": alias_pass,
        },
        "worker_read_audit": worker_read_audit,
    }
    games.clear()
    del games, keep_game, arms_by_id
    gc.collect()
    return context
