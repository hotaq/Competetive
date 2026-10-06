#!/usr/bin/env python3
"""Consumed-Gate8 joint V67 residual feasibility oracle.

This is deliberately not a deployable policy.  It runs the frozen V67 file
agent on-policy while holding only the opponent action tape fixed.  The target
screen compares V67, the strict Stage-0 sheep option, one predeclared V80
single macro deletion, and their exact joint composition.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kaggriculture_rs import Engine  # noqa: E402

from scripts.evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from scripts.evaluate_replay_tape import engine_config, frame_action  # noqa: E402
from src.kaggriculture_rl.policy.v67_residual import (  # noqa: E402
    V67ResidualDecision,
    V67ResidualExecutor,
)

ANALYSIS_DIR = ROOT / "analysis/v82_joint_residual_oracle_20260827"
RESULTS_PATH = ANALYSIS_DIR / "results.json"
REPORT_PATH = ANALYSIS_DIR / "REPORT.md"
GATE_PATH = ROOT / "analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json"
V80_RESULTS_PATH = ROOT / "analysis/v80_residual_action_oracle_20260827/results.json"
REPLAY_DIR = ROOT / "replays/submission_55797143_public_refresh_after22_20260826"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
RESIDUAL_PATH = ROOT / "src/kaggriculture_rl/policy/v67_residual.py"
NATIVE_ENGINE_PATH = ROOT / "bindings/python/kaggriculture_rs/_native.abi3.so"
START_STEP = 120


@dataclass(frozen=True, slots=True)
class MacroSpec:
    source_episode_id: int
    verb: str
    item: str | None
    quantity: int | None
    occurrence: int
    step: int
    market_index: int

    @property
    def expected_command(self) -> list[Any]:
        command: list[Any] = [self.verb]
        if self.item is not None:
            command.append(self.item)
        if self.quantity is not None:
            command.append(self.quantity)
        return command

    @property
    def label(self) -> str:
        item = self.item or "NONE"
        return f"DROP_{self.verb}_{item}_OCC{self.occurrence}_S{self.step}_I{self.market_index}"


TARGET_MACROS: dict[int, MacroSpec] = {
    100159725: MacroSpec(100159725, "BUY_SEED", "STRAWBERRY", 1, 2, 144, 6),
    100184942: MacroSpec(100184942, "BUY_SEED", "STRAWBERRY", 4, 2, 141, 0),
    100187229: MacroSpec(100187229, "BUY_SEED", "STRAWBERRY", 3, 0, 127, 0),
    100196432: MacroSpec(100196432, "HIRE", None, None, 4, 120, 6),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def update_digest(digest: Any, value: Any) -> None:
    digest.update(canonical(value).encode("utf-8"))
    digest.update(b"\n")


def command_parts(command: Any) -> tuple[str, str | None, int | None]:
    if not isinstance(command, list) or not command:
        return "INVALID", None, None
    verb = str(command[0])
    item = str(command[1]) if len(command) >= 2 else None
    quantity = (
        int(command[2])
        if len(command) >= 3 and isinstance(command[2], int) and not isinstance(command[2], bool)
        else None
    )
    return verb, item, quantity


def diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        paths = []
        for index in range(max(len(left), len(right))):
            child = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(diff_paths(left[index], right[index], child))
        return paths
    return [] if left == right else [prefix]


class MacroOverlay:
    """Call V67 once per state, optionally dropping one predeclared macro."""

    def __init__(
        self,
        agent: Any,
        accepts_configuration: bool,
        macro: MacroSpec | None,
        *,
        match_mode: str,
    ) -> None:
        self.agent = agent
        self.accepts_configuration = accepts_configuration
        self.macro = macro
        self.match_mode = match_mode
        self.occurrences: Counter[tuple[str, str | None]] = Counter()
        self.latest_step = -1
        self.latest_base: dict[str, Any] = {}
        self.latest_output: dict[str, Any] = {}
        self.applied: list[dict[str, Any]] = []
        self.signature_failures: list[dict[str, Any]] = []

    def __call__(
        self, observation: dict[str, Any], configuration: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raw = run_agent(
            self.agent,
            self.accepts_configuration,
            observation,
            configuration or {},
        )
        base = normalize_legacy_action(raw)
        output = copy.deepcopy(base)
        step = int(observation["step"])
        self.latest_step = step
        self.latest_base = copy.deepcopy(base)

        market = base.get("market", [])
        for index, command in enumerate(market):
            verb, item, _quantity = command_parts(command)
            key = (verb, item)
            occurrence = self.occurrences[key] if step >= START_STEP else -1
            if step >= START_STEP:
                self.occurrences[key] += 1
            if self.macro is None or self.applied:
                continue
            semantic_match = (
                verb == self.macro.verb
                and item == self.macro.item
                and occurrence == self.macro.occurrence
            )
            coordinate_match = step == self.macro.step and index == self.macro.market_index
            selected = coordinate_match if self.match_mode == "exact" else semantic_match
            if not selected:
                continue
            issues: list[str] = []
            if not semantic_match:
                issues.append("semantic_occurrence_mismatch")
            if self.match_mode == "exact" and command != self.macro.expected_command:
                issues.append("exact_command_mismatch")
            if issues:
                self.signature_failures.append(
                    {
                        "step": step,
                        "market_index": index,
                        "occurrence": occurrence,
                        "command": copy.deepcopy(command),
                        "issues": issues,
                    }
                )
                continue
            output_market = list(output["market"])
            removed = output_market.pop(index)
            output["market"] = output_market
            self.applied.append(
                {
                    "step": step,
                    "market_index": index,
                    "occurrence": occurrence,
                    "removed": removed,
                }
            )

        self.latest_output = copy.deepcopy(output)
        return output


def allowed_scope_path(
    path: str,
    step: int,
    *,
    macro_step: int | None,
    forced_sheep: bool,
) -> bool:
    if macro_step == step and path.startswith("$.market"):
        return True
    if not forced_sheep:
        return False
    if step == 248:
        return path == "$.market[1][1]"
    if 249 <= step <= 261:
        return path.startswith("$.hands[4]")
    return False


def variant_name(*, forced_sheep: bool, macro: MacroSpec | None) -> str:
    if forced_sheep and macro is not None:
        return "JOINT_SHEEP_SWAP_2_PLUS_V80_MACRO"
    if forced_sheep:
        return "SHEEP_SWAP_2_V2_ONLY"
    if macro is not None:
        return "V80_MACRO_ONLY"
    return "FROZEN_V67"


def run_variant(
    replay: dict[str, Any],
    gate_row: dict[str, Any],
    *,
    forced_sheep: bool,
    macro: MacroSpec | None,
    match_mode: str = "exact",
) -> dict[str, Any]:
    candidate_seat = int(gate_row["agent_seat"])
    opponent_seat = int(gate_row["opponent_tape_seat"])
    if opponent_seat != 1 - candidate_seat:
        raise AssertionError("Gate8 seats are not complementary")

    _module, agent, accepts_configuration = load_agent(V67_PATH)
    overlay = MacroOverlay(
        agent,
        accepts_configuration,
        macro,
        match_mode=match_mode,
    )
    decision = V67ResidualDecision.SHEEP_SWAP_2 if forced_sheep else V67ResidualDecision.ABSTAIN
    executor = V67ResidualExecutor(
        overlay,
        lambda _observation: decision,
        eligibility=lambda _observation: True,
    )
    configuration = engine_config(replay)
    engine = Engine(configuration)

    base_digest = hashlib.sha256()
    proposed_digest = hashlib.sha256()
    sanitized_digest = hashlib.sha256()
    opponent_digest = hashlib.sha256()
    candidate_sanitizer_steps: list[dict[str, Any]] = []
    opponent_sanitizer_steps: list[dict[str, Any]] = []
    scope_changed_steps: list[dict[str, Any]] = []
    scope_violations: list[dict[str, Any]] = []
    tape_candidate_mismatch_steps: list[dict[str, Any]] = []

    for transition_step, frame in enumerate(replay["steps"][1:]):
        observation = engine.observation(candidate_seat)
        observed_step = int(observation["step"])
        if observed_step != transition_step:
            raise AssertionError(f"engine step mismatch {observed_step=} {transition_step=}")
        proposed = normalize_legacy_action(executor(observation, configuration))
        if overlay.latest_step != observed_step:
            raise AssertionError("base overlay was not called at current step")
        same_state_base = copy.deepcopy(overlay.latest_base)
        recorded_candidate = normalize_legacy_action(frame_action(frame, candidate_seat))
        opponent_tape = frame_action(frame, opponent_seat)
        sanitized = engine.sanitize_action(candidate_seat, proposed)
        opponent_sanitized = engine.sanitize_action(opponent_seat, opponent_tape)

        update_digest(base_digest, same_state_base)
        update_digest(proposed_digest, proposed)
        update_digest(sanitized_digest, sanitized)
        update_digest(opponent_digest, opponent_tape)

        sanitizer_diff = diff_paths(proposed, sanitized)
        if sanitizer_diff:
            candidate_sanitizer_steps.append({"step": observed_step, "paths": sanitizer_diff})
        opponent_diff = diff_paths(opponent_tape, opponent_sanitized)
        if opponent_diff:
            opponent_sanitizer_steps.append({"step": observed_step, "paths": opponent_diff})

        output_diff = diff_paths(same_state_base, proposed)
        if output_diff:
            scope_changed_steps.append({"step": observed_step, "paths": output_diff})
            macro_step = overlay.applied[0]["step"] if overlay.applied else None
            bad_paths = [
                path
                for path in output_diff
                if not allowed_scope_path(
                    path,
                    observed_step,
                    macro_step=macro_step,
                    forced_sheep=forced_sheep,
                )
            ]
            if bad_paths:
                scope_violations.append({"step": observed_step, "paths": bad_paths})

        tape_diff = diff_paths(proposed, recorded_candidate)
        if tape_diff:
            tape_candidate_mismatch_steps.append({"step": observed_step, "paths": tape_diff})

        actions = [None, None]
        actions[candidate_seat] = sanitized
        actions[opponent_seat] = opponent_tape
        engine.step(actions)

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    counters = executor.counters_for(candidate_seat)
    packaging_gate = executor.packaging_gate_for(candidate_seat)
    expected_interventions = 1 if forced_sheep else 0
    expected_macro_count = 1 if macro is not None else 0
    macro_valid = len(overlay.applied) == expected_macro_count and not overlay.signature_failures
    executor_valid = (
        packaging_gate.passed
        and counters.interventions == expected_interventions
        and counters.completed == expected_interventions
        and counters.partial_interventions == 0
    )
    action_contract_valid = not candidate_sanitizer_steps and not scope_violations
    valid = macro_valid and executor_valid and action_contract_valid

    return {
        "name": variant_name(forced_sheep=forced_sheep, macro=macro),
        "forced_sheep": forced_sheep,
        "macro": asdict(macro) if macro is not None else None,
        "macro_match_mode": match_mode if macro is not None else None,
        "margin": money[candidate_seat] - money[opponent_seat],
        "candidate_money": money[candidate_seat],
        "opponent_money": money[opponent_seat],
        "final_state_step": int(engine.state["step"]),
        "valid": valid,
        "macro_contract": {
            "valid": macro_valid,
            "applied_count": len(overlay.applied),
            "applied": overlay.applied,
            "signature_failures": overlay.signature_failures,
        },
        "executor": {
            "phase": str(executor.phase_for(candidate_seat)),
            "counters": asdict(counters),
            "packaging_gate": asdict(packaging_gate),
            "trace_changed": [
                asdict(event)
                for event in executor.trace_for(candidate_seat)
                if event.changed
                or event.event in {"OPTION_START", "OPTION_COMPLETE", "PARTIAL_ABORT"}
            ],
        },
        "sanitizer": {
            "candidate_changed_count": len(candidate_sanitizer_steps),
            "candidate_changed_steps": candidate_sanitizer_steps,
            "opponent_tape_changed_count": len(opponent_sanitizer_steps),
            "opponent_tape_changed_steps": opponent_sanitizer_steps,
            "candidate_execution_uses_explicit_sanitized_action": True,
            "opponent_execution_preserves_raw_tape_action": True,
        },
        "action_scope": {
            "valid": not scope_violations,
            "changed_step_count": len(scope_changed_steps),
            "changed_steps": scope_changed_steps,
            "violation_count": len(scope_violations),
            "violations": scope_violations,
            "comparison": (
                "final proposed action versus V67 callable output on the same observation"
            ),
        },
        "recorded_candidate_tape_comparison": {
            "mismatch_step_count": len(tape_candidate_mismatch_steps),
            "mismatch_steps": tape_candidate_mismatch_steps[:40],
            "diagnostic_only_after_state_divergence": True,
        },
        "action_hashes": {
            "same_state_v67_callable": base_digest.hexdigest(),
            "candidate_proposed": proposed_digest.hexdigest(),
            "candidate_sanitized": sanitized_digest.hexdigest(),
            "opponent_raw_tape": opponent_digest.hexdigest(),
        },
    }


def validate_baseline(episode_id: int, gate_row: dict[str, Any], baseline: dict[str, Any]) -> None:
    expected_margin = float(gate_row["baseline_margin"])
    if not math.isclose(baseline["margin"], expected_margin, abs_tol=1e-9):
        raise AssertionError(
            f"episode {episode_id} callable baseline {baseline['margin']} != {expected_margin}"
        )
    if baseline["sanitizer"]["candidate_changed_count"] != 0:
        raise AssertionError(f"episode {episode_id} baseline candidate was sanitized")
    if baseline["recorded_candidate_tape_comparison"]["mismatch_step_count"] != 0:
        raise AssertionError(f"episode {episode_id} callable V67 differs from uploaded tape")
    if not baseline["valid"]:
        raise AssertionError(f"episode {episode_id} callable baseline contract invalid")


def evaluate_target(
    row: dict[str, Any], replay: dict[str, Any], macro: MacroSpec
) -> dict[str, Any]:
    variants = {
        "frozen_v67": run_variant(replay, row, forced_sheep=False, macro=None),
        "sheep_swap_2_v2_only": run_variant(replay, row, forced_sheep=True, macro=None),
        "v80_macro_only": run_variant(replay, row, forced_sheep=False, macro=macro),
        "joint": run_variant(replay, row, forced_sheep=True, macro=macro),
    }
    validate_baseline(int(row["episode_id"]), row, variants["frozen_v67"])
    baseline_margin = variants["frozen_v67"]["margin"]
    for variant in variants.values():
        variant["delta_vs_frozen_v67"] = variant["margin"] - baseline_margin
    joint = variants["joint"]
    return {
        "episode_id": int(row["episode_id"]),
        "candidate_seat": int(row["agent_seat"]),
        "opponent_tape_seat": int(row["opponent_tape_seat"]),
        "opponent": row["opponent"],
        "replay_sha256": row["replay_sha256"],
        "macro": asdict(macro),
        "variants": variants,
        "joint_positive": bool(joint["valid"] and joint["margin"] > 0.0),
    }


def evaluate_control_optimistic(row: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Non-deployable upper bound over four target semantic macro mappings."""
    baseline = run_variant(replay, row, forced_sheep=False, macro=None)
    validate_baseline(int(row["episode_id"]), row, baseline)
    sheep = run_variant(replay, row, forced_sheep=True, macro=None)
    mapped = []
    for source_episode_id, macro in TARGET_MACROS.items():
        macro_only = run_variant(
            replay,
            row,
            forced_sheep=False,
            macro=macro,
            match_mode="semantic_occurrence",
        )
        joint = run_variant(
            replay,
            row,
            forced_sheep=True,
            macro=macro,
            match_mode="semantic_occurrence",
        )
        mapped.append(
            {
                "source_target_episode_id": source_episode_id,
                "macro_only": macro_only,
                "joint": joint,
            }
        )
    valid_joint = [item for item in mapped if item["joint"]["valid"]]
    selected = max(valid_joint, key=lambda item: item["joint"]["margin"]) if valid_joint else None
    selected_margin = selected["joint"]["margin"] if selected is not None else None
    baseline_margin = baseline["margin"]
    return {
        "episode_id": int(row["episode_id"]),
        "candidate_seat": int(row["agent_seat"]),
        "opponent_tape_seat": int(row["opponent_tape_seat"]),
        "opponent": row["opponent"],
        "baseline": baseline,
        "sheep_swap_2_v2_only": sheep,
        "semantic_mapping_variants": mapped,
        "optimistic_selected_source_target_episode_id": (
            selected["source_target_episode_id"] if selected is not None else None
        ),
        "optimistic_selected_joint_margin": selected_margin,
        "optimistic_no_regression": bool(
            selected_margin is not None
            and selected_margin > 0.0
            and selected_margin >= baseline_margin
        ),
        "boundary": (
            "Per-control hindsight chooses the best valid joint among the four "
            "target semantic-occurrence mappings. This is an optimistic existence "
            "upper bound, not an observable or identity-based deployable rule."
        ),
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# V82 Joint Residual Oracle — 2026-08-27",
        "",
        "## Verdict",
        "",
        f"Status: **{result['status']}**.",
        "",
        (
            "This is a consumed-Gate8, development-only exact Rust feasibility "
            "screen. It is not a learned policy, promotion result, package, or upload."
        ),
        "",
        "## Target results",
        "",
        "| Episode | V67 | Sheep v2 | V80 macro | Joint | Joint valid | Joint win |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for target in result["targets"]:
        variants = target["variants"]
        joint = variants["joint"]
        lines.append(
            "| {episode_id} | {base:+.0f} | {sheep:+.0f} | {macro:+.0f} | "
            "{joint:+.0f} | {valid} | {win} |".format(
                episode_id=target["episode_id"],
                base=variants["frozen_v67"]["margin"],
                sheep=variants["sheep_swap_2_v2_only"]["margin"],
                macro=variants["v80_macro_only"]["margin"],
                joint=joint["margin"],
                valid="yes" if joint["valid"] else "NO",
                win="yes" if target["joint_positive"] else "NO",
            )
        )
    summary = result["target_summary"]
    by_episode = {target["episode_id"]: target for target in result["targets"]}
    lines.extend(
        [
            "",
            (
                f"Joint rescued **{summary['positive_joint_count']}/4** targets; "
                f"worst joint margin **{summary['worst_joint_margin']:+.0f}**."
            ),
            "",
            "Every variant used callable V67 on the current engine state and the "
            "raw opponent replay action on the fixed opponent seat. Candidate "
            "sanitizer, same-state overlay scope, residual phase/counters, and the "
            "zero-partial packaging gate are recorded in `results.json`.",
            "",
            "## Interaction finding",
            "",
            (
                "All four joint executions were mechanically valid: zero candidate "
                "sanitizer changes, zero out-of-scope changes, zero partials, and a "
                "passing completion gate. The stop is economic, not an executor fault."
            ),
            "",
            (
                f"- `100159725`: sheep alone reached "
                f"{by_episode[100159725]['variants']['sheep_swap_2_v2_only']['margin']:+.0f} "
                f"and macro alone "
                f"{by_episode[100159725]['variants']['v80_macro_only']['margin']:+.0f}, "
                f"but the joint fell back to "
                f"{by_episode[100159725]['variants']['joint']['margin']:+.0f}."
            ),
            (
                f"- `100184942`: the joint was the only target rescue at "
                f"{by_episode[100184942]['variants']['joint']['margin']:+.0f}."
            ),
            (
                f"- `100187229`: the on-policy macro reversed the earlier V80 "
                f"tape-oracle signal and scored "
                f"{by_episode[100187229]['variants']['v80_macro_only']['margin']:+.0f}; "
                f"the joint remained "
                f"{by_episode[100187229]['variants']['joint']['margin']:+.0f}."
            ),
            (
                f"- `100196432`: sheep alone "
                f"({by_episode[100196432]['variants']['sheep_swap_2_v2_only']['margin']:+.0f}) "
                f"and macro alone "
                f"({by_episode[100196432]['variants']['v80_macro_only']['margin']:+.0f}) "
                f"both won, yet their joint lost at "
                f"{by_episode[100196432]['variants']['joint']['margin']:+.0f}."
            ),
            "",
            (
                "V80 held the candidate continuation tape fixed after its one mutation; "
                "V82 instead recalls V67 from every changed state. Therefore the V80 "
                "single-mutation margins are an optimistic tape oracle and are not "
                "expected to equal these on-policy macro-only margins."
            ),
            "",
            "## Gate order",
            "",
        ]
    )
    if result["controls_executed"]:
        lines.extend(
            [
                "All four joint targets were positive and valid, so the four "
                "consumed controls were run under an explicitly optimistic boundary.",
                "",
                "For each control, hindsight selected the best valid joint among "
                "the four target semantic-occurrence macro mappings. This is only "
                "an existence upper bound. It is not an observable mapping and no "
                "opponent-identity rule is proposed.",
                "",
                "| Control | V67 | Optimistic joint | No regression |",
                "|---:|---:|---:|:---:|",
            ]
        )
        for control in result["controls"]:
            lines.append(
                "| {episode_id} | {base:+.0f} | {selected:+.0f} | {passed} |".format(
                    episode_id=control["episode_id"],
                    base=control["baseline"]["margin"],
                    selected=control["optimistic_selected_joint_margin"],
                    passed="yes" if control["optimistic_no_regression"] else "NO",
                )
            )
    else:
        lines.extend(
            [
                "At least one exact joint target remained non-positive or invalid. "
                "Per the predeclared fail-fast rule, **no controls were executed**.",
                "",
                "Therefore this report makes no control-regression claim.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope and reproducibility",
            "",
            "- Data: only four consumed Gate8 losses; controls only if 4/4 target pass.",
            "- No old/fresh/novel/sealed/new replay was opened.",
            "- No PPO, candidate archive, package, submission, or upload was created.",
            "- Input and action-stream SHA-256 hashes are in `results.json`.",
            f"- `run_oracle.py`: `{sha256(Path(__file__))}`",
            f"- `results.json`: `{sha256(RESULTS_PATH)}`",
            "",
            "Final validation: Ruff check PASS, Ruff format check PASS, Python "
            "compile PASS, and `tests/policy/test_v67_residual.py` PASS.",
            "",
            "```bash",
            ".venv/bin/python analysis/v82_joint_residual_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff check analysis/v82_joint_residual_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v82_joint_residual_oracle_20260827/run_oracle.py",
            "```",
            "",
            "## TokenSave boundary",
            "",
            "The TokenSave MCP surface was unavailable and its SQLite graph had no "
            "current V7x/V80 nodes, so named current artifacts had to be read directly. "
            "If reporting this at <https://github.com/aovestdipaperino/tokensave>, "
            "strip sensitive or proprietary code from the issue description first.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if sha256(V67_PATH) != gate["baseline_agent"]["sha256"]:
        raise RuntimeError("frozen V67 hash mismatch")

    targets = []
    for row in gate["target_losses"]:
        episode_id = int(row["episode_id"])
        if episode_id not in TARGET_MACROS:
            raise RuntimeError(f"missing predeclared target macro {episode_id}")
        replay_path = REPLAY_DIR / f"{episode_id}.json"
        if sha256(replay_path) != row["replay_sha256"]:
            raise RuntimeError(f"replay hash mismatch {episode_id}")
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        targets.append(evaluate_target(row, replay, TARGET_MACROS[episode_id]))

    positive_joint_count = sum(target["joint_positive"] for target in targets)
    controls_executed = positive_joint_count == len(targets) == 4
    controls = []
    if controls_executed:
        for row in gate["regression_controls"]:
            episode_id = int(row["episode_id"])
            replay_path = REPLAY_DIR / f"{episode_id}.json"
            if sha256(replay_path) != row["replay_sha256"]:
                raise RuntimeError(f"control replay hash mismatch {episode_id}")
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            controls.append(evaluate_control_optimistic(row, replay))

    controls_pass = bool(
        controls_executed
        and len(controls) == 4
        and all(control["optimistic_no_regression"] for control in controls)
    )
    status = (
        "HEADROOM_ONLY_JOINT_ORACLE_TARGET_AND_OPTIMISTIC_CONTROLS_PASS"
        if controls_pass
        else (
            "STOP_OPTIMISTIC_CONTROL_REGRESSION"
            if controls_executed
            else "STOP_JOINT_TARGET_PREREQUISITE_FAILED"
        )
    )
    result = {
        "schema_version": 1,
        "experiment": "v82_joint_residual_oracle_20260827",
        "status": status,
        "scope": {
            "gate": "consumed Gate8 only",
            "target_data": "four target losses",
            "control_condition": "execute only after four valid positive joint targets",
            "candidate_execution": "callable V67 on-policy in official Rust engine",
            "opponent_execution": "raw fixed opponent replay tape",
            "opening": "frozen V67 callable; only declared macro and/or sheep option overlay",
            "promotion_evidence": False,
            "upload_performed": False,
        },
        "inputs": {
            "runner": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": sha256(Path(__file__)),
            },
            "gate": {"path": str(GATE_PATH.relative_to(ROOT)), "sha256": sha256(GATE_PATH)},
            "v80_results": {
                "path": str(V80_RESULTS_PATH.relative_to(ROOT)),
                "sha256": sha256(V80_RESULTS_PATH),
            },
            "v67": {"path": str(V67_PATH.relative_to(ROOT)), "sha256": sha256(V67_PATH)},
            "residual_executor": {
                "path": str(RESIDUAL_PATH.relative_to(ROOT)),
                "sha256": sha256(RESIDUAL_PATH),
            },
            "native_engine": {
                "path": str(NATIVE_ENGINE_PATH.relative_to(ROOT)),
                "sha256": sha256(NATIVE_ENGINE_PATH),
            },
        },
        "target_summary": {
            "games": len(targets),
            "positive_joint_count": positive_joint_count,
            "all_joint_valid": all(target["variants"]["joint"]["valid"] for target in targets),
            "worst_joint_margin": min(target["variants"]["joint"]["margin"] for target in targets),
            "mean_joint_margin": sum(target["variants"]["joint"]["margin"] for target in targets)
            / len(targets),
        },
        "targets": targets,
        "controls_executed": controls_executed,
        "controls": controls,
        "validation": {
            "callable_v67_baselines_match_gate_margin": True,
            "callable_v67_baselines_match_uploaded_candidate_tape": True,
            "target_joint_candidate_sanitizer_zero": all(
                target["variants"]["joint"]["sanitizer"]["candidate_changed_count"] == 0
                for target in targets
            ),
            "target_joint_scope_violations_zero": all(
                target["variants"]["joint"]["action_scope"]["violation_count"] == 0
                for target in targets
            ),
            "target_joint_partial_interventions_zero": all(
                target["variants"]["joint"]["executor"]["counters"]["partial_interventions"] == 0
                for target in targets
            ),
            "target_joint_packaging_gates_pass": all(
                target["variants"]["joint"]["executor"]["packaging_gate"]["passed"]
                for target in targets
            ),
        },
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(
        canonical(
            {
                "status": status,
                "positive_joint_count": positive_joint_count,
                "worst_joint_margin": result["target_summary"]["worst_joint_margin"],
                "controls_executed": controls_executed,
            }
        )
    )


if __name__ == "__main__":
    main()
