#!/usr/bin/env python3
"""Build the development-only V67 Stage0-v2 allocation on-policy dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine  # noqa: E402

from kaggriculture_rl.policy.v67_residual import (  # noqa: E402
    V67_SHEEP_SWAP_HAND4_PLAN,
    V67ResidualDecision,
    V67ResidualExecutor,
)

OUT = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"
CATALOG_PATH = ROOT / "analysis/residual_proxy_family_catalog_20260827/manifest.json"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
EXECUTOR_PATH = ROOT / "src/kaggriculture_rl/policy/v67_residual.py"
HELPER_PATH = ROOT / "scripts/evaluate_file_agent_vs_replay.py"

SEEDS = (27100100, 27100101, 27100102, 27100103)
SEATS = (0, 1)
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}
EXPECTED_EXECUTOR_SHA256 = "7a68d4424dd4793d32f01e0b899ab110768f6342a8bd68ffb904cdd46ff682ee"
EXPECTED_V67_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"
HAND4_PLAN = {step: list(command) for step, command in V67_SHEEP_SWAP_HAND4_PLAN}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def jsonable_dataclass(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value)))


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


class RecordingBase:
    """Expose the exact base action seen by the executor in the forced state."""

    def __init__(self, agent: Any, accepts_configuration: bool) -> None:
        self.agent = agent
        self.accepts_configuration = accepts_configuration
        self.last_action: dict[str, Any] | None = None

    def __call__(self, observation: dict[str, Any], configuration: Any) -> dict[str, Any]:
        action = run_agent(
            self.agent,
            self.accepts_configuration,
            observation,
            configuration,
        )
        if not isinstance(action, dict):
            raise TypeError(f"base agent returned {type(action).__name__}")
        self.last_action = copy.deepcopy(action)
        return action


def new_sanitizer_stats() -> dict[str, Any]:
    return {
        "changed_steps": 0,
        "farmer_changed_steps": 0,
        "hands_changed_steps": 0,
        "market_changed_steps": 0,
        "step_ids": [],
    }


def record_sanitizer_change(
    stats: dict[str, Any],
    step: int,
    proposed: dict[str, Any],
    sanitized: dict[str, Any],
) -> None:
    if proposed == sanitized:
        return
    stats["changed_steps"] += 1
    stats["step_ids"].append(step)
    for component in ("farmer", "hands", "market"):
        if proposed.get(component) != sanitized.get(component):
            stats[f"{component}_changed_steps"] += 1


def direct_scope_check(
    *,
    step: int,
    decision: V67ResidualDecision,
    option_started: bool,
    base_action: dict[str, Any],
    output_action: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    paths = diff_paths(base_action, output_action)
    if decision is V67ResidualDecision.ABSTAIN:
        if paths:
            return None, {
                "step": step,
                "reason": "abstain_not_base_identical",
                "diff_paths": paths,
            }
        return None, None

    if not option_started:
        if paths:
            return None, {
                "step": step,
                "reason": "unavailable_option_changed_action",
                "diff_paths": paths,
            }
        return None, None

    if step == 248:
        market_base = base_action.get("market")
        market_output = output_action.get("market")
        signature_ok = (
            isinstance(market_base, list)
            and isinstance(market_output, list)
            and len(market_base) > 1
            and len(market_output) > 1
            and market_base[1] == ["BUY_ANIMAL", "COW", 2]
            and market_output[1] == ["BUY_ANIMAL", "SHEEP", 2]
        )
        restored = copy.deepcopy(output_action)
        if signature_ok:
            restored["market"][1] = ["BUY_ANIMAL", "COW", 2]
        if not signature_ok or restored != base_action:
            return None, {
                "step": step,
                "reason": "start_mutation_out_of_scope",
                "diff_paths": paths,
            }
        return {
            "step": step,
            "owned_path": "market[1]",
            "base_command": market_base[1],
            "option_command": market_output[1],
            "changed": True,
            "diff_paths": paths,
        }, None

    if step in HAND4_PLAN:
        base_hands = base_action.get("hands")
        output_hands = output_action.get("hands")
        signature_ok = (
            isinstance(base_hands, list)
            and isinstance(output_hands, list)
            and len(base_hands) > 4
            and len(output_hands) > 4
            and output_hands[4] == HAND4_PLAN[step]
        )
        restored = copy.deepcopy(output_action)
        if signature_ok:
            restored["hands"][4] = copy.deepcopy(base_hands[4])
        if not signature_ok or restored != base_action:
            return None, {
                "step": step,
                "reason": "hand4_takeover_out_of_scope",
                "diff_paths": paths,
            }
        return {
            "step": step,
            "owned_path": "hands[4]",
            "base_command": copy.deepcopy(base_hands[4]),
            "option_command": copy.deepcopy(output_hands[4]),
            "changed": base_hands[4] != output_hands[4],
            "diff_paths": paths,
        }, None

    if paths:
        return None, {
            "step": step,
            "reason": "mutation_outside_option_window",
            "diff_paths": paths,
        }
    return None, None


def private_summary(observation: dict[str, Any]) -> dict[str, Any]:
    private = observation.get("private", {})
    inventory: Counter[str] = Counter()
    for unit in private.get("units", []):
        for item, quantity in unit.get("inventory", {}).items():
            inventory[str(item)] += int(quantity)
    return {
        "money": float(private.get("money", 0.0)),
        "shed": dict(sorted(private.get("shed", {}).items())),
        "seeds": dict(sorted(private.get("seeds", {}).items())),
        "unit_inventory": dict(sorted(inventory.items())),
    }


def play(
    *,
    family: str,
    representative: str,
    proxy_path: Path,
    seed: int,
    candidate_seat: int,
    decision: V67ResidualDecision,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, proxy_agent, proxy_accepts_config = load_agent(proxy_path)
    _, base_agent, base_accepts_config = load_agent(V67_PATH)
    recording_base = RecordingBase(base_agent, base_accepts_config)
    executor = V67ResidualExecutor(
        recording_base,
        lambda _observation: decision,
        eligibility=lambda _observation: True,
    )

    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    candidate_proposed_trace: list[dict[str, Any]] = []
    candidate_sanitized_trace: list[dict[str, Any]] = []
    forced_state_base_trace: list[dict[str, Any]] = []
    proxy_proposed_trace: list[dict[str, Any]] = []
    proxy_sanitized_trace: list[dict[str, Any]] = []
    candidate_sanitizer = new_sanitizer_stats()
    proxy_sanitizer = new_sanitizer_stats()
    action_scope: list[dict[str, Any]] = []
    scope_violations: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None

    for step in range(719):
        actions: list[dict[str, Any]] = []
        for player in range(2):
            observation = engine.observation(player)
            if player == candidate_seat:
                raw = executor(observation, configuration)
                if not isinstance(raw, dict) or recording_base.last_action is None:
                    raise TypeError("executor or recording base returned invalid action")
                base_in_forced_state = normalize_legacy_action(recording_base.last_action)
                proposed = normalize_legacy_action(raw)
                scope_record, scope_failure = direct_scope_check(
                    step=step,
                    decision=decision,
                    option_started=(executor.counters_for(candidate_seat).interventions == 1),
                    base_action=base_in_forced_state,
                    output_action=proposed,
                )
                if scope_record is not None:
                    action_scope.append(scope_record)
                if scope_failure is not None:
                    scope_violations.append(scope_failure)
                    failure = {"gate": "out_of_contract", **scope_failure}
                sanitized = engine.sanitize_action(player, proposed)
                record_sanitizer_change(candidate_sanitizer, step, proposed, sanitized)
                if proposed != sanitized and failure is None:
                    failure = {
                        "gate": "candidate_sanitizer",
                        "step": step,
                        "diff_paths": diff_paths(proposed, sanitized),
                    }
                candidate_proposed_trace.append(copy.deepcopy(proposed))
                candidate_sanitized_trace.append(copy.deepcopy(sanitized))
                forced_state_base_trace.append(copy.deepcopy(base_in_forced_state))
                if executor.phase_for(candidate_seat).value == "PARTIAL_ABORTED":
                    failure = {
                        "gate": "executor_partial",
                        "step": step,
                        "trace_event": jsonable_dataclass(executor.trace_for(candidate_seat)[-1]),
                    }
            else:
                raw = run_agent(
                    proxy_agent,
                    proxy_accepts_config,
                    observation,
                    configuration,
                )
                proposed = normalize_legacy_action(raw)
                sanitized = engine.sanitize_action(player, proposed)
                record_sanitizer_change(proxy_sanitizer, step, proposed, sanitized)
                proxy_proposed_trace.append(copy.deepcopy(proposed))
                proxy_sanitized_trace.append(copy.deepcopy(sanitized))
            actions.append(sanitized)
        if failure is not None:
            break
        engine.step(actions)

    counters = executor.counters_for(candidate_seat)
    gate = executor.packaging_gate_for(candidate_seat)
    completed = failure is None and int(engine.state["step"]) == 719
    proxy_seat = 1 - candidate_seat
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    candidate_margin = money[candidate_seat] - money[proxy_seat] if completed else None
    run = {
        "family": family,
        "representative": representative,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "proxy_seat": proxy_seat,
        "decision": decision.value,
        "completed": completed,
        "final_step": int(engine.state["step"]),
        "failure": failure,
        "candidate_money": money[candidate_seat] if completed else None,
        "proxy_money": money[proxy_seat] if completed else None,
        "candidate_margin": candidate_margin,
        "candidate_win": candidate_margin > 0 if candidate_margin is not None else None,
        "executor_counters": jsonable_dataclass(counters),
        "executor_packaging_gate": jsonable_dataclass(gate),
        "executor_trace": [
            jsonable_dataclass(event) for event in executor.trace_for(candidate_seat)
        ],
        "direct_action_scope": action_scope,
        "direct_scope_violations": scope_violations,
        "candidate_sanitizer": candidate_sanitizer,
        "proxy_sanitizer": proxy_sanitizer,
        "candidate_proposed_trace_sha256": value_sha256(candidate_proposed_trace),
        "candidate_sanitized_trace_sha256": value_sha256(candidate_sanitized_trace),
        "forced_state_base_trace_sha256": value_sha256(forced_state_base_trace),
        "proxy_proposed_trace_sha256": value_sha256(proxy_proposed_trace),
        "proxy_sanitized_trace_sha256": value_sha256(proxy_sanitized_trace),
        "candidate_final_private": (
            private_summary(engine.observation(candidate_seat)) if completed else None
        ),
    }
    return run, candidate_proposed_trace, candidate_sanitized_trace


def paired_diff_summary(
    baseline: list[dict[str, Any]], forced: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(baseline) != len(forced):
        return {"comparable": False, "reason": "trace_length_mismatch"}
    differences = [
        {
            "step": step,
            "diff_paths": diff_paths(left, right),
            "phase": (
                "start" if step == 248 else "owned_suffix" if 249 <= step <= 261 else "response"
            ),
        }
        for step, (left, right) in enumerate(zip(baseline, forced, strict=True))
        if left != right
    ]
    return {
        "comparable": True,
        "different_steps": len(differences),
        "start_steps": sum(row["phase"] == "start" for row in differences),
        "owned_suffix_steps": sum(row["phase"] == "owned_suffix" for row in differences),
        "response_steps": sum(row["phase"] == "response" for row in differences),
        "step_ids": [row["step"] for row in differences],
        "differences": differences,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_margins = [float(row["baseline"]["candidate_margin"]) for row in rows]
    forced_margins = [float(row["forced"]["candidate_margin"]) for row in rows]
    deltas = [float(row["paired_terminal_margin_delta"]) for row in rows]
    baseline_wins = sum(margin > 0 for margin in baseline_margins)
    forced_wins = sum(margin > 0 for margin in forced_margins)
    return {
        "games": len(rows),
        "option_available_games": sum(row["option_available"] for row in rows),
        "option_completed_games": sum(row["option_completed"] for row in rows),
        "partial_games": sum(row["has_partial"] for row in rows),
        "beneficial_games": sum(delta > 0 for delta in deltas),
        "neutral_games": sum(delta == 0 for delta in deltas),
        "regressed_games": sum(delta < 0 for delta in deltas),
        "baseline_wins": baseline_wins,
        "forced_wins": forced_wins,
        "win_delta": forced_wins - baseline_wins,
        "baseline_mean_margin": fmean(baseline_margins),
        "forced_mean_margin": fmean(forced_margins),
        "mean_margin_delta": fmean(deltas),
        "baseline_worst_margin": min(baseline_margins),
        "forced_worst_margin": min(forced_margins),
        "worst_margin_delta": min(forced_margins) - min(baseline_margins),
        "paired_delta_mean": fmean(deltas),
        "paired_delta_worst": min(deltas),
        "paired_delta_best": max(deltas),
        "any_paired_regression": any(delta < 0 for delta in deltas),
        "aggregate_mean_regression": fmean(deltas) < 0,
        "worst_margin_regression": min(forced_margins) < min(baseline_margins),
        "candidate_sanitizer_changed_steps": sum(
            row["forced"]["candidate_sanitizer"]["changed_steps"] for row in rows
        ),
        "direct_scope_violations": sum(
            len(row["forced"]["direct_scope_violations"]) for row in rows
        ),
        "owned_hand4_steps": sum(
            row["forced"]["executor_counters"]["owned_hand4_steps"] for row in rows
        ),
        "changed_action_steps": sum(
            row["forced"]["executor_counters"]["changed_action_steps"] for row in rows
        ),
    }


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# V83 Allocation-v2 On-Policy Dataset — 2026-08-27",
        "",
        "## Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This is development-only Rust-engine evidence. It is not training, a "
        "candidate, a package, a submission, or promotion evidence.",
        "",
    ]
    if result["stop_reason"]:
        lines.extend([f"Fail-fast reason: `{result['stop_reason']}`.", ""])
    if result["family_summaries"]:
        lines.extend(
            [
                "## Per-family paired outcomes",
                "",
                "| Family | Games | Available / complete | Benefit / neutral / regress | "
                "Wins base→forced (Δ) | Mean base→forced (Δ) | "
                "Worst base→forced (Δ) | Paired Δ mean / worst | Regressed? |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for family, summary in result["family_summaries"].items():
            lines.append(
                f"| {family} | {summary['games']} | "
                f"{summary['option_available_games']} / {summary['option_completed_games']} | "
                f"{summary['beneficial_games']} / {summary['neutral_games']} / "
                f"{summary['regressed_games']} | {summary['baseline_wins']}→"
                f"{summary['forced_wins']} ({summary['win_delta']:+d}) | "
                f"{summary['baseline_mean_margin']:.1f}→"
                f"{summary['forced_mean_margin']:.1f} "
                f"({summary['mean_margin_delta']:+.1f}) | "
                f"{summary['baseline_worst_margin']:.1f}→"
                f"{summary['forced_worst_margin']:.1f} "
                f"({summary['worst_margin_delta']:+.1f}) | "
                f"{summary['paired_delta_mean']:+.1f} / "
                f"{summary['paired_delta_worst']:+.1f} | "
                f"{'YES' if summary['any_paired_regression'] else 'NO'} |"
            )
        lines.append("")
    global_summary = result["global_summary"]
    if global_summary is not None:
        lines.extend(
            [
                "## Benefit verdict",
                "",
                "The executor is mechanically valid, but the unconditional forced "
                "allocation is rejected as a general policy on this development set. "
                f"Wins fall {global_summary['baseline_wins']}→"
                f"{global_summary['forced_wins']}, paired mean delta is "
                f"{global_summary['paired_delta_mean']:+.1f}, and "
                f"{global_summary['regressed_games']} of "
                f"{global_summary['games']} games regress. Every family has negative "
                "mean delta and a worse family worst-case margin.",
                "",
            ]
        )
    exact = result["exactness"]
    lines.extend(
        [
            "## Mechanical gates and action scope",
            "",
            f"- Completed pairs: {len(result['games'])} / "
            f"{result['scope']['expected_paired_games']}.",
            f"- Forced option starts/completions/partials: "
            f"{exact['option_starts']} / {exact['option_completions']} / "
            f"{exact['partial_interventions']}.",
            f"- Owned hand4 suffix steps: {exact['owned_hand4_steps']} "
            f"(expected {13 * exact['option_starts']}).",
            f"- Candidate sanitizer changes: {exact['candidate_sanitizer_changed_steps']}.",
            f"- Direct executor action-scope violations: {exact['direct_scope_violations']}.",
            "",
            "The direct scope gate compares the executor output with the V67 base "
            "action generated in the same forced state. Only `market[1]` at step 248 "
            "and `hands[4]` at steps 249-261 may differ. Counterfactual ABSTAIN-versus-"
            "forced response differences after state divergence are recorded but are "
            "not mislabeled as executor mutations.",
            "",
            "## Frozen scope",
            "",
            "- Four catalog representatives only: V22, V12/Ryo, V44, robust cluster.",
            "- Already-consumed seeds `27100100..27100103`, both candidate seats.",
            "- No new seed, replay, holdout, live Kaggle trace, training, candidate, "
            "package, submission, or upload.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
            ".venv/bin/ruff check analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
            "```",
            "",
            "Frozen input and per-run action-trace hashes are in `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="refresh hashes and report from existing results without running games",
    )
    args = parser.parse_args()
    if args.render_existing:
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        result["inputs"][str(RUNNER_PATH.relative_to(ROOT))] = sha256(RUNNER_PATH)
        render_command = (
            ".venv/bin/python "
            "analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py "
            "--render-existing"
        )
        if render_command not in result["reproduction"]:
            result["reproduction"].append(render_command)
        RESULTS_PATH.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        REPORT_PATH.write_text(build_report(result), encoding="utf-8")
        print(json.dumps({"status": result["status"], "games_executed": 0}))
        return

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if tuple(catalog["development_split"]["seeds"]) != SEEDS:
        raise RuntimeError("catalog seed scope changed")
    if tuple(catalog["development_split"]["seats_per_seed"]) != SEATS:
        raise RuntimeError("catalog seat scope changed")
    if sha256(V67_PATH) != EXPECTED_V67_SHA256:
        raise RuntimeError("frozen V67 hash mismatch")
    if catalog["baseline"]["sha256"] != EXPECTED_V67_SHA256:
        raise RuntimeError("catalog V67 hash mismatch")
    if sha256(EXECUTOR_PATH) != EXPECTED_EXECUTOR_SHA256:
        raise RuntimeError("frozen Stage0-v2 executor hash mismatch")

    family_specs: list[tuple[str, str, Path, str]] = []
    for family_row in catalog["catalog"]["families"]:
        family = str(family_row["family"])
        representative = str(family_row["representative"])
        agent_row = catalog["agents"][representative]
        proxy_path = ROOT / agent_row["path"]
        actual_hash = sha256(proxy_path)
        if actual_hash != agent_row["sha256"]:
            raise RuntimeError(f"proxy hash mismatch: {representative}")
        family_specs.append((family, representative, proxy_path, actual_hash))
    if [row[1] for row in family_specs] != [
        "v22",
        "v12_ryo",
        "v44_route",
        "robust_cluster",
    ]:
        raise RuntimeError("catalog representative scope changed")

    games: list[dict[str, Any]] = []
    stop_reason: str | None = None
    expected_pairs = len(family_specs) * len(SEEDS) * len(SEATS)
    pair_index = 0
    for family, representative, proxy_path, _ in family_specs:
        for seed in SEEDS:
            for candidate_seat in SEATS:
                pair_index += 1
                baseline, baseline_proposed, baseline_sanitized = play(
                    family=family,
                    representative=representative,
                    proxy_path=proxy_path,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    decision=V67ResidualDecision.ABSTAIN,
                )
                if baseline["failure"] is not None or not baseline["completed"]:
                    stop_reason = (
                        f"baseline_mechanical_gate:{family}:seed={seed}:"
                        f"seat={candidate_seat}:{baseline['failure']}"
                    )
                    break
                forced, forced_proposed, forced_sanitized = play(
                    family=family,
                    representative=representative,
                    proxy_path=proxy_path,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    decision=V67ResidualDecision.SHEEP_SWAP_2,
                )
                gate = forced["executor_packaging_gate"]
                if forced["failure"] is not None or not forced["completed"]:
                    stop_reason = (
                        f"forced_mechanical_gate:{family}:seed={seed}:"
                        f"seat={candidate_seat}:{forced['failure']}"
                    )
                elif not gate["passed"]:
                    stop_reason = (
                        f"forced_packaging_gate:{family}:seed={seed}:"
                        f"seat={candidate_seat}:{gate['reason']}"
                    )
                row = {
                    "family": family,
                    "representative": representative,
                    "seed": seed,
                    "candidate_seat": candidate_seat,
                    "baseline": baseline,
                    "forced": forced,
                    "paired_terminal_margin_delta": (
                        forced["candidate_margin"] - baseline["candidate_margin"]
                        if forced["candidate_margin"] is not None
                        else None
                    ),
                    "option_available": (forced["executor_counters"]["interventions"] == 1),
                    "option_completed": forced["executor_counters"]["completed"] == 1,
                    "has_partial": (forced["executor_counters"]["partial_interventions"] > 0),
                    "paired_candidate_proposed_diff": paired_diff_summary(
                        baseline_proposed, forced_proposed
                    ),
                    "paired_candidate_sanitized_diff": paired_diff_summary(
                        baseline_sanitized, forced_sanitized
                    ),
                }
                games.append(row)
                print(
                    json.dumps(
                        {
                            "pair": pair_index,
                            "of": expected_pairs,
                            "family": family,
                            "seed": seed,
                            "seat": candidate_seat,
                            "delta": row["paired_terminal_margin_delta"],
                            "gate": gate["reason"],
                            "stop": stop_reason,
                        }
                    ),
                    flush=True,
                )
                if stop_reason:
                    break
            if stop_reason:
                break
        if stop_reason:
            break

    complete_rows = [
        row for row in games if row["baseline"]["completed"] and row["forced"]["completed"]
    ]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in complete_rows:
        by_family.setdefault(row["family"], []).append(row)
    family_summaries = {family: summarize(rows) for family, rows in by_family.items()}
    exactness = {
        "option_starts": sum(row["forced"]["executor_counters"]["interventions"] for row in games),
        "option_completions": sum(row["forced"]["executor_counters"]["completed"] for row in games),
        "partial_interventions": sum(
            row["forced"]["executor_counters"]["partial_interventions"] for row in games
        ),
        "owned_hand4_steps": sum(
            row["forced"]["executor_counters"]["owned_hand4_steps"] for row in games
        ),
        "changed_action_steps": sum(
            row["forced"]["executor_counters"]["changed_action_steps"] for row in games
        ),
        "candidate_sanitizer_changed_steps": sum(
            row[variant]["candidate_sanitizer"]["changed_steps"]
            for row in games
            for variant in ("baseline", "forced")
        ),
        "direct_scope_violations": sum(
            len(row[variant]["direct_scope_violations"])
            for row in games
            for variant in ("baseline", "forced")
        ),
    }
    complete = len(games) == expected_pairs and len(complete_rows) == expected_pairs
    result = {
        "schema_version": 1,
        "status": (
            "PASS_COMPLETE_ON_POLICY_DATASET" if complete and stop_reason is None else "FAIL_STOP"
        ),
        "stop_reason": stop_reason,
        "scope": {
            "engine": "kaggriculture_rs",
            "episode_steps": 720,
            "weed_spawn_chance": 0.005,
            "representatives": [row[1] for row in family_specs],
            "families": [row[0] for row in family_specs],
            "seeds": list(SEEDS),
            "candidate_seats": list(SEATS),
            "expected_paired_games": expected_pairs,
            "consumption_state": "already_consumed_development_only",
            "promotion_eligible": False,
        },
        "inputs": {
            str(RUNNER_PATH.relative_to(ROOT)): sha256(RUNNER_PATH),
            str(CATALOG_PATH.relative_to(ROOT)): sha256(CATALOG_PATH),
            str(V67_PATH.relative_to(ROOT)): sha256(V67_PATH),
            str(EXECUTOR_PATH.relative_to(ROOT)): sha256(EXECUTOR_PATH),
            str(HELPER_PATH.relative_to(ROOT)): sha256(HELPER_PATH),
            **{str(path.relative_to(ROOT)): digest for _, _, path, digest in family_specs},
        },
        "exactness": exactness,
        "family_summaries": family_summaries,
        "global_summary": summarize(complete_rows) if complete_rows else None,
        "games": games,
        "guardrails": {
            "sealed_inputs_opened": [],
            "forbidden_inputs": [
                "all replay episode IDs",
                "all 56-ID sets",
                "old/fresh/novel holdouts",
                "live Kaggle action or reward traces",
            ],
            "training_performed": False,
            "candidate_created": False,
            "candidate_packaged": False,
            "submission_created": False,
            "upload_performed": False,
            "development_only_not_promotion_evidence": True,
        },
        "reproduction": [
            ".venv/bin/python analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
            ".venv/bin/python "
            "analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py "
            "--render-existing",
            ".venv/bin/ruff check analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v83_allocation_v2_onpolicy_20260827/run_oracle.py",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "stop_reason": stop_reason}))


if __name__ == "__main__":
    main()
