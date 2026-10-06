#!/usr/bin/env python3
"""Run the exact development-only V67 allocation residual on-policy oracle.

The runner is intentionally limited to the four representatives and four
already-consumed synthetic seeds declared in the residual proxy catalog.  It
does not open replay artifacts, holdouts, or Kaggle traces, and it does not
train or package an agent.
"""

from __future__ import annotations

import argparse
import copy
import gzip
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
    V67ResidualDecision,
    V67ResidualExecutor,
)

OUT = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"
CATALOG_PATH = ROOT / "analysis/residual_proxy_family_catalog_20260827/manifest.json"
EXACT_ACTIONS_PATH = ROOT / "analysis/residual_proxy_family_catalog_20260827/exact_actions.json.gz"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
EXECUTOR_PATH = ROOT / "src/kaggriculture_rl/policy/v67_residual.py"
HELPER_PATH = ROOT / "scripts/evaluate_file_agent_vs_replay.py"

SEEDS = (27100100, 27100101, 27100102, 27100103)
SEATS = (0, 1)
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}
EXPECTED_EXECUTOR_SHA256 = "3400d38b6d9ec9a2a8c4321bfa7a2e28e375ccdb4dca3dfe5d9aa94fade4c296"
EXPECTED_V67_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"

LINKED = {
    248: ("market", 1, ["BUY_ANIMAL", "COW", 2], ["BUY_ANIMAL", "SHEEP", 2]),
    249: ("hands", 4, ["PICKUP", "COW", 2], ["PICKUP", "SHEEP", 2]),
    256: ("hands", 4, ["PLACE", "COW", 1], ["PLACE", "SHEEP", 1]),
    261: ("hands", 4, ["PLACE", "COW", 1], ["PLACE", "SHEEP", 1]),
}


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


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value)))


def _new_sanitizer_stats() -> dict[str, Any]:
    return {
        "changed_steps": 0,
        "farmer_changed_steps": 0,
        "hands_changed_steps": 0,
        "market_changed_steps": 0,
        "step_ids": [],
    }


def _record_sanitizer_change(
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


def _diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        paths = []
        for index in range(max(len(left), len(right))):
            child = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[index], right[index], child))
        return paths
    return [] if left == right else [prefix]


def _private_summary(observation: dict[str, Any]) -> dict[str, Any]:
    private = observation.get("private", {})
    unit_inventory: Counter[str] = Counter()
    for unit in private.get("units", []):
        for item, quantity in unit.get("inventory", {}).items():
            unit_inventory[str(item)] += int(quantity)
    return {
        "money": float(private.get("money", 0.0)),
        "shed": dict(sorted(private.get("shed", {}).items())),
        "seeds": dict(sorted(private.get("seeds", {}).items())),
        "unit_inventory": dict(sorted(unit_inventory.items())),
    }


def _play(
    *,
    family: str,
    representative: str,
    proxy_path: Path,
    seed: int,
    candidate_seat: int,
    decision: V67ResidualDecision,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, proxy_agent, proxy_accepts_config = load_agent(proxy_path)
    _, base_agent, _ = load_agent(V67_PATH)
    executor = V67ResidualExecutor(
        base_agent,
        lambda _observation: decision,
        eligibility=lambda _observation: True,
    )

    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    candidate_proposed_trace: list[dict[str, Any]] = []
    candidate_sanitized_trace: list[dict[str, Any]] = []
    proxy_proposed_trace: list[dict[str, Any]] = []
    proxy_sanitized_trace: list[dict[str, Any]] = []
    candidate_sanitizer = _new_sanitizer_stats()
    proxy_sanitizer = _new_sanitizer_stats()

    for step in range(719):
        actions: list[dict[str, Any]] = []
        for player in range(2):
            observation = engine.observation(player)
            if player == candidate_seat:
                raw = executor(observation, configuration)
                if not isinstance(raw, dict):
                    raise TypeError(f"candidate returned {type(raw).__name__}")
                proposed = normalize_legacy_action(raw)
                sanitized = engine.sanitize_action(player, proposed)
                candidate_proposed_trace.append(copy.deepcopy(proposed))
                candidate_sanitized_trace.append(copy.deepcopy(sanitized))
                _record_sanitizer_change(candidate_sanitizer, step, proposed, sanitized)
            else:
                raw = run_agent(
                    proxy_agent,
                    proxy_accepts_config,
                    observation,
                    configuration,
                )
                proposed = normalize_legacy_action(raw)
                sanitized = engine.sanitize_action(player, proposed)
                proxy_proposed_trace.append(copy.deepcopy(proposed))
                proxy_sanitized_trace.append(copy.deepcopy(sanitized))
                _record_sanitizer_change(proxy_sanitizer, step, proposed, sanitized)
            actions.append(sanitized)
        engine.step(actions)

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    proxy_seat = 1 - candidate_seat
    candidate_margin = money[candidate_seat] - money[proxy_seat]
    counters = executor.counters_for(candidate_seat)
    packaging_gate = executor.packaging_gate_for(candidate_seat)
    trace_events = [_jsonable_dataclass(event) for event in executor.trace_for(candidate_seat)]
    run = {
        "family": family,
        "representative": representative,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "proxy_seat": proxy_seat,
        "decision": decision.value,
        "final_step": int(engine.state["step"]),
        "candidate_money": money[candidate_seat],
        "proxy_money": money[proxy_seat],
        "candidate_margin": candidate_margin,
        "candidate_win": candidate_margin > 0,
        "executor_counters": _jsonable_dataclass(counters),
        "executor_packaging_gate": _jsonable_dataclass(packaging_gate),
        "executor_trace": trace_events,
        "candidate_sanitizer": candidate_sanitizer,
        "proxy_sanitizer": proxy_sanitizer,
        "candidate_proposed_trace_sha256": value_sha256(candidate_proposed_trace),
        "candidate_sanitized_trace_sha256": value_sha256(candidate_sanitized_trace),
        "proxy_proposed_trace_sha256": value_sha256(proxy_proposed_trace),
        "proxy_sanitized_trace_sha256": value_sha256(proxy_sanitized_trace),
        "candidate_final_private": _private_summary(engine.observation(candidate_seat)),
    }
    return run, candidate_proposed_trace, candidate_sanitized_trace


def _stream_contract(
    baseline: list[dict[str, Any]],
    forced: list[dict[str, Any]],
    stream: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(baseline) != 719 or len(forced) != 719:
        raise RuntimeError("candidate trace length is not 719")
    linked_differences: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for step, (left, right) in enumerate(zip(baseline, forced, strict=True)):
        if left == right:
            continue
        paths = _diff_paths(left, right)
        if step not in LINKED:
            violations.append(
                {
                    "stream": stream,
                    "step": step,
                    "reason": "difference_outside_linked_steps",
                    "diff_paths": paths,
                    "baseline": left,
                    "forced": right,
                }
            )
            continue
        component, index, expected_cow, expected_sheep = LINKED[step]
        left_commands = left.get(component)
        right_commands = right.get(component)
        signature_ok = (
            isinstance(left_commands, list)
            and isinstance(right_commands, list)
            and len(left_commands) > index
            and len(right_commands) > index
            and left_commands[index] == expected_cow
            and right_commands[index] == expected_sheep
        )
        restored = copy.deepcopy(right)
        if signature_ok:
            restored[component][index] = copy.deepcopy(expected_cow)
        if not signature_ok or restored != left:
            violations.append(
                {
                    "stream": stream,
                    "step": step,
                    "reason": "linked_step_has_out_of_contract_difference",
                    "diff_paths": paths,
                    "baseline": left,
                    "forced": right,
                }
            )
            continue
        linked_differences.append(
            {
                "stream": stream,
                "step": step,
                "command_path": f"{component}[{index}]",
                "leaf_diff_paths": paths,
                "baseline_command": expected_cow,
                "forced_command": expected_sheep,
            }
        )
    return linked_differences, violations


def _find_catalog_game(
    exact_actions: dict[str, list[dict[str, Any]]], game: dict[str, Any]
) -> dict[str, Any]:
    expected_proxy_seat = 1 - int(game["candidate_seat"])
    matches = [
        row
        for row in exact_actions[game["representative"]]
        if int(row["seed"]) == int(game["seed"]) and int(row["proxy_seat"]) == expected_proxy_seat
    ]
    if len(matches) != 1:
        raise RuntimeError("catalog baseline trace is not uniquely identified")
    return matches[0]


def _reconstruct_forced_trace(
    game: dict[str, Any],
    baseline_trace: list[dict[str, Any]],
    stream: str,
) -> list[dict[str, Any]]:
    forced_trace = copy.deepcopy(baseline_trace)
    for difference in game[f"linked_{stream}_differences"]:
        step = int(difference["step"])
        component, index, _, expected_sheep = LINKED[step]
        forced_trace[step][component][index] = copy.deepcopy(expected_sheep)
    for violation in game["out_of_contract_action_differences"]:
        if violation["stream"] == stream:
            forced_trace[int(violation["step"])] = copy.deepcopy(violation["forced"])
    return forced_trace


def _action_view(action: dict[str, Any]) -> dict[str, Any]:
    hands = action.get("hands", [])
    return {
        "hand4": copy.deepcopy(hands[4]) if len(hands) > 4 else None,
        "market": copy.deepcopy(action.get("market", [])),
    }


def _attach_action_windows(result: dict[str, Any]) -> None:
    with gzip.open(EXACT_ACTIONS_PATH, "rt", encoding="utf-8") as handle:
        exact_actions = json.load(handle)
    for game in result["games"]:
        catalog_game = _find_catalog_game(exact_actions, game)
        streams: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for stream in ("proposed", "sanitized"):
            baseline_trace = catalog_game[f"baseline_{stream}"]
            baseline_hash = game["baseline"][f"candidate_{stream}_trace_sha256"]
            if value_sha256(baseline_trace) != baseline_hash:
                raise RuntimeError(f"catalog {stream} baseline trace hash mismatch")
            forced_trace = _reconstruct_forced_trace(game, baseline_trace, stream)
            forced_hash = game["forced"][f"candidate_{stream}_trace_sha256"]
            if value_sha256(forced_trace) != forced_hash:
                raise RuntimeError(f"reconstructed forced {stream} trace hash mismatch")
            streams[stream] = (baseline_trace, forced_trace)

        window: list[dict[str, Any]] = []
        for step in range(248, 262):
            baseline_proposed, forced_proposed = streams["proposed"]
            baseline_sanitized, forced_sanitized = streams["sanitized"]
            window.append(
                {
                    "step": step,
                    "baseline": {
                        "proposed": _action_view(baseline_proposed[step]),
                        "sanitized": _action_view(baseline_sanitized[step]),
                    },
                    "forced": {
                        "proposed": _action_view(forced_proposed[step]),
                        "sanitized": _action_view(forced_sanitized[step]),
                    },
                    "proposed_diff_paths": _diff_paths(
                        baseline_proposed[step], forced_proposed[step]
                    ),
                    "sanitized_diff_paths": _diff_paths(
                        baseline_sanitized[step], forced_sanitized[step]
                    ),
                }
            )
        game["action_window_248_261"] = window

    result["exactness"]["action_windows_248_261"] = len(result["games"])
    result["exactness"]["out_of_contract_proposed_differences"] = sum(
        difference["stream"] == "proposed"
        for game in result["games"]
        for difference in game["out_of_contract_action_differences"]
    )
    result["exactness"]["out_of_contract_sanitized_differences"] = sum(
        difference["stream"] == "sanitized"
        for game in result["games"]
        for difference in game["out_of_contract_action_differences"]
    )
    result["scope"]["unrun_pairs_due_fail_fast"] = result["scope"]["expected_paired_games"] - len(
        result["games"]
    )
    measured_families = Counter(game["family"] for game in result["games"])
    result["family_execution_status"] = {
        family: {
            "games_executed": measured_families[family],
            "status": (
                "FAIL_STOP_PARTIAL" if measured_families[family] else "NOT_RUN_AFTER_FAIL_FAST"
            ),
        }
        for family in result["scope"]["families"]
    }
    result["inputs"][str(EXACT_ACTIONS_PATH.relative_to(ROOT))] = sha256(EXACT_ACTIONS_PATH)
    result["inputs"][str(RUNNER_PATH.relative_to(ROOT))] = sha256(RUNNER_PATH)
    result["guardrails"]["development_catalog_exact_actions_opened"] = str(
        EXACT_ACTIONS_PATH.relative_to(ROOT)
    )
    enrich_command = (
        ".venv/bin/python "
        "analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py "
        "--enrich-existing"
    )
    if enrich_command not in result["reproduction"]:
        result["reproduction"].append(enrich_command)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_margins = [float(row["baseline"]["candidate_margin"]) for row in rows]
    forced_margins = [float(row["forced"]["candidate_margin"]) for row in rows]
    deltas = [float(row["paired_terminal_margin_delta"]) for row in rows]
    baseline_wins = sum(margin > 0 for margin in baseline_margins)
    forced_wins = sum(margin > 0 for margin in forced_margins)
    return {
        "games": len(rows),
        "option_available_games": sum(row["option_available"] for row in rows),
        "option_completed_games": sum(row["option_completed"] for row in rows),
        "partial_intervention_games": sum(row["has_partial"] for row in rows),
        "beneficial_games": sum(delta > 0 for delta in deltas),
        "neutral_games": sum(delta == 0 for delta in deltas),
        "regressed_games": sum(delta < 0 for delta in deltas),
        "baseline_wins": baseline_wins,
        "forced_wins": forced_wins,
        "win_delta": forced_wins - baseline_wins,
        "baseline_mean_margin": fmean(baseline_margins),
        "forced_mean_margin": fmean(forced_margins),
        "mean_margin_delta": fmean(forced_margins) - fmean(baseline_margins),
        "baseline_worst_margin": min(baseline_margins),
        "forced_worst_margin": min(forced_margins),
        "worst_margin_delta": min(forced_margins) - min(baseline_margins),
        "paired_delta_mean": fmean(deltas),
        "paired_delta_worst": min(deltas),
        "paired_delta_best": max(deltas),
        "any_paired_regression": any(delta < 0 for delta in deltas),
        "aggregate_mean_regression": fmean(deltas) < 0,
        "worst_margin_regression": min(forced_margins) < min(baseline_margins),
    }


def _build_report_legacy(result: dict[str, Any]) -> str:
    lines = [
        "# V81 Allocation On-Policy Oracle — 2026-08-27",
        "",
        "## Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This is a development-only paired Rust-engine oracle. It does not train, "
        "create, package, or upload a candidate and is not promotion evidence.",
        "",
    ]
    if result["stop_reason"]:
        lines.extend(
            [
                f"Fail-fast stop reason: `{result['stop_reason']}`.",
                "",
            ]
        )
    if result["family_summaries"]:
        lines.extend(
            [
                "## Availability versus benefit",
                "",
                "Availability means the strict executor found the exact V67 start "
                "signature and started the four-command option. Benefit is separate: "
                "the forced run must improve paired terminal margin.",
                "",
                "| Family | Games | Available / complete | Benefit / neutral / regress | "
                "Wins base→forced (Δ) | Mean base→forced (Δ) | Worst base→forced (Δ) | "
                "Paired Δ mean / worst | Any regression |",
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
    exact = result["exactness"]
    lines.extend(
        [
            "## Executor and action-contract evidence",
            "",
            f"- Completed paired games: {len(result['games'])} / "
            f"{result['scope']['expected_paired_games']}.",
            f"- Forced option starts/completions/partials: "
            f"{exact['forced_option_starts']} / {exact['forced_option_completions']} / "
            f"{exact['forced_partial_interventions']}.",
            f"- Candidate sanitizer-changed steps, ABSTAIN/forced: "
            f"{exact['baseline_candidate_sanitizer_changed_steps']} / "
            f"{exact['forced_candidate_sanitizer_changed_steps']}.",
            f"- Exact linked proposed/sanitized command differences: "
            f"{exact['linked_proposed_differences']} / "
            f"{exact['linked_sanitized_differences']}.",
            f"- Out-of-contract candidate action differences: "
            f"{exact['out_of_contract_action_differences']}.",
            "",
            "Proxy actions are allowed to react on-policy to the changed state. The "
            "four-path constraint applies to the candidate's proposed and sanitized "
            "action streams.",
            "",
            "## Frozen scope",
            "",
            "- Representatives: `v22`, `v12_ryo`, `v44_route`, `robust_cluster`.",
            "- Already-consumed seeds only: `27100100..27100103`; both candidate seats.",
            "- No replay episode, 56-ID set, old/fresh/novel holdout, or live Kaggle "
            "action/reward trace was opened.",
            "- No training, archive creation, packaging, submission, or upload.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff check analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            "```",
            "",
            "Input hashes and per-run exact action-trace hashes are stored in `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _build_report(result: dict[str, Any]) -> str:
    exact = result["exactness"]
    lines = [
        "# V81 Allocation On-Policy Oracle — 2026-08-27",
        "",
        "## Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This is a development-only paired Rust-engine oracle. It does not train, "
        "create, package, or upload a candidate and is not promotion evidence.",
        "",
        f"Fail-fast stop reason: `{result['stop_reason']}`.",
        "",
        "## Availability versus benefit",
        "",
        "Availability means the exact V67 step-248 signature was present and the "
        "strict option started. Benefit separately means positive paired terminal "
        "margin delta.",
        "",
        "| Family | Games | Available / complete | Benefit / neutral / regress | "
        "Wins base→forced (Δ) | Mean base→forced (Δ) | Worst base→forced (Δ) | "
        "Paired Δ mean / worst | Any regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
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
    lines.extend(
        [
            "",
            "Only the first family has a result: fail-fast left the other three "
            "families unmeasured, so no cross-family benefit claim is possible.",
            "",
            "## Executor and action-contract evidence",
            "",
            f"- Completed paired games: {len(result['games'])} / "
            f"{result['scope']['expected_paired_games']}.",
            f"- Forced option starts/completions/partials: "
            f"{exact['forced_option_starts']} / {exact['forced_option_completions']} / "
            f"{exact['forced_partial_interventions']}.",
            f"- Candidate sanitizer-changed steps, ABSTAIN/forced: "
            f"{exact['baseline_candidate_sanitizer_changed_steps']} / "
            f"{exact['forced_candidate_sanitizer_changed_steps']}.",
            f"- Exact linked proposed/sanitized command differences: "
            f"{exact['linked_proposed_differences']} / "
            f"{exact['linked_sanitized_differences']}.",
            f"- Out-of-contract candidate action differences: "
            f"{exact['out_of_contract_action_differences']} "
            "(proposed and sanitized counted separately).",
            "",
        ]
    )

    failed_game = result["games"][-1]
    partial_events = [
        event
        for event in failed_game["forced"]["executor_trace"]
        if event["event"] == "PARTIAL_ABORT"
    ]
    if partial_events:
        event = partial_events[-1]
        lines.extend(
            [
                "## First exact failure",
                "",
                f"The option rewrote step 248, then aborted at step {event['step']} "
                f"with `{event['reason']}`. The on-policy V67 base emitted "
                "`hands[4]=PASS` after the SHEEP purchase instead of the frozen "
                "`PICKUP COW 2` token, so the current executor cannot complete "
                "the chain.",
                "",
                f"This diagnostic game regressed by "
                f"{failed_game['paired_terminal_margin_delta']:+.0f} terminal margin "
                f"(baseline {failed_game['baseline']['candidate_margin']:+.0f}, "
                f"forced {failed_game['forced']['candidate_margin']:+.0f}).",
                "",
            ]
        )

    lines.extend(
        [
            "## Exact action window, steps 248-261",
            "",
            "Every movement step is included. Values show candidate hand 4 and the "
            "complete market list for baseline versus forced, before and after Rust "
            "sanitization.",
            "",
            "| Step | Base proposed H4 | Forced proposed H4 | Base sanitized H4 | "
            "Forced sanitized H4 | Base proposed market | Forced proposed market |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in failed_game["action_window_248_261"]:
        lines.append(
            f"| {row['step']} | "
            f"`{_compact(row['baseline']['proposed']['hand4'])}` | "
            f"`{_compact(row['forced']['proposed']['hand4'])}` | "
            f"`{_compact(row['baseline']['sanitized']['hand4'])}` | "
            f"`{_compact(row['forced']['sanitized']['hand4'])}` | "
            f"`{_compact(row['baseline']['proposed']['market'])}` | "
            f"`{_compact(row['forced']['proposed']['market'])}` |"
        )
    lines.extend(
        [
            "",
            "Proxy actions may react on-policy to the changed state. The four-path "
            "constraint applies to the candidate's proposed and sanitized streams.",
            "",
            "## Frozen scope",
            "",
            "- Representatives: `v22`, `v12_ryo`, `v44_route`, `robust_cluster`.",
            "- Already-consumed seeds only: `27100100..27100103`; both seats.",
            "- No replay, holdout, or live Kaggle action/reward trace was opened.",
            "- No training, candidate, package, submission, or upload was created.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/python "
            "analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py "
            "--enrich-existing",
            ".venv/bin/ruff check analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            "```",
            "",
            "Input hashes and exact action-trace hashes are in `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(result: dict[str, Any]) -> None:
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_build_report(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enrich-existing",
        action="store_true",
        help="attach exact action window to existing fail-fast result; run no games",
    )
    args = parser.parse_args()
    if args.enrich_existing:
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        _attach_action_windows(result)
        _write_artifacts(result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stop_reason": result["stop_reason"],
                    "games_executed": 0,
                    "action_windows_attached": len(result["games"]),
                }
            )
        )
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
        raise RuntimeError("validated Stage0 executor hash mismatch")

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
    if [representative for _, representative, _, _ in family_specs] != [
        "v22",
        "v12_ryo",
        "v44_route",
        "robust_cluster",
    ]:
        raise RuntimeError("catalog representative scope changed")

    games: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for family, representative, proxy_path, _ in family_specs:
        for seed in SEEDS:
            for candidate_seat in SEATS:
                baseline, baseline_proposed, baseline_sanitized = _play(
                    family=family,
                    representative=representative,
                    proxy_path=proxy_path,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    decision=V67ResidualDecision.ABSTAIN,
                )
                forced, forced_proposed, forced_sanitized = _play(
                    family=family,
                    representative=representative,
                    proxy_path=proxy_path,
                    seed=seed,
                    candidate_seat=candidate_seat,
                    decision=V67ResidualDecision.SHEEP_SWAP_2,
                )
                proposed_linked, proposed_violations = _stream_contract(
                    baseline_proposed, forced_proposed, "proposed"
                )
                sanitized_linked, sanitized_violations = _stream_contract(
                    baseline_sanitized, forced_sanitized, "sanitized"
                )
                counters = forced["executor_counters"]
                gate = forced["executor_packaging_gate"]
                row = {
                    "family": family,
                    "representative": representative,
                    "seed": seed,
                    "candidate_seat": candidate_seat,
                    "baseline": baseline,
                    "forced": forced,
                    "paired_terminal_margin_delta": (
                        forced["candidate_margin"] - baseline["candidate_margin"]
                    ),
                    "option_available": counters["interventions"] == 1,
                    "option_completed": counters["completed"] == 1,
                    "has_partial": counters["partial_interventions"] > 0,
                    "linked_proposed_differences": proposed_linked,
                    "linked_sanitized_differences": sanitized_linked,
                    "out_of_contract_action_differences": [
                        *proposed_violations,
                        *sanitized_violations,
                    ],
                }
                games.append(row)
                if counters["partial_interventions"] > 0 or not gate["passed"]:
                    stop_reason = (
                        f"executor_partial_or_pending:{family}:seed={seed}:"
                        f"seat={candidate_seat}:{gate['reason']}"
                    )
                elif proposed_violations or sanitized_violations:
                    stop_reason = (
                        f"out_of_contract_candidate_action_diff:{family}:"
                        f"seed={seed}:seat={candidate_seat}"
                    )
                if stop_reason:
                    break
            if stop_reason:
                break
        if stop_reason:
            break

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in games:
        by_family.setdefault(row["family"], []).append(row)
    family_summaries = {family: _summarize(rows) for family, rows in by_family.items()}
    complete = len(games) == len(family_specs) * len(SEEDS) * len(SEATS)
    exactness = {
        "forced_option_starts": sum(
            row["forced"]["executor_counters"]["interventions"] for row in games
        ),
        "forced_option_completions": sum(
            row["forced"]["executor_counters"]["completed"] for row in games
        ),
        "forced_partial_interventions": sum(
            row["forced"]["executor_counters"]["partial_interventions"] for row in games
        ),
        "baseline_candidate_sanitizer_changed_steps": sum(
            row["baseline"]["candidate_sanitizer"]["changed_steps"] for row in games
        ),
        "forced_candidate_sanitizer_changed_steps": sum(
            row["forced"]["candidate_sanitizer"]["changed_steps"] for row in games
        ),
        "linked_proposed_differences": sum(
            len(row["linked_proposed_differences"]) for row in games
        ),
        "linked_sanitized_differences": sum(
            len(row["linked_sanitized_differences"]) for row in games
        ),
        "out_of_contract_action_differences": sum(
            len(row["out_of_contract_action_differences"]) for row in games
        ),
    }
    result = {
        "schema_version": 1,
        "status": (
            "PASS_EXACT_ON_POLICY_DATASET" if complete and stop_reason is None else "FAIL_STOP"
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
            "expected_paired_games": len(family_specs) * len(SEEDS) * len(SEATS),
            "consumption_state": "already_consumed_development_only",
            "promotion_eligible": False,
        },
        "inputs": {
            str(CATALOG_PATH.relative_to(ROOT)): sha256(CATALOG_PATH),
            str(V67_PATH.relative_to(ROOT)): sha256(V67_PATH),
            str(EXECUTOR_PATH.relative_to(ROOT)): sha256(EXECUTOR_PATH),
            str(HELPER_PATH.relative_to(ROOT)): sha256(HELPER_PATH),
            **{str(path.relative_to(ROOT)): digest for _, _, path, digest in family_specs},
        },
        "exactness": exactness,
        "family_summaries": family_summaries,
        "global_summary": _summarize(games) if games else None,
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
            ".venv/bin/python analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff check analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
            ".venv/bin/ruff format --check "
            "analysis/v81_allocation_onpolicy_oracle_20260827/run_oracle.py",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _attach_action_windows(result)
    _write_artifacts(result)
    print(json.dumps({"status": result["status"], "stop_reason": stop_reason}))


if __name__ == "__main__":
    main()
