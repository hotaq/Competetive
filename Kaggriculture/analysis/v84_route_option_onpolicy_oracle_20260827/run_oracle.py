#!/usr/bin/env python3
"""Run the frozen V84 route-option on-policy action-support falsifier.

This script uses only synthetic, already-consumed development seeds.  It calls
all opponents live from the changing Rust-engine state and never reads replay
actions, rewards, or sealed holdouts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import kaggriculture_rs  # noqa: E402
from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine  # noqa: E402

OUT = Path(__file__).resolve().parent
RUNNER_PATH = Path(__file__).resolve()
RESULTS_PATH = OUT / "results.json"
REPORT_PATH = OUT / "REPORT.md"
CATALOG_PATH = ROOT / "analysis/residual_proxy_family_catalog_20260827/manifest.json"
CATALOG_RESULTS_PATH = ROOT / "analysis/residual_proxy_family_catalog_20260827/screen_results.json"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
V59_PATH = ROOT / "submissions/meta_v59_causal_route_selector_b120_local/main.py"
HELPER_PATH = ROOT / "scripts/evaluate_file_agent_vs_replay.py"

EXPECTED_V67_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"
SEEDS = (27100100, 27100101, 27100102, 27100103)
SEATS = (0, 1)
BRANCH_STEP = 120
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}

OPTIONS = (
    "ABSTAIN_KEEP_V67",
    "FORCE_BASE_SUFFIX",
    "FORCE_A120_SUFFIX",
    "FORCE_B120_SUFFIX",
    "FORCE_C120_SUFFIX",
)
FORCED_ROUTES = {
    "FORCE_BASE_SUFFIX": "base",
    "FORCE_A120_SUFFIX": "a120",
    "FORCE_B120_SUFFIX": "b120",
    "FORCE_C120_SUFFIX": "c120",
}
FAMILIES = (
    (
        "v16_route_family",
        "v22",
        ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py",
    ),
    (
        "ryo_selector_family",
        "v12_ryo",
        ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py",
    ),
    (
        "route_99612231_family",
        "v44_route",
        ROOT / "submissions/meta_v44_route99612231_orphan_standalone/main.py",
    ),
    (
        "robust_cluster_family",
        "robust_cluster",
        ROOT / "submissions/robust_cluster_tape_v10_standalone/main.py",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--limit-keys",
        type=int,
        default=0,
        help="Smoke-test only; a nonzero limit can never pass the experiment gate.",
    )
    parser.add_argument(
        "--reference-core-sha256",
        default="",
        help="Hash from an independent full first run used to verify reproduction.",
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Regenerate REPORT.md from the existing deterministic results.json.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def sanitizer_stats() -> dict[str, Any]:
    return {
        "changed_steps": 0,
        "farmer_changed_steps": 0,
        "hands_changed_steps": 0,
        "market_changed_steps": 0,
        "step_ids": [],
    }


def record_sanitizer(
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


def catalog_baseline(representative: str, seed: int, candidate_seat: int) -> dict[str, Any]:
    payload = json.loads(CATALOG_RESULTS_PATH.read_text(encoding="utf-8"))
    proxy_seat = 1 - candidate_seat
    matches = [
        game
        for game in payload["results"][representative]["games"]
        if int(game["seed"]) == seed and int(game["proxy_seat"]) == proxy_seat
    ]
    if len(matches) != 1:
        raise AssertionError(f"catalog key mismatch: {representative=} {seed=} {candidate_seat=}")
    return matches[0]


class ForcedRouteV67:
    """Force V67's embedded V59 route choice without bypassing V67 overlays."""

    def __init__(
        self,
        agent: Any,
        accepts_configuration: bool,
        embedded_v59: Any,
        option: str,
    ) -> None:
        self.agent = agent
        self.accepts_configuration = accepts_configuration
        self.embedded_v59 = embedded_v59
        self.option = option
        self.route = FORCED_ROUTES.get(option)
        self.starts = 0
        self.selected_route: str | None = None

        suffix_keys = set(embedded_v59._SUFFIX_AGENTS)
        if suffix_keys != {"a120", "b120", "c120"}:
            raise AssertionError(f"unexpected V59 suffix catalog: {suffix_keys}")

    def __call__(
        self, observation: dict[str, Any], configuration: dict[str, Any]
    ) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        player = int(observation.get("player", 0) or 0)
        choices = self.embedded_v59._CHOICE_BY_PLAYER

        if self.route is not None and step == BRANCH_STEP:
            if player in choices:
                raise AssertionError(f"route was selected before forced branch: {choices[player]}")
            choices[player] = self.route
            self.starts += 1

        action = run_agent(
            self.agent,
            self.accepts_configuration,
            observation,
            configuration,
        )
        if step >= BRANCH_STEP:
            self.selected_route = choices.get(player)
            if self.route is not None and self.selected_route != self.route:
                raise AssertionError(
                    f"forced route drift: expected {self.route}, got {self.selected_route}"
                )
        return action


def play_option(
    *,
    family: str,
    representative: str,
    proxy_path: Path,
    seed: int,
    candidate_seat: int,
    option: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Match the frozen catalog import order: opponent first, then a fresh V67
    # standalone bundle.  Captured function/module references remain isolated
    # even when later standalone imports replace sys.modules names.
    _, proxy_agent, proxy_accepts_configuration = load_agent(proxy_path)
    _, v67_agent, v67_accepts_configuration = load_agent(V67_PATH)
    embedded_v59 = v67_agent.__globals__.get("v59")
    if embedded_v59 is None:
        raise AssertionError("frozen V67 overlay does not expose embedded v59")
    candidate = ForcedRouteV67(
        v67_agent,
        v67_accepts_configuration,
        embedded_v59,
        option,
    )

    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    candidate_proposed: list[dict[str, Any]] = []
    candidate_sanitized: list[dict[str, Any]] = []
    proxy_proposed: list[dict[str, Any]] = []
    proxy_sanitized: list[dict[str, Any]] = []
    candidate_sanitizer = sanitizer_stats()
    proxy_sanitizer = sanitizer_stats()
    exception: dict[str, Any] | None = None

    for step in range(719):
        actions: list[dict[str, Any]] = []
        try:
            for player in range(2):
                observation = engine.observation(player)
                if player == candidate_seat:
                    raw = candidate(observation, configuration)
                    proposed = normalize_legacy_action(raw)
                    sanitized = engine.sanitize_action(player, proposed)
                    candidate_proposed.append(copy.deepcopy(proposed))
                    candidate_sanitized.append(copy.deepcopy(sanitized))
                    record_sanitizer(candidate_sanitizer, step, proposed, sanitized)
                else:
                    raw = run_agent(
                        proxy_agent,
                        proxy_accepts_configuration,
                        observation,
                        configuration,
                    )
                    proposed = normalize_legacy_action(raw)
                    sanitized = engine.sanitize_action(player, proposed)
                    proxy_proposed.append(copy.deepcopy(proposed))
                    proxy_sanitized.append(copy.deepcopy(sanitized))
                    record_sanitizer(proxy_sanitizer, step, proposed, sanitized)
                actions.append(sanitized)
            engine.step(actions)
        except Exception as error:  # Evidence records the exact mechanical failure.
            exception = {
                "step": step,
                "type": type(error).__name__,
                "message": str(error),
            }
            break

    final_step = int(engine.state["step"])
    completed = final_step == 719
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    option_started = candidate.starts == 1
    option_completed = (
        option in FORCED_ROUTES
        and option_started
        and completed
        and candidate.selected_route == FORCED_ROUTES[option]
    )
    partial = int(option in FORCED_ROUTES and option_started and not option_completed)

    result = {
        "option": option,
        "family": family,
        "representative": representative,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "opponent_seat": 1 - candidate_seat,
        "completed_transitions": final_step,
        "final_step": final_step,
        "candidate_money": money[candidate_seat],
        "opponent_money": money[1 - candidate_seat],
        "candidate_margin": money[candidate_seat] - money[1 - candidate_seat],
        "candidate_sanitizer": candidate_sanitizer,
        "opponent_sanitizer": proxy_sanitizer,
        "fallback_events": 0,
        "exception_events": int(exception is not None),
        "exception": exception,
        "option_starts": int(option_started),
        "option_completions": int(option_completed),
        "partial_option_events": partial,
        "selected_route": candidate.selected_route,
        "trace_sha256": {
            "candidate_proposed": value_sha256(candidate_proposed),
            "candidate_sanitized": value_sha256(candidate_sanitized),
            "candidate_prefix_0_119_proposed": value_sha256(candidate_proposed[:BRANCH_STEP]),
            "opponent_proposed": value_sha256(proxy_proposed),
            "opponent_sanitized": value_sha256(proxy_sanitized),
        },
    }
    return result, candidate_proposed


def run_key(task: tuple[str, str, str, int, int]) -> dict[str, Any]:
    family, representative, proxy_path_text, seed, candidate_seat = task
    proxy_path = Path(proxy_path_text)
    outcomes: list[dict[str, Any]] = []
    proposed_by_option: dict[str, list[dict[str, Any]]] = {}
    for option in OPTIONS:
        outcome, proposed = play_option(
            family=family,
            representative=representative,
            proxy_path=proxy_path,
            seed=seed,
            candidate_seat=candidate_seat,
            option=option,
        )
        outcomes.append(outcome)
        proposed_by_option[option] = proposed

    abstain = proposed_by_option["ABSTAIN_KEEP_V67"]
    prefix_parity: dict[str, Any] = {}
    difference_contract: dict[str, Any] = {}
    for option in FORCED_ROUTES:
        forced = proposed_by_option[option]
        prefix_equal = forced[:BRANCH_STEP] == abstain[:BRANCH_STEP]
        differences = [
            step
            for step, (left, right) in enumerate(zip(abstain, forced, strict=False))
            if left != right
        ]
        prefix_parity[option] = {
            "equal": prefix_equal,
            "abstain_prefix_sha256": value_sha256(abstain[:BRANCH_STEP]),
            "forced_prefix_sha256": value_sha256(forced[:BRANCH_STEP]),
        }
        difference_contract[option] = {
            "different_steps": len(differences),
            "first_difference_step": differences[0] if differences else None,
            "last_difference_step": differences[-1] if differences else None,
            "no_difference_before_120": not differences or differences[0] >= BRANCH_STEP,
            "first_difference_paths": diff_paths(abstain[differences[0]], forced[differences[0]])
            if differences
            else [],
        }

    expected = catalog_baseline(representative, seed, candidate_seat)
    abstain_outcome = next(row for row in outcomes if row["option"] == "ABSTAIN_KEEP_V67")
    abstain_catalog_parity = {
        "candidate_proposed_hash": abstain_outcome["trace_sha256"]["candidate_proposed"]
        == expected["trace_sha256"]["baseline_proposed"],
        "candidate_sanitized_hash": abstain_outcome["trace_sha256"]["candidate_sanitized"]
        == expected["trace_sha256"]["baseline_sanitized"],
        "opponent_proposed_hash": abstain_outcome["trace_sha256"]["opponent_proposed"]
        == expected["trace_sha256"]["proxy_proposed"],
        "opponent_sanitized_hash": abstain_outcome["trace_sha256"]["opponent_sanitized"]
        == expected["trace_sha256"]["proxy_sanitized"],
        "candidate_money": abstain_outcome["candidate_money"] == float(expected["baseline_money"]),
        "opponent_money": abstain_outcome["opponent_money"] == float(expected["proxy_money"]),
        "candidate_margin": abstain_outcome["candidate_margin"] == -float(expected["proxy_margin"]),
        "final_step": abstain_outcome["final_step"] == int(expected["final_step"]),
    }
    abstain_catalog_parity["all"] = all(abstain_catalog_parity.values())

    margins = {row["option"]: float(row["candidate_margin"]) for row in outcomes}
    best_margin = max(margins.values())
    best_options = sorted(option for option, value in margins.items() if value == best_margin)
    baseline_margin = margins["ABSTAIN_KEEP_V67"]
    non_abstain_best = max(margins[option] for option in FORCED_ROUTES)
    return {
        "key": {
            "family": family,
            "representative": representative,
            "seed": seed,
            "candidate_seat": candidate_seat,
        },
        "outcomes": outcomes,
        "prefix_parity": prefix_parity,
        "difference_contract": difference_contract,
        "abstain_catalog_parity": abstain_catalog_parity,
        "oracle": {
            "best_margin": best_margin,
            "best_options": best_options,
            "positive": best_margin > 0,
        },
        "baseline_loss": baseline_margin <= 0,
        "baseline_margin": baseline_margin,
        "best_non_abstain_margin": non_abstain_best,
        "baseline_loss_rescued": baseline_margin <= 0 and non_abstain_best > 0,
    }


def smallest_cover(games: list[dict[str, Any]]) -> dict[str, Any]:
    for size in range(1, len(OPTIONS) + 1):
        covers: list[list[str]] = []
        for option_set in itertools.combinations(OPTIONS, size):
            if all(
                max(
                    float(outcome["candidate_margin"])
                    for outcome in game["outcomes"]
                    if outcome["option"] in option_set
                )
                > 0
                for game in games
            ):
                covers.append(list(option_set))
        if covers:
            return {"size": size, "sets": covers, "exists": True}
    return {"size": None, "sets": [], "exists": False}


def summarize(games: list[dict[str, Any]], full_grid: bool) -> dict[str, Any]:
    outcomes = [outcome for game in games for outcome in game["outcomes"]]
    family_summaries: dict[str, Any] = {}
    for family, _, _ in FAMILIES:
        rows = [game for game in games if game["key"]["family"] == family]
        if not rows:
            continue
        oracle_margins = [float(row["oracle"]["best_margin"]) for row in rows]
        baseline_margins = [float(row["baseline_margin"]) for row in rows]
        family_summaries[family] = {
            "keys": len(rows),
            "oracle_wins": sum(value > 0 for value in oracle_margins),
            "oracle_worst_margin": min(oracle_margins),
            "oracle_mean_margin": fmean(oracle_margins),
            "baseline_wins": sum(value > 0 for value in baseline_margins),
            "baseline_worst_margin": min(baseline_margins),
            "baseline_mean_margin": fmean(baseline_margins),
        }

    forced_prefix_checks = [check for game in games for check in game["prefix_parity"].values()]
    difference_checks = [check for game in games for check in game["difference_contract"].values()]
    candidate_sanitizer_changes = sum(
        int(row["candidate_sanitizer"]["changed_steps"]) for row in outcomes
    )
    exceptions = sum(int(row["exception_events"]) for row in outcomes)
    fallbacks = sum(int(row["fallback_events"]) for row in outcomes)
    partials = sum(int(row["partial_option_events"]) for row in outcomes)
    baseline_losses = [row for row in games if row["baseline_loss"]]

    mechanical = {
        "full_32_key_grid": full_grid and len(games) == 32,
        "completed_719_transitions": len(outcomes) == 160
        and all(int(row["final_step"]) == 719 for row in outcomes),
        "candidate_sanitizer_changed_steps": candidate_sanitizer_changes,
        "fallback_events": fallbacks,
        "exception_events": exceptions,
        "partial_option_events": partials,
        "v67_prefix_parity_checks": len(forced_prefix_checks),
        "v67_prefix_parity_all": len(forced_prefix_checks) == 128
        and all(check["equal"] for check in forced_prefix_checks),
        "abstain_catalog_parity_checks": len(games),
        "abstain_catalog_parity_all": len(games) == 32
        and all(game["abstain_catalog_parity"]["all"] for game in games),
        "non_abstain_difference_contract_checks": len(difference_checks),
        "non_abstain_differences_no_earlier_than_120": len(difference_checks) == 128
        and all(check["no_difference_before_120"] for check in difference_checks),
    }
    economic = {
        "oracle_positive_keys": sum(row["oracle"]["positive"] for row in games),
        "oracle_total_keys": len(games),
        "oracle_32_of_32_positive": len(games) == 32
        and all(row["oracle"]["positive"] for row in games),
        "positive_worst_every_family": len(family_summaries) == 4
        and all(float(summary["oracle_worst_margin"]) > 0 for summary in family_summaries.values()),
        "v67_development_losses": len(baseline_losses),
        "v67_losses_rescued": sum(row["baseline_loss_rescued"] for row in baseline_losses),
        "all_six_v67_losses_rescued": len(baseline_losses) == 6
        and all(row["baseline_loss_rescued"] for row in baseline_losses),
        "smallest_positive_cover": smallest_cover(games),
    }
    return {
        "mechanical": mechanical,
        "economic": economic,
        "families": family_summaries,
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "run_oracle.py": RUNNER_PATH,
        "frozen_v67": V67_PATH,
        "v59_route_source": V59_PATH,
        "catalog_manifest": CATALOG_PATH,
        "catalog_results": CATALOG_RESULTS_PATH,
        "evaluation_helper": HELPER_PATH,
        "rust_python_extension": Path(kaggriculture_rs.__file__),
    }
    for _, representative, path in FAMILIES:
        paths[f"opponent_{representative}"] = path
    return {name: sha256(path) for name, path in sorted(paths.items())}


def build_report(result: dict[str, Any], results_sha256: str) -> str:
    summary = result["summary"]
    mechanical = summary["mechanical"]
    economic = summary["economic"]
    reproduction = result["determinism"]["reproduction"]
    completed_games = sum(
        int(row["final_step"]) == 719 for game in result["games"] for row in game["outcomes"]
    )
    lines = [
        "# V84 Route-Option On-Policy Oracle — 2026-08-27",
        "",
        "## Verdict",
        "",
        f"Status: **{result['status']}**.",
        "",
        result["stop_reason"],
        "",
        "This is development-only Rust-engine falsification evidence. It did not train "
        "a model, open a replay/holdout, build a submission, or upload anything.",
        "",
        "## Frozen protocol",
        "",
        "- Options fixed before results: `ABSTAIN_KEEP_V67`, `FORCE_BASE_SUFFIX`, "
        "`FORCE_A120_SUFFIX`, `FORCE_B120_SUFFIX`, `FORCE_C120_SUFFIX`.",
        "- Forced choices are injected only into frozen V67's captured embedded V59 "
        "choice state at step 120; V67's full overlay remains callable.",
        "- Four live executable proxy families, seeds 27100100..27100103, both "
        "candidate seats: 32 keys and 160 unique option games.",
        "- Opponents are called live from changed state; no opponent replay tape is used.",
        "",
        "## Mechanical gates",
        "",
        f"- Complete 719-transition games: {completed_games} / 160.",
        f"- Candidate sanitizer / fallback / exception / partial events: "
        f"{mechanical['candidate_sanitizer_changed_steps']} / "
        f"{mechanical['fallback_events']} / {mechanical['exception_events']} / "
        f"{mechanical['partial_option_events']}.",
        f"- V67 proposed-action prefix parity through step 119: "
        f"{mechanical['v67_prefix_parity_checks']} checks, "
        f"all={mechanical['v67_prefix_parity_all']}.",
        f"- ABSTAIN full action+score parity against frozen catalog V67: "
        f"{mechanical['abstain_catalog_parity_checks']} checks, "
        f"all={mechanical['abstain_catalog_parity_all']}.",
        f"- Non-abstain differences never begin before step 120: "
        f"{mechanical['non_abstain_difference_contract_checks']} checks, "
        f"all={mechanical['non_abstain_differences_no_earlier_than_120']}.",
        f"- Independent full rerun core hash match: {reproduction['verified']} "
        f"(`{result['determinism']['core_sha256']}`).",
        "",
        "## Economic action-support gate",
        "",
        f"- Hindsight oracle positive: {economic['oracle_positive_keys']} / "
        f"{economic['oracle_total_keys']} keys (required 32/32).",
        f"- Positive oracle worst margin in every family: "
        f"{economic['positive_worst_every_family']}.",
        f"- V67 losses rescued by non-abstain options: "
        f"{economic['v67_losses_rescued']} / {economic['v67_development_losses']} "
        "(required 6/6).",
        f"- Smallest all-key positive fixed-action cover: "
        f"{economic['smallest_positive_cover']['size']}.",
        "",
        "| Family | Keys | V67 wins | Oracle wins | V67 worst | Oracle worst | Oracle mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, _, _ in FAMILIES:
        family_summary = summary["families"].get(family)
        if family_summary is None:
            continue
        lines.append(
            f"| {family} | {family_summary['keys']} | "
            f"{family_summary['baseline_wins']} | {family_summary['oracle_wins']} | "
            f"{family_summary['baseline_worst_margin']:.0f} | "
            f"{family_summary['oracle_worst_margin']:.0f} | "
            f"{family_summary['oracle_mean_margin']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails and hashes",
            "",
            "- Sealed inputs opened: none.",
            "- Old/fresh/novel replay actions or rewards opened: none.",
            "- Training/checkpoint/candidate/package/submission/upload: none.",
            f"- Frozen V67 SHA-256: `{result['sources']['frozen_v67']}`.",
            f"- Results SHA-256: `{results_sha256}`.",
            f"- Deterministic 32-key outcome core SHA-256: "
            f"`{result['determinism']['core_sha256']}`.",
            "",
            "The route catalog is frozen for this decision. No option may be added after "
            "seeing failed keys.",
            "",
        ]
    )
    return "\n".join(lines)


def build_result(
    games: list[dict[str, Any]],
    *,
    full_grid: bool,
    reference_core_sha256: str,
) -> dict[str, Any]:
    games = sorted(
        games,
        key=lambda row: (
            row["key"]["family"],
            row["key"]["seed"],
            row["key"]["candidate_seat"],
        ),
    )
    summary = summarize(games, full_grid)
    core_sha256 = value_sha256(games)
    reproduction_verified = bool(reference_core_sha256) and (reference_core_sha256 == core_sha256)
    if reference_core_sha256 and not reproduction_verified:
        raise AssertionError(
            "independent rerun core hash mismatch: "
            f"expected {reference_core_sha256}, got {core_sha256}"
        )

    mechanical = summary["mechanical"]
    mechanical_pass_without_reproduction = (
        mechanical["full_32_key_grid"]
        and mechanical["completed_719_transitions"]
        and mechanical["candidate_sanitizer_changed_steps"] == 0
        and mechanical["fallback_events"] == 0
        and mechanical["exception_events"] == 0
        and mechanical["partial_option_events"] == 0
        and mechanical["v67_prefix_parity_all"]
        and mechanical["abstain_catalog_parity_all"]
        and mechanical["non_abstain_differences_no_earlier_than_120"]
    )
    economic = summary["economic"]
    economic_pass = (
        economic["oracle_32_of_32_positive"]
        and economic["positive_worst_every_family"]
        and economic["all_six_v67_losses_rescued"]
    )
    mechanical_pass = mechanical_pass_without_reproduction and reproduction_verified

    if not full_grid:
        status = "incomplete_smoke_only"
        stop_reason = "Limited smoke run; this output cannot satisfy any V84 gate."
    elif not mechanical_pass_without_reproduction:
        status = "fail_stop_before_training"
        stop_reason = (
            "Mechanical support gate failed. Stop branch A before any checkpoint or training."
        )
    elif not reproduction_verified:
        status = "pending_independent_reproduction"
        stop_reason = (
            "The unique 160-game grid completed, but the required independent full-rerun "
            "hash comparison is still pending."
        )
    elif not economic_pass:
        status = "fail_stop_before_training"
        stop_reason = (
            "The predeclared route catalog cannot meet the positive-worst action-support "
            "gate. Stop branch A before any checkpoint or training; do not add routes post hoc."
        )
    else:
        status = "pass_action_support_only"
        stop_reason = (
            "The cheap action-support falsifier passed. This permits only the separately "
            "gated observable-selection test, not promotion or submission."
        )

    hashes = source_hashes()
    if hashes["frozen_v67"] != EXPECTED_V67_SHA256:
        raise AssertionError(
            f"frozen V67 hash drift: {hashes['frozen_v67']} != {EXPECTED_V67_SHA256}"
        )
    return {
        "schema_version": 1,
        "status": status,
        "stop_reason": stop_reason,
        "protocol": {
            "predeclared_options": list(OPTIONS),
            "branch_step": BRANCH_STEP,
            "families": [family for family, _, _ in FAMILIES],
            "representatives": [representative for _, representative, _ in FAMILIES],
            "seeds": list(SEEDS),
            "candidate_seats": list(SEATS),
            "expected_keys": 32,
            "expected_unique_games": 160,
            "engine": "official kaggriculture_rs Rust engine",
            "action_execution": (
                "normalize_legacy_action then Engine.sanitize_action then Engine.step"
            ),
            "opponent_execution": "live callable from each changed observation",
        },
        "sources": hashes,
        "games": games,
        "summary": summary,
        "gates": {
            "mechanical_pass_without_reproduction": mechanical_pass_without_reproduction,
            "mechanical_pass": mechanical_pass,
            "economic_action_support_pass": economic_pass,
        },
        "determinism": {
            "core_sha256": core_sha256,
            "reproduction": {
                "reference_core_sha256": reference_core_sha256 or None,
                "verified": reproduction_verified,
            },
        },
        "guardrails": {
            "sealed_inputs_opened": [],
            "replay_actions_or_rewards_opened": [],
            "old_fresh_novel_holdouts_opened": [],
            "training_performed": False,
            "checkpoint_created": False,
            "candidate_created": False,
            "submission_created": False,
            "upload_performed": False,
            "development_only": True,
            "post_hoc_options_added": False,
        },
    }


def main() -> None:
    args = parse_args()
    if args.render_existing:
        result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        REPORT_PATH.write_text(build_report(result, sha256(RESULTS_PATH)), encoding="utf-8")
        return

    tasks = [
        (family, representative, str(path), seed, seat)
        for family, representative, path in FAMILIES
        for seed in SEEDS
        for seat in SEATS
    ]
    full_grid = args.limit_keys <= 0
    if args.limit_keys > 0:
        tasks = tasks[: args.limit_keys]
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    games: list[dict[str, Any]] = []

    def record_progress(index: int, game: dict[str, Any]) -> None:
        games.append(game)
        key = game["key"]
        print(
            f"[{index}/{len(tasks)}] {key['family']} seed={key['seed']} "
            f"seat={key['candidate_seat']} oracle={game['oracle']['best_margin']:.0f}",
            flush=True,
        )

    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            record_progress(index, run_key(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_key, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                record_progress(index, future.result())

    result = build_result(
        games,
        full_grid=full_grid,
        reference_core_sha256=args.reference_core_sha256,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_PATH, result)
    REPORT_PATH.write_text(build_report(result, sha256(RESULTS_PATH)), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "core_sha256": result["determinism"]["core_sha256"],
                "results_sha256": sha256(RESULTS_PATH),
                "report_sha256": sha256(REPORT_PATH),
                "oracle_positive_keys": result["summary"]["economic"]["oracle_positive_keys"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
