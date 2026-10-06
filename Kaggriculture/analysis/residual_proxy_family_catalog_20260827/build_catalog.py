#!/usr/bin/env python3
"""Build a development-only executable opponent-family catalog.

Only the predeclared synthetic seeds in this file are evaluated.  No replay
manifest or Kaggle episode artifact is read.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine  # noqa: E402

OUT = Path(__file__).resolve().parent
SEEDS = (27100100, 27100101, 27100102, 27100103)
SEATS = (0, 1)
CONFIG_BASE = {"episodeSteps": 720, "weedSpawnChance": 0.005}
BASELINE = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    key: str
    path: Path
    architecture_family: str
    rationale: str


SPECS = (
    AgentSpec(
        "v16",
        ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
        "v16_route_family",
        "fixed coherent V16 RC5 route with observation-only repairs",
    ),
    AgentSpec(
        "v22",
        ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py",
        "v16_route_family",
        "V16-derived route with land-3 and terminal rule; grouped with V16",
    ),
    AgentSpec(
        "v12_ryo",
        ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py",
        "ryo_selector_family",
        "state selector over repaired Ryo-derived continuations after step 120",
    ),
    AgentSpec(
        "v44_route",
        ROOT / "submissions/meta_v44_route99612231_orphan_standalone/main.py",
        "route_99612231_family",
        "distinct orphan-recovered route 99612231",
    ),
    AgentSpec(
        "robust_cluster",
        ROOT / "submissions/robust_cluster_tape_v10_standalone/main.py",
        "robust_cluster_family",
        "cluster-selected multi-tape continuation policy",
    ),
    AgentSpec(
        "live_response_ppo_v8",
        ROOT / "submissions/live_response_ppo_v8/main.py",
        "live_response_profile_family",
        "profile-conditioned PPO; screened for standalone compatibility only",
    ),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_hash(actions: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(actions)).hexdigest()


def _standalone_smoke(path: Path, observation: dict[str, Any]) -> dict[str, Any]:
    """Import and call a copied agent under Python isolated mode."""
    runner = """
import importlib.util
import inspect
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("isolated_candidate", root / "main.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot create import spec")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
agent = getattr(module, "agent")
observation = json.loads((root / "observation.json").read_text(encoding="utf-8"))
configuration = {"episodeSteps": 720, "seed": 27100100, "weedSpawnChance": 0.005}
action = (
    agent(observation, configuration)
    if len(inspect.signature(agent).parameters) >= 2
    else agent(observation)
)
if not isinstance(action, dict):
    raise TypeError(f"agent returned {type(action).__name__}")
print(json.dumps(action, sort_keys=True, separators=(",", ":")))
""".lstrip()
    with tempfile.TemporaryDirectory(prefix="kaggriculture-standalone-") as tmp:
        temp = Path(tmp)
        shutil.copy2(path, temp / "main.py")
        (temp / "runner.py").write_text(runner, encoding="utf-8")
        (temp / "observation.json").write_text(
            json.dumps(observation, sort_keys=True), encoding="utf-8"
        )
        completed = subprocess.run(
            [sys.executable, "-I", str(temp / "runner.py")],
            cwd=temp,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    stderr_tail = completed.stderr.strip().splitlines()[-1:] or []
    if stderr_tail and "live_response_ppo_v8.npz" in stderr_tail[0]:
        stderr_tail = ["FileNotFoundError: missing external model live_response_ppo_v8.npz"]
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr_tail": stderr_tail,
        "action_sha256": (
            hashlib.sha256(completed.stdout.strip().encode("utf-8")).hexdigest()
            if completed.returncode == 0
            else None
        ),
    }


def _new_stats() -> dict[str, Any]:
    return {
        "steps": 0,
        "active_steps": 0,
        "idle_steps": 0,
        "actor_slots": 0,
        "active_unit_commands": 0,
        "sanitizer_changed_steps": 0,
        "farmer_changed_steps": 0,
        "hands_changed_steps": 0,
        "market_changed_steps": 0,
        "unit_operations": Counter(),
        "market_operations": Counter(),
        "market_quantity": Counter(),
        "timed_operations": Counter(),
        "operation_first_step": {},
        "operation_last_step": {},
    }


def _record_action(
    stats: dict[str, Any],
    proposed: dict[str, Any],
    sanitized: dict[str, Any],
    step: int,
) -> None:
    stats["steps"] += 1
    if proposed != sanitized:
        stats["sanitizer_changed_steps"] += 1
    if proposed.get("farmer") != sanitized.get("farmer"):
        stats["farmer_changed_steps"] += 1
    if proposed.get("hands") != sanitized.get("hands"):
        stats["hands_changed_steps"] += 1
    if proposed.get("market") != sanitized.get("market"):
        stats["market_changed_steps"] += 1

    unit_commands = [
        sanitized.get("farmer", ["PASS"]),
        *sanitized.get("hands", []),
    ]
    market_orders = sanitized.get("market", [])
    stats["actor_slots"] += len(unit_commands)
    active = bool(market_orders)
    time_bin = min(step // 120, 5)

    for command in unit_commands:
        operation = str(command[0]) if isinstance(command, list) and command else "INVALID"
        stats["unit_operations"][operation] += 1
        stats["timed_operations"][f"b{time_bin}:unit:{operation}"] += 1
        if operation != "PASS":
            active = True
            stats["active_unit_commands"] += 1
        key = f"unit:{operation}"
        stats["operation_first_step"].setdefault(key, step)
        stats["operation_last_step"][key] = step

    for order in market_orders:
        operation = str(order[0]) if isinstance(order, list) and order else "INVALID"
        item = str(order[1]) if isinstance(order, list) and len(order) >= 2 else "NONE"
        quantity = 1
        if isinstance(order, list) and len(order) >= 3:
            try:
                quantity = int(order[2])
            except (TypeError, ValueError):
                quantity = 0
        market_key = f"{operation}:{item}"
        stats["market_operations"][market_key] += 1
        stats["market_quantity"][market_key] += quantity
        stats["timed_operations"][f"b{time_bin}:market:{market_key}"] += quantity
        key = f"market:{market_key}"
        stats["operation_first_step"].setdefault(key, step)
        stats["operation_last_step"][key] = step

    stats["active_steps" if active else "idle_steps"] += 1


def _jsonable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    steps = int(stats["steps"])
    result: dict[str, Any] = {}
    for key, value in stats.items():
        result[key] = dict(sorted(value.items())) if isinstance(value, Counter) else value
    result["active_step_rate"] = stats["active_steps"] / steps if steps else 0.0
    result["idle_step_rate"] = stats["idle_steps"] / steps if steps else 1.0
    result["sanitizer_changed_step_rate"] = (
        stats["sanitizer_changed_steps"] / steps if steps else 1.0
    )
    return result


def _private_summary(observation: dict[str, Any]) -> dict[str, Any]:
    private = observation.get("private", {})
    inventory = Counter()
    for unit_inventory in private.get("inventories", []):
        if isinstance(unit_inventory, dict):
            inventory.update(
                {str(item): int(quantity) for item, quantity in unit_inventory.items()}
            )
    return {
        "shed": dict(sorted(private.get("shed", {}).items())),
        "seeds": dict(sorted(private.get("seeds", {}).items())),
        "unit_inventory": dict(sorted(inventory.items())),
    }


def _play(spec: AgentSpec, seed: int, proxy_seat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _, proxy, proxy_accepts_config = load_agent(spec.path)
    _, baseline, baseline_accepts_config = load_agent(BASELINE)
    configuration = {**CONFIG_BASE, "seed": seed}
    engine = Engine(configuration)
    traces: dict[str, list[dict[str, Any]]] = {
        "proxy_proposed": [],
        "proxy_sanitized": [],
        "baseline_proposed": [],
        "baseline_sanitized": [],
    }
    proxy_stats = _new_stats()
    baseline_stats = _new_stats()

    for step in range(719):
        actions: list[dict[str, Any]] = []
        for player in range(2):
            is_proxy = player == proxy_seat
            policy = proxy if is_proxy else baseline
            accepts_config = proxy_accepts_config if is_proxy else baseline_accepts_config
            observation = engine.observation(player)
            proposed = normalize_legacy_action(
                run_agent(policy, accepts_config, observation, configuration)
            )
            sanitized = engine.sanitize_action(player, proposed)
            prefix = "proxy" if is_proxy else "baseline"
            traces[f"{prefix}_proposed"].append(proposed)
            traces[f"{prefix}_sanitized"].append(sanitized)
            _record_action(
                proxy_stats if is_proxy else baseline_stats,
                proposed,
                sanitized,
                step,
            )
            actions.append(sanitized)
        engine.step(actions)

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    baseline_seat = 1 - proxy_seat
    game = {
        "seed": seed,
        "proxy_seat": proxy_seat,
        "final_step": int(engine.state["step"]),
        "proxy_money": money[proxy_seat],
        "baseline_money": money[baseline_seat],
        "proxy_margin": money[proxy_seat] - money[baseline_seat],
        "proxy_final_private": _private_summary(engine.observation(proxy_seat)),
        "trace_sha256": {key: _action_hash(value) for key, value in traces.items()},
        "proxy_stats": _jsonable_stats(proxy_stats),
        "baseline_stats": _jsonable_stats(baseline_stats),
    }
    trace_record = {"seed": seed, "proxy_seat": proxy_seat, **traces}
    return game, trace_record


def _merge_stats(games: list[dict[str, Any]], key: str) -> dict[str, Any]:
    merged = _new_stats()
    additive = {
        "steps",
        "active_steps",
        "idle_steps",
        "actor_slots",
        "active_unit_commands",
        "sanitizer_changed_steps",
        "farmer_changed_steps",
        "hands_changed_steps",
        "market_changed_steps",
    }
    counter_keys = {
        "unit_operations",
        "market_operations",
        "market_quantity",
        "timed_operations",
    }
    for game in games:
        stats = game[key]
        for name in additive:
            merged[name] += int(stats[name])
        for name in counter_keys:
            merged[name].update(stats[name])
        for operation, step in stats["operation_first_step"].items():
            if operation not in merged["operation_first_step"]:
                merged["operation_first_step"][operation] = int(step)
            else:
                merged["operation_first_step"][operation] = min(
                    merged["operation_first_step"][operation], int(step)
                )
        for operation, step in stats["operation_last_step"].items():
            merged["operation_last_step"][operation] = max(
                merged["operation_last_step"].get(operation, -1), int(step)
            )
    return _jsonable_stats(merged)


def _summary(games: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(game["proxy_margin"]) for game in games]
    return {
        "games": len(games),
        "wins_vs_v67": sum(value > 0 for value in margins),
        "draws_vs_v67": sum(value == 0 for value in margins),
        "losses_vs_v67": sum(value < 0 for value in margins),
        "win_rate_vs_v67": sum(value > 0 for value in margins) / len(margins),
        "mean_margin_vs_v67": fmean(margins),
        "median_margin_vs_v67": median(margins),
        "worst_margin_vs_v67": min(margins),
        "all_complete": all(int(game["final_step"]) == 719 for game in games),
    }


def _l1_histogram_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_counts = Counter(left["timed_operations"])
    right_counts = Counter(right["timed_operations"])
    keys = set(left_counts) | set(right_counts)
    left_total = sum(left_counts.values()) or 1
    right_total = sum(right_counts.values()) or 1
    return 0.5 * sum(
        abs(left_counts[key] / left_total - right_counts[key] / right_total) for key in keys
    )


def _trace_mismatch(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    total = 0
    mismatches = 0
    right_index = {(row["seed"], row["proxy_seat"]): row for row in right}
    for left_row in left:
        right_row = right_index[(left_row["seed"], left_row["proxy_seat"])]
        for left_action, right_action in zip(
            left_row["proxy_sanitized"], right_row["proxy_sanitized"], strict=True
        ):
            total += 1
            mismatches += left_action != right_action
    return mismatches / total if total else math.nan


def main() -> None:
    initial = Engine({**CONFIG_BASE, "seed": SEEDS[0]}).observation(0)
    standalone = {"v67_baseline": _standalone_smoke(BASELINE, initial)}
    standalone.update({spec.key: _standalone_smoke(spec.path, initial) for spec in SPECS})

    results: dict[str, Any] = {}
    trace_bundle: dict[str, list[dict[str, Any]]] = {}
    for spec in SPECS:
        if not standalone[spec.key]["passed"]:
            results[spec.key] = {
                "path": str(spec.path.relative_to(ROOT)),
                "source_sha256": _sha256(spec.path),
                "architecture_family": spec.architecture_family,
                "rationale": spec.rationale,
                "status": "rejected_non_standalone",
                "standalone_smoke": standalone[spec.key],
            }
            continue

        games: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for seed in SEEDS:
            for seat in SEATS:
                game, trace = _play(spec, seed, seat)
                games.append(game)
                traces.append(trace)
        aggregate_proxy = _merge_stats(games, "proxy_stats")
        aggregate_baseline = _merge_stats(games, "baseline_stats")
        summary = _summary(games)
        non_idle = aggregate_proxy["active_step_rate"] >= 0.05
        usable = bool(summary["all_complete"] and non_idle)
        results[spec.key] = {
            "path": str(spec.path.relative_to(ROOT)),
            "source_sha256": _sha256(spec.path),
            "architecture_family": spec.architecture_family,
            "rationale": spec.rationale,
            "status": "usable" if usable else "rejected_broken_or_idle",
            "standalone_smoke": standalone[spec.key],
            "summary": summary,
            "aggregate_proxy_stats": aggregate_proxy,
            "aggregate_v67_stats": aggregate_baseline,
            "games": games,
        }
        trace_bundle[spec.key] = traces

    usable_keys = [key for key, value in results.items() if value["status"] == "usable"]
    pairwise: dict[str, Any] = {}
    for index, left in enumerate(usable_keys):
        for right in usable_keys[index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "sanitized_action_mismatch_rate": _trace_mismatch(
                    trace_bundle[left], trace_bundle[right]
                ),
                "timed_operation_l1_distance": _l1_histogram_distance(
                    results[left]["aggregate_proxy_stats"],
                    results[right]["aggregate_proxy_stats"],
                ),
            }

    payload = {
        "schema_version": 1,
        "engine": "kaggriculture_rs",
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": _sha256(BASELINE),
        "development_seeds": list(SEEDS),
        "seats_per_seed": list(SEATS),
        "games_per_executable_proxy": len(SEEDS) * len(SEATS),
        "standalone_smoke": standalone,
        "results": results,
        "pairwise_behavior_distance": pairwise,
        "forbidden_inputs_opened": [],
    }
    result_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (OUT / "screen_results.json").write_bytes(result_bytes)

    trace_bytes = _canonical_bytes(trace_bundle) + b"\n"
    (OUT / "exact_actions.json.gz").write_bytes(gzip.compress(trace_bytes, mtime=0))
    print(
        json.dumps(
            {
                "usable": usable_keys,
                "rejected": [key for key, value in results.items() if value["status"] != "usable"],
                "screen_results_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "exact_actions_uncompressed_sha256": hashlib.sha256(trace_bytes).hexdigest(),
                "exact_actions_gzip_sha256": _sha256(OUT / "exact_actions.json.gz"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
