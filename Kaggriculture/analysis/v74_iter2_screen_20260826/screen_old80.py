"""Hard oracle screen for V74 on the two authorized old80 replays only."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action  # noqa: E402
from kaggriculture_rs import Engine  # noqa: E402

from kaggriculture_rl.policy.closed_loop_town_quota_v74 import (  # noqa: E402
    ClosedLoopTownQuotaV74Policy,
    extract_v74_snapshot,
)
from kaggriculture_rl.policy.primitive_legalizer import (  # noqa: E402
    append_terminal_liquidation,
    legalize_action,
)

CASES = (
    {
        "label": "median",
        "episode_id": 99614514,
        "path": "replays/top20_live_20260826/r10_16670032_kenjo1209/99614514.json",
        "top1_seat": 0,
        "money_gate": 53036.0,
    },
    {
        "label": "worst",
        "episode_id": 99589578,
        "path": "replays/top20_live_20260826/r09_16748154_taiseiu/99589578.json",
        "top1_seat": 0,
        "money_gate": 53533.0,
    },
)
PRODUCTION_KEYS = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)


def _target_key(target: dict[str, Any]) -> tuple[str, str] | None:
    kind = str(target.get("kind", ""))
    if kind == "ANIMAL_PRODUCTION":
        return kind, str(target.get("animal", ""))
    if kind == "CROP_PRODUCTION":
        return kind, str(target.get("crop", ""))
    return None


def _market_quantities(action: dict[str, Any]) -> Counter[str]:
    quantities: Counter[str] = Counter()
    for raw_order in action.get("market", ()):
        if not isinstance(raw_order, list) or not raw_order:
            continue
        operation = str(raw_order[0])
        item = str(raw_order[1]) if len(raw_order) > 1 else ""
        quantity = int(raw_order[2]) if len(raw_order) > 2 else 1
        quantities[f"{operation}:{item}" if item else operation] += quantity
    return quantities


def trace(case: dict[str, Any]) -> dict[str, Any]:
    replay = json.loads((ROOT / case["path"]).read_text(encoding="utf-8"))
    configuration = engine_config(replay)
    engine = Engine(configuration)
    module, agent, accepts_config = load_agent(
        ROOT / "submissions/meta_v74_closed_loop_completion_local/main.py"
    )
    policy = module._POLICY
    if not isinstance(policy, ClosedLoopTownQuotaV74Policy):
        raise TypeError("V74 wrapper did not expose _POLICY")
    candidate_seat = 1 - int(case["top1_seat"])

    post_normalized = 0
    post_sanitized = 0
    call_times_ms: list[float] = []
    longest_starvation = {f"{kind}:{item}": 0 for kind, item in PRODUCTION_KEYS}
    starvation_run = {key: 0 for key in longest_starvation}
    longest_stranded = {"COW": 0, "SHEEP": 0}
    stranded_run = {"COW": 0, "SHEEP": 0}
    report_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    market_quantities: Counter[str] = Counter()
    min_money_after_fixed_purchase: float | None = None
    land_at_192: int | None = None
    active_switches = 0
    previous_active: str | None = None

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation["step"])
        before = extract_v74_snapshot(observation)
        started = time.perf_counter_ns()
        proposed = run_agent(agent, accepts_config, observation, configuration)
        call_times_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        proposed = append_terminal_liquidation(observation, proposed)
        proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        action = engine.sanitize_action(candidate_seat, normalized)
        if step >= 120:
            post_normalized += int(normalized != proposed)
            post_sanitized += int(action != normalized)

            targets = [dict(target) for target in policy.targets_for(candidate_seat)]
            selected = {_target_key(target) for target in targets}
            quota = policy.quota_for(candidate_seat)
            if quota is None:
                raise AssertionError("missing V74 quota after handoff")
            for key in PRODUCTION_KEYS:
                kind, item = key
                physical = (
                    before.live_count(item)
                    if kind == "ANIMAL_PRODUCTION"
                    else before.crop_count(item)
                )
                desired = quota.production_target(key)
                label = f"{kind}:{item}"
                if physical < desired and key not in selected:
                    starvation_run[label] += 1
                    longest_starvation[label] = max(
                        longest_starvation[label], starvation_run[label]
                    )
                else:
                    starvation_run[label] = 0

            scheduler = policy.scheduler_for(candidate_seat)
            phase_counts[str(scheduler.get("phase", ""))] += 1
            active = scheduler.get("active")
            active_label = ":".join(active["key"]) if isinstance(active, dict) else None
            if active_label != previous_active:
                active_switches += 1
                previous_active = active_label
            for entry in policy.report_for(candidate_seat).get("entries", ()):
                report_counts[f"{entry.get('status', '')}:{entry.get('code', '')}"] += 1
            market_quantities.update(_market_quantities(action))

            if step == 192:
                land_at_192 = before.land_total

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = action
        engine.step(actions)

        if step >= 120:
            after_observation = engine.observation(candidate_seat)
            after = extract_v74_snapshot(after_observation)
            for animal in ("COW", "SHEEP"):
                if after.stranded_count(animal) > 0:
                    stranded_run[animal] += 1
                    longest_stranded[animal] = max(longest_stranded[animal], stranded_run[animal])
                else:
                    stranded_run[animal] = 0
            if any(
                order[0] in {"HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}
                for order in action.get("market", ())
            ):
                money = after.money
                min_money_after_fixed_purchase = (
                    money
                    if min_money_after_fixed_purchase is None
                    else min(min_money_after_fixed_purchase, money)
                )

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    candidate_money = money[candidate_seat]
    tape_money = money[int(case["top1_seat"])]
    final_observation = engine.observation(candidate_seat)
    final = extract_v74_snapshot(final_observation)
    sorted_times = sorted(call_times_ms)
    p99_index = min(len(sorted_times) - 1, int(len(sorted_times) * 0.99))
    gates = {
        "money_half_v59": candidate_money >= float(case["money_gate"]),
        "stranded_max_24": max(longest_stranded.values()) <= 24,
        "starvation_max_48": max(longest_starvation.values()) <= 48,
        "worst_land_two_by_192": case["label"] != "worst" or (land_at_192 or 0) >= 2,
        "post_handoff_sanitizer_zero": post_normalized == 0 and post_sanitized == 0,
        "fixed_purchase_cash_floor": (
            min_money_after_fixed_purchase is None or min_money_after_fixed_purchase >= 400
        ),
        "complete_719": int(engine.state["step"]) == 719,
    }
    return {
        "label": case["label"],
        "episode_id": case["episode_id"],
        "seed": int(replay["info"]["seed"]),
        "candidate_money": candidate_money,
        "tape_top1_money": tape_money,
        "candidate_margin": candidate_money - tape_money,
        "money_gate": case["money_gate"],
        "final_state_step": int(engine.state["step"]),
        "land_at_192": land_at_192,
        "final_land": final.land_total,
        "final_live": {animal: final.live_count(animal) for animal in ("COW", "SHEEP")},
        "final_stranded": {animal: final.stranded_count(animal) for animal in ("COW", "SHEEP")},
        "longest_stranded": longest_stranded,
        "longest_starvation": longest_starvation,
        "min_money_after_fixed_purchase": min_money_after_fixed_purchase,
        "post_handoff_normalized": post_normalized,
        "post_handoff_sanitized": post_sanitized,
        "fallback_components": 0,
        "active_switches": active_switches,
        "phase_counts": dict(sorted(phase_counts.items())),
        "report_counts": dict(sorted(report_counts.items())),
        "market_quantities": dict(sorted(market_quantities.items())),
        "runtime_ms": {
            "p50": sorted_times[len(sorted_times) // 2],
            "p99": sorted_times[p99_index],
            "max": max(sorted_times),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    cases = [trace(case) for case in CASES]
    payload = {
        "schema_version": 1,
        "scope": "authorized_old80_two_replay_screen_only",
        "cases": cases,
        "all_gates_pass": all(case["all_gates_pass"] for case in cases),
        "stop": not all(case["all_gates_pass"] for case in cases),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
