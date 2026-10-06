"""Two-replay hard screen for the V75 maintenance-admission hypothesis."""

from __future__ import annotations

import json
import sys
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
    extract_v74_snapshot,
)
from kaggriculture_rl.policy.closed_loop_town_quota_v75 import (  # noqa: E402
    ClosedLoopTownQuotaV75Policy,
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
KEYS = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)


def _selected(targets: tuple[dict[str, Any], ...]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for target in targets:
        kind = str(target.get("kind", ""))
        if kind == "ANIMAL_PRODUCTION":
            selected.add((kind, str(target.get("animal", ""))))
        elif kind == "CROP_PRODUCTION":
            selected.add((kind, str(target.get("crop", ""))))
    return selected


def _has_fixed_purchase(action: dict[str, Any]) -> bool:
    return any(
        order[0] in {"HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}
        for order in action.get("market", ())
    )


def trace(case: dict[str, Any]) -> dict[str, Any]:
    replay = json.loads((ROOT / case["path"]).read_text(encoding="utf-8"))
    configuration = engine_config(replay)
    engine = Engine(configuration)
    module, agent, accepts_config = load_agent(
        ROOT / "submissions/meta_v75_maintenance_admission_local/main.py"
    )
    policy = module._POLICY
    if not isinstance(policy, ClosedLoopTownQuotaV75Policy):
        raise TypeError("V75 wrapper did not expose _POLICY")
    top1_seat = int(case["top1_seat"])
    candidate_seat = 1 - top1_seat

    normalized_count = 0
    sanitized_count = 0
    stranded_run = {animal: 0 for animal in ("COW", "SHEEP")}
    longest_stranded = dict(stranded_run)
    starvation_run = {f"{kind}:{item}": 0 for kind, item in KEYS}
    longest_starvation = dict(starvation_run)
    land_at_192: int | None = None
    min_money_after_purchase: float | None = None

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation["step"])
        before = extract_v74_snapshot(observation)
        proposed = run_agent(agent, accepts_config, observation, configuration)
        proposed = append_terminal_liquidation(observation, proposed)
        proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        action = engine.sanitize_action(candidate_seat, normalized)

        if step >= 120:
            normalized_count += int(normalized != proposed)
            sanitized_count += int(action != normalized)
            quota = policy.quota_for(candidate_seat)
            if quota is None:
                raise AssertionError("missing V75 quota")
            selected = _selected(policy.targets_for(candidate_seat))
            for kind, item in KEYS:
                current = (
                    before.live_count(item)
                    if kind == "ANIMAL_PRODUCTION"
                    else before.crop_count(item)
                )
                label = f"{kind}:{item}"
                if current < quota.production_target((kind, item)) and (kind, item) not in selected:
                    starvation_run[label] += 1
                    longest_starvation[label] = max(
                        longest_starvation[label], starvation_run[label]
                    )
                else:
                    starvation_run[label] = 0
            if step == 192:
                land_at_192 = before.land_total

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = action
        engine.step(actions)

        if step >= 120:
            after = extract_v74_snapshot(engine.observation(candidate_seat))
            for animal in ("COW", "SHEEP"):
                if after.stranded_count(animal) > 0:
                    stranded_run[animal] += 1
                    longest_stranded[animal] = max(longest_stranded[animal], stranded_run[animal])
                else:
                    stranded_run[animal] = 0
            if _has_fixed_purchase(action):
                min_money_after_purchase = (
                    after.money
                    if min_money_after_purchase is None
                    else min(min_money_after_purchase, after.money)
                )

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    candidate_money = money[candidate_seat]
    tape_money = money[top1_seat]
    final = extract_v74_snapshot(engine.observation(candidate_seat))
    instrumentation = policy.instrumentation_for(candidate_seat)
    gates = {
        "money": candidate_money >= float(case["money_gate"]),
        "worst_land_at_192": case["label"] != "worst" or (land_at_192 or 0) >= 2,
        "post_handoff_sanitizer": normalized_count == 0 and sanitized_count == 0,
        "stranded": max(longest_stranded.values()) <= 24,
        "starvation": max(longest_starvation.values()) <= 48,
        "cash_floor": min_money_after_purchase is None or min_money_after_purchase >= 400,
        "complete": int(engine.state["step"]) == 719,
    }
    return {
        "label": case["label"],
        "episode_id": case["episode_id"],
        "seed": int(replay["info"]["seed"]),
        "candidate_money": candidate_money,
        "tape_top1_money": tape_money,
        "candidate_margin": candidate_money - tape_money,
        "money_gate": case["money_gate"],
        "land_at_192": land_at_192,
        "final_land": final.land_total,
        "final_live": {animal: final.live_count(animal) for animal in ("COW", "SHEEP")},
        "final_stranded": {animal: final.stranded_count(animal) for animal in ("COW", "SHEEP")},
        "longest_stranded": longest_stranded,
        "longest_starvation": longest_starvation,
        "min_money_after_fixed_purchase": min_money_after_purchase,
        "post_handoff_normalized": normalized_count,
        "post_handoff_sanitized": sanitized_count,
        "fallback_components": 0,
        "instrumentation": instrumentation,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    cases = [trace(case) for case in CASES]
    passed = all(case["all_gates_pass"] for case in cases)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "authorized_old80_two_replay_screen_only",
                "cases": cases,
                "all_gates_pass": passed,
                "stop": not passed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
