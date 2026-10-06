"""Paired causal trace for frozen V59 versus local V74 on two old80 cases.

The scope is intentionally hard-coded.  This diagnostic reads exactly two
replays, never mutates either agent, and does not enumerate any dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
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
from kaggriculture_rl.policy.primitive_legalizer import (  # noqa: E402
    append_terminal_liquidation,
    legalize_action,
)

PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
ANIMALS = ("COW", "SHEEP")
CHECKPOINTS = (120, 144, 192, 240, 360, 480, 600, 719)


@dataclass(frozen=True, slots=True)
class Case:
    label: str
    episode_id: int
    replay: str
    top1_seat: int


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    agent: str


CASES = (
    Case(
        label="median",
        episode_id=99614514,
        replay="replays/top20_live_20260826/r10_16670032_kenjo1209/99614514.json",
        top1_seat=0,
    ),
    Case(
        label="worst",
        episode_id=99589578,
        replay="replays/top20_live_20260826/r09_16748154_taiseiu/99589578.json",
        top1_seat=0,
    ),
)
CANDIDATES = (
    Candidate(
        label="V59",
        agent="submissions/meta_v59_causal_route_selector_b120_local/main.py",
    ),
    Candidate(
        label="V74",
        agent="submissions/meta_v74_closed_loop_completion_local/main.py",
    ),
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _nonzero(mapping: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(mapping.items()) if int(value) != 0}


def _inventory_total(observation: dict[str, Any]) -> dict[str, int]:
    total: Counter[str] = Counter()
    private = _mapping(observation.get("private"))
    raw_inventories = private.get("inventories", ())
    if isinstance(raw_inventories, list):
        for raw_inventory in raw_inventories:
            for item, quantity in _mapping(raw_inventory).items():
                total[str(item)] += int(quantity)
    return _nonzero(dict(total))


def _state(observation: dict[str, Any], player: int) -> dict[str, Any]:
    farm = _mapping(observation["farms"][player])
    private = _mapping(observation.get("private"))
    snapshot = extract_v74_snapshot(observation)
    return {
        "step": int(observation["step"]),
        "day": int(observation["day"]),
        "hour": int(observation["hour"]),
        "money": float(farm.get("money", 0.0)),
        "land": snapshot.land_total,
        "hands": snapshot.hands,
        "hires_today": int(farm.get("hires_today", 0)),
        "live_animals": {animal: snapshot.live_count(animal) for animal in ANIMALS},
        "owned_animals": {animal: snapshot.owned_count(animal) for animal in ANIMALS},
        "stranded_animals": {animal: snapshot.stranded_count(animal) for animal in ANIMALS},
        "live_crops": {
            crop: snapshot.crop_count(crop) for crop in ("WHEAT", "STRAWBERRY", "MELON")
        },
        "shed": _nonzero(_mapping(private.get("shed"))),
        "unit_inventory": _inventory_total(observation),
        "seeds": _nonzero(_mapping(private.get("seeds"))),
        "market_inventory": _nonzero(_mapping(observation["market"].get("inventory"))),
        "market_prices": {item: int(observation["market"]["prices"][item]) for item in PRODUCTS},
        "town_shops": list(observation["town"].get("unlocked_shops", ())),
    }


def _orders(action: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for raw_order in action.get("market", ()):
        if not isinstance(raw_order, list) or not raw_order:
            continue
        operation = str(raw_order[0])
        item = str(raw_order[1]) if len(raw_order) > 1 else ""
        quantity = int(raw_order[2]) if len(raw_order) > 2 else 1
        result[f"{operation}:{item}" if item else operation] += quantity
    return result


def _economic_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state["money"],
        state["land"],
        state["hands"],
        tuple(state["live_animals"].items()),
        tuple(state["owned_animals"].items()),
        tuple(state["stranded_animals"].items()),
        tuple(state["live_crops"].items()),
        tuple(state["shed"].items()),
        tuple(state["unit_inventory"].items()),
        tuple(state["seeds"].items()),
    )


def _trace(case: Case, candidate: Candidate) -> dict[str, Any]:
    replay = json.loads((ROOT / case.replay).read_text(encoding="utf-8"))
    configuration = engine_config(replay)
    engine = Engine(configuration)
    module, agent, accepts_config = load_agent(ROOT / candidate.agent)
    candidate_seat = 1 - case.top1_seat
    records: list[dict[str, Any]] = []
    normalized_steps = 0
    sanitized_steps = 0

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        before = _state(observation, candidate_seat)
        proposed = run_agent(agent, accepts_config, observation, configuration)
        policy_state: dict[str, Any] = {}
        policy = getattr(module, "_POLICY", None)
        if policy is not None:
            if hasattr(policy, "targets_for"):
                policy_state["targets"] = list(policy.targets_for(candidate_seat))
            if hasattr(policy, "report_for"):
                policy_state["report"] = policy.report_for(candidate_seat)
            if hasattr(policy, "scheduler_for"):
                policy_state["scheduler"] = policy.scheduler_for(candidate_seat)
        choices = getattr(module, "_CHOICE_BY_PLAYER", None)
        if isinstance(choices, dict):
            policy_state["route_choice"] = choices.get(candidate_seat)
        proposed = append_terminal_liquidation(observation, proposed)
        proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        action = engine.sanitize_action(candidate_seat, normalized)
        normalized_steps += normalized != proposed
        sanitized_steps += action != normalized

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = action
        engine.step(actions)
        after = _state(engine.observation(candidate_seat), candidate_seat)
        records.append(
            {
                "step": before["step"],
                "before": before,
                "after": after,
                "action": action,
                "orders": dict(sorted(_orders(action).items())),
                "money_delta": after["money"] - before["money"],
                "policy": policy_state,
            }
        )

    if int(engine.state["step"]) != 719:
        raise AssertionError(f"{candidate.label}/{case.episode_id} ended at {engine.state['step']}")

    checkpoints = _checkpoints(records)
    day_rows = _days(records)
    return {
        "candidate": asdict(candidate),
        "candidate_seat": candidate_seat,
        "final_money": records[-1]["after"]["money"],
        "opponent_money": float(engine.state["farms"][case.top1_seat]["money"]),
        "margin": (
            records[-1]["after"]["money"] - float(engine.state["farms"][case.top1_seat]["money"])
        ),
        "normalized_steps": normalized_steps,
        "sanitized_steps": sanitized_steps,
        "checkpoints": checkpoints,
        "days": day_rows,
        "top_positive_market_steps": sorted(
            (
                {
                    "step": row["step"],
                    "money_delta": row["money_delta"],
                    "orders": row["orders"],
                    "before_money": row["before"]["money"],
                    "after_money": row["after"]["money"],
                    "before_shed": row["before"]["shed"],
                    "after_shed": row["after"]["shed"],
                }
                for row in records
                if row["money_delta"] > 0 and row["orders"]
            ),
            key=lambda row: (-row["money_delta"], row["step"]),
        )[:20],
        "records": records,
    }


def _checkpoints(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states = {row["step"]: row["before"] for row in records}
    states[719] = records[-1]["after"]
    result: list[dict[str, Any]] = []
    previous = 0
    cumulative: Counter[str] = Counter()
    for checkpoint in CHECKPOINTS:
        interval = [row for row in records if previous <= row["step"] < checkpoint]
        interval_orders: Counter[str] = Counter()
        for row in interval:
            interval_orders.update(row["orders"])
            cumulative.update(row["orders"])
        start_money = (
            states[previous]["money"] if previous in states else records[0]["before"]["money"]
        )
        end_money = states[checkpoint]["money"]
        result.append(
            {
                "step": checkpoint,
                "state": states[checkpoint],
                "interval": [previous, checkpoint - 1],
                "interval_money_delta": end_money - start_money,
                "interval_positive_cash": sum(
                    max(0.0, float(row["money_delta"])) for row in interval
                ),
                "interval_negative_cash": sum(
                    min(0.0, float(row["money_delta"])) for row in interval
                ),
                "peak_hands": max((row["before"]["hands"] for row in interval), default=0),
                "interval_orders": dict(sorted(interval_orders.items())),
                "cumulative_orders": dict(sorted(cumulative.items())),
            }
        )
        previous = checkpoint
    return result


def _days(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["step"] >= 120:
            grouped[row["before"]["day"]].append(row)
    result: list[dict[str, Any]] = []
    for day, rows in sorted(grouped.items()):
        orders: Counter[str] = Counter()
        for row in rows:
            orders.update(row["orders"])
        result.append(
            {
                "day": day,
                "steps": [rows[0]["step"], rows[-1]["step"]],
                "start_money": rows[0]["before"]["money"],
                "end_money": rows[-1]["after"]["money"],
                "money_delta": rows[-1]["after"]["money"] - rows[0]["before"]["money"],
                "positive_cash": sum(max(0.0, row["money_delta"]) for row in rows),
                "negative_cash": sum(min(0.0, row["money_delta"]) for row in rows),
                "peak_hands": max(row["before"]["hands"] for row in rows),
                "orders": dict(sorted(orders.items())),
                "end_live_animals": rows[-1]["after"]["live_animals"],
                "end_live_crops": rows[-1]["after"]["live_crops"],
                "end_shed": rows[-1]["after"]["shed"],
            }
        )
    return result


def _first_divergences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = left["records"]
    right_rows = right["records"]
    result: dict[str, Any] = {}
    predicates = {
        "action": lambda a, b: a["action"] != b["action"],
        "market_action": lambda a, b: a["orders"] != b["orders"],
        "pre_economic_state": lambda a, b: (
            _economic_signature(a["before"]) != _economic_signature(b["before"])
        ),
        "money_delta": lambda a, b: a["money_delta"] != b["money_delta"],
        "post_economic_state": lambda a, b: (
            _economic_signature(a["after"]) != _economic_signature(b["after"])
        ),
    }
    for label, predicate in predicates.items():
        for minimum, suffix in ((0, "all"), (120, "post_handoff")):
            for a, b in zip(left_rows, right_rows, strict=True):
                if a["step"] >= minimum and predicate(a, b):
                    result[f"{label}_{suffix}"] = {
                        "step": a["step"],
                        "v59": {
                            "money_delta": a["money_delta"],
                            "orders": a["orders"],
                            "before": a["before"],
                            "after": a["after"],
                            "action": a["action"],
                        },
                        "v74": {
                            "money_delta": b["money_delta"],
                            "orders": b["orders"],
                            "before": b["before"],
                            "after": b["after"],
                            "action": b["action"],
                        },
                    }
                    break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases: list[dict[str, Any]] = []
    for case in CASES:
        traces = {candidate.label: _trace(case, candidate) for candidate in CANDIDATES}
        cases.append(
            {
                "case": asdict(case),
                "traces": traces,
                "first_divergences": _first_divergences(traces["V59"], traces["V74"]),
            }
        )
    payload = {
        "schema_version": 1,
        "scope": "exactly_two_old80_replays",
        "checkpoints_are_pre_action_states_except_719_final": True,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
