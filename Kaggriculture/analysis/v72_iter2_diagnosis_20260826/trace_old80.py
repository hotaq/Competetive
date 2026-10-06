"""Reproduce two V72 old80 cases and summarize post-handoff failure dynamics.

This is a development-only diagnostic.  It reads only the frozen old80 replay
files named below and never mutates the candidate controller or replay corpus.
"""

from __future__ import annotations

import copy
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

from kaggriculture_rl.policy.closed_loop_town_quota import (  # noqa: E402
    ClosedLoopTownQuotaPolicy,
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
SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COSTS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_PRICES = (1000, 2000, 4000)
PRODUCTION_KEYS = (
    "ANIMAL_PRODUCTION:COW",
    "ANIMAL_PRODUCTION:SHEEP",
    "CROP_PRODUCTION:WHEAT",
    "CROP_PRODUCTION:STRAWBERRY",
    "CROP_PRODUCTION:MELON",
)


@dataclass(frozen=True, slots=True)
class Case:
    label: str
    episode_id: int
    team: str
    replay: str
    top1_seat: int
    expected_candidate_money: float
    expected_tape_money: float
    expected_margin: float


CASES = (
    Case(
        label="median",
        episode_id=99614514,
        team="Kenjo1209",
        replay="replays/top20_live_20260826/r10_16670032_kenjo1209/99614514.json",
        top1_seat=0,
        expected_candidate_money=1682.0,
        expected_tape_money=135589.0,
        expected_margin=-133907.0,
    ),
    Case(
        label="worst",
        episode_id=99589578,
        team="taiseiu",
        replay="replays/top20_live_20260826/r09_16748154_taiseiu/99589578.json",
        top1_seat=0,
        expected_candidate_money=1258.0,
        expected_tape_money=198124.0,
        expected_margin=-196866.0,
    ),
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _target_key(target: dict[str, Any]) -> str:
    kind = str(target.get("kind", ""))
    detail = {
        "ANIMAL_PRODUCTION": "animal",
        "CROP_PRODUCTION": "crop",
        "SELLING": "item",
        "INVENTORY_BUFFER": "item",
        "LAND_EXPANSION": "quadrant",
        "RECOVERY": "mode",
    }.get(kind)
    return f"{kind}:{target.get(detail, '')}" if detail else kind


def _nonzero(mapping: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in mapping.items():
        try:
            quantity = int(value)
        except (TypeError, ValueError):
            continue
        if quantity:
            result[str(key)] = quantity
    return dict(sorted(result.items()))


def _farm_state(observation: dict[str, Any], player: int) -> dict[str, Any]:
    farm = observation["farms"][player]
    private = _mapping(observation.get("private"))
    tiles = farm.get("tiles", ())
    tile_kinds: Counter[str] = Counter()
    live_animals: Counter[str] = Counter()
    live_crops: Counter[str] = Counter()
    yield_units: Counter[str] = Counter()
    if isinstance(tiles, list):
        for row in tiles:
            if not isinstance(row, list):
                continue
            for raw_tile in row:
                if not isinstance(raw_tile, dict):
                    continue
                kind = str(raw_tile.get("kind", ""))
                crop = str(raw_tile.get("crop", ""))
                animal = str(raw_tile.get("animal", ""))
                if kind:
                    tile_kinds[kind] += 1
                if crop:
                    live_crops[crop] += 1
                if animal:
                    live_animals[animal] += 1
                units = int(raw_tile.get("yield_units", 0) or 0)
                if units:
                    yield_units[crop or animal or kind] += units

    inventories = private.get("inventories", ())
    inventory_total: Counter[str] = Counter()
    unit_inventories: list[dict[str, int]] = []
    if isinstance(inventories, list):
        for raw_inventory in inventories:
            inventory = _nonzero(_mapping(raw_inventory))
            unit_inventories.append(inventory)
            inventory_total.update(inventory)

    return {
        "step": int(observation.get("step", 0)),
        "day": int(observation.get("day", 0)),
        "hour": int(observation.get("hour", 0)),
        "money": float(farm.get("money", 0.0)),
        "hands": len(farm.get("hands", ())),
        "hires_today": int(farm.get("hires_today", 0)),
        "land": len(farm.get("unlocked_quadrants", ())),
        "farmer": copy.deepcopy(farm.get("farmer")),
        "hand_positions": copy.deepcopy(farm.get("hands", ())),
        "tile_kinds": dict(sorted(tile_kinds.items())),
        "live_animals": dict(sorted(live_animals.items())),
        "live_crops": dict(sorted(live_crops.items())),
        "yield_units": dict(sorted(yield_units.items())),
        "shed": _nonzero(_mapping(private.get("shed"))),
        "seeds": _nonzero(_mapping(private.get("seeds"))),
        "inventory_total": dict(sorted(inventory_total.items())),
        "unit_inventories": unit_inventories,
    }


def _production_key(targets: list[dict[str, Any]]) -> str | None:
    for target in targets:
        key = _target_key(target)
        if key in PRODUCTION_KEYS:
            return key
    return None


def _report_rows(targets: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entries = report.get("entries", ())
    if not isinstance(entries, list):
        return rows
    for raw_entry in entries:
        entry = _mapping(raw_entry)
        index = int(entry.get("target_index", -1))
        target = targets[index] if 0 <= index < len(targets) else {}
        rows.append(
            {
                "target_index": index,
                "target_key": _target_key(target),
                "status": str(entry.get("status", "")),
                "code": str(entry.get("code", "")),
                "message": str(entry.get("message", "")),
            }
        )
    return rows


def _market_quantities(action: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_order in action.get("market", ()):
        if not isinstance(raw_order, list) or not raw_order:
            continue
        operation = str(raw_order[0])
        quantity = int(raw_order[2]) if len(raw_order) >= 3 else 1
        detail = str(raw_order[1]) if len(raw_order) >= 2 else ""
        counts[f"{operation}:{detail}" if detail else operation] += quantity
    return counts


def _listed_fixed_purchase_cost(action: dict[str, Any], before: dict[str, Any]) -> float:
    """Cost from listed fixed-price orders; excludes dynamic-price products.

    The legalizer may keep a partially fillable listed quantity, so this is an
    order-ledger upper bound rather than claimed realized cash flow.  Realized
    net movement is recorded separately from engine money before/after.
    """

    cost = 0.0
    hires_today = int(before["hires_today"])
    land = int(before["land"])
    for raw_order in action.get("market", ()):
        if not isinstance(raw_order, list) or not raw_order:
            continue
        operation = str(raw_order[0])
        if operation == "HIRE":
            a, b = 1, 1
            for _ in range(hires_today):
                a, b = b, a + b
            cost += a
            hires_today += 1
        elif operation == "BUY_LAND":
            index = max(0, land - 1)
            if index < len(LAND_PRICES):
                cost += LAND_PRICES[index]
                land += 1
        elif len(raw_order) >= 3:
            item = str(raw_order[1])
            quantity = int(raw_order[2])
            if operation == "BUY_SEED":
                cost += SEED_COSTS[item] * quantity
            elif operation == "BUY_ANIMAL":
                cost += ANIMAL_COSTS[item] * quantity
    return cost


def _composition_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state["hands"],
        state["land"],
        tuple(state["live_animals"].items()),
        tuple(state["live_crops"].items()),
        tuple(state["shed"].items()),
        tuple(state["inventory_total"].items()),
    )


def _longest_true_run(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _summarize(case: Case, records: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts: Counter[str] = Counter()
    target_runs: list[tuple[str | None, int]] = []
    report_counts: Counter[str] = Counter()
    market_quantities: Counter[str] = Counter()
    cash_by_market_signature: defaultdict[str, float] = defaultdict(float)
    cash_delta_negative = 0.0
    cash_delta_positive = 0.0
    no_market_positive = 0.0
    production_planned_without_same_step_progress = 0
    production_planned_total = 0
    listed_fixed_purchase_cost = 0.0
    first: dict[str, int] = {}
    current_target: str | None = None
    current_run = 0

    for row in records:
        key = row["production_target"]
        if key is not None:
            target_counts[key] += 1
            first.setdefault(f"target:{key}", row["step"])
        if key == current_target:
            current_run += 1
        else:
            if current_run:
                target_runs.append((current_target, current_run))
            current_target = key
            current_run = 1

        market = _market_quantities(row["action"])
        market_quantities.update(market)
        listed_fixed_purchase_cost += _listed_fixed_purchase_cost(row["action"], row["before"])
        signature = "+".join(sorted(market)) if market else "NO_MARKET"
        cash_by_market_signature[signature] += row["money_delta"]
        if row["money_delta"] < 0:
            cash_delta_negative += row["money_delta"]
        elif row["money_delta"] > 0:
            cash_delta_positive += row["money_delta"]
            if not market:
                no_market_positive += row["money_delta"]

        for report_row in row["report_entries"]:
            report_counts[
                f"{report_row['target_key']}|{report_row['status']}|{report_row['code']}"
            ] += 1
            if report_row["target_key"] in PRODUCTION_KEYS and report_row["status"] == "PLANNED":
                production_planned_total += 1
                if row["composition_before"] == row["composition_after"]:
                    production_planned_without_same_step_progress += 1

        before = row["before"]
        after = row["after"]
        if after["money"] < 1000:
            first.setdefault("money_below_1000", row["step"])
        if after["money"] < 500:
            first.setdefault("money_below_500", row["step"])
        if not after["live_animals"]:
            first.setdefault("all_live_animals_zero", row["step"])
        if any(
            entry["target_key"].startswith("LAND_EXPANSION") and entry["status"] != "PLANNED"
            for entry in row["report_entries"]
        ):
            first.setdefault("land_expansion_blocked", row["step"])
        farmer_inventory = after["unit_inventories"][0] if after["unit_inventories"] else {}
        if farmer_inventory.get("WHEAT", 0):
            first.setdefault("farmer_holds_wheat", row["step"])
        for animal in ("COW", "SHEEP"):
            if before["live_animals"].get(animal, 0) == 0 < after["live_animals"].get(animal, 0):
                first.setdefault(f"first_live:{animal}", row["step"])
        if before["land"] < after["land"]:
            first.setdefault("first_land_expansion", row["step"])

    if current_run:
        target_runs.append((current_target, current_run))

    quota_deficit_without_selection: dict[str, dict[str, int]] = {}
    for key in PRODUCTION_KEYS:
        kind, item = key.split(":", 1)
        flags: list[bool] = []
        deficit_steps = 0
        selected_while_deficit = 0
        for row in records:
            quota = row["quota"]
            before = row["before"]
            desired = {
                "COW": quota.get("cows", 0),
                "SHEEP": quota.get("sheep", 0),
                "WHEAT": quota.get("wheat", 0),
                "STRAWBERRY": quota.get("strawberry", 0),
                "MELON": quota.get("melon", 0),
            }[item]
            current = (
                before["live_animals"].get(item, 0)
                if kind == "ANIMAL_PRODUCTION"
                else before["live_crops"].get(item, 0)
            )
            deficit = current < desired
            if deficit:
                deficit_steps += 1
                if row["production_target"] == key:
                    selected_while_deficit += 1
            flags.append(deficit and row["production_target"] != key)
        quota_deficit_without_selection[key] = {
            "physical_deficit_steps": deficit_steps,
            "selected_while_physically_deficit": selected_while_deficit,
            "longest_unselected_physical_deficit_run": _longest_true_run(flags),
        }

    def maxima(path: str, keys: tuple[str, ...]) -> dict[str, int]:
        return {key: max(int(row["after"][path].get(key, 0)) for row in records) for key in keys}

    notable: list[dict[str, Any]] = []
    previous_target: str | None = None
    for row in records:
        target_changed = row["production_target"] != previous_target
        blocked = any(entry["status"] != "PLANNED" for entry in row["report_entries"])
        has_market = bool(row["action"].get("market"))
        composition_changed = row["composition_before"] != row["composition_after"]
        if (
            row["step"] <= 132
            or row["step"] % 24 == 0
            or blocked
            or has_market
            or composition_changed
            or target_changed
        ):
            notable.append(
                {
                    key: row[key]
                    for key in (
                        "step",
                        "money_before",
                        "money_after",
                        "money_delta",
                        "production_target",
                        "targets",
                        "report_entries",
                        "action",
                        "before",
                        "after",
                        "backoff",
                    )
                }
            )
        previous_target = row["production_target"]

    final = records[-1]["after"]
    farmer_wheat_flags = []
    for row in records:
        inventories = row["after"]["unit_inventories"]
        farmer_wheat_flags.append(bool(inventories and inventories[0].get("WHEAT", 0)))
    longest_non_null_target_run = max(
        (length for key, length in target_runs if key is not None), default=0
    )
    one_step_target_runs = sum(1 for key, length in target_runs if key and length == 1)
    return {
        "label": case.label,
        "team": case.team,
        "episode_id": case.episode_id,
        "seed": records[0]["seed"],
        "top1_seat": case.top1_seat,
        "candidate_seat": 1 - case.top1_seat,
        "steps_traced": len(records),
        "final": final,
        "target_selection_counts": dict(sorted(target_counts.items())),
        "target_run_count": len(target_runs),
        "one_step_target_runs": one_step_target_runs,
        "longest_non_null_target_run": longest_non_null_target_run,
        "report_counts": dict(sorted(report_counts.items())),
        "market_quantities": dict(sorted(market_quantities.items())),
        "cash_flow": {
            "sum_negative_step_deltas": cash_delta_negative,
            "sum_positive_step_deltas": cash_delta_positive,
            "positive_without_manual_market": no_market_positive,
            "net_by_market_signature": dict(sorted(cash_by_market_signature.items())),
            "listed_fixed_purchase_cost_upper_bound": listed_fixed_purchase_cost,
        },
        "production_planned_total": production_planned_total,
        "production_planned_without_same_step_composition_progress": (
            production_planned_without_same_step_progress
        ),
        "physical_quota_starvation": quota_deficit_without_selection,
        "first_events": first,
        "max_live_animals": maxima("live_animals", ("COW", "SHEEP")),
        "max_live_crops": maxima("live_crops", ("WHEAT", "STRAWBERRY", "MELON")),
        "max_shed_products": maxima("shed", PRODUCTS),
        "max_shed_animals": maxima("shed", ("COW", "SHEEP")),
        "longest_farmer_holds_wheat_run": _longest_true_run(farmer_wheat_flags),
        "notable_events": notable,
    }


def trace(case: Case) -> dict[str, Any]:
    replay_path = ROOT / case.replay
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    configuration = engine_config(replay)
    engine = Engine(configuration)
    module, agent, accepts_config = load_agent(
        ROOT / "submissions/meta_v72_closed_loop_town_quota_local/main.py"
    )
    policy = module._POLICY
    if not isinstance(policy, ClosedLoopTownQuotaPolicy):
        raise TypeError("V72 wrapper did not expose ClosedLoopTownQuotaPolicy as _POLICY")
    candidate_seat = 1 - case.top1_seat
    records: list[dict[str, Any]] = []

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        proposed = run_agent(agent, accepts_config, observation, configuration)
        proposed = append_terminal_liquidation(observation, proposed)
        proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        action = engine.sanitize_action(candidate_seat, normalized)
        step = int(observation.get("step", 0))

        targets = [dict(target) for target in policy.targets_for(candidate_seat)]
        quota_raw = policy.quota_for(candidate_seat)
        quota = asdict(quota_raw) if quota_raw is not None else {}
        report = policy.report_for(candidate_seat)
        before = _farm_state(observation, candidate_seat)
        backoff = {
            f"{kind}:{item}": until
            for (kind, item), until in sorted(policy.backoff_for(candidate_seat).items())
        }

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = action
        engine.step(actions)

        if step >= 120:
            after_observation = engine.observation(candidate_seat)
            after = _farm_state(after_observation, candidate_seat)
            records.append(
                {
                    "seed": int(replay["info"]["seed"]),
                    "step": step,
                    "money_before": before["money"],
                    "money_after": after["money"],
                    "money_delta": after["money"] - before["money"],
                    "quota": quota,
                    "targets": targets,
                    "production_target": _production_key(targets),
                    "report_entries": _report_rows(targets, report),
                    "backoff": backoff,
                    "action": action,
                    "before": before,
                    "after": after,
                    "composition_before": _composition_signature(before),
                    "composition_after": _composition_signature(after),
                }
            )

    final_money = float(engine.state["farms"][candidate_seat]["money"])
    tape_money = float(engine.state["farms"][case.top1_seat]["money"])
    margin = final_money - tape_money
    if (final_money, tape_money, margin) != (
        case.expected_candidate_money,
        case.expected_tape_money,
        case.expected_margin,
    ):
        raise AssertionError(
            "trace did not reproduce frozen old80 result: "
            f"got {(final_money, tape_money, margin)}, expected "
            f"{(case.expected_candidate_money, case.expected_tape_money, case.expected_margin)}"
        )
    summary = _summarize(case, records)
    summary["reproduction"] = {
        "candidate_money": final_money,
        "tape_top1_money": tape_money,
        "candidate_margin": margin,
        "final_state_step": int(engine.state["step"]),
        "matches_frozen_old80": True,
    }
    return summary


def main() -> None:
    payload = {
        "schema_version": 1,
        "scope": "old80_only_two_representative_cases",
        "candidate": "submissions/meta_v72_closed_loop_town_quota_local/main.py",
        "action_steps": "120..718 plus state at 719",
        "cases": [trace(case) for case in CASES],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
