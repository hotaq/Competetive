"""Run the observable marginal-value model on the pre-consumed gate8 only."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from marginal_value import ANIMALS, compare_cow_sheep, lockstep_sell

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
GATE_PATH = ROOT / "analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json"
EXPECTED_GATE_SHA256 = "4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2"
EXPECTED_BASELINE_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"
PLACEMENT_STEP = 248


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_at(replay: dict[str, Any], step: int, seat: int) -> dict[str, Any]:
    own = dict(replay["steps"][step][seat]["observation"])
    shared = replay["steps"][step][0]["observation"]
    own.setdefault("step", shared.get("step", step))
    own.setdefault("day", shared["day"])
    own.setdefault("hour", shared["hour"])
    own["player"] = seat
    return own


def _market_at(replay: dict[str, Any], step: int) -> dict[str, Any]:
    return replay["steps"][step][0]["observation"]["market"]


def _realized_path_value(
    replay: dict[str, Any],
    configuration: dict[str, Any],
    seat: int,
    value: dict[str, Any],
) -> dict[str, float]:
    """Ex-post price-path sanity check; never an input to the decision model."""

    product = value["product"]
    revenue = 0.0
    for event in value["product_events"]:
        step = min(int(event["sale_step"]), len(replay["steps"]) - 1)
        inventory = float(_market_at(replay, step)["inventory"][product])
        quantities = (int(event["units"]), 0) if seat == 0 else (0, int(event["units"]))
        _, proceeds = lockstep_sell(inventory, quantities, product)
        revenue += proceeds[seat]

    turns = max(1, int(configuration.get("turnsPerDay", 24)))
    placement_day = int(value["placement_day"])
    feed_actions = int(value["feed_actions"])
    feed_cost = 0.0
    for day in range(placement_day, placement_day + feed_actions):
        step = min(
            max(PLACEMENT_STEP, day * turns + turns // 2),
            len(replay["steps"]) - 1,
        )
        feed_cost += float(_market_at(replay, step)["prices"]["WHEAT"])
    net = revenue - feed_cost - float(value["purchase_cost"])
    return {
        "marginal_product_revenue": round(revenue, 6),
        "feed_opportunity_cost": round(feed_cost, 6),
        "net_before_labor_and_fertilizer": round(net, 6),
    }


def _step248_animal_orders(action: dict[str, Any]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for order in action.get("market", ()):
        if (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "BUY_ANIMAL"
            and order[1] in ANIMALS
        ):
            result[str(order[1])] += int(order[2])
    return dict(sorted(result.items()))


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def main() -> None:
    assert _sha256(GATE_PATH) == EXPECTED_GATE_SHA256
    gate = json.loads(GATE_PATH.read_text())
    baseline = ROOT / gate["baseline_agent"]["path"]
    assert gate["baseline_agent"]["sha256"] == EXPECTED_BASELINE_SHA256
    assert _sha256(baseline) == EXPECTED_BASELINE_SHA256

    source_dir = (ROOT / gate["source_manifest"]).parent
    entries = [("target_loss", item) for item in gate["target_losses"]] + [
        ("regression_control", item) for item in gate["regression_controls"]
    ]
    assert len(entries) == 8

    cases: list[dict[str, Any]] = []
    for category, entry in entries:
        episode_id = int(entry["episode_id"])
        replay_path = source_dir / f"{episode_id}.json"
        assert _sha256(replay_path) == entry["replay_sha256"]
        replay = json.loads(replay_path.read_text())
        configuration = dict(replay["configuration"])
        seat = int(entry["agent_seat"])
        observation = _observation_at(replay, PLACEMENT_STEP, seat)
        assert int(observation["day"]) == PLACEMENT_STEP // int(configuration["turnsPerDay"])

        comparison = compare_cow_sheep(
            observation,
            configuration,
            placement_step=PLACEMENT_STEP,
            seat=seat,
        )
        realized = {
            animal.lower(): _realized_path_value(
                replay,
                configuration,
                seat,
                comparison[animal.lower()],
            )
            for animal in ("COW", "SHEEP")
        }
        realized_delta = (
            realized["sheep"]["net_before_labor_and_fertilizer"]
            - realized["cow"]["net_before_labor_and_fertilizer"]
        )
        # Kaggle replay record i+1 stores the action selected from observation i.
        actual_action = replay["steps"][PLACEMENT_STEP + 1][seat].get("action") or {}
        cases.append(
            {
                "episode_id": episode_id,
                "category": category,
                "agent_seat": seat,
                "baseline_margin": float(entry["baseline_margin"]),
                "replay_sha256": entry["replay_sha256"],
                "step248": {
                    "day": int(observation["day"]),
                    "hour": int(observation["hour"]),
                    "shops": list(observation["town"]["unlocked_shops"]),
                    "prices": {
                        item: int(observation["market"]["prices"][item])
                        for item in ("WHEAT", "MILK", "WOOL", "FERTILIZER")
                    },
                    "inventory": {
                        item: int(observation["market"]["inventory"][item])
                        for item in ("WHEAT", "MILK", "WOOL", "FERTILIZER")
                    },
                    "farm_money": [float(farm["money"]) for farm in observation["farms"]],
                    "visible_live_animals": comparison["cow"]["visible_live_animals"],
                    "actual_candidate_buy_animal_orders": _step248_animal_orders(actual_action),
                    "replay_action_record_index": PLACEMENT_STEP + 1,
                    "empty_pasture": comparison["cow"]["empty_matching_structure_at_observation"],
                },
                "observable_projection": comparison,
                "realized_price_path_sanity_only": {
                    "cow": realized["cow"],
                    "sheep": realized["sheep"],
                    "sheep_minus_cow_net_before_labor_and_fertilizer": round(realized_delta, 6),
                    "preferred": (
                        "SHEEP" if realized_delta > 0 else "COW" if realized_delta < 0 else "TIE"
                    ),
                    "warning": "ex-post replay prices; audit sanity only, never a policy input",
                },
            }
        )

    categories = ("target_loss", "regression_control")
    summary: dict[str, Any] = {}
    for category in categories:
        selected = [case for case in cases if case["category"] == category]
        projected_choices = Counter(
            case["observable_projection"]["preferred_by_mechanical_cash_value"] for case in selected
        )
        realized_choices = Counter(
            case["realized_price_path_sanity_only"]["preferred"] for case in selected
        )
        summary[category] = {
            "cases": len(selected),
            "projected_preference_counts": dict(sorted(projected_choices.items())),
            "realized_path_preference_counts": dict(sorted(realized_choices.items())),
            "mean_projected_sheep_minus_cow": _mean(
                [
                    float(
                        case["observable_projection"][
                            "sheep_minus_cow_net_before_labor_and_fertilizer"
                        ]
                    )
                    for case in selected
                ]
            ),
            "mean_realized_sheep_minus_cow": _mean(
                [
                    float(
                        case["realized_price_path_sanity_only"][
                            "sheep_minus_cow_net_before_labor_and_fertilizer"
                        ]
                    )
                    for case in selected
                ]
            ),
            "current_cow_order_cases": sum(
                int(case["step248"]["actual_candidate_buy_animal_orders"].get("COW", 0) > 0)
                for case in selected
            ),
            "current_cow_order_cases_where_projection_prefers_sheep": sum(
                int(
                    case["step248"]["actual_candidate_buy_animal_orders"].get("COW", 0) > 0
                    and case["observable_projection"]["preferred_by_mechanical_cash_value"]
                    == "SHEEP"
                )
                for case in selected
            ),
        }

    absolute_errors: dict[str, list[float]] = {"cow": [], "sheep": []}
    for case in cases:
        for animal in ("cow", "sheep"):
            projected = float(
                case["observable_projection"][animal]["net_before_labor_and_fertilizer"]
            )
            realized_net = float(
                case["realized_price_path_sanity_only"][animal]["net_before_labor_and_fertilizer"]
            )
            absolute_errors[animal].append(abs(projected - realized_net))

    output = {
        "analysis": "observable_market_livestock_marginal_value",
        "status": "read_only_gate8_sanity_complete",
        "decision_scope": "economic audit only; no candidate, promotion, or upload",
        "data_boundary": {
            "gate_path": str(GATE_PATH.relative_to(ROOT)),
            "gate_sha256": EXPECTED_GATE_SHA256,
            "baseline_path": str(baseline.relative_to(ROOT)),
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "episodes": [int(entry["episode_id"]) for _, entry in entries],
            "forbidden_sets_read": [],
        },
        "mechanical_schedule_step248": {
            "cow_product_days": [
                event["refresh_day"]
                for event in cases[0]["observable_projection"]["cow"]["product_events"]
            ],
            "cow_units": cases[0]["observable_projection"]["cow"]["product_units"],
            "sheep_product_days": [
                event["refresh_day"]
                for event in cases[0]["observable_projection"]["sheep"]["product_events"]
            ],
            "sheep_units": cases[0]["observable_projection"]["sheep"]["product_units"],
            "feed_actions_each": cases[0]["observable_projection"]["cow"]["feed_actions"],
            "max_fertilizer_units_each": cases[0]["observable_projection"]["cow"][
                "max_fertilizer_units"
            ],
        },
        "summary": summary,
        "projection_vs_realized_net_mae": {
            animal: _mean(errors) for animal, errors in absolute_errors.items()
        },
        "cases": cases,
    }
    output_path = OUT_DIR / "gate8_results.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "status",
                    "mechanical_schedule_step248",
                    "summary",
                    "projection_vs_realized_net_mae",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
