"""Screen single-event bidirectional market timing interventions on frozen leagues.

This is deliberately a development-only oracle.  It discovers observable events on
the baseline trajectory, then replays one timing intervention at a time.  It never
uses opponent identity or future opponent actions.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, play_against_tape, run_agent

PRODUCTS = (
    "CARROT",
    "EGG",
    "MELON",
    "MILK",
    "STRAWBERRY",
    "TOMATO",
    "WHEAT",
    "WOOL",
)
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
SEED_COST = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--league-manifest", type=Path, action="append", required=True)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--cap", type=int, default=2)
    parser.add_argument("--step-from", type=int, default=120)
    parser.add_argument("--step-to", type=int, default=695)
    parser.add_argument("--max-events-per-kind", type=int, default=0)
    parser.add_argument("--baseline-margin-min", type=float, default=float("-inf"))
    parser.add_argument("--stop-episode-on-rescue", action="store_true")
    parser.add_argument(
        "--event-kinds", default="reserve,early_exit", help="Comma-separated kinds"
    )
    return parser.parse_args()


def config_value(configuration: dict[str, Any], name: str, default: int) -> int:
    return max(1, int(configuration.get(name, default) or default))


def town_demand(
    observation: dict[str, Any],
    configuration: dict[str, Any],
    item: str,
    step: int,
) -> int:
    demand = int(
        item != "FERTILIZER"
        and step % config_value(configuration, "townCenterSellInterval", 24) == 0
    )
    if step % config_value(configuration, "townShopSellInterval", 4) != 0:
        return demand
    for shop in observation.get("town", {}).get("unlocked_shops", []):
        products = SHOP_PRODUCTS.get(str(shop), ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def demand_horizon(
    observation: dict[str, Any],
    configuration: dict[str, Any],
    item: str,
    step: int,
    horizon: int = 24,
) -> int:
    return sum(
        town_demand(observation, configuration, item, future_step)
        for future_step in range(step, step + horizon + 1)
    )


def planned_buy_cost(
    action: dict[str, Any], observation: dict[str, Any]
) -> int:
    prices = observation.get("market", {}).get("prices", {})
    total = 0
    for order in action.get("market", []):
        if not isinstance(order, list) or len(order) < 3:
            continue
        operation, item, raw_quantity = str(order[0]), str(order[1]), order[2]
        quantity = max(0, int(raw_quantity))
        if operation == "BUY_PRODUCT":
            total += quantity * max(0, int(prices.get(item, 0) or 0))
        elif operation == "BUY_SEED":
            total += quantity * SEED_COST.get(item, 0)
        elif operation == "BUY_ANIMAL":
            total += quantity * ANIMAL_COST.get(item, 0)
    return total


def sell_quantity(action: dict[str, Any], item: str) -> int:
    return sum(
        max(0, int(order[2]))
        for order in action.get("market", [])
        if isinstance(order, list)
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] == item
    )


def change_sell(action: dict[str, Any], item: str, delta: int) -> dict[str, Any]:
    result = copy.deepcopy(action)
    market = [list(order) for order in result.get("market", [])]
    if delta < 0:
        remaining = -delta
        for order in market:
            if len(order) >= 3 and order[:2] == ["SELL", item]:
                reduction = min(remaining, max(0, int(order[2])))
                order[2] = max(0, int(order[2])) - reduction
                remaining -= reduction
                if remaining == 0:
                    break
        market = [
            order
            for order in market
            if not (len(order) >= 3 and order[:2] == ["SELL", item] and order[2] <= 0)
        ]
    elif delta > 0:
        existing = next(
            (
                order
                for order in market
                if len(order) >= 3 and order[:2] == ["SELL", item]
            ),
            None,
        )
        if existing is not None:
            existing[2] = max(0, int(existing[2])) + delta
        else:
            market.append(["SELL", item, delta])
    result["market"] = market
    return result


def report_losses(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    losses: dict[tuple[int, int], dict[str, Any]] = {}
    for team in report["teams"]:
        for episode in team["episodes"]:
            if float(episode["candidate_margin"]) < 0:
                key = (int(team["team_id"]), int(episode["episode_id"]))
                losses[key] = {
                    "baseline_margin": float(episode["candidate_margin"]),
                    "team_id": int(team["team_id"]),
                    "team_name": str(team["team_name"]),
                    "snapshot_rank": int(team["snapshot_rank"]),
                }
    return losses


def replay_rows(
    league_path: Path, losses: dict[tuple[int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    league = json.loads(league_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for team in league["teams"]:
        team_dir = league_path.parent / team["directory"]
        manifest = json.loads((team_dir / "manifest.json").read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            episode_id = int(row["episode_id"])
            key = (int(team["team_id"]), episode_id)
            if key not in losses:
                continue
            selected.append(
                {
                    **losses[key],
                    "episode_id": episode_id,
                    "top1_seat": int(row.get("agent_seat", row.get("top1_seat"))),
                    "replay_path": team_dir / f"{episode_id}.json",
                }
            )
    return selected


def collect_episode(
    replay: dict[str, Any],
    episode_id: int,
    top1_seat: int,
    base_agent: Any,
    accepts_config: bool,
    *,
    cap: int,
    step_from: int,
    step_to: int,
) -> tuple[Any, list[dict[str, Any]]]:
    observations: dict[int, dict[str, Any]] = {}
    actions: dict[int, dict[str, Any]] = {}
    configuration = replay["configuration"]

    def logging_agent(
        observation: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        action = copy.deepcopy(run_agent(base_agent, accepts_config, observation, config))
        step = int(observation.get("step", 0) or 0)
        observations[step] = copy.deepcopy(observation)
        actions[step] = copy.deepcopy(action)
        return action

    baseline = play_against_tape(
        replay,
        episode_id=episode_id,
        top1_seat=top1_seat,
        agent=logging_agent,
        accepts_config=True,
    )

    events: list[dict[str, Any]] = []
    for step in sorted(observations):
        if step < max(step_from, 24) or step > step_to or step + 1 not in actions:
            continue
        observation = observations[step]
        action = actions[step]
        prior_observation = observations.get(step - 24)
        if prior_observation is None:
            continue
        prices = observation.get("market", {}).get("prices", {})
        prior_prices = prior_observation.get("market", {}).get("prices", {})
        shed = observation.get("private", {}).get("shed", {})
        player = int(observation.get("player", 0) or 0)
        own_money = float(observation.get("farms", [])[player].get("money", 0) or 0)
        cash_gap = max(0.0, float(planned_buy_cost(action, observation)) - own_money)
        future_action = actions[step + 1]

        for item in PRODUCTS:
            price = float(prices.get(item, 0) or 0)
            slope24 = price - float(prior_prices.get(item, price) or price)
            current_demand = town_demand(observation, configuration, item, step)
            horizon_demand = demand_horizon(
                observation, configuration, item, step, horizon=24
            )
            available = max(0, int(shed.get(item, 0) or 0))
            coverage = available / max(1, horizon_demand)
            current_sell = sell_quantity(action, item)
            future_sell = sell_quantity(future_action, item)

            if (
                current_sell > 0
                and current_demand > 0
                and slope24 > 0
                and coverage >= 1.0
                and cash_gap == 0
            ):
                events.append(
                    {
                        "kind": "reserve",
                        "step": step,
                        "item": item,
                        "quantity": min(cap, current_sell),
                        "base_sell": current_sell,
                        "price": price,
                        "slope24": slope24,
                        "town_demand": current_demand,
                        "demand24": horizon_demand,
                        "shed": available,
                        "shed_coverage24": coverage,
                        "own_money": own_money,
                        "cash_gap": cash_gap,
                    }
                )

            if (
                future_sell > 0
                and current_demand == 0
                and slope24 < 0
                and available > 0
                and (coverage >= 1.0 or cash_gap > 0)
            ):
                events.append(
                    {
                        "kind": "early_exit",
                        "step": step,
                        "item": item,
                        "quantity": min(cap, future_sell, available),
                        "base_sell": future_sell,
                        "price": price,
                        "slope24": slope24,
                        "town_demand": current_demand,
                        "demand24": horizon_demand,
                        "shed": available,
                        "shed_coverage24": coverage,
                        "own_money": own_money,
                        "cash_gap": cash_gap,
                    }
                )
    return baseline, events


def evaluate_event(
    replay: dict[str, Any],
    episode_id: int,
    top1_seat: int,
    base_agent: Any,
    accepts_config: bool,
    event: dict[str, Any],
) -> Any:
    event_step = int(event["step"])
    item = str(event["item"])
    quantity = int(event["quantity"])
    direction = 1 if event["kind"] == "early_exit" else -1

    def event_agent(
        observation: dict[str, Any], configuration: dict[str, Any]
    ) -> dict[str, Any]:
        action = copy.deepcopy(
            run_agent(base_agent, accepts_config, observation, configuration)
        )
        step = int(observation.get("step", 0) or 0)
        if step == event_step:
            return change_sell(action, item, direction * quantity)
        if step == event_step + 1:
            return change_sell(action, item, -direction * quantity)
        return action

    return play_against_tape(
        replay,
        episode_id=episode_id,
        top1_seat=top1_seat,
        agent=event_agent,
        accepts_config=True,
    )


def main() -> None:
    args = parse_args()
    if len(args.league_manifest) != len(args.baseline_report):
        raise SystemExit("--league-manifest and --baseline-report counts must match")

    _, base_agent, accepts_config = load_agent(args.agent)
    selected_kinds = {kind for kind in args.event_kinds.split(",") if kind}
    unknown_kinds = selected_kinds - {"reserve", "early_exit"}
    if unknown_kinds:
        raise SystemExit(f"unknown --event-kinds: {sorted(unknown_kinds)}")
    episode_rows: list[dict[str, Any]] = []
    for league_path, report_path in zip(
        args.league_manifest, args.baseline_report, strict=True
    ):
        episode_rows.extend(replay_rows(league_path, report_losses(report_path)))

    episodes: list[dict[str, Any]] = []
    rescue_ids: set[int] = set()
    rescue_teams: set[int] = set()
    regressions = 0
    for row in episode_rows:
        if float(row["baseline_margin"]) < args.baseline_margin_min:
            continue
        replay = json.loads(row["replay_path"].read_text(encoding="utf-8"))
        baseline, events = collect_episode(
            replay,
            int(row["episode_id"]),
            int(row["top1_seat"]),
            base_agent,
            accepts_config,
            cap=args.cap,
            step_from=args.step_from,
            step_to=args.step_to,
        )
        events = [event for event in events if event["kind"] in selected_kinds]
        if abs(float(baseline.candidate_margin) - row["baseline_margin"]) > 1e-9:
            raise RuntimeError(
                f"baseline drift for {row['episode_id']}: "
                f"{baseline.candidate_margin} != {row['baseline_margin']}"
            )

        if args.max_events_per_kind > 0:
            limited: list[dict[str, Any]] = []
            for kind in ("reserve", "early_exit"):
                by_kind = [event for event in events if event["kind"] == kind]
                by_kind.sort(key=lambda event: int(event["step"]), reverse=True)
                limited.extend(by_kind[: args.max_events_per_kind])
            events = limited

        evaluated: list[dict[str, Any]] = []
        if not args.collect_only:
            for event in events:
                result = evaluate_event(
                    replay,
                    int(row["episode_id"]),
                    int(row["top1_seat"]),
                    base_agent,
                    accepts_config,
                    event,
                )
                candidate_margin = float(result.candidate_margin)
                evaluated.append(
                    {
                        **event,
                        "candidate_margin": candidate_margin,
                        "margin_delta": candidate_margin - float(baseline.candidate_margin),
                        "rescued": candidate_margin > 0,
                        "sanitized_steps": int(result.sanitized_steps),
                        "market_sanitized_steps": int(result.market_sanitized_steps),
                    }
                )
                if candidate_margin > 0:
                    rescue_ids.add(int(row["episode_id"]))
                    rescue_teams.add(int(row["team_id"]))
                    if args.stop_episode_on_rescue:
                        break
        episodes.append(
            {
                **{key: value for key, value in row.items() if key != "replay_path"},
                "baseline": asdict(baseline),
                "event_count": len(events),
                "events": evaluated if not args.collect_only else events,
            }
        )

    payload = {
        "schema_version": 1,
        "candidate": str(args.agent),
        "development_only": True,
        "league_manifests": [str(path) for path in args.league_manifest],
        "baseline_reports": [str(path) for path in args.baseline_report],
        "screen": {
            "loss_episodes": len(episodes),
            "events": sum(row["event_count"] for row in episodes),
            "rescued_unique_episodes": len(rescue_ids),
            "rescued_unique_teams": len(rescue_teams),
            "regressions": regressions,
            "pass": len(rescue_ids) >= 3 and len(rescue_teams) >= 3 and regressions <= 1,
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["screen"], indent=2))


if __name__ == "__main__":
    main()
