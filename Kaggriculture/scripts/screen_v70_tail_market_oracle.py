"""Screen the causal ceiling of quantity-conserving tail SELL scheduling.

This is a development-only oracle: it chooses the best intervention from a
fixed causal menu separately for each old/fresh replay.  The menu never reads
the opponent's current/future action and never borrows inventory across steps.
It is useful only as a ceiling check before implementing a deployable policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, play_against_tape
from search_single_market_mutations import collect_actions

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
PREMIUM = ("MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "EGG")
PARAMS: dict[str, tuple[int, int, str, float, str, float]] = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
EQUILIBRIUM_INVENTORY = 10_000
PRICE_FLOOR = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--old-league", type=Path, required=True)
    parser.add_argument("--fresh-league", type=Path, required=True)
    parser.add_argument("--shallow-limit", type=float, default=5_000.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _shape(kind: str, value: float, throughput: float) -> float:
    if kind == "linear":
        return value
    if kind == "sq":
        return value * value
    if kind == "sqrt":
        return math.sqrt(value)
    if kind == "log":
        return math.log1p(value)
    if kind == "log10":
        return math.log10(1.0 + value)
    if kind == "hinge" and throughput > 0:
        normalized = value / throughput
        return normalized + 4.0 * max(0.0, normalized - 1.0) ** 2
    return value


def market_price(item: str, inventory: int) -> int:
    base, throughput_i, below_func, below_target, above_func, above_target = PARAMS[item]
    throughput = float(throughput_i)
    if inventory < EQUILIBRIUM_INVENTORY:
        amplitude = below_target * base / _shape(below_func, throughput, throughput)
        price = base + amplitude * _shape(
            below_func, float(EQUILIBRIUM_INVENTORY - inventory), throughput
        )
    else:
        amplitude = above_target * base / _shape(above_func, throughput, throughput)
        price = base - amplitude * _shape(
            above_func, float(inventory - EQUILIBRIUM_INVENTORY), throughput
        )
    return max(PRICE_FLOOR, round(price))


def _pickup_reserve(action: dict[str, Any], item: str) -> int:
    reserve = 0
    for command in [action.get("farmer", ["PASS"]), *action.get("hands", [])]:
        if isinstance(command, list) and len(command) >= 2 and command[:2] == ["PICKUP", item]:
            reserve += max(0, int(command[2])) if len(command) >= 3 else 1
    return reserve


def _sell_score(observation: dict[str, Any], item: str, quantity: int, rank: str) -> float:
    market = observation.get("market", {})
    inventory = int(market.get("inventory", {}).get(item, EQUILIBRIUM_INVENTORY) or 0)
    prices = market.get("prices", {})
    current = float(prices.get(item, market_price(item, inventory)) or 0)
    if rank == "price":
        return current
    if rank == "average_revenue":
        simulated = sum(market_price(item, inventory + offset) for offset in range(quantity))
        return simulated / max(1, quantity)
    # Prefer scarce, high-value products whose next units lose price fastest.
    next_price = market_price(item, inventory + max(1, quantity))
    elasticity = max(0.0, current - float(next_price))
    return current + 4.0 * elasticity


def mutate_action(
    observation: dict[str, Any],
    action: dict[str, Any],
    *,
    items: tuple[str, ...],
    quantity_cap: int | None,
    placement: str,
    rank: str,
) -> dict[str, Any]:
    result = copy.deepcopy(action)
    original_market = [list(order) for order in result.get("market", [])]
    shed = observation.get("private", {}).get("shed", {})
    scheduled: dict[str, int] = {}
    for order in original_market:
        if len(order) >= 3 and order[:1] == ["SELL"]:
            item = str(order[1])
            scheduled[item] = scheduled.get(item, 0) + max(0, int(order[2]))

    free_slots = max(0, 10 - len(original_market))
    proposals: list[list[Any]] = []
    for item in items:
        available = max(
            0,
            int(shed.get(item, 0) or 0) - scheduled.get(item, 0) - _pickup_reserve(result, item),
        )
        if quantity_cap is not None:
            available = min(available, quantity_cap)
        if available > 0:
            proposals.append(["SELL", item, available])
    proposals.sort(
        key=lambda order: _sell_score(observation, str(order[1]), int(order[2]), rank),
        reverse=True,
    )
    proposals = proposals[:free_slots]
    if placement == "prepend":
        result["market"] = [*proposals, *original_market]
    else:
        result["market"] = [*original_market, *proposals]
    return result


def reorder_existing_sells(
    observation: dict[str, Any], action: dict[str, Any], *, rank: str, sell_first: bool
) -> dict[str, Any]:
    result = copy.deepcopy(action)
    market = [list(order) for order in result.get("market", [])]
    sells = [order for order in market if len(order) >= 3 and order[:1] == ["SELL"]]
    sells.sort(
        key=lambda order: _sell_score(observation, str(order[1]), int(order[2]), rank),
        reverse=True,
    )
    if sell_first:
        nonsells = [order for order in market if not (len(order) >= 3 and order[:1] == ["SELL"])]
        result["market"] = [*sells, *nonsells]
        return result
    iterator = iter(sells)
    result["market"] = [
        next(iterator) if len(order) >= 3 and order[:1] == ["SELL"] else order for order in market
    ]
    return result


def make_wrapper(
    base_actions: dict[int, dict[str, Any]],
    spec: dict[str, Any],
) -> Callable[[dict[str, Any], Any], dict[str, Any]]:
    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        action = copy.deepcopy(base_actions.get(step, {}))
        if spec["kind"] == "one_shot" and step == spec["step"]:
            return mutate_action(
                observation,
                action,
                items=PREMIUM if spec["items"] == "premium" else PRODUCTS,
                quantity_cap=spec["cap"],
                placement=spec["placement"],
                rank=spec["rank"],
            )
        if spec["kind"] == "reorder" and step >= 672:
            return reorder_existing_sells(
                observation,
                action,
                rank=spec["rank"],
                sell_first=spec["sell_first"],
            )
        return action

    return agent


def causal_menu() -> list[dict[str, Any]]:
    menu: list[dict[str, Any]] = []
    # Bounded causal menu: final post-demand timings, broad/capped batches,
    # append/prepend placement, and three observable-only rank functions.
    for step in (717, 713, 709, 695):
        for items in ("premium", "all"):
            for cap in (None, 4):
                menu.append(
                    {
                        "kind": "one_shot",
                        "step": step,
                        "items": items,
                        "cap": cap,
                        "placement": "prepend",
                        "rank": "elasticity",
                    }
                )
    for items in ("premium", "all"):
        for cap in (1, 2, 8):
            menu.append(
                {
                    "kind": "one_shot",
                    "step": 717,
                    "items": items,
                    "cap": cap,
                    "placement": "prepend",
                    "rank": "elasticity",
                }
            )
        for placement in ("prepend", "append"):
            for rank in ("price", "average_revenue"):
                menu.append(
                    {
                        "kind": "one_shot",
                        "step": 717,
                        "items": items,
                        "cap": None,
                        "placement": placement,
                        "rank": rank,
                    }
                )
    for rank in ("elasticity", "price", "average_revenue"):
        for sell_first in (False, True):
            menu.append(
                {
                    "kind": "reorder",
                    "rank": rank,
                    "sell_first": sell_first,
                }
            )
    return menu


def _team_directories(league_path: Path) -> dict[int, Path]:
    league = json.loads(league_path.read_text(encoding="utf-8"))
    return {
        int(team["team_id"]): league_path.parent / str(team["directory"])
        for team in league["teams"]
    }


def _shallow_rows(
    report_path: Path, league_path: Path, split: str, shallow_limit: float
) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    directories = _team_directories(league_path)
    rows: list[dict[str, Any]] = []
    for team in report["teams"]:
        for episode in team["episodes"]:
            margin = float(episode["candidate_margin"])
            if not (-shallow_limit <= margin < 0):
                continue
            rows.append(
                {
                    "split": split,
                    "team_id": int(team["team_id"]),
                    "team_name": str(team["team_name"]),
                    "episode_id": int(episode["episode_id"]),
                    "top1_seat": int(episode["top1_seat"]),
                    "baseline_margin": margin,
                    "replay": directories[int(team["team_id"])]
                    / f"{int(episode['episode_id'])}.json",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    _, base_agent, base_accepts_config = load_agent(args.agent)
    rows = [
        *_shallow_rows(args.old_report, args.old_league, "old80", args.shallow_limit),
        *_shallow_rows(args.fresh_report, args.fresh_league, "fresh95", args.shallow_limit),
    ]
    rows.sort(key=lambda row: (abs(float(row["baseline_margin"])), row["episode_id"]))
    menu = causal_menu()
    screened: list[dict[str, Any]] = []
    for row in rows:
        replay = json.loads(Path(row["replay"]).read_text(encoding="utf-8"))
        base_actions = collect_actions(
            replay,
            top1_seat=int(row["top1_seat"]),
            agent=base_agent,
            accepts_config=base_accepts_config,
        )
        fixed_baseline = play_against_tape(
            replay,
            episode_id=int(row["episode_id"]),
            top1_seat=int(row["top1_seat"]),
            agent=make_wrapper(base_actions, {"kind": "baseline"}),
            accepts_config=True,
        )
        if float(fixed_baseline.candidate_margin) != float(row["baseline_margin"]):
            raise RuntimeError(
                "fixed-action baseline mismatch for "
                f"{row['episode_id']}: report={row['baseline_margin']} "
                f"replayed={fixed_baseline.candidate_margin}"
            )
        best_margin = float(row["baseline_margin"])
        best_spec: dict[str, Any] | None = None
        evaluated = 0
        for spec in menu:
            wrapper = make_wrapper(base_actions, spec)
            result = play_against_tape(
                replay,
                episode_id=int(row["episode_id"]),
                top1_seat=int(row["top1_seat"]),
                agent=wrapper,
                accepts_config=True,
            )
            evaluated += 1
            if result.candidate_margin > best_margin:
                best_margin = float(result.candidate_margin)
                best_spec = spec
            if best_margin > 0:
                break
        screened.append(
            {
                **{key: value for key, value in row.items() if key != "replay"},
                "best_margin": best_margin,
                "improvement": best_margin - float(row["baseline_margin"]),
                "fixed_action_baseline_margin": float(fixed_baseline.candidate_margin),
                "fixed_action_baseline_exact": True,
                "rescued": best_margin > 0,
                "best_spec": best_spec,
                "menu_evaluations": evaluated,
            }
        )
        print(
            f"{row['split']} {row['team_name']} {row['episode_id']}: "
            f"{row['baseline_margin']:+.0f} -> {best_margin:+.0f}"
        )

    rescued = [row for row in screened if row["rescued"]]
    unique_rescued_episodes = len({row["episode_id"] for row in rescued})
    unique_rescued_teams = len({row["team_id"] for row in rescued})
    ceiling_pass = unique_rescued_episodes >= 3 and unique_rescued_teams >= 3
    payload = {
        "schema_version": 1,
        "kind": "development_only_per_replay_causal_oracle",
        "agent": str(args.agent),
        "datasets": {
            "old_report": str(args.old_report),
            "fresh_report": str(args.fresh_report),
            "old_league": str(args.old_league),
            "fresh_league": str(args.fresh_league),
        },
        "constraints": {
            "step_from": 672,
            "opponent_action_features": False,
            "future_action_features": False,
            "team_episode_seed_features": False,
            "cross_step_inventory_borrowing": False,
            "fixed_baseline_proposals_replayed": True,
            "oracle_selects_best_fixed_causal_intervention_per_replay": True,
        },
        "gate": {
            "required_unique_rescued_episodes": 3,
            "required_unique_rescued_teams": 3,
            "allowed_regressions": 1,
            "oracle_regressions": 0,
            "unique_rescued_episodes": unique_rescued_episodes,
            "unique_rescued_teams": unique_rescued_teams,
            "pass": ceiling_pass,
        },
        "menu_size": len(menu),
        "shallow_limit": args.shallow_limit,
        "episodes": screened,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
