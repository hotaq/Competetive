"""State-aware cow-purchase cap layered over the causal V59 controller."""

from __future__ import annotations

import copy
from typing import Any

from submissions.meta_v59_causal_route_selector_b120_local.main import agent as base_agent


def _owned_cows(observation: dict[str, Any]) -> int:
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    cows = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") == "COW":
                cows += 1
    private = observation.get("private", {})
    cows += int(private.get("shed", {}).get("COW", 0) or 0)
    return cows


def make_agent(cow_cap: int):
    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        action = copy.deepcopy(base_agent(observation, configuration))
        cows = _owned_cows(observation)
        market = []
        for order in action.get("market", []):
            if len(order) >= 3 and order[0] == "BUY_ANIMAL" and order[1] == "COW":
                allowed = max(0, cow_cap - cows)
                quantity = min(allowed, max(0, int(order[2])))
                if quantity:
                    market.append(["BUY_ANIMAL", "COW", quantity])
                    cows += quantity
            else:
                market.append(order)
        action["market"] = market
        return action

    return agent
