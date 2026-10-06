"""Two-step V59-compatible opening variants for causal splice screening."""

from __future__ import annotations

from typing import Any


def make_agent(buy_wheat: int, sell_wheat: int):
    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        del configuration
        step = int(observation.get("step", 0))
        if step == 0:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [
                    ["BUY_PRODUCT", "WHEAT", buy_wheat],
                    ["HIRE"],
                    ["HIRE"],
                    ["BUY_SEED", "MELON", 7],
                    ["BUY_SEED", "WHEAT", 5],
                    ["BUY_ANIMAL", "SHEEP", 4],
                ],
            }
        if step == 1:
            return {
                "farmer": ["PICKUP", "SHEEP", 4],
                "hands": [["WEST"], ["NORTH"]],
                "market": [["SELL", "WHEAT", sell_wheat]],
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def make_appended_agent(extra_wheat: int):
    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        del configuration
        step = int(observation.get("step", 0))
        if step == 0:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [
                    ["BUY_PRODUCT", "WHEAT", 4],
                    ["HIRE"],
                    ["HIRE"],
                    ["BUY_SEED", "MELON", 7],
                    ["BUY_SEED", "WHEAT", 5],
                    ["BUY_ANIMAL", "SHEEP", 4],
                    ["BUY_PRODUCT", "WHEAT", extra_wheat],
                ],
            }
        if step == 1:
            private = observation.get("private", {})
            wheat = int(private.get("shed", {}).get("WHEAT", 0))
            market = [["SELL", "WHEAT", wheat - 4]] if wheat > 4 else []
            return {
                "farmer": ["PICKUP", "SHEEP", 4],
                "hands": [["WEST"], ["NORTH"]],
                "market": market,
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent
