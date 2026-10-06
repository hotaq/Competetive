from __future__ import annotations

from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


def test_crop_heuristic_maintains_target_and_sells_excess() -> None:
    policy = CropHeuristicPolicy(Crop.WHEAT, target_quantity=2, sell_reserve=1)

    targets = policy({"shed": {"WHEAT": 5}})

    assert targets == [
        {
            "schema_version": 1,
            "kind": "CROP_PRODUCTION",
            "crop": "WHEAT",
            "target_quantity": 2,
        },
        {
            "schema_version": 1,
            "kind": "SELLING",
            "item": "WHEAT",
            "quantity": 4,
        },
    ]


def test_typed_crop_loop_harvests_deposits_and_sells() -> None:
    engine = Engine(
        {
            "episodeSteps": 120,
            "seed": 19,
            "weedSpawnChance": 0.0,
        }
    )
    policy = CropHeuristicPolicy(Crop.WHEAT, target_quantity=1)
    observation: dict[str, Any] = {"shed": {crop.value: 0 for crop in Crop}}
    seen: set[str] = set()

    for _ in range(80):
        planned = engine.plan(0, policy(observation))
        player = planned["joint_action"]["actions"][0]
        seen.add(player["farmer"][0])
        seen.update(order[0] for order in player["market"])
        result = engine.step_typed(planned["joint_action"])
        observation = result["observations"][0]

    assert {"BUY_SEED", "PLANT", "WATER", "HARVEST", "DROP", "SELL"} <= seen
    assert engine.state["farms"][0]["money"] > 3_000
