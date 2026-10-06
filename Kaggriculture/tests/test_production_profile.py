from __future__ import annotations

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.crop_heuristic import CropHeuristicPolicy
from kaggriculture_rl.policy.production_profile import (
    INCUMBENT_PROFILE,
    ProductionProfile,
    StickyProfileController,
    profile_targets,
)


def test_incumbent_profile_is_exact_melon6_macro_policy() -> None:
    observation = Engine().observation(0)
    observation["private"]["shed"]["MELON"] = 3

    assert profile_targets(INCUMBENT_PROFILE, observation) == CropHeuristicPolicy(
        "MELON", 6, 0
    )({"shed": observation["private"]["shed"]})


def test_mixed_profile_emits_funding_and_survival_targets_before_crop() -> None:
    observation = Engine().observation(0)
    observation["private"]["shed"]["MILK"] = 2
    profile = ProductionProfile(
        "mixed",
        cows=2,
        workers=3,
        wheat_buffer=6,
        land_quadrants=1,
    )

    targets = profile_targets(profile, observation)
    kinds = [target["kind"] for target in targets]

    assert kinds == [
        "SELLING",
        "WORKFORCE",
        "INVENTORY_BUFFER",
        "LAND_EXPANSION",
        "ANIMAL_PRODUCTION",
        "CROP_PRODUCTION",
    ]
    assert targets[0]["item"] == "MILK"
    assert targets[3]["quadrant"] == "NE"


def test_profile_plan_survives_engine_sanitizer() -> None:
    engine = Engine({"seed": 23, "weedSpawnChance": 0.0})
    profile = ProductionProfile(
        "cow2",
        cows=2,
        workers=2,
        wheat_buffer=4,
    )

    for _ in range(120):
        planned = engine.plan(0, profile_targets(profile, engine.observation(0)))
        action = planned["joint_action"]["actions"][0]
        assert engine.sanitize_action(0, action) == action
        actions = planned["joint_action"]["actions"]
        engine.step(actions)


def test_sticky_controller_never_abandons_irreversible_investments() -> None:
    controller = StickyProfileController()
    invested = controller.select(
        ProductionProfile(
            "cow4",
            cows=4,
            workers=4,
            wheat_buffer=8,
            land_quadrants=1,
        )
    )
    switched = controller.select(INCUMBENT_PROFILE)

    assert invested.cows == switched.cows == 4
    assert switched.workers == 4
    assert switched.wheat_buffer == 8
    assert switched.land_quadrants == 1
    assert switched.crop.value == "MELON"
