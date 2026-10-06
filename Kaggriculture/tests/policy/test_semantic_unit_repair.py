from kaggriculture_rs import Engine

from kaggriculture_rl.policy.primitive_legalizer import (
    legalize_action,
    recover_rejected_unit_intents,
)


def _recover(
    observation: dict,
    proposed: dict,
    *,
    allowed_operations: frozenset[str] | None = None,
) -> dict:
    legalized = legalize_action(observation, proposed)
    return recover_rejected_unit_intents(
        observation,
        proposed,
        legalized,
        allowed_operations=allowed_operations,
    )


def test_rejected_water_harvests_a_ready_crop_without_moving() -> None:
    observation = Engine({"seed": 7}).observation(0)
    x, y = observation["farms"][0]["farmer"]
    observation["day"] = 4
    observation["farms"][0]["tiles"][y][x] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 3,
        "watered_today": True,
        "fertilized_until_day": -1,
    }
    proposed = {"farmer": ["WATER"], "hands": [], "market": []}

    assert legalize_action(observation, proposed)["farmer"] == ["PASS"]
    assert _recover(observation, proposed)["farmer"] == ["HARVEST"]


def test_rejected_harvest_digs_a_weed() -> None:
    observation = Engine({"seed": 11}).observation(0)
    x, y = observation["farms"][0]["farmer"]
    observation["farms"][0]["tiles"][y][x] = {"kind": "WEED"}
    proposed = {"farmer": ["HARVEST"], "hands": [], "market": []}

    assert _recover(observation, proposed)["farmer"] == ["DIG"]


def test_intent_repair_never_replaces_an_explicit_pass() -> None:
    observation = Engine({"seed": 13}).observation(0)
    x, y = observation["farms"][0]["farmer"]
    observation["farms"][0]["tiles"][y][x] = {"kind": "WEED"}
    proposed = {"farmer": ["PASS"], "hands": [], "market": []}

    assert _recover(observation, proposed)["farmer"] == ["PASS"]


def test_intent_repair_never_turns_rejected_movement_into_tile_work() -> None:
    observation = Engine({"seed": 17}).observation(0)
    observation["farms"][0]["farmer"] = [0, 0]
    observation["farms"][0]["tiles"][0][0] = {"kind": "WEED"}
    proposed = {"farmer": ["NORTH"], "hands": [], "market": []}

    assert legalize_action(observation, proposed)["farmer"] == ["PASS"]
    assert _recover(observation, proposed)["farmer"] == ["PASS"]


def test_intent_repair_can_be_restricted_to_a_safe_operation_set() -> None:
    observation = Engine({"seed": 19}).observation(0)
    x, y = observation["farms"][0]["farmer"]
    observation["day"] = 4
    observation["farms"][0]["tiles"][y][x] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 3,
        "watered_today": True,
        "fertilized_until_day": -1,
    }
    proposed = {"farmer": ["WATER"], "hands": [], "market": []}

    repaired = _recover(
        observation,
        proposed,
        allowed_operations=frozenset({"WATER"}),
    )

    assert repaired["farmer"] == ["PASS"]
