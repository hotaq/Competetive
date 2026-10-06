from __future__ import annotations

import json

from kaggriculture_rs import Engine


def _pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def test_existing_json_engine_contract_remains_available() -> None:
    engine = Engine({"seed": 17, "weedSpawnChance": 0.0})
    before = engine.state

    after = engine.step([_pass_action(), _pass_action()])

    assert before["step"] == 0
    assert after["step"] == 1
    assert engine.observation(0)["player"] == 0


def test_typed_observation_matches_json_compatibility_path() -> None:
    engine = Engine({"seed": 17, "weedSpawnChance": 0.0})
    scripted = (
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
        {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        {"farmer": ["WATER"], "hands": [], "market": [["HIRE"]]},
    )

    for action in scripted:
        for player in (0, 1):
            assert engine.observation(player) == json.loads(
                engine._native.observation_json(player)
            )
        engine.step_typed(
            {
                "schema_version": 1,
                "actions": [action, _pass_action()],
            }
        )

    for player in (0, 1):
        assert engine.observation(player) == json.loads(
            engine._native.observation_json(player)
        )


def test_typed_legality_mask_and_sanitizer_share_engine_rules() -> None:
    engine = Engine({"seed": 17, "weedSpawnChance": 0.0})

    assert engine.unit_action_mask(
        0,
        0,
        [],
        [["PASS"], ["PLANT", "WHEAT"], ["CARE"]],
    ) == [True, False, False]
    assert engine.market_order_mask(
        0,
        _pass_action(),
        [],
        [["SELL", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1], ["HIRE"]],
    ) == [False, True, True]

    assert engine.sanitize_action(
        0,
        {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [["CARE"]],
            "market": [["SELL", "WHEAT", 1], ["HIRE"]],
        },
    ) == {
        "farmer": ["PASS"],
        "hands": [["PASS"]],
        "market": [["HIRE"]],
    }
