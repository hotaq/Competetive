from __future__ import annotations

from typing import Any

import kaggriculture_rs
import pytest
from kaggriculture_rs import Engine, plan_observation

from kaggriculture_rl.schemas import (
    validate_execution_report,
    validate_joint_action,
    validate_observation,
)


def _pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _buy_wheat(engine: Engine, quantity: int) -> None:
    engine.step(
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_PRODUCT", "WHEAT", quantity]],
            },
            _pass_action(),
        ]
    )


def test_observation_snapshot_planner_matches_stateful_engine() -> None:
    config = {
        "seed": 91,
        "weedSpawnChance": 0.0,
        "maxMarketOrdersPerTurn": 6,
        "farmHandCostMult": 2,
    }
    engine = Engine(config)
    targets = [
        {"schema_version": 1, "kind": "SELLING", "item": "MILK", "quantity": 4},
        {"schema_version": 1, "kind": "WORKFORCE", "target_count": 4},
        {
            "schema_version": 1,
            "kind": "INVENTORY_BUFFER",
            "item": "WHEAT",
            "target_quantity": 8,
        },
        {"schema_version": 1, "kind": "LAND_EXPANSION", "quadrant": "NE"},
        {
            "schema_version": 1,
            "kind": "ANIMAL_PRODUCTION",
            "animal": "COW",
            "target_count": 4,
        },
        {
            "schema_version": 1,
            "kind": "CROP_PRODUCTION",
            "crop": "MELON",
            "target_quantity": 6,
        },
    ]

    for _ in range(48):
        actions: list[dict[str, object]] = []
        for player in range(2):
            expected = engine.plan(player, targets)
            actual = plan_observation(engine.observation(player), targets, config)
            assert actual == expected
            actions.append(expected["joint_action"]["actions"][player])
        engine.step(actions)


def test_typed_selling_plan_and_apply_are_contract_valid() -> None:
    engine = Engine({"seed": 7, "weedSpawnChance": 0.0})
    _buy_wheat(engine, 4)
    before = engine.observation(0)["private"]["shed"]["WHEAT"]

    planned = engine.plan(
        0,
        [{"schema_version": 1, "kind": "SELLING", "item": "WHEAT", "quantity": 3}],
    )

    validate_joint_action(planned["joint_action"])
    validate_execution_report(planned["report"])
    assert planned["report"]["status"] == "PLANNED"
    assert planned["joint_action"]["actions"][0]["market"] == [["SELL", "WHEAT", 3]]
    assert engine.observation(0)["private"]["shed"]["WHEAT"] == before

    result = engine.step_typed(planned["joint_action"])

    for observation in result["observations"]:
        validate_observation(observation)
    assert result["observations"][0]["shed"]["WHEAT"] == before - 3


def test_workforce_target_returns_explicit_legal_hires() -> None:
    engine = Engine()
    before = engine.state

    planned = engine.plan(
        0,
        [{"schema_version": 1, "kind": "WORKFORCE", "target_count": 2}],
    )

    assert engine.state == before
    assert planned["joint_action"]["actions"][0]["market"] == [["HIRE"], ["HIRE"]]
    assert planned["report"]["status"] == "PLANNED"
    assert planned["report"]["fallback_count"] == 0
    assert planned["report"]["entries"][0]["code"] == "KG-EXEC-PLANNED"


@pytest.mark.parametrize(
    ("target", "code"),
    [
        (
            {"schema_version": 2, "kind": "SELLING", "item": "MILK", "quantity": 1},
            "KG-SCHEMA-VERSION",
        ),
        (
            {"schema_version": 1, "kind": "UNKNOWN", "item": "MILK", "quantity": 1},
            "KG-SCHEMA-ENUM",
        ),
        (
            {
                "schema_version": 1,
                "kind": "SELLING",
                "item": "MILK",
                "quantity": 1,
                "unexpected": True,
            },
            "KG-SCHEMA-INVALID",
        ),
    ],
)
def test_invalid_target_fails_closed_before_mutation(target: dict[str, Any], code: str) -> None:
    engine = Engine()
    before = engine.state

    with pytest.raises(ValueError, match=code):
        engine.plan(0, [target])

    assert engine.state == before


def test_typed_hot_path_does_not_call_python_json_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine()

    def fail_json(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("typed hot path called Python JSON codec")

    for name in ("dumps", "loads"):
        monkeypatch.setattr(
            kaggriculture_rs.json,
            name,
            fail_json,
        )

    planned = engine.plan(
        0,
        [{"schema_version": 1, "kind": "WORKFORCE", "target_count": 2}],
    )
    result = engine.step_typed(planned["joint_action"])

    assert result["step"] == 1
