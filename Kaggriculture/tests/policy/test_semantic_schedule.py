from kaggriculture_rs import Engine

from kaggriculture_rl.policy.semantic_schedule import (
    SemanticFarmSchedule,
    SemanticFarmStage,
    top1_imitation_schedule,
)


def test_schedule_selects_latest_stage_and_emits_scheduled_crop_output() -> None:
    observation = Engine({"seed": 23}).observation(0)
    observation["step"] = 150
    observation["hour"] = 2
    schedule = top1_imitation_schedule()

    stage = schedule.stage(observation)
    targets = schedule(observation)

    assert stage.start_step == 144
    assert [target["crop"] for target in targets if target["kind"] == "CROP_PRODUCTION"] == [
        "STRAWBERRY"
    ]


def test_schedule_rejects_duplicate_stage_steps() -> None:
    stage = SemanticFarmStage(96, (("WHEAT", 4),))

    try:
        SemanticFarmSchedule((stage, stage))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate stage steps must be rejected")


def test_schedule_rotates_production_priority_across_the_day() -> None:
    observation = Engine({"seed": 29}).observation(0)
    observation["step"] = 150
    observation["hour"] = 0
    schedule = top1_imitation_schedule()
    first = [
        target
        for target in schedule(observation)
        if target["kind"] in {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}
    ]
    observation["hour"] = 1
    second = [
        target
        for target in schedule(observation)
        if target["kind"] in {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}
    ]

    assert first[0]["animal"] == "COW"
    assert second[0]["animal"] == "SHEEP"


def test_schedule_terminates_investment_and_liquidates_on_the_last_day() -> None:
    observation = Engine({"seed": 31}).observation(0)
    observation["step"] = 700
    observation["private"]["shed"]["WHEAT"] = 7
    targets = top1_imitation_schedule()(observation)

    assert [target["kind"] for target in targets] == ["SELLING", "RECOVERY"]
    assert targets[0]["quantity"] == 7
    assert targets[1]["mode"] == "INVENTORY"
