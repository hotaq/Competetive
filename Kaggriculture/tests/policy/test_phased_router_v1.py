from __future__ import annotations

import copy

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.phased_router_v1 import (
    BOOK_SOURCE,
    PhasedRouterV1,
    load_opening,
)

CONFIG = {
    "episodeSteps": 720,
    "seed": 2053877166,
    "weedSpawnChance": 0.0,
}


def test_opening_book_has_16_steps() -> None:
    opening = load_opening()
    assert len(opening) == 16
    assert BOOK_SOURCE["submission_id"] == 55929317
    first_market = opening[1]["market"]
    hires = [order for order in first_market if order[0] == "HIRE"]
    assert len(hires) == 4
    assert opening[1]["farmer"] == ["BUILD_PASTURE"]


def test_book_phase_replays_crop_dusta_prefix() -> None:
    policy = PhasedRouterV1()
    opening = load_opening()
    engine = Engine(dict(CONFIG))
    for step in range(16):
        for seat in (0, 1):
            observation = engine.observation(seat)
            assert int(observation.get("step", -1)) == step
            action = policy.agent(copy.deepcopy(observation), CONFIG)
            assert set(action.keys()) == {"farmer", "hands", "market"}
            expected = copy.deepcopy(opening[step])
            assert action["farmer"] == expected["farmer"]
            assert action["market"] == expected["market"]
        engine.step(
            [
                policy.agent(engine.observation(0), CONFIG),
                policy.agent(engine.observation(1), CONFIG),
            ]
        )


def test_route_locks_in_self_play() -> None:
    policy = PhasedRouterV1()
    policy.reset()
    engine = Engine(dict(CONFIG))
    for _ in range(719):
        actions = [
            policy.agent(engine.observation(0), CONFIG),
            policy.agent(engine.observation(1), CONFIG),
        ]
        for action in actions:
            assert set(action.keys()) == {"farmer", "hands", "market"}
        engine.step(actions)
    locked = [policy.route_selection_for(seat) for seat in (0, 1)]
    for selection in locked:
        assert selection["selected"] in ("moon", "mutoy", "munib")
        assert selection["step"] is not None and selection["step"] <= 240


def test_reset_clears_route_selection() -> None:
    policy = PhasedRouterV1()
    policy.reset()
    engine = Engine(dict(CONFIG))
    for _ in range(130):
        engine.step(
            [
                policy.agent(engine.observation(0), CONFIG),
                policy.agent(engine.observation(1), CONFIG),
            ]
        )
    policy.reset(0)
    assert policy.route_selection_for(0)["selected"] == ""
    policy.reset()
    assert policy.route_selection_for(1)["selected"] == ""
