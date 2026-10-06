from __future__ import annotations

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.phased_router_v3 import (
    COMMIT_STEP,
    PhasedRouterV3,
    classify_opponent,
)

CONFIG = {
    "episodeSteps": 720,
    "seed": 208248505,
    "weedSpawnChance": 0.0,
}


def test_classify_opponent_by_hires() -> None:
    assert classify_opponent(0, 3000.0) == "yusuke"
    assert classify_opponent(4, 3000.0) == "v21"
    assert classify_opponent(5, 20.0) == "v21"
    assert classify_opponent(6, 1850.0) == "v21"


def test_commit_step_constant() -> None:
    assert COMMIT_STEP == 1


def test_scout_is_resource_neutral_then_commit() -> None:
    router = PhasedRouterV3()
    router.reset()
    engine = Engine(dict(CONFIG))
    scout = router.agent(engine.observation(0), CONFIG)
    assert scout == {"farmer": ["PASS"], "hands": [], "market": []}
    assert router.committed_for(0) is None
    engine.step([scout, dict(scout)])
    assert router.committed_for(0) is None
    action = router.agent(engine.observation(0), CONFIG)
    assert router.committed_for(0) == "yusuke"
    selection = router.route_selection_for(0)
    assert selection["selected"] == "yusuke"
    assert selection["step"] == 1
    assert selection["signal"]["hires_today"] == 0
    assert set(action.keys()) == {"farmer", "hands", "market"}


def test_v3_completes_full_game() -> None:
    router = PhasedRouterV3()
    router.reset()
    engine = Engine(dict(CONFIG))
    for _ in range(719):
        actions = [
            router.agent(engine.observation(0), CONFIG),
            router.agent(engine.observation(1), CONFIG),
        ]
        for action in actions:
            assert set(action.keys()) == {"farmer", "hands", "market"}
        engine.step(actions)
    assert int(engine.state.get("step", 0)) == 719
    assert router.committed_for(0) == "yusuke"
    assert router.committed_for(1) == "yusuke"


def test_reset_clears_commitment() -> None:
    router = PhasedRouterV3()
    router.reset()
    engine = Engine(dict(CONFIG))
    for _ in range(5):
        engine.step(
            [
                router.agent(engine.observation(0), CONFIG),
                router.agent(engine.observation(1), CONFIG),
            ]
        )
    assert router.committed_for(0) is not None
    router.reset(0)
    assert router.committed_for(0) is None
    assert router.route_selection_for(0)["selected"] == ""
