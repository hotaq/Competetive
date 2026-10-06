from __future__ import annotations

import pytest
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.phased_router_v2 import (
    BUNDLE_SOURCE,
    DustaSchedule,
    PhasedRouterV2,
)

CONFIG = {
    "episodeSteps": 720,
    "seed": 847999973,
    "weedSpawnChance": 0.0,
}


def test_schedule_sources_are_pinned() -> None:
    assert BUNDLE_SOURCE["dusta_s0_episode"] == 105391557
    assert BUNDLE_SOURCE["dusta_s1_episode"] == 105366473
    dusta = DustaSchedule()
    engine = Engine(dict(CONFIG))
    for seat in (0, 1):
        action = dusta.agent(engine.observation(seat), CONFIG)
        assert set(action.keys()) == {"farmer", "hands", "market"}


def test_schedule_matches_recorded_opening() -> None:
    dusta = DustaSchedule()
    engine = Engine(dict(CONFIG))
    first = dusta.agent(engine.observation(0), CONFIG)
    assert first == {"farmer": ["PASS"], "hands": [], "market": []}
    engine.step([first, dict(first)])
    action = dusta.agent(engine.observation(0), CONFIG)
    assert action["farmer"] == ["BUILD_PASTURE"]
    hires = [order for order in action["market"] if order[0] == "HIRE"]
    assert len(hires) == 4
def test_bundle_picker_defaults_and_override() -> None:
    router = PhasedRouterV2()
    assert router.bundle_for(0) == "yusuke"
    assert router.bundle_for(1) == "yusuke"
    forced = PhasedRouterV2(bundle_by_seat={0: "dusta", 1: "dusta"})
    assert forced.bundle_for(0) == "dusta"
    assert forced.bundle_for(1) == "dusta"
    assert forced.route_selection_for(0)["selected"] == "dusta"
    with pytest.raises(ValueError):
        PhasedRouterV2(bundle_by_seat={0: "nope"})


def test_dusta_bundle_completes_full_game() -> None:
    router = PhasedRouterV2(bundle_by_seat={0: "dusta", 1: "dusta"})
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


def test_yusuke_bundle_is_selectable() -> None:
    from kaggriculture_rl.policy.phased_router_v2 import BUNDLE_SOURCE, BUNDLES

    assert "yusuke" in BUNDLES
    assert BUNDLE_SOURCE["yusuke_s0_episode"] == 102574290
    assert BUNDLE_SOURCE["yusuke_s1_episode"] == 102589974
    router = PhasedRouterV2(bundle_by_seat={0: "yusuke", 1: "yusuke"})
    assert router.bundle_for(0) == "yusuke"
    assert router.route_selection_for(1)["selected"] == "yusuke"
    engine = Engine(dict(CONFIG))
    for _ in range(30):
        actions = [
            router.agent(engine.observation(0), CONFIG),
            router.agent(engine.observation(1), CONFIG),
        ]
        for action in actions:
            assert set(action.keys()) == {"farmer", "hands", "market"}
        engine.step(actions)
