from __future__ import annotations

import copy

import pytest
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.shop_router_v1 import (
    DEFAULT_ROUTES_PATH,
    PASS_ACTION,
    ShopRouterV1,
    align_hands,
    load_routes,
    observation_step,
    route_index_for,
    splice,
)

CONFIG = {"episodeSteps": 720, "seed": 847999973, "weedSpawnChance": 0.0}


@pytest.fixture(scope="module")
def data() -> dict:
    return load_routes(DEFAULT_ROUTES_PATH)


def test_vendored_routes_have_expected_shape(data: dict) -> None:
    assert len(data["routes"]) == 10
    assert all(len(route) == 719 for route in data["routes"])
    assert data["first_branch_step"] == 72 and data["second_branch_step"] == 144
    assert len(data["first_routes"]) == 8 and len(data["second_routes"]) == 64
    specialists = sum(
        1 for pair, idx in data["second_routes"].items()
        if idx != data["first_routes"][pair.split("__", 1)[0]]
    )
    assert specialists == 23


def test_routes_share_prefix_up_to_their_branch(data: dict) -> None:
    routes = data["routes"]
    base = routes[0]
    for index, route in enumerate(routes[1:], start=1):
        first_diff = next(t for t in range(719) if route[t] != base[t])
        boundary = data["first_branch_step"] if index == 1 else data["second_branch_step"]
        assert first_diff >= boundary, (index, first_diff)


def test_route_index_rule(data: dict) -> None:
    kw = dict(
        first_routes=data["first_routes"], second_routes=data["second_routes"],
        first_branch_step=72, second_branch_step=144,
    )
    assert route_index_for(0, [], **kw) == 0
    assert route_index_for(71, ["YARN_STORE"], **kw) == 0
    assert route_index_for(72, ["YARN_STORE"], **kw) == 1
    assert route_index_for(72, ["BAKERY"], **kw) == 0
    assert route_index_for(143, ["BAKERY", "PET_CAFE"], **kw) == 0
    assert route_index_for(144, ["BAKERY", "PET_CAFE"], **kw) == 3
    assert route_index_for(144, ["BAKERY", "PET_CAFE"], max_shops=1, **kw) == 0
    assert route_index_for(144, ["YARN_STORE", "BAKERY"], max_shops=0, **kw) == 0
    # unknown second key falls back to the first-shop route, not to a foreign route
    assert route_index_for(144, ["YARN_STORE", "NOPE"], **kw) == 1


def test_observation_step_falls_back_to_day_and_hour() -> None:
    assert observation_step({"step": 90}) == 90
    assert observation_step({"day": 3, "hour": 18}) == 90
    assert observation_step({"step": 5000}) == 718
    assert observation_step({"step": -3}) == 0


def test_align_hands_pads_and_trims() -> None:
    obs = {"player": 1, "farms": [{"hands": []}, {"hands": [{}, {}, {}]}]}
    padded = align_hands({"farmer": ["PASS"], "hands": [["NORTH"]], "market": []}, obs)
    assert padded["hands"] == [["NORTH"], ["PASS"], ["PASS"]]
    four = {"farmer": ["PASS"], "hands": [["A"], ["B"], ["C"], ["D"]], "market": []}
    trimmed = align_hands(four, obs)
    assert trimmed["hands"] == [["A"], ["B"], ["C"]]


def test_splice_keeps_base_prefix_and_donor_suffix() -> None:
    base = [["b", i] for i in range(10)]
    donor = [["d", i] for i in range(10)]
    out = splice(base, donor, 4)
    assert out[:4] == base[:4] and out[4:] == donor[4:]
    with pytest.raises(ValueError):
        splice(base, donor, 0)
    with pytest.raises(ValueError):
        splice(base, donor[:5], 4)


def test_agent_is_robust_to_garbage() -> None:
    router = ShopRouterV1.from_file()
    assert router.agent(None) == PASS_ACTION
    assert router.agent({"step": "x"}) == PASS_ACTION


def test_router_plays_full_game_on_both_seats_and_routes_on_revealed_shops() -> None:
    router = ShopRouterV1.from_file()
    engine = Engine(dict(CONFIG))
    seen: dict[int, list[int]] = {0: [], 1: []}
    for _ in range(719):
        actions = []
        for seat in (0, 1):
            observation = engine.observation(seat)
            action = router.agent(observation, CONFIG)
            assert set(action.keys()) == {"farmer", "hands", "market"}
            seen[seat].append(router.route_index(observation))
            actions.append(action)
        engine.step(actions)
    assert int(engine.state.get("step", 0)) == 719
    for seat in (0, 1):
        assert set(seen[seat][:72]) == {0}
        # once the second shop is revealed the route is fixed for the rest of the game
        assert len(set(seen[seat][144:])) == 1
    shops = engine.observation(0)["town"]["unlocked_shops"]
    assert len(shops) >= 2
    expected = router.route_index({"step": 700, "town": {"unlocked_shops": shops[:2]}})
    assert seen[0][-1] == expected == seen[1][-1]


def test_max_shops_zero_is_the_base_tape(data: dict) -> None:
    base_only = ShopRouterV1(data, max_shops=0)
    obs = {"step": 300, "player": 0, "farms": [{"hands": []}],
           "town": {"unlocked_shops": ["YARN_STORE", "BAKERY"]}}
    expected = copy.deepcopy(data["routes"][0][300])
    expected["hands"] = []
    assert base_only.agent(obs) == expected
    assert base_only.route_index(obs) == 0
    assert ShopRouterV1(data, max_shops=2).route_index(obs) == 1
