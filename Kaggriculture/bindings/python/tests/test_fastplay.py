"""The Rust game loop must reproduce the Python harness game for game."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from kaggriculture_rs import ShopRouter, Tape, play_batch, play_fast

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ROOT / "artifacts" / "shop_router_v1" / "yusuke_routes.json"
OPPONENTS = ROOT / "analysis" / "v96_shop_router_20260905" / "opponents"
SCRIPTS = ROOT / "scripts"
CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]


def pool_tape(name: str) -> Tape:
    actions = json.loads((OPPONENTS / f"{name}.json").read_text(encoding="utf-8"))["actions"]
    return Tape(json.dumps(actions[1:720]))  # pool convention: actions[k+1] is step k


@pytest.fixture(scope="module")
def routes_json() -> str:
    return ROUTES.read_text(encoding="utf-8")


def test_tape_and_router_construct(routes_json: str) -> None:
    tape = pool_tape("kenjo1209__ep102449287__seat1")
    assert len(tape) == 719
    router = ShopRouter(routes_json, max_shops=2)
    assert router.route_count == 10 and router.max_shops == 2
    with pytest.raises(ValueError):
        ShopRouter(routes_json, max_shops=3)
    with pytest.raises(ValueError):
        Tape("{}")


def test_route_index_matches_python_rule(routes_json: str) -> None:
    from kaggriculture_rl.policy.shop_router_v1 import load_routes, route_index_for

    data = load_routes(ROUTES)
    kw = dict(first_routes=data["first_routes"], second_routes=data["second_routes"],
              first_branch_step=72, second_branch_step=144)
    for depth in (0, 1, 2):
        router = ShopRouter(routes_json, max_shops=depth)
        for step in (0, 71, 72, 143, 144, 300, 718):
            assert router.route_index(step, []) == route_index_for(step, [], max_shops=depth, **kw)
            for a, b in itertools.product(SHOPS, SHOPS):
                expected = route_index_for(step, [a, b], max_shops=depth, **kw)
                assert router.route_index(step, [a, b]) == expected, (depth, step, a, b)
                one = route_index_for(step, [a], max_shops=depth, **kw)
                assert router.route_index(step, [a]) == one


def _tape_agent(actions: list) -> object:
    def agent(observation, configuration=None):
        del configuration
        step = int(observation["step"] if isinstance(observation, dict) else observation.step)
        return actions[min(max(step, 0), 718)]
    return agent


def test_play_fast_matches_the_official_kaggle_env(routes_json: str) -> None:
    """The official env is the oracle; the Rust loop must reproduce it to the coin."""
    from kaggle_environments import make

    from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1

    kenjo_path = OPPONENTS / "kenjo1209__ep102449287__seat1.json"
    kenjo = json.loads(kenjo_path.read_text(encoding="utf-8"))["actions"][1:720]
    text = (ROOT / "artifacts" / "yusuke_schedule_s0.jsonl").read_text(encoding="utf-8")
    live0 = [json.loads(line) for line in text.splitlines() if line.strip()][:719]
    router_py = ShopRouterV1.from_file(max_shops=2)
    router_rs = ShopRouter(routes_json, max_shops=2)
    cases = [
        ("live_tape", 15300001, 0, _tape_agent(live0), Tape(json.dumps(live0))),
        ("router", 15300002, 0, router_py.agent, router_rs),
        ("router", 15300003, 1, router_py.agent, router_rs),
    ]
    for label, seed, seat, py_agent, rs_player in cases:
        env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720})
        agents = [None, None]
        agents[seat] = py_agent
        agents[1 - seat] = _tape_agent(kenjo)
        env.run(agents)
        official = [float(state.reward) for state in env.state]
        config = json.dumps({**CONFIG, "seed": seed})
        result = play_fast(config, seat, rs_player, Tape(json.dumps(kenjo)))
        assert result["money"] == official, (label, seed, seat, result["money"], official)
        assert result["steps"] == 719


def test_play_fast_rejects_bad_players(routes_json: str) -> None:
    router = ShopRouter(routes_json)
    with pytest.raises(ValueError):
        play_fast(json.dumps({**CONFIG, "seed": 1}), 2, router, router)
    with pytest.raises(ValueError):
        play_fast(json.dumps({**CONFIG, "seed": 1}), 0, router, "not a player")


def test_play_batch_matches_play_fast_and_is_order_stable(routes_json: str) -> None:
    router = ShopRouter(routes_json, max_shops=2)
    tape = pool_tape("kenjo1209__ep102449287__seat1")
    jobs = [(seed, seat, 0, 1) for seed in (15300001, 15300002, 15300003) for seat in (0, 1)]
    batch = play_batch(json.dumps(CONFIG), [router, tape], jobs, 4)
    assert len(batch) == len(jobs)
    for (seed, seat, _, _), result in zip(jobs, batch, strict=True):
        single = play_fast(json.dumps({**CONFIG, "seed": seed}), seat, router, tape)
        assert result == single, (seed, seat)
    with pytest.raises(ValueError):
        play_batch(json.dumps(CONFIG), [router, tape], [(1, 0, 0, 5)], 1)
    with pytest.raises(ValueError):
        play_batch(json.dumps(CONFIG), [router, tape], [(1, 3, 0, 1)], 1)
