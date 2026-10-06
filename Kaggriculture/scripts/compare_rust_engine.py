"""Turn-by-turn differential test: Rust engine vs installed Kaggle reference."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
RUST_ENGINE = ROOT / "target" / "debug" / "kg-engine"
OBSERVATION_KEYS = ("player", "step", "day", "hour", "farms", "private", "market", "town")
SELLABLE_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)


class RustEngine:
    def __init__(self) -> None:
        self._process = subprocess.Popen(
            [str(RUST_ENGINE)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        response = json.loads(self._process.stdout.readline())
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Rust engine request failed"))
        return response

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.terminate()
        self._process.wait(timeout=5)


def _pass_action() -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _actions(step: int) -> list[dict[str, Any]]:
    p0 = _pass_action()
    p1 = _pass_action()

    if step == 0:
        p0["market"] = [
            ["BUY_SEED", "WHEAT", 2],
            ["BUY_PRODUCT", "WHEAT", 3],
            ["HIRE"],
            ["BUY_ANIMAL", "GOOSE", 1],
        ]
        p1["market"] = [
            ["BUY_SEED", "CARROT", 2],
            ["BUY_PRODUCT", "FERTILIZER", 2],
            ["HIRE"],
            ["BUY_LAND"],
        ]
    elif step == 1:
        p0.update(farmer=["PLANT", "WHEAT"], hands=[["PICKUP", "GOOSE", 1]])
        p1.update(farmer=["PLANT", "CARROT"], hands=[["WEST"]])
    elif step == 2:
        p0.update(farmer=["WATER"], hands=[["PICKUP", "WHEAT", 2]])
        p1.update(farmer=["WATER"], hands=[["NORTH"]])
    elif step == 3:
        p0.update(farmer=["NORTH"], hands=[["WEST"]])
        p1.update(farmer=["NORTH"], hands=[["BUILD_PASTURE"]])
    elif step == 4:
        p0.update(farmer=["EAST"], hands=[["NORTH"]])
        p1.update(farmer=["PLANT", "CARROT"], hands=[["DIG"]])
    elif step == 5:
        p0.update(farmer=["WEST"], hands=[["BUILD_COOP"]])
        p1.update(farmer=["WATER"], hands=[["SOUTH"]])
    elif step == 6:
        p0.update(farmer=["SOUTH"], hands=[["PLACE", "GOOSE"]])
        p1["market"] = [["BUY_LAND"], ["BUY_SEED", "MELON", 1]]
    elif step == 7:
        p0.update(farmer=["WATER"], hands=[["FEED"]])
        p1["market"] = [["HIRE"], ["BUY_PRODUCT", "WHEAT", 1]]
    elif step == 8:
        p0["hands"] = [["CARE"]]
        p1["hands"] = [["EAST"], ["PICKUP", "WHEAT", 1]]
    elif step == 9:
        p0["hands"] = [["COLLECT_FERTILIZER"]]
        p1["hands"] = [["PASS"], ["WEST"]]
    elif step == 10:
        # Two WHEAT requests with no seed left must both become silent no-ops.
        p0.update(farmer=["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])
        p1["market"] = [["BUY_PRODUCT", "INVALID", 3], ["SELL", "FERTILIZER", 1]]
    elif step == 12:
        p0["market"] = [["BUY_PRODUCT", "FERTILIZER", 1]]
    elif step == 13:
        p0["farmer"] = ["PICKUP", "FERTILIZER", 1]
    elif step == 14:
        p0["farmer"] = ["FERTILIZE"]
    elif step == 24:
        p0["market"] = [["HIRE"], ["BUY_SEED", "TOMATO", 1]]
        p1["market"] = [["HIRE"], ["BUY_ANIMAL", "SHEEP", 1]]
    elif step == 25:
        p0["farmer"] = ["PLANT", "TOMATO"]
        p1["farmer"] = ["DIG"]
    elif step == 26:
        p0["farmer"] = ["WATER"]

    return [p0, p1]


def _python_observation(state: list[Any], player: int) -> dict[str, Any]:
    observation = json.loads(json.dumps(state[player].observation))
    # kaggle-environments stores the shared step counter only on observation 0.
    observation.setdefault("step", state[0].observation.step)
    return {key: observation[key] for key in OBSERVATION_KEYS}


def _rust_observation(state: dict[str, Any], player: int) -> dict[str, Any]:
    return {
        "player": player,
        "step": state["step"],
        "day": state["day"],
        "hour": state["hour"],
        "farms": state["farms"],
        "private": state["privates"][player],
        "market": state["market"],
        "town": state["town"],
    }


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        # JSON does not distinguish integral floats semantically.
        if isinstance(left, int | float) and isinstance(right, int | float) and left == right:
            return None
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _assert_parity(python_state: list[Any], rust_state: dict[str, Any], label: str) -> None:
    for player in range(2):
        difference = _first_difference(
            _python_observation(python_state, player),
            _rust_observation(rust_state, player),
        )
        if difference:
            raise AssertionError(f"{label}, player {player}: {difference}")


def compare(turns: int = 72) -> None:
    if not RUST_ENGINE.exists():
        raise RuntimeError("Build the Rust binary first: cargo build --bin kg-engine")
    config = {
        # Kaggle counts the initialized state as one episode step, so N action
        # turns require episodeSteps=N+1.
        "episodeSteps": turns + 1,
        "boardSize": 10,
        "startingMoney": 3000,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.05,
        "townShopUnlockInterval": 2,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "farmHandCostMult": 1,
        "seed": 1729,
    }
    reference = make("kaggriculture", configuration=config, debug=True)
    python_state = reference.reset(2)
    rust = RustEngine()
    try:
        rust_state = rust.request({"op": "reset", "config": config})["state"]
        _assert_parity(python_state, rust_state, "initial state")

        for turn in range(turns):
            actions = _actions(turn)
            python_state = reference.step(actions)
            rust_state = rust.request({"op": "step", "actions": actions})["state"]
            _assert_parity(python_state, rust_state, f"after action turn {turn}")
    finally:
        rust.close()


def _random_unit_action(rng: random.Random) -> list[Any]:
    simple = [
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "PASS",
        "DROP",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    ]
    category = rng.randrange(4)
    if category == 0:
        return [rng.choice(simple)]
    if category == 1:
        return ["PLANT", rng.choice(["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])]
    item = rng.choice(
        [
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
            "EGG",
            "MILK",
            "WOOL",
            "FERTILIZER",
            "GOOSE",
            "COW",
            "SHEEP",
        ]
    )
    return [rng.choice(["PICKUP", "PLACE"]), item, rng.randint(1, 4)]


def _random_market_order(rng: random.Random) -> list[Any]:
    operation = rng.choice(["HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"])
    if operation in {"HIRE", "BUY_LAND"}:
        return [operation]
    if operation == "BUY_SEED":
        item = rng.choice(["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])
    elif operation == "BUY_PRODUCT":
        item = rng.choice(["WHEAT", "FERTILIZER", "CARROT"])
    elif operation == "BUY_ANIMAL":
        item = rng.choice(["GOOSE", "COW", "SHEEP"])
    else:
        item = rng.choice(SELLABLE_PRODUCTS)
    return [operation, item, rng.randint(1, 5)]


def _random_player_action(rng: random.Random) -> dict[str, Any]:
    return {
        "farmer": _random_unit_action(rng),
        "hands": [_random_unit_action(rng) for _ in range(rng.randrange(4))],
        "market": [_random_market_order(rng) for _ in range(rng.randrange(6))],
    }


def compare_random(episodes: int = 8, turns: int = 120) -> None:
    """Property-style differential sweep over legal-shaped random action traces."""
    if not RUST_ENGINE.exists():
        raise RuntimeError("Build the Rust binary first: cargo build --bin kg-engine")
    rng = random.Random(90_210)
    rust = RustEngine()
    try:
        for episode in range(episodes):
            config: dict[str, Any] = {
                "episodeSteps": turns + 1,
                "seed": 10_000 + episode,
                "weedSpawnChance": 0.03,
                "shedCapacity": 12,
                "maxMarketOrdersPerTurn": 4,
                "townShopUnlockInterval": 2,
                "townShopSellInterval": 3,
                "townCenterSellInterval": 12,
            }
            if episode == 0:
                config["marketParams"] = {
                    "WHEAT": {"below_target": 0.5},
                    "WOOL": {"above_target": 0.95},
                }
            reference = make("kaggriculture", configuration=config, debug=True)
            python_state = reference.reset(2)
            rust_state = rust.request({"op": "reset", "config": config})["state"]
            _assert_parity(python_state, rust_state, f"random episode {episode} initial")
            for turn in range(turns):
                actions = [_random_player_action(rng), _random_player_action(rng)]
                python_state = reference.step(actions)
                rust_state = rust.request({"op": "step", "actions": actions})["state"]
                _assert_parity(
                    python_state,
                    rust_state,
                    f"random episode {episode} after turn {turn}",
                )
    finally:
        rust.close()


def compare_typed_json() -> None:
    """Verify typed plan/apply matches the retained JSON compatibility path."""
    from kaggriculture_rs import Engine as BindingEngine

    config = {"seed": 7, "weedSpawnChance": 0.0}
    typed_engine = BindingEngine(config)
    json_engine = BindingEngine(config)
    setup = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 4]],
        },
        _pass_action(),
    ]
    typed_engine.step(setup)
    json_engine.step(setup)
    planned = typed_engine.plan(
        0,
        [{"schema_version": 1, "kind": "SELLING", "item": "WHEAT", "quantity": 3}],
    )

    typed_result = typed_engine.step_typed(planned["joint_action"])
    json_state = json_engine.step(planned["joint_action"]["actions"])
    for player, typed_observation in enumerate(typed_result["observations"]):
        expected = {
            "schema_version": 1,
            "player": player,
            "step": json_state["step"],
            "day": json_state["day"],
            "hour": json_state["hour"],
            "shed": {
                item: max(0, json_state["privates"][player]["shed"].get(item, 0))
                for item in SELLABLE_PRODUCTS
            },
        }
        difference = _first_difference(expected, typed_observation)
        if difference:
            raise AssertionError(f"typed-vs-JSON player {player}: {difference}")


if __name__ == "__main__":
    compare_typed_json()
    compare()
    compare_random()
    print("Rust/Python parity passed: deterministic full-state trace plus 960 randomized turns.")
