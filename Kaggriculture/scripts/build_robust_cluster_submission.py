"""Render the robust-cluster policy as one self-contained Kaggle agent file."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import textwrap
import zlib
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-population", type=Path, required=True)
    parser.add_argument("--program-population", type=Path, required=True)
    parser.add_argument("--robust-program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path, index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"standalone_trace_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def main() -> None:
    args = parse_args()
    model = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    required = {
        int(model["base_action_id"]),
        *(int(value) for value in model["special_action_ids"].values()),
        *(int(value) for value in model["branch163"]["first_actions"]),
        *(
            int(value)
            for stage in model["stages"].values()
            for value in stage["decisions"]
        ),
    }
    paths: dict[int, Path] = {}
    for manifest_path in (
        args.raw_population,
        args.program_population,
        args.robust_program,
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for agent in manifest["agents"]:
            paths[int(agent["episode_id"])] = Path(agent["agent_file"])
    missing = required.difference(paths)
    if missing:
        raise ValueError(f"missing traces: {sorted(missing)}")
    traces = {
        str(action_id): load_trace(paths[action_id], index)
        for index, action_id in enumerate(sorted(required))
    }
    payload = base64.b85encode(
        zlib.compress(
            json.dumps(
                {"model": model, "traces": traces}, separators=(",", ":")
            ).encode("utf-8"),
            level=9,
        )
    ).decode("ascii")
    source = STANDALONE_TEMPLATE.replace("__PAYLOAD__", repr(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(textwrap.dedent(source), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "traces": len(traces),
            }
        )
    )


STANDALONE_TEMPLATE = r'''"""Self-contained robust-cluster Kaggriculture agent."""

import base64
import copy
import json
import math
import zlib


_DATA = json.loads(zlib.decompress(base64.b85decode(__PAYLOAD__)))
_MODEL = _DATA["model"]
_TRACES = {int(key): value for key, value in _DATA["traces"].items()}
_SELECTED = {}
_MODE = {}
_LAST_STEP = {}

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER", *ANIMALS)
MAX_RETRIEVER_UNITS = 13


def _tile_features(farm):
    kinds = {}
    crops = {}
    animals = {}
    ready = 0
    needs_water = 0
    fertilizer = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "EMPTY"))
            kinds[kind] = kinds.get(kind, 0) + 1
            crop = tile.get("crop")
            if crop is not None:
                crops[str(crop)] = crops.get(str(crop), 0) + 1
            animal = tile.get("animal")
            if animal is not None:
                animals[str(animal)] = animals.get(str(animal), 0) + 1
            ready += int(float(tile.get("yield_units", 0) or 0) > 0)
            needs_water += int(
                kind == "PLANT" and not tile.get("watered_today", False)
            )
            fertilizer += int(float(tile.get("fertilizer_available", 0) or 0) > 0)
    return [
        len(farm.get("hands", [])) / 16.0,
        len(farm.get("unlocked_quadrants", [])) / 4.0,
        float(farm.get("hires_today", 0) or 0) / 16.0,
        kinds.get("WEED", 0) / 100.0,
        kinds.get("PASTURE", 0) / 30.0,
        kinds.get("COOP", 0) / 30.0,
        ready / 30.0,
        needs_water / 30.0,
        fertilizer / 30.0,
        *(crops.get(crop, 0) / 30.0 for crop in CROPS),
        *(animals.get(animal, 0) / 30.0 for animal in ANIMALS),
    ]


def _features(observation):
    player = int(observation.get("player", 0))
    opponent = 1 - player
    farms = observation["farms"]
    candidate_farm = farms[player]
    opponent_farm = farms[opponent]
    candidate_money = float(candidate_farm.get("money", 0) or 0)
    opponent_money = float(opponent_farm.get("money", 0) or 0)
    prices = observation.get("market", {}).get("prices", {})
    private = observation.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    inventory = {}
    inventories = private.get("inventories", [])
    for unit_inventory in inventories:
        for item, quantity in unit_inventory.items():
            inventory[str(item)] = inventory.get(str(item), 0) + int(quantity)
    result = [
        float(observation.get("step", 0) or 0) / 719.0,
        float(observation.get("day", 0) or 0) / 30.0,
        float(observation.get("hour", 0) or 0) / 24.0,
        candidate_money / 200000.0,
        opponent_money / 200000.0,
        (candidate_money - opponent_money) / 200000.0,
        *(float(prices.get(item, 0) or 0) / 2000.0 for item in ITEMS),
        *_tile_features(candidate_farm),
        *_tile_features(opponent_farm),
        *(float(shed.get(item, 0) or 0) / 200.0 for item in ITEMS),
        *(float(seeds.get(crop, 0) or 0) / 50.0 for crop in CROPS),
        *(float(inventory.get(item, 0) or 0) / 100.0 for item in ITEMS),
    ]
    positions = [candidate_farm.get("farmer"), *candidate_farm.get("hands", [])]
    for index in range(MAX_RETRIEVER_UNITS):
        if index < len(positions) and positions[index] is not None:
            position = positions[index]
            result.extend((float(position[0]) / 9.0, float(position[1]) / 9.0, 1.0))
            unit_inventory = inventories[index] if index < len(inventories) else {}
            result.extend(
                float(unit_inventory.get(item, 0) or 0) / 50.0 for item in ITEMS
            )
        else:
            result.extend((-1.0, -1.0, 0.0))
            result.extend(0.0 for _ in ITEMS)
    result.append(float(player))
    return result


def _nearest_index(model, features):
    mean = model["feature_mean"]
    scale = model["feature_scale"]
    normalized = [
        (value - mean[index]) / scale[index]
        for index, value in enumerate(features)
    ]
    best_index = 0
    best_distance = float("inf")
    for row_index, prototype in enumerate(model["features"]):
        distance = math.sqrt(
            sum(
                (prototype[index] - normalized[index]) ** 2
                for index in range(len(normalized))
            )
        )
        if distance < best_distance:
            best_index = row_index
            best_distance = distance
    return best_index, best_distance


def _nearest(model, features, value_key):
    index, distance = _nearest_index(model, features)
    return model[value_key][index], distance


def agent(observation, configuration=None):
    del configuration
    player = int(observation.get("player", 0))
    step = int(observation.get("step", 0) or 0)
    previous = _LAST_STEP.get(player, -1)
    if step < previous or (step == 0 and previous != 0):
        _SELECTED.pop(player, None)
        _MODE.pop(player, None)
    _LAST_STEP[player] = step
    _SELECTED.setdefault(player, int(_MODEL["base_action_id"]))
    _MODE.setdefault(player, "base")
    features = None
    if step == 97:
        features = _features(observation)
        mode, distance = _nearest(_MODEL["branch97"], features, "modes")
        if distance < 1e-10 and mode != "base":
            _MODE[player] = mode
            _SELECTED[player] = int(_MODEL["special_action_ids"][mode])
    if step == 163 and _MODE[player] == "base":
        features = _features(observation)
        row, _ = _nearest_index(_MODEL["branch163"], features)
        _SELECTED[player] = int(_MODEL["branch163"]["first_actions"][row])
        replan = _MODEL["branch163"]["replans"][row]
        if replan:
            _MODE[player] = replan
    mode = _MODE[player]
    if mode == "special_cluster" and step == 216:
        mode = "special_cluster_216"
    stage = _MODEL["stages"].get(mode)
    if stage is not None and step == int(stage["branch_step"]):
        if features is None:
            features = _features(observation)
        selected, _ = _nearest(stage, features, "decisions")
        _SELECTED[player] = int(selected)
        _MODE[player] = "committed"
    trace = _TRACES[_SELECTED[player]]
    return copy.deepcopy(trace[min(step, len(trace) - 1)])
'''


if __name__ == "__main__":
    main()
