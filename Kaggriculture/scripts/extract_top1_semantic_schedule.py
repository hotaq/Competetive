"""Summarize observable Top-1 farm targets at semantic decision steps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
DEFAULT_STEPS = (96, 120, 144, 168, 216, 288, 360, 480, 600, 696)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--step", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _farm_metrics(observation: dict[str, Any], seat: int) -> dict[str, float]:
    farm = observation["farms"][seat]
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    structures: Counter[str] = Counter()
    weeds = 0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", ""))
            structures[kind] += 1
            if kind == "WEED":
                weeds += 1
            if tile.get("crop") is not None:
                crops[str(tile["crop"])] += 1
            if tile.get("animal") is not None:
                animals[str(tile["animal"])] += 1
    private = observation.get("private", {})
    shed = private.get("shed", {})
    metrics: dict[str, float] = {
        "money": float(farm.get("money", 0.0)),
        "workers": float(len(farm.get("hands", []))),
        "hires_today": float(farm.get("hires_today", 0)),
        "land_quadrants": float(len(farm.get("unlocked_quadrants", []))),
        "weeds": float(weeds),
        "pastures": float(structures["PASTURE"]),
        "coops": float(structures["COOP"]),
        "shed_wheat": float(shed.get("WHEAT", 0) or 0),
    }
    metrics.update({f"crop_{crop.lower()}": float(crops[crop]) for crop in CROPS})
    metrics.update(
        {f"animal_{animal.lower()}": float(animals[animal]) for animal in ANIMALS}
    )
    return metrics


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    args = parse_args()
    steps = tuple(sorted(set(args.step or DEFAULT_STEPS)))
    if not steps or min(steps) < 0 or max(steps) > 719:
        raise SystemExit("steps must be within 0..719")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["episodes"][: args.limit or None]
    values: dict[int, dict[str, list[float]]] = {
        step: defaultdict(list) for step in steps
    }
    episode_ids: list[int] = []
    for index, row in enumerate(rows, start=1):
        episode_id = int(row["episode_id"])
        seat = int(row["top1_seat"])
        replay = json.loads(
            (args.manifest.parent / f"{episode_id}.json").read_text(encoding="utf-8")
        )
        episode_ids.append(episode_id)
        for step in steps:
            observation = replay["steps"][step][seat]["observation"]
            for name, value in _farm_metrics(observation, seat).items():
                values[step][name].append(value)
        if index % 8 == 0 or index == len(rows):
            print(
                json.dumps({"completed": index, "total": len(rows)}, sort_keys=True),
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "episodes": len(rows),
        "episode_ids": episode_ids,
        "steps": {
            str(step): {
                name: _summary(metric_values)
                for name, metric_values in sorted(values[step].items())
            }
            for step in steps
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "episodes": len(rows), "steps": steps}))


if __name__ == "__main__":
    main()
