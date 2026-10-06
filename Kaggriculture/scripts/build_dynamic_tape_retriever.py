"""Build compact state prototypes for dynamic replay-tape retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)

RETRIEVER_FEATURE_DIM = len(
    encode_retriever_features(
        {
            "player": 0,
            "farms": [
                {"farmer": [0, 0], "hands": [], "tiles": []},
                {"farmer": [0, 0], "hands": [], "tiles": []},
            ],
            "private": {},
        }
    )
)
SELF_FEATURE_INDICES = np.asarray(
    [*range(0, 4), *range(6, 35), *range(52, RETRIEVER_FEATURE_DIM)],
    dtype=np.int64,
)
ALL_FEATURE_INDICES = np.arange(RETRIEVER_FEATURE_DIM, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--feature-view", choices=("self", "all"), default="self")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_entries(manifest_path: Path) -> dict[int, tuple[Path, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        int(entry["episode_id"]): (
            manifest_path.parent / f"{int(entry['episode_id'])}.json",
            int(entry["top1_seat"]),
        )
        for entry in manifest["episodes"]
    }


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    population_ids = [int(agent["episode_id"]) for agent in population["agents"]]

    excluded: set[int] = set()
    for manifest_path in args.exclude_manifest:
        excluded.update(manifest_entries(manifest_path))

    sources: dict[int, tuple[Path, int]] = {}
    for manifest_path in args.manifest:
        sources.update(manifest_entries(manifest_path))
    action_ids = [
        episode_id
        for episode_id in population_ids
        if episode_id in sources and episode_id not in excluded
    ]
    if not action_ids:
        raise ValueError("no population episodes remain after filtering")

    indices = (
        SELF_FEATURE_INDICES
        if args.feature_view == "self"
        else ALL_FEATURE_INDICES
    )
    episodes: list[np.ndarray] = []
    step_count: int | None = None
    for position, episode_id in enumerate(action_ids, start=1):
        replay_path, seat = sources[episode_id]
        replay: dict[str, Any] = json.loads(replay_path.read_text(encoding="utf-8"))
        observations = [frame[seat]["observation"] for frame in replay["steps"][:-1]]
        features = np.stack(
            [encode_retriever_features(observation)[indices] for observation in observations]
        )
        if step_count is None:
            step_count = len(features)
        elif len(features) != step_count:
            raise ValueError(
                f"episode {episode_id} has {len(features)} states, expected {step_count}"
            )
        episodes.append(features.astype(np.float32))
        print(json.dumps({"loaded": position, "episode_id": episode_id}))

    prototype_features = np.stack(episodes)
    feature_scale = np.std(prototype_features, axis=(0, 1), dtype=np.float64)
    metadata = {
        "population": str(args.population),
        "manifests": [str(path) for path in args.manifest],
        "excluded_manifests": [str(path) for path in args.exclude_manifest],
        "feature_view": args.feature_view,
        "episodes": len(action_ids),
        "steps": int(prototype_features.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prototype_features=prototype_features,
        feature_scale=feature_scale,
        feature_indices=indices,
        action_ids=np.asarray(action_ids, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": str(args.output)}))


if __name__ == "__main__":
    main()
