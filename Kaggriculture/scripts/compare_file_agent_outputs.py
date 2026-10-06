"""Compare two file agents on identical reconstructed observations from one replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    configuration = engine_config(replay)
    _, left, left_accepts_config = load_agent(args.left)
    _, right, right_accepts_config = load_agent(args.right)
    engine = Engine(configuration)
    mismatches: list[dict[str, Any]] = []
    exact_steps = 0

    for frame_index, frame in enumerate(replay["steps"][1:], start=1):
        observation = engine.observation(args.seat)
        left_action = normalize_legacy_action(
            run_agent(left, left_accepts_config, observation, configuration)
        )
        right_action = normalize_legacy_action(
            run_agent(right, right_accepts_config, observation, configuration)
        )
        if left_action == right_action:
            exact_steps += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "frame": frame_index,
                    "step": int(observation.get("step", 0) or 0),
                    "left": left_action,
                    "right": right_action,
                }
            )
        engine.step([frame_action(frame, 0), frame_action(frame, 1)])

    report = {
        "left": str(args.left),
        "right": str(args.right),
        "replay": str(args.replay),
        "seat": args.seat,
        "steps": len(replay["steps"]) - 1,
        "exact_steps": exact_steps,
        "mismatch_steps": len(replay["steps"]) - 1 - exact_steps,
        "first_mismatches": mismatches,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
