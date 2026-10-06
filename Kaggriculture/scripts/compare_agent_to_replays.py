"""Compare a recovered agent's actions with its recorded public replays."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def first_component(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    for component in ("farmer", "hands", "market"):
        if expected.get(component) != actual.get(component):
            return component
    return None


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    total_steps = 0
    exact_steps = 0
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["episodes"]:
            episode_id = int(record["episode_id"])
            replay = json.loads(
                (manifest_path.parent / f"{episode_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            seat = int(record.get("agent_seat", record["top1_seat"]))
            _, agent, accepts_config = load_agent(args.agent_file)
            engine = Engine(engine_config(replay))
            episode_exact = 0
            first_mismatch: dict[str, Any] | None = None
            for step, frame in enumerate(replay["steps"][1:]):
                observation = engine.observation(seat)
                proposed = run_agent(
                    agent,
                    accepts_config,
                    observation,
                    engine_config(replay),
                )
                actual = engine.sanitize_action(
                    seat, normalize_legacy_action(proposed)
                )
                expected = engine.sanitize_action(
                    seat, normalize_legacy_action(frame_action(frame, seat))
                )
                if actual == expected:
                    episode_exact += 1
                elif first_mismatch is None:
                    first_mismatch = {
                        "step": step,
                        "component": first_component(expected, actual),
                        "expected": expected,
                        "actual": actual,
                    }
                engine.step([frame_action(frame, 0), frame_action(frame, 1)])
            steps = len(replay["steps"]) - 1
            total_steps += steps
            exact_steps += episode_exact
            rows.append(
                {
                    "episode_id": episode_id,
                    "steps": steps,
                    "exact_steps": episode_exact,
                    "exact_rate": episode_exact / steps,
                    "first_mismatch": first_mismatch,
                }
            )

    result = {
        "agent_file": str(args.agent_file),
        "episodes": len(rows),
        "total_steps": total_steps,
        "exact_steps": exact_steps,
        "exact_rate": exact_steps / total_steps,
        "exact_episodes": sum(row["first_mismatch"] is None for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
