"""Trace observable economic state for one file agent against one replay tape."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from kaggriculture_rs import Engine

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action  # noqa: E402

from kaggriculture_rl.policy.primitive_legalizer import (  # noqa: E402
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import (  # noqa: E402
    OrphanStructureRecoveryController,
)

DEFAULT_STEPS = (
    0,
    72,
    96,
    120,
    144,
    168,
    192,
    240,
    288,
    336,
    384,
    432,
    480,
    528,
    576,
    624,
    672,
    696,
    708,
    718,
    719,
)


def _parse_steps(raw: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(value) for value in raw.split(",") if value.strip()}))
    if not steps or steps[0] < 0 or steps[-1] > 719:
        raise ValueError("snapshot steps must be within 0..719")
    return steps


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _snapshot(
    observation: dict[str, Any],
    *,
    candidate_seat: int,
    proposed_action: dict[str, Any] | None,
    executed_action: dict[str, Any] | None,
) -> dict[str, Any]:
    own = observation["farms"][candidate_seat]
    opponent = observation["farms"][1 - candidate_seat]
    private = observation.get("private", {})

    def composition(farm: dict[str, Any]) -> dict[str, dict[str, int]]:
        kinds: Counter[str] = Counter()
        crops: Counter[str] = Counter()
        animals: Counter[str] = Counter()
        for row in farm.get("tiles", []):
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if tile.get("kind"):
                    kinds[str(tile["kind"])] += 1
                if tile.get("crop"):
                    crops[str(tile["crop"])] += 1
                if tile.get("animal"):
                    animals[str(tile["animal"])] += 1
        return {
            "tile_kinds": dict(kinds),
            "crops": dict(crops),
            "animals": dict(animals),
        }

    return {
        "step": int(observation.get("step", 0)),
        "day": int(observation.get("day", 0)),
        "hour": int(observation.get("hour", 0)),
        "candidate_money": float(own.get("money", 0.0)),
        "opponent_money": float(opponent.get("money", 0.0)),
        "margin": float(own.get("money", 0.0)) - float(opponent.get("money", 0.0)),
        "candidate_hands": len(own.get("hands", ())),
        "opponent_hands": len(opponent.get("hands", ())),
        "candidate_land": len(own.get("unlocked_quadrants", ())),
        "opponent_land": len(opponent.get("unlocked_quadrants", ())),
        "candidate_farmer": _plain(own.get("farmer")),
        "opponent_farmer": _plain(opponent.get("farmer")),
        "candidate_composition": composition(own),
        "opponent_composition": composition(opponent),
        "shed": _plain(private.get("shed", {})),
        "seeds": _plain(private.get("seeds", {})),
        "inventories": _plain(private.get("inventories", [])),
        "market": _plain(observation.get("market", {})),
        "proposed_action": _plain(proposed_action),
        "executed_action": _plain(executed_action),
    }


def trace_episode(
    agent_path: Path,
    replay_path: Path,
    *,
    top1_seat: int,
    snapshot_steps: tuple[int, ...] = DEFAULT_STEPS,
    candidate_repair: str = "none",
) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    _, agent, accepts_config = load_agent(agent_path)
    configuration = engine_config(replay)
    engine = Engine(configuration)
    weed_recovery = WeedRecoveryController()
    orphan_recovery = OrphanStructureRecoveryController()
    candidate_seat = 1 - top1_seat
    wanted = set(snapshot_steps)
    snapshots: list[dict[str, Any]] = []
    market_events: list[dict[str, Any]] = []
    candidate_unit_operations: Counter[str] = Counter()
    opponent_unit_operations: Counter[str] = Counter()

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation.get("step", 0))
        proposed = run_agent(agent, accepts_config, observation, configuration)
        if candidate_repair != "none":
            proposed = weed_recovery.repair(observation, proposed)
            if candidate_repair == "orphan":
                proposed = orphan_recovery.apply(observation, proposed)
            proposed = append_terminal_liquidation(observation, proposed)
            proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        executed = engine.sanitize_action(candidate_seat, normalized)
        if step in wanted:
            snapshots.append(
                _snapshot(
                    observation,
                    candidate_seat=candidate_seat,
                    proposed_action=proposed,
                    executed_action=executed,
                )
            )
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = executed
        opponent_action = actions[top1_seat]
        for side_action, counter in (
            (executed, candidate_unit_operations),
            (opponent_action, opponent_unit_operations),
        ):
            commands = [
                side_action.get("farmer", ["PASS"]),
                *side_action.get("hands", []),
            ]
            counter.update(
                str(command[0])
                for command in commands
                if isinstance(command, list) and command
            )
        if executed.get("market") or opponent_action.get("market"):
            market_events.append(
                {
                    "step": step,
                    "day": int(observation.get("day", step // 24)),
                    "prices": _plain(observation.get("market", {}).get("prices", {})),
                    "candidate_money": float(
                        observation["farms"][candidate_seat].get("money", 0.0)
                    ),
                    "opponent_money": float(
                        observation["farms"][top1_seat].get("money", 0.0)
                    ),
                    "candidate_orders": _plain(executed.get("market", [])),
                    "opponent_orders": _plain(opponent_action.get("market", [])),
                }
            )
        engine.step(actions)

    final_observation = engine.observation(candidate_seat)
    if 719 in wanted:
        snapshots.append(
            _snapshot(
                final_observation,
                candidate_seat=candidate_seat,
                proposed_action=None,
                executed_action=None,
            )
        )
    return {
        "agent_file": str(agent_path),
        "replay_file": str(replay_path),
        "seed": int(replay["info"]["seed"]),
        "top1_seat": top1_seat,
        "candidate_seat": candidate_seat,
        "candidate_repair": candidate_repair,
        "snapshots": snapshots,
        "market_events": market_events,
        "candidate_unit_operations": dict(candidate_unit_operations),
        "opponent_unit_operations": dict(opponent_unit_operations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument("replay_file", type=Path)
    parser.add_argument("--top1-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--candidate-repair",
        choices=("none", "weed", "orphan"),
        default="none",
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = trace_episode(
        args.agent_file,
        args.replay_file,
        top1_seat=args.top1_seat,
        snapshot_steps=_parse_steps(args.steps),
        candidate_repair=args.candidate_repair,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
