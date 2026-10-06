"""Validate public replays, then play a crop heuristic against Top-1 action tapes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.learning.live_response_ppo import LiveResponsePPOPolicy
from kaggriculture_rl.learning.profile_ppo import ProfilePPOPolicy
from kaggriculture_rl.policy import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


@dataclass(frozen=True, slots=True)
class TapeResult:
    episode_id: int
    seed: int
    top1_seat: int
    steps: int
    original_top1_money: float
    original_opponent_money: float
    candidate_money: float
    tape_top1_money: float
    candidate_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("replays/top1_current/manifest.json")
    )
    parser.add_argument("--crop", type=Crop, choices=list(Crop), default=Crop.MELON)
    parser.add_argument("--plots", type=int, default=6)
    parser.add_argument("--sell-reserve", type=int, default=0)
    parser.add_argument(
        "--profile-checkpoint",
        type=Path,
        help="evaluate adaptive profile PPO instead of the crop heuristic",
    )
    parser.add_argument("--limit", type=int, default=0, help="zero uses every manifest replay")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--live-response-checkpoint",
        type=Path,
        help="evaluate live-response PPO instead of crop heuristic",
    )
    return parser.parse_args()


def engine_config(replay: dict[str, Any]) -> dict[str, Any]:
    config = dict(replay.get("configuration") or {})
    seed = replay.get("info", {}).get("seed")
    if seed is None:
        raise ValueError("replay is missing info.seed")
    config["seed"] = int(seed)
    return config


def strip_runtime_fields(observation: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(observation)
    cleaned.pop("remainingOverageTime", None)
    return cleaned


def replay_observation(
    observation: dict[str, Any], shared_step: int
) -> dict[str, Any]:
    """Restore the shared step omitted from non-zero Kaggle seats."""

    cleaned = strip_runtime_fields(observation)
    cleaned.setdefault("step", shared_step)
    return cleaned


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in expected:
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: lengths {len(expected)} != {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            difference = first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def frame_action(frame: list[dict[str, Any]], seat: int) -> dict[str, Any]:
    action = frame[seat].get("action")
    if not isinstance(action, dict):
        raise ValueError(f"replay frame has no player-{seat} action")
    return action


def validate_original_replay(replay: dict[str, Any]) -> tuple[int, tuple[float, float]]:
    """Require exact public observations after each recorded transition."""

    engine = Engine(engine_config(replay))
    frames = replay.get("steps")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("replay has no transition frames")

    transitions = 0
    for frame_index, frame in enumerate(frames[1:], 1):
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        engine.step(actions)
        transitions += 1
        shared_step = int(frame[0]["observation"]["step"])
        for seat in range(2):
            expected = replay_observation(frame[seat]["observation"], shared_step)
            actual = strip_runtime_fields(engine.observation(seat))
            difference = first_difference(expected, actual)
            if difference is not None:
                raise RuntimeError(
                    f"replay diverged at frame={frame_index} seat={seat}: {difference}"
                )

    expected_rewards = tuple(float(value) for value in replay.get("rewards", []))
    actual_rewards = tuple(float(farm["money"]) for farm in engine.state["farms"])
    if expected_rewards != actual_rewards:
        raise RuntimeError(f"final rewards diverged: {expected_rewards} != {actual_rewards}")
    return transitions, (actual_rewards[0], actual_rewards[1])


def play_against_tape(
    replay: dict[str, Any],
    *,
    episode_id: int,
    top1_seat: int,
    policy: CropHeuristicPolicy | ProfilePPOPolicy | LiveResponsePPOPolicy,
) -> TapeResult:
    engine = Engine(engine_config(replay))
    candidate_seat = 1 - top1_seat
    if isinstance(policy, (ProfilePPOPolicy, LiveResponsePPOPolicy)):
        policy.reset()
    frames = replay["steps"]
    steps = 0

    for frame in frames[1:]:
        full_observation = engine.observation(candidate_seat)
        if isinstance(policy, CropHeuristicPolicy):
            targets = policy({"shed": full_observation["private"]["shed"]})
        else:
            targets = policy(full_observation)
        planned = engine.plan(candidate_seat, targets)
        actions = planned["joint_action"]["actions"]
        actions[top1_seat] = frame_action(frame, top1_seat)
        result = engine.step_typed({"schema_version": 1, "actions": actions})
        steps += 1
        if result["done"]:
            break

    money = tuple(float(farm["money"]) for farm in engine.state["farms"])
    original = tuple(float(value) for value in replay["rewards"])
    return TapeResult(
        episode_id=episode_id,
        seed=int(replay["info"]["seed"]),
        top1_seat=top1_seat,
        steps=steps,
        original_top1_money=original[top1_seat],
        original_opponent_money=original[candidate_seat],
        candidate_money=money[candidate_seat],
        tape_top1_money=money[top1_seat],
        candidate_margin=money[candidate_seat] - money[top1_seat],
    )


def summarize(results: list[TapeResult]) -> dict[str, Any]:
    margins = [result.candidate_margin for result in results]
    candidate_money = [result.candidate_money for result in results]
    tape_money = [result.tape_top1_money for result in results]
    return {
        "episodes": len(results),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "mean_candidate_money": fmean(candidate_money),
        "median_candidate_money": median(candidate_money),
        "worst_candidate_money": min(candidate_money),
        "mean_tape_top1_money": fmean(tape_money),
        "mean_margin": fmean(margins),
        "median_margin": median(margins),
        "worst_margin": min(margins),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episode_rows = manifest["episodes"]
    if args.limit > 0:
        episode_rows = episode_rows[: args.limit]
    policy: CropHeuristicPolicy | ProfilePPOPolicy | LiveResponsePPOPolicy
    if args.profile_checkpoint is not None and args.live_response_checkpoint is not None:
        raise ValueError("choose only one checkpoint type")
    if args.profile_checkpoint is not None:
        policy = ProfilePPOPolicy(args.profile_checkpoint)
    elif args.live_response_checkpoint is not None:
        policy = LiveResponsePPOPolicy(args.live_response_checkpoint)
    else:
        policy = CropHeuristicPolicy(args.crop, args.plots, args.sell_reserve)
    results = []

    for row in episode_rows:
        replay_path = args.manifest.parent / f"{row['episode_id']}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        transitions, rewards = validate_original_replay(replay)
        if transitions != len(replay["steps"]) - 1 or rewards != tuple(replay["rewards"]):
            raise RuntimeError(f"replay validation failed for {row['episode_id']}")
        result = play_against_tape(
            replay,
            episode_id=int(row["episode_id"]),
            top1_seat=int(row["top1_seat"]),
            policy=policy,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    payload = {
        "source": {
            "competition": manifest["competition"],
            "leader": manifest["leader"],
            "submission": manifest["submission"],
            "manifest_created_at": manifest["created_at"],
        },
        "candidate": (
            {
                "crop": args.crop.value,
                "plots": args.plots,
                "sell_reserve": args.sell_reserve,
            }
            if args.profile_checkpoint is None
            else {
                "profile_checkpoint": str(args.profile_checkpoint),
                "model": policy.metadata["model"],
            }
        ),
        "validation": {
            "all_original_replays_exact": True,
            "action_alignment": "steps[1:] actions produce observations in the same frame",
        },
        "summary": summarize(results),
        "episodes": [asdict(result) for result in results],
        "limitation": (
            "Recorded Top-1 actions are an action tape, not a live policy; changing the opponent "
            "changes state and can make later tape actions stale or illegal."
        ),
    }
    if args.live_response_checkpoint is not None:
        payload["candidate"] = {
            "live_response_checkpoint": str(args.live_response_checkpoint),
            "model": "live-response-contextual-ppo-v2",
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
