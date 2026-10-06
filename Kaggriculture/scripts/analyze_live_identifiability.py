"""Find when live opponent profiles become distinguishable from public state."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.evaluation.live_profile_league import WarmupProfilePolicy
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    INCUMBENT_PROFILE,
    ProductionProfile,
)


def _token_counts(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(value, str):
        counts[value] += 1
    elif isinstance(value, Mapping):
        for child in value.values():
            counts.update(_token_counts(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            counts.update(_token_counts(child))
    return counts


def observation_trace(
    opponent_profile: ProductionProfile,
    *,
    seed: int,
    candidate_seat: int,
    through_step: int,
) -> dict[int, dict[str, Any]]:
    engine = Engine(
        {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    )
    candidate = WarmupProfilePolicy(INCUMBENT_PROFILE)
    opponent = WarmupProfilePolicy(opponent_profile)
    opponent_seat = 1 - candidate_seat
    trace: dict[int, dict[str, Any]] = {}
    while True:
        candidate_observation = engine.observation(candidate_seat)
        step = int(candidate_observation["step"])
        trace[step] = candidate_observation["farms"][opponent_seat]
        if step >= through_step:
            return trace
        opponent_observation = engine.observation(opponent_seat)
        candidate_plan = engine.plan(
            candidate_seat, candidate(candidate_observation)
        )
        opponent_plan = engine.plan(opponent_seat, opponent(opponent_observation))
        actions = candidate_plan["joint_action"]["actions"]
        actions[opponent_seat] = opponent_plan["joint_action"]["actions"][
            opponent_seat
        ]
        engine.step(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=220_000)
    parser.add_argument("--through-step", type=int, default=90)
    args = parser.parse_args()

    traces = {
        profile.name: observation_trace(
            profile,
            seed=args.seed,
            candidate_seat=0,
            through_step=args.through_step,
        )
        for profile in ADAPTIVE_PROFILES
    }
    previous_groups: list[list[str]] | None = None
    transitions = []
    for step in sorted(next(iter(traces.values()))):
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for name, trace in traces.items():
            grouped[json.dumps(trace[step], sort_keys=True)].append(name)
        groups = sorted(sorted(names) for names in grouped.values())
        if groups != previous_groups:
            transitions.append({"step": step, "groups": groups})
            previous_groups = groups

    final_step = max(next(iter(traces.values())))
    signatures = {}
    for name, trace in traces.items():
        farm = trace[final_step]
        counts = _token_counts(farm)
        signatures[name] = {
            "money": farm["money"],
            "hands": len(farm["hands"]),
            "unlocked_quadrants": len(farm["unlocked_quadrants"]),
            "tokens": {
                token: counts[token]
                for token in ("COW", "SHEEP", "GOOSE", "MELON", "WHEAT")
            },
        }
    print(
        json.dumps(
            {
                "seed": args.seed,
                "transitions": transitions,
                "signatures_at_final_step": signatures,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
