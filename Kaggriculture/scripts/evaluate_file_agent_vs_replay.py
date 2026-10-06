"""Evaluate a raw Kaggle agent file against exact-validated replay action tapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from types import ModuleType
from typing import Any

from evaluate_replay_tape import (
    engine_config,
    frame_action,
    validate_original_replay,
)
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

Agent = Callable[..., dict[str, Any]]
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class FileAgentTapeResult:
    episode_id: int
    seed: int
    top1_seat: int
    steps: int
    replay_frames: int
    completed_transitions: int
    final_state_step: int
    original_top1_money: float
    original_opponent_money: float
    candidate_money: float
    tape_top1_money: float
    candidate_margin: float
    fallback_components: int
    normalized_steps: int
    sanitized_steps: int
    farmer_sanitized_steps: int
    hands_sanitized_steps: int
    market_sanitized_steps: int
    opening_normalized_steps: int
    post_handoff_normalized_steps: int
    opening_sanitized_steps: int
    post_handoff_sanitized_steps: int
    route_decisions: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_file", type=Path)
    parser.add_argument(
        "--fallback-agent-file",
        type=Path,
        help="replace invalid primary components with this agent's legal components",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("replays/top1_current/manifest.json"),
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        default=[],
        help="evaluate only selected episode id; repeatable",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--phase-boundary",
        type=int,
        default=120,
        help="first observation step attributed to the post-handoff policy",
    )
    parser.add_argument(
        "--candidate-repair",
        choices=("none", "weed", "orphan"),
        default="none",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_agent(path: Path) -> tuple[ModuleType, Agent, bool]:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        f"kaggriculture_file_agent_{hashlib.sha256(str(resolved).encode()).hexdigest()[:12]}",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    inserted_root = str(ROOT) not in sys.path
    if inserted_root:
        sys.path.insert(0, str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_root:
            sys.path.remove(str(ROOT))
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError(f"{resolved} does not export a callable agent")
    accepts_config = len(inspect.signature(agent).parameters) >= 2
    return module, agent, accepts_config


def run_agent(
    agent: Agent,
    accepts_config: bool,
    observation: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    action = agent(observation, configuration) if accepts_config else agent(observation)
    if not isinstance(action, dict):
        raise TypeError(f"agent returned {type(action).__name__}, expected dict")
    return action


def policy_trace(agent: Agent, player: int, step: int) -> dict[str, Any] | None:
    namespace = getattr(agent, "__globals__", {})
    policy = namespace.get("_V18_POLICY") if isinstance(namespace, dict) else None
    if policy is None or not hasattr(policy, "decision_for"):
        return None
    decision = policy.decision_for(player)
    if decision is None:
        return None
    route = getattr(decision.route, "value", str(decision.route))
    status = policy.plan_status_for(player) if hasattr(policy, "plan_status_for") else None
    return {
        "step": step,
        "gate": int(decision.gate),
        "route": str(route),
        "reason": str(decision.reason),
        "planner_status": status,
    }


def normalize_legacy_action(action: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize commands whose Kaggle runtime defaults quantity to one."""

    def normalize_unit(command: Any) -> Any:
        if isinstance(command, list) and len(command) == 2 and command[0] in {"PICKUP", "PLACE"}:
            return [*command, 1]
        return command

    return {
        "farmer": normalize_unit(action.get("farmer", ["PASS"])),
        "hands": [normalize_unit(command) for command in action.get("hands", [])],
        "market": list(action.get("market", [])),
    }


def replace_invalid_components(
    proposed: dict[str, Any],
    sanitized: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Use fallback only where strict sanitization changed the primary action."""
    replaced = 0
    if sanitized["farmer"] != proposed["farmer"]:
        farmer = fallback["farmer"]
        replaced += 1
    else:
        farmer = proposed["farmer"]

    hands: list[Any] = []
    for index, legal in enumerate(sanitized["hands"]):
        primary = proposed["hands"][index] if index < len(proposed["hands"]) else None
        fallback_hand = fallback["hands"][index] if index < len(fallback["hands"]) else legal
        if primary is None or legal != primary:
            hands.append(fallback_hand)
            replaced += 1
        else:
            hands.append(primary)

    if sanitized["market"] != proposed["market"]:
        market = fallback["market"]
        replaced += 1
    else:
        market = proposed["market"]
    return {"farmer": farmer, "hands": hands, "market": market}, replaced


def play_against_tape(
    replay: dict[str, Any],
    *,
    episode_id: int,
    top1_seat: int,
    agent: Agent,
    accepts_config: bool,
    fallback_agent: Agent | None = None,
    fallback_accepts_config: bool = False,
    phase_boundary: int = 120,
    candidate_repair: str = "none",
) -> FileAgentTapeResult:
    configuration = engine_config(replay)
    engine = Engine(configuration)
    weed_recovery = WeedRecoveryController()
    orphan_recovery = OrphanStructureRecoveryController()
    candidate_seat = 1 - top1_seat
    fallback_components = 0
    normalized_steps = 0
    sanitized_steps = 0
    farmer_sanitized_steps = 0
    hands_sanitized_steps = 0
    market_sanitized_steps = 0
    opening_normalized_steps = 0
    post_handoff_normalized_steps = 0
    opening_sanitized_steps = 0
    post_handoff_sanitized_steps = 0
    route_decisions: list[dict[str, Any]] = []
    recorded_route_gates: set[int] = set()
    steps = 0

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        proposed = run_agent(agent, accepts_config, observation, configuration)
        if candidate_repair != "none":
            proposed = weed_recovery.repair(observation, proposed)
            if candidate_repair == "orphan":
                proposed = orphan_recovery.apply(observation, proposed)
            proposed = append_terminal_liquidation(observation, proposed)
            proposed = legalize_action(observation, proposed, configuration)
        trace = policy_trace(agent, candidate_seat, int(observation.get("step", 0)))
        if trace is not None and trace["gate"] not in recorded_route_gates:
            route_decisions.append(trace)
            recorded_route_gates.add(trace["gate"])
        normalized = normalize_legacy_action(proposed)
        was_normalized = int(normalized != proposed)
        normalized_steps += was_normalized
        candidate_action = engine.sanitize_action(candidate_seat, normalized)
        was_sanitized = int(candidate_action != normalized)
        sanitized_steps += was_sanitized
        farmer_sanitized_steps += int(candidate_action["farmer"] != normalized["farmer"])
        hands_sanitized_steps += int(candidate_action["hands"] != normalized["hands"])
        market_sanitized_steps += int(candidate_action["market"] != normalized["market"])
        if int(observation.get("step", 0)) < phase_boundary:
            opening_normalized_steps += was_normalized
            opening_sanitized_steps += was_sanitized
        else:
            post_handoff_normalized_steps += was_normalized
            post_handoff_sanitized_steps += was_sanitized
        if fallback_agent is not None and candidate_action != normalized:
            fallback_proposed = run_agent(
                fallback_agent,
                fallback_accepts_config,
                observation,
                engine_config(replay),
            )
            fallback_normalized = normalize_legacy_action(fallback_proposed)
            fallback_action = engine.sanitize_action(candidate_seat, fallback_normalized)
            merged, replaced = replace_invalid_components(
                normalized,
                candidate_action,
                fallback_action,
            )
            candidate_action = engine.sanitize_action(candidate_seat, merged)
            fallback_components += replaced

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)
        steps += 1

    final_money = [float(farm["money"]) for farm in engine.state["farms"]]
    original_rewards = [float(value) for value in replay["rewards"]]
    candidate_money = final_money[candidate_seat]
    tape_top1_money = final_money[top1_seat]
    return FileAgentTapeResult(
        episode_id=episode_id,
        seed=int(replay["info"]["seed"]),
        top1_seat=top1_seat,
        steps=steps,
        replay_frames=len(replay["steps"]),
        completed_transitions=steps,
        final_state_step=int(engine.state["step"]),
        original_top1_money=original_rewards[top1_seat],
        original_opponent_money=original_rewards[candidate_seat],
        candidate_money=candidate_money,
        tape_top1_money=tape_top1_money,
        candidate_margin=candidate_money - tape_top1_money,
        fallback_components=fallback_components,
        normalized_steps=normalized_steps,
        sanitized_steps=sanitized_steps,
        farmer_sanitized_steps=farmer_sanitized_steps,
        hands_sanitized_steps=hands_sanitized_steps,
        market_sanitized_steps=market_sanitized_steps,
        opening_normalized_steps=opening_normalized_steps,
        post_handoff_normalized_steps=post_handoff_normalized_steps,
        opening_sanitized_steps=opening_sanitized_steps,
        post_handoff_sanitized_steps=post_handoff_sanitized_steps,
        route_decisions=tuple(route_decisions),
    )


def summarize(results: list[FileAgentTapeResult]) -> dict[str, Any]:
    margins = [result.candidate_margin for result in results]
    money = [result.candidate_money for result in results]
    ordered_margins = sorted(margins)
    tail_count = max(1, (len(ordered_margins) + 9) // 10)
    route_counts_by_gate: dict[str, dict[str, int]] = {}
    for result in results:
        for decision in result.route_decisions:
            gate = str(decision["gate"])
            route = str(decision["route"])
            counts = route_counts_by_gate.setdefault(gate, {})
            counts[route] = counts.get(route, 0) + 1
    return {
        "games": len(results),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "win_rate": sum(margin > 0 for margin in margins) / len(margins),
        "mean_margin": fmean(margins),
        "median_margin": median(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered_margins[:tail_count]),
        "mean_money": fmean(money),
        "worst_money": min(money),
        "fallback_components": sum(result.fallback_components for result in results),
        "normalized_steps": sum(result.normalized_steps for result in results),
        "sanitized_steps": sum(result.sanitized_steps for result in results),
        "farmer_sanitized_steps": sum(result.farmer_sanitized_steps for result in results),
        "hands_sanitized_steps": sum(result.hands_sanitized_steps for result in results),
        "market_sanitized_steps": sum(result.market_sanitized_steps for result in results),
        "opening_normalized_steps": sum(result.opening_normalized_steps for result in results),
        "post_handoff_normalized_steps": sum(
            result.post_handoff_normalized_steps for result in results
        ),
        "opening_sanitized_steps": sum(result.opening_sanitized_steps for result in results),
        "post_handoff_sanitized_steps": sum(
            result.post_handoff_sanitized_steps for result in results
        ),
        "route_counts_by_gate": route_counts_by_gate,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["episodes"]
    if args.episode_id:
        selected = set(args.episode_id)
        rows = [row for row in rows if int(row["episode_id"]) in selected]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("manifest contains no selected episodes")

    _, agent, accepts_config = load_agent(args.agent_file)
    fallback_agent: Agent | None = None
    fallback_accepts_config = False
    if args.fallback_agent_file is not None:
        _, fallback_agent, fallback_accepts_config = load_agent(args.fallback_agent_file)
    results: list[FileAgentTapeResult] = []
    for row in rows:
        replay_path = args.manifest.parent / f"{row['episode_id']}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        transitions, rewards = validate_original_replay(replay)
        if transitions != len(replay["steps"]) - 1:
            raise RuntimeError(f"replay transition validation failed: {row['episode_id']}")
        if rewards != tuple(float(value) for value in replay["rewards"]):
            raise RuntimeError(f"replay reward validation failed: {row['episode_id']}")
        result = play_against_tape(
            replay,
            episode_id=int(row["episode_id"]),
            top1_seat=int(
                row["top1_seat"] if "top1_seat" in row else row["agent_seat"]
            ),
            agent=agent,
            accepts_config=accepts_config,
            fallback_agent=fallback_agent,
            fallback_accepts_config=fallback_accepts_config,
            phase_boundary=args.phase_boundary,
            candidate_repair=args.candidate_repair,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    payload = {
        "schema_version": 4,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "competition": manifest["competition"],
            "leader": manifest.get("leader", {"team_name": manifest.get("team_name", "unknown")}),
            "submission": manifest["submission"],
            "manifest_created_at": manifest["created_at"],
            "manifest_file": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        },
        "candidate": {
            "agent_file": str(args.agent_file),
            "sha256": hashlib.sha256(args.agent_file.read_bytes()).hexdigest(),
            "fallback_agent_file": (
                str(args.fallback_agent_file) if args.fallback_agent_file is not None else None
            ),
            "candidate_repair": args.candidate_repair,
        },
        "validation": {
            "all_original_replays_exact": True,
            "candidate_actions_sanitized": True,
            "action_alignment": "steps[1:] actions produce observations in the same frame",
            "phase_boundary": args.phase_boundary,
        },
        "summary": summarize(results),
        "episodes": [asdict(result) for result in results],
        "limitation": (
            "Recorded Top-1 actions are an action tape, not a live policy; changing "
            "the opponent can make later tape actions stale or illegal."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
