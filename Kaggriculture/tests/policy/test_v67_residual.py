from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kaggriculture_rl.policy.v67_residual import (
    V67_SHEEP_SWAP_HAND4_PLAN,
    V67ResidualDecision,
    V67ResidualExecutor,
    V67ResidualPhase,
)

ROOT = Path(__file__).resolve().parents[2]
HAND4_PLAN = dict(V67_SHEEP_SWAP_HAND4_PLAN)
POSITIONS = {
    248: [4, 4],
    249: [4, 4],
    250: [4, 4],
    251: [4, 4],
    252: [3, 4],
    253: [3, 3],
    254: [3, 4],
    255: [3, 5],
    256: [3, 5],
    257: [3, 5],
    258: [3, 5],
    259: [3, 5],
    260: [4, 5],
    261: [4, 5],
}


def _inventory_for(step: int) -> dict[str, int]:
    if step == 248:
        return {"MELON": 6}
    if step == 249:
        return {}
    if step == 250:
        return {"SHEEP": 2}
    if 251 <= step <= 256:
        return {"SHEEP": 2, "WHEAT": 2}
    if step == 257:
        return {"SHEEP": 1, "WHEAT": 2}
    if 258 <= step <= 261:
        return {"SHEEP": 1, "WHEAT": 1}
    return {}


def _observation(step: int, player: int = 0) -> dict[str, Any]:
    hand_positions = [[0, 0] for _ in range(12)]
    hand_positions[4] = copy.deepcopy(POSITIONS.get(step, [4, 4]))
    farms = [
        {"money": 2_000.0, "farmer": [3, 3], "hands": copy.deepcopy(hand_positions)},
        {"money": 2_000.0, "farmer": [3, 3], "hands": copy.deepcopy(hand_positions)},
    ]
    inventories: list[dict[str, int]] = [{} for _ in range(13)]
    inventories[5] = _inventory_for(step)
    return {
        "step": step,
        "player": player,
        "day": step // 24,
        "hour": step % 24,
        "farms": farms,
        "private": {
            "inventories": inventories,
            "shed": {
                "COW": 0,
                "SHEEP": 2 if step == 249 else 0,
                "WHEAT": 7,
            },
        },
    }


def _action_for(step: int) -> dict[str, Any]:
    action: dict[str, Any] = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(12)],
        "market": [],
        "diagnostic": {"preserve": [step]},
    }
    if step == 248:
        action["hands"][4] = ["DROP"]
        action["market"] = [
            ["SELL", "MELON", 6],
            ["BUY_ANIMAL", "COW", 2],
        ]
    elif step in HAND4_PLAN:
        action["hands"][4] = list(HAND4_PLAN[step])
        if step in (249, 256, 261):
            action["hands"][4] = ["PASS"]
    return action


class RecordingAgent:
    def __init__(self, actions: dict[int, dict[str, Any]]) -> None:
        self.actions = actions
        self.calls: list[tuple[dict[str, Any], Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        self.calls.append((copy.deepcopy(observation), copy.deepcopy(configuration)))
        return self.actions[int(observation["step"])]


def _executor(
    actions: dict[int, dict[str, Any]],
    *,
    decision: V67ResidualDecision | str = V67ResidualDecision.SHEEP_SWAP_2,
    eligibility: Callable[[dict[str, Any]], bool] = lambda _: True,
) -> tuple[V67ResidualExecutor, RecordingAgent]:
    agent = RecordingAgent(actions)
    executor = V67ResidualExecutor(
        agent,
        lambda _: decision,
        eligibility=eligibility,
    )
    return executor, agent


def _diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        paths = []
        for index in range(max(len(left), len(right))):
            child = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[index], right[index], child))
        return paths
    return [] if left == right else [prefix]


def test_abstain_is_structurally_and_json_byte_identical() -> None:
    base = _action_for(248)
    executor, _ = _executor({248: base}, decision=V67ResidualDecision.ABSTAIN)

    actual = executor(_observation(248), {"seed": 7})

    assert actual == base
    assert actual is not base
    assert json.dumps(actual, separators=(",", ":")) == json.dumps(base, separators=(",", ":"))
    assert executor.phase_for(0) is V67ResidualPhase.IDLE
    assert executor.counters_for(0).abstentions == 1
    assert executor.counters_for(0).interventions == 0
    assert executor.packaging_gate_for(0).passed is True


def test_default_eligibility_is_fail_closed_and_does_not_call_model() -> None:
    called = False

    def model(_: dict[str, Any]) -> V67ResidualDecision:
        nonlocal called
        called = True
        return V67ResidualDecision.SHEEP_SWAP_2

    base = _action_for(248)
    executor = V67ResidualExecutor(RecordingAgent({248: base}), model)

    assert executor(_observation(248)) == base
    assert called is False
    assert executor.counters_for(0).interventions == 0


def test_complete_hand4_takeover_owns_exact_suffix_and_preserves_other_components() -> None:
    steps = tuple(range(248, 262))
    actions = {step: _action_for(step) for step in steps}
    frozen = copy.deepcopy(actions)
    executor, _ = _executor(actions)

    actual = {step: executor(_observation(step)) for step in steps}

    assert actual[248]["market"][1] == ["BUY_ANIMAL", "SHEEP", 2]
    for step, command in V67_SHEEP_SWAP_HAND4_PLAN:
        assert actual[step]["hands"][4] == list(command)
        restored = copy.deepcopy(actual[step])
        restored["hands"][4] = frozen[step]["hands"][4]
        assert restored == frozen[step]
    restored_start = copy.deepcopy(actual[248])
    restored_start["market"][1][1] = "COW"
    assert restored_start == frozen[248]
    assert actions == frozen

    counters = executor.counters_for(0)
    assert executor.phase_for(0) is V67ResidualPhase.COMPLETE
    assert counters.interventions == 1
    assert counters.owned_hand4_steps == 13
    assert counters.changed_action_steps == 4
    assert counters.rewritten_tokens == 4
    assert counters.completed == 1
    assert counters.partial_interventions == 0
    assert executor.packaging_gate_for(0).passed is True
    assert [event.event for event in executor.trace_for(0)] == [
        "OPTION_START",
        *(["HAND4_TAKEOVER"] * 12),
        "OPTION_COMPLETE",
    ]
    assert sum(event.changed for event in executor.trace_for(0)) == 4


def test_episode_rollback_resets_only_that_players_pending_takeover() -> None:
    actions = {0: _action_for(0), 248: _action_for(248), 249: _action_for(249)}
    executor, _ = _executor(actions)

    executor(_observation(248, player=0))
    executor(_observation(248, player=1))
    executor(_observation(0, player=0))

    assert executor.phase_for(0) is V67ResidualPhase.IDLE
    assert executor.phase_for(1) is V67ResidualPhase.TAKEOVER_ACTIVE
    assert executor(_observation(249, player=0)) == actions[249]
    assert executor(_observation(249, player=1))["hands"][4] == ["PICKUP", "SHEEP", 2]
    assert executor.counters_for(0).episode_resets == 1
    assert executor.counters_for(1).episode_resets == 0


@pytest.mark.parametrize(
    "bad_order",
    (
        ["BUY_ANIMAL", "COW", 1],
        ["BUY_ANIMAL", "SHEEP", 2],
        ["BUY_ANIMAL", "COW", 2, "extra"],
    ),
)
def test_start_signature_mismatch_abstains_before_purchase(bad_order: list[Any]) -> None:
    action = _action_for(248)
    action["market"][1] = bad_order
    model_calls = 0

    def model(_: dict[str, Any]) -> V67ResidualDecision:
        nonlocal model_calls
        model_calls += 1
        return V67ResidualDecision.SHEEP_SWAP_2

    executor = V67ResidualExecutor(
        RecordingAgent({248: action}),
        model,
        eligibility=lambda _: True,
    )

    assert executor(_observation(248)) == action
    assert model_calls == 0
    assert executor.phase_for(0) is V67ResidualPhase.IDLE
    assert executor.counters_for(0).interventions == 0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (lambda obs, action: action["hands"].__setitem__(4, ["PASS"]), "unexpected_step248_hand4"),
        (
            lambda obs, action: obs["farms"][0].__setitem__("money", 999),
            "insufficient_visible_money",
        ),
        (
            lambda obs, action: obs["private"]["shed"].__setitem__("WHEAT", 1),
            "insufficient_start_wheat",
        ),
        (lambda obs, action: obs["farms"][0]["hands"].__setitem__(4, [3, 4]), "hand4_not_at_shed"),
    ),
)
def test_observable_start_preconditions_fail_closed_before_model(
    mutate: Callable[[dict[str, Any], dict[str, Any]], None], reason: str
) -> None:
    observation = _observation(248)
    action = _action_for(248)
    mutate(observation, action)
    model_calls = 0

    def model(_: dict[str, Any]) -> V67ResidualDecision:
        nonlocal model_calls
        model_calls += 1
        return V67ResidualDecision.SHEEP_SWAP_2

    executor = V67ResidualExecutor(RecordingAgent({248: action}), model, eligibility=lambda _: True)

    assert executor(observation) == action
    assert model_calls == 0
    assert executor.counters_for(0).abstentions == 1
    assert executor.trace_for(0)[-1].reason == f"start_precondition:{reason}"


def test_impossible_state_after_start_is_partial_and_fails_packaging() -> None:
    actions = {248: _action_for(248), 249: _action_for(249)}
    executor, _ = _executor(actions)

    executor(_observation(248))
    assert executor.packaging_gate_for(0).reason == "FAIL_PENDING_INTERVENTION"
    impossible = _observation(249)
    impossible["private"]["shed"]["SHEEP"] = 0

    assert executor(impossible) == actions[249]
    assert executor.phase_for(0) is V67ResidualPhase.PARTIAL_ABORTED
    assert executor.counters_for(0).partial_interventions == 1
    assert executor.counters_for(0).abstentions == 0
    assert executor.trace_for(0)[-1].event == "PARTIAL_ABORT"
    assert executor.trace_for(0)[-1].reason == (
        "takeover_precondition:purchased_sheep_not_observable"
    )
    assert executor.packaging_gate_for(0).reason == "FAIL_PARTIAL_INTERVENTION"


def test_missed_suffix_step_is_partial_not_abstention() -> None:
    actions = {248: _action_for(248), 250: _action_for(250)}
    executor, _ = _executor(actions)

    executor(_observation(248))
    executor(_observation(250))

    assert executor.phase_for(0) is V67ResidualPhase.PARTIAL_ABORTED
    assert executor.counters_for(0).partial_interventions == 1
    assert executor.counters_for(0).abstentions == 0
    assert executor.trace_for(0)[-1].reason == "missed_takeover_step:249"


def test_returned_actions_and_callback_inputs_are_deep_copies() -> None:
    base = _action_for(248)
    frozen_action = copy.deepcopy(base)
    observation = _observation(248)
    frozen_observation = copy.deepcopy(observation)

    def mutating_eligibility(candidate_observation: dict[str, Any]) -> bool:
        candidate_observation["private"]["shed"]["WHEAT"] = 99
        return True

    def mutating_model(candidate_observation: dict[str, Any]) -> V67ResidualDecision:
        candidate_observation["private"]["shed"]["SHEEP"] = 99
        return V67ResidualDecision.SHEEP_SWAP_2

    executor = V67ResidualExecutor(
        RecordingAgent({248: base}),
        mutating_model,
        eligibility=mutating_eligibility,
    )

    actual = executor(observation)
    actual["market"][0][2] = 999

    assert observation == frozen_observation
    assert base == frozen_action


def test_market_capacity_is_preserved_and_over_limit_fails_closed() -> None:
    at_limit = _action_for(248)
    at_limit["market"].extend([["HIRE"] for _ in range(8)])
    over_limit = copy.deepcopy(at_limit)
    over_limit["market"].append(["HIRE"])
    executor, _ = _executor({248: at_limit})

    rewritten = executor(_observation(248))

    assert len(rewritten["market"]) == 10
    assert rewritten["market"][1] == ["BUY_ANIMAL", "SHEEP", 2]
    assert rewritten["market"][:1] + rewritten["market"][2:] == (
        at_limit["market"][:1] + at_limit["market"][2:]
    )

    rejected, _ = _executor({248: over_limit})
    assert rejected(_observation(248)) == over_limit
    assert rejected.counters_for(0).interventions == 0


def test_repeated_option_step_is_idempotent_without_double_counting() -> None:
    actions = {248: _action_for(248), 249: _action_for(249)}
    executor, _ = _executor(actions)

    first_start = executor(_observation(248))
    second_start = executor(_observation(248))
    first_pickup = executor(_observation(249))
    second_pickup = executor(_observation(249))

    assert first_start == second_start
    assert first_pickup == second_pickup
    assert executor.counters_for(0).interventions == 1
    assert executor.counters_for(0).owned_hand4_steps == 1
    assert len(executor.trace_for(0)) == 2


def test_invalid_or_raising_callbacks_fail_closed() -> None:
    action = _action_for(248)

    def raises(_: dict[str, Any]) -> bool:
        raise RuntimeError("model unavailable")

    eligibility_failure = V67ResidualExecutor(
        RecordingAgent({248: action}),
        lambda _: V67ResidualDecision.SHEEP_SWAP_2,
        eligibility=raises,
    )
    assert eligibility_failure(_observation(248)) == action
    assert eligibility_failure.counters_for(0).eligibility_failures == 1

    invalid_decision = V67ResidualExecutor(
        RecordingAgent({248: action}),
        lambda _: "BUY_COWS",
        eligibility=lambda _: True,
    )
    assert invalid_decision(_observation(248)) == action
    assert invalid_decision.counters_for(0).decision_failures == 1


def test_first_catalog_game_v22_raw_terminal_and_action_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("kaggriculture_rs")
    from kaggriculture_rs import Engine

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from evaluate_file_agent_vs_replay import (
        load_agent,
        normalize_legacy_action,
        run_agent,
    )

    v67_path = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
    v22_path = ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone/main.py"
    if not v67_path.exists() or not v22_path.exists():
        pytest.skip("frozen V67/V22 catalog agents are not present")
    configuration = {"episodeSteps": 720, "weedSpawnChance": 0.005, "seed": 27100100}

    def play(decision: V67ResidualDecision) -> dict[str, Any]:
        _, base_agent, base_accepts = load_agent(v67_path)
        _, proxy_agent, proxy_accepts = load_agent(v22_path)
        latest_base: dict[str, Any] = {}

        def captured_base(observation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            action = run_agent(base_agent, base_accepts, observation, config)
            latest_base["action"] = copy.deepcopy(action)
            return action

        executor = V67ResidualExecutor(
            captured_base,
            lambda _: decision,
            eligibility=lambda _: True,
        )
        engine = Engine(configuration)
        sanitizer_changes: list[int] = []
        diff_steps: list[int] = []

        for step in range(719):
            candidate_observation = engine.observation(0)
            proposed = normalize_legacy_action(executor(candidate_observation, configuration))
            base = normalize_legacy_action(latest_base["action"])
            paths = _diff_paths(base, proposed)
            if paths:
                diff_steps.append(step)
            if step == 248 and decision is V67ResidualDecision.SHEEP_SWAP_2:
                assert paths == ["$.market[1][1]"]
            elif 249 <= step <= 261 and decision is V67ResidualDecision.SHEEP_SWAP_2:
                assert all(path.startswith("$.hands[4]") for path in paths)
            else:
                assert paths == []

            sanitized = engine.sanitize_action(0, proposed)
            if sanitized != proposed:
                sanitizer_changes.append(step)
            proxy_observation = engine.observation(1)
            proxy_proposed = normalize_legacy_action(
                run_agent(proxy_agent, proxy_accepts, proxy_observation, configuration)
            )
            engine.step([sanitized, engine.sanitize_action(1, proxy_proposed)])

        return {
            "step": int(engine.state["step"]),
            "money": [float(farm["money"]) for farm in engine.state["farms"]],
            "sanitizer_changes": sanitizer_changes,
            "diff_steps": diff_steps,
            "executor": executor,
        }

    abstain = play(V67ResidualDecision.ABSTAIN)
    forced = play(V67ResidualDecision.SHEEP_SWAP_2)

    assert abstain["step"] == forced["step"] == 719
    assert abstain["sanitizer_changes"] == forced["sanitizer_changes"] == []
    assert abstain["diff_steps"] == []
    assert forced["diff_steps"] == [248, 249, 256, 261]
    assert abstain["money"] == [65_318.0, 68_903.0]
    assert forced["money"] == [62_706.0, 69_652.0]

    forced_executor = forced["executor"]
    assert forced_executor.phase_for(0) is V67ResidualPhase.COMPLETE
    assert forced_executor.counters_for(0).owned_hand4_steps == 13
    assert forced_executor.counters_for(0).changed_action_steps == 4
    assert forced_executor.counters_for(0).partial_interventions == 0
    assert forced_executor.packaging_gate_for(0).passed is True
