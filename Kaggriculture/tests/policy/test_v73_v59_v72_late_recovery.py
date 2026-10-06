from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = str(PROJECT_ROOT)
root_was_importable = project_root in sys.path
if not root_was_importable:
    sys.path.insert(0, project_root)
try:
    overlay = importlib.import_module(
        "artifacts.v73_v59_v72_late_recovery_20260826.overlay"
    )
finally:
    if not root_was_importable:
        sys.path.remove(project_root)


class _LatePolicy:
    def __init__(self) -> None:
        self.reset_calls: list[int] = []

    def reset(self, player: int) -> None:
        self.reset_calls.append(player)

    def __call__(
        self, observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        return {
            "farmer": ["MOVE", "N"],
            "hands": [],
            "market": [["SELL", "WHEAT", 1]],
        }


def _observation(step: int, player: int = 0) -> dict[str, Any]:
    return {"step": step, "player": player}


def test_exact_branch_boundary_uses_v59_before_and_v72_at_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overlay.v59,
        "agent",
        lambda observation, configuration=None: {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        },
    )
    late = _LatePolicy()
    agent = overlay.make_agent(600, late_policy_factory=lambda: late)

    assert agent(_observation(599))["farmer"] == ["PASS"]
    assert agent(_observation(600))["farmer"] == ["MOVE", "N"]


def test_outputs_are_copied_from_both_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    base_action = {"farmer": ["PASS"], "hands": [], "market": []}
    monkeypatch.setattr(
        overlay.v59,
        "agent",
        lambda observation, configuration=None: base_action,
    )
    late = _LatePolicy()
    agent = overlay.make_agent(648, late_policy_factory=lambda: late)

    before = agent(_observation(647))
    after = agent(_observation(648))
    before["farmer"][0] = "MUTATED"
    after["market"][0][2] = 99

    assert base_action["farmer"] == ["PASS"]
    assert late(_observation(648))["market"] == [["SELL", "WHEAT", 1]]


def test_late_policy_reset_is_player_isolated_and_rewind_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        overlay.v59,
        "agent",
        lambda observation, configuration=None: {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        },
    )
    late = _LatePolicy()
    agent = overlay.make_agent(480, late_policy_factory=lambda: late)

    agent(_observation(0, player=0))
    agent(_observation(0, player=1))
    agent(_observation(600, player=0))
    agent(_observation(500, player=0))

    assert late.reset_calls == [0, 1, 0]


@pytest.mark.parametrize("branch_step", [480, 600, 648, 672])
def test_supported_branch_steps_have_no_identity_or_seed_input(
    monkeypatch: pytest.MonkeyPatch, branch_step: int
) -> None:
    seen: list[dict[str, Any]] = []

    def base(observation: dict[str, Any], configuration: Any = None):
        seen.append(copy.deepcopy(observation))
        return {"farmer": ["PASS"], "hands": [], "market": []}

    monkeypatch.setattr(overlay.v59, "agent", base)
    agent = overlay.make_agent(branch_step, late_policy_factory=_LatePolicy)
    observation = _observation(branch_step - 1)
    agent(observation, {"seed": 123})

    assert seen == [observation]

