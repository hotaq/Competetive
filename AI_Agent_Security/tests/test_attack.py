from __future__ import annotations

import attack
from aicomp_sdk.attacks import AttackCandidate, AttackRunConfig


def test_attack_delegates_to_content_guided_search(monkeypatch) -> None:
    expected = [AttackCandidate.from_messages(["open reference"])]
    monkeypatch.setattr(
        attack.ContentGuidedSearch,
        "run",
        lambda self, env, config: expected,
    )

    candidates = attack.AttackAlgorithm().run(
        object(),
        AttackRunConfig(time_budget_s=10, max_steps=8),
    )

    assert candidates == expected

