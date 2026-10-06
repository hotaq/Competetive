from __future__ import annotations

import pytest

from scripts.build_static_attack import MAX_CANDIDATES, render_attack


def test_rendered_static_attack_is_self_contained() -> None:
    source = render_attack(3)
    namespace: dict[str, object] = {}

    exec(compile(source, "generated-attack.py", "exec"), namespace)
    attack = namespace["AttackAlgorithm"]()
    candidates = attack.run(None, None)

    assert len(candidates) == 3
    assert "status-0000.invalid" in candidates[0].user_messages[0]
    assert "status-0002.invalid" in candidates[2].user_messages[0]
    assert all("SECRET_MARKER" in candidate.user_messages[0] for candidate in candidates)


def test_rendered_static_attack_enforces_candidate_cap() -> None:
    with pytest.raises(ValueError, match="between 1"):
        render_attack(0)
    with pytest.raises(ValueError, match="between 1"):
        render_attack(MAX_CANDIDATES + 1)
