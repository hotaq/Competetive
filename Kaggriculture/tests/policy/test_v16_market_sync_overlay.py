from __future__ import annotations

from pathlib import Path

from kaggriculture_rl.policy.v16_market_sync_overlay import _sync_market, build_agent


def test_sync_market_respects_pickup_existing_sell_and_cap() -> None:
    action = {
        "farmer": ["PICKUP", "MILK", 1],
        "hands": [["PASS"]],
        "market": [["SELL", "STRAWBERRY", 1]],
    }
    observation = {
        "private": {"shed": {"STRAWBERRY": 5, "MILK": 5}},
        "market": {"prices": {"STRAWBERRY": 200, "MILK": 250}},
    }

    result = _sync_market(
        action,
        observation,
        ("STRAWBERRY", "MILK"),
        minimum_price=160,
        maximum_quantity=3,
    )

    assert result["market"] == [
        ["SELL", "MILK", 3],
        ["SELL", "STRAWBERRY", 3],
        ["SELL", "STRAWBERRY", 1],
    ]
    assert action["market"] == [["SELL", "STRAWBERRY", 1]]


def test_build_agent_uses_gate_for_matching_step(tmp_path: Path) -> None:
    base_path = tmp_path / "base.py"
    base_path.write_text(
        "def agent(observation):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
    )
    policy = build_agent(
        base_path=base_path,
        module_name="test_v16_market_sync_gate",
        target_steps={7: ("MILK",)},
        minimum_price=160,
        maximum_quantity=3,
        state_gates={7: lambda observation: bool(observation.get("enabled"))},
    )
    observation = {
        "step": 7,
        "enabled": False,
        "private": {"shed": {"MILK": 4}},
        "market": {"prices": {"MILK": 200}},
    }

    assert policy(observation, None)["market"] == []
    observation["enabled"] = True
    assert policy(observation, None)["market"] == [["SELL", "MILK", 3]]
