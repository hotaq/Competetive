from kaggriculture_rs import Engine

from kaggriculture_rl.policy.production_profile import ADAPTIVE_PROFILES
from kaggriculture_rl.policy.reactive_profile import (
    COW4_BASELINE_PROFILE,
    COW5_ROBUST_PROFILE,
    COW6_ROBUST_PROFILE,
    COW10_LAND2_PROFILE,
    ReactiveProfilePolicy,
)


def _observation_with_hands(
    count: int,
    *,
    step: int = 73,
    money_gap: float = 0.0,
) -> dict:
    observation = Engine({"seed": 7}).observation(0)
    observation["step"] = step
    observation["farms"][1]["hands"] = [[0, index] for index in range(count)]
    observation["farms"][1]["money"] -= money_gap
    return observation


def test_reactive_profile_waits_then_uses_public_opponent_investment() -> None:
    policy = ReactiveProfilePolicy()

    assert (
        policy.selected_profile(_observation_with_hands(0, step=72))
        == COW4_BASELINE_PROFILE
    )
    assert policy.selected_profile(_observation_with_hands(0)) == COW10_LAND2_PROFILE
    policy.reset()
    assert policy.selected_profile(_observation_with_hands(2)) == COW6_ROBUST_PROFILE
    policy.reset()
    assert policy.selected_profile(_observation_with_hands(4)) == COW4_BASELINE_PROFILE
    policy.reset()
    assert (
        policy.selected_profile(_observation_with_hands(4, money_gap=200.0))
        == COW5_ROBUST_PROFILE
    )
    policy.reset()
    assert policy.selected_profile(_observation_with_hands(3)) == COW4_BASELINE_PROFILE


def test_reactive_profile_commitment_is_sticky() -> None:
    policy = ReactiveProfilePolicy()

    assert policy.selected_profile(_observation_with_hands(2)) == COW6_ROBUST_PROFILE
    assert policy.selected_profile(_observation_with_hands(0)) == COW6_ROBUST_PROFILE
    assert len(ADAPTIVE_PROFILES) == 6
