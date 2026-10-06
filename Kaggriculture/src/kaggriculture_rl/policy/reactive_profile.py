"""Observable wait-then-respond policy for the live production-profile league."""

from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.production_profile import (
    INCUMBENT_PROFILE,
    Crop,
    ProductionProfile,
    StickyProfileController,
    profile_targets,
)

COW8_LAND1_PROFILE = ProductionProfile(
    "melon8_cow8_w8_b16_l1",
    Crop.MELON,
    8,
    cows=8,
    workers=8,
    wheat_buffer=16,
    land_quadrants=1,
)
COW10_LAND2_PROFILE = ProductionProfile(
    "melon10_cow8_w8_b16_l2",
    Crop.MELON,
    10,
    cows=8,
    workers=8,
    wheat_buffer=16,
    land_quadrants=2,
)
COW6_ROBUST_PROFILE = ProductionProfile(
    "melon6_cow6_w6_b18",
    Crop.MELON,
    6,
    cows=6,
    workers=6,
    wheat_buffer=18,
)
COW5_ROBUST_PROFILE = ProductionProfile(
    "melon6_cow5_w7_b10",
    Crop.MELON,
    6,
    cows=5,
    workers=7,
    wheat_buffer=10,
)
COW4_BASELINE_PROFILE = ProductionProfile(
    "melon6_cow4_w4",
    Crop.MELON,
    6,
    cows=4,
    workers=4,
    wheat_buffer=8,
)
REACTIVE_PROFILES = (
    INCUMBENT_PROFILE,
    COW4_BASELINE_PROFILE,
    COW10_LAND2_PROFILE,
    COW8_LAND1_PROFILE,
    COW6_ROBUST_PROFILE,
    COW5_ROBUST_PROFILE,
)


class ReactiveProfilePolicy:
    """Start from safe cow4, then escalate from public opponent investment.

    The six-profile live league is indistinguishable through step 72. At step
    73, public hand counts and the public money gap separate the live plans
    without using hidden state or an opponent label.
    """

    def __init__(self, observe_through_step: int = 72) -> None:
        self.observe_through_step = observe_through_step
        self._controller = StickyProfileController()
        self._committed: ProductionProfile | None = None

    def reset(self) -> None:
        self._controller.reset()
        self._committed = None

    @staticmethod
    def _opponent_hands(observation: dict[str, Any]) -> int:
        player = int(observation.get("player", 0))
        farms = observation.get("farms", ())
        opponent = 1 - player
        if not isinstance(farms, list | tuple) or opponent >= len(farms):
            return 0
        farm = farms[opponent]
        if not isinstance(farm, dict):
            return 0
        hands = farm.get("hands", ())
        return len(hands) if isinstance(hands, list | tuple) else 0

    @staticmethod
    def _money_gap(observation: dict[str, Any]) -> float:
        player = int(observation.get("player", 0))
        farms = observation.get("farms", ())
        if not isinstance(farms, list | tuple) or len(farms) < 2:
            return 0.0
        own = farms[player]
        opponent = farms[1 - player]
        if not isinstance(own, dict) or not isinstance(opponent, dict):
            return 0.0
        return float(own.get("money", 0.0)) - float(
            opponent.get("money", 0.0)
        )

    def selected_profile(self, observation: dict[str, Any]) -> ProductionProfile:
        step = int(observation.get("step", 0))
        if step < self.observe_through_step:
            return INCUMBENT_PROFILE
        if step == self.observe_through_step:
            return COW4_BASELINE_PROFILE
        if self._committed is None:
            hands = self._opponent_hands(observation)
            if hands == 0:
                self._committed = COW10_LAND2_PROFILE
            elif hands >= 4 and self._money_gap(observation) < 100.0:
                self._committed = COW4_BASELINE_PROFILE
            elif hands >= 4:
                self._committed = COW5_ROBUST_PROFILE
            elif hands == 3:
                self._committed = COW4_BASELINE_PROFILE
            else:
                self._committed = COW6_ROBUST_PROFILE
        return self._committed

    def __call__(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        profile = self.selected_profile(observation)
        return profile_targets(self._controller.select(profile), observation)
