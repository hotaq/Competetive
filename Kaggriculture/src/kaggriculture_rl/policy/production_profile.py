"""State-aware production profiles for executor-backed agents.

Profiles are deliberately semantic: the Rust executor owns path finding,
legality, reservations, and low-level actions.  A policy only declares the
farm it wants to maintain.  This keeps the RL action space small enough to
train while allowing it to switch crop and investment plans as the match
changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kaggriculture_rl.schemas.models import SCHEMA_VERSION, Animal, Crop, Product

_SELLABLE_OUTPUTS = (
    Product.MELON,
    Product.STRAWBERRY,
    Product.TOMATO,
    Product.CARROT,
    Product.WHEAT,
    Product.EGG,
    Product.MILK,
    Product.WOOL,
    Product.FERTILIZER,
)
_LAND_ORDER = ("NE", "SW", "SE")


@dataclass(frozen=True, slots=True)
class ProductionProfile:
    """A compact, persistent target farm used as one semantic RL action."""

    name: str
    crop: Crop | str = Crop.MELON
    plots: int = 6
    geese: int = 0
    cows: int = 0
    sheep: int = 0
    workers: int = 0
    wheat_buffer: int = 0
    land_quadrants: int = 0
    sell_reserve: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop", Crop(self.crop))
        integer_fields = (
            self.plots,
            self.geese,
            self.cows,
            self.sheep,
            self.workers,
            self.wheat_buffer,
            self.land_quadrants,
            self.sell_reserve,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("production profile targets must be non-negative")
        if self.land_quadrants > len(_LAND_ORDER):
            raise ValueError("land_quadrants cannot exceed three")
        if not self.name:
            raise ValueError("profile name must not be empty")

    @property
    def animal_count(self) -> int:
        return self.geese + self.cows + self.sheep


def _player_view(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    player = int(observation.get("player", 0))
    farms = observation.get("farms", ())
    farm: Mapping[str, Any] = {}
    if isinstance(farms, (list, tuple)) and 0 <= player < len(farms):
        candidate = farms[player]
        if isinstance(candidate, Mapping):
            farm = candidate

    private = observation.get("private", {})
    return farm, private if isinstance(private, Mapping) else {}


def profile_targets(
    profile: ProductionProfile,
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Translate a profile into ordered macro targets for one snapshot.

    Sales run first so proceeds can fund the same turn.  Animal care runs
    before crops so irreversible livestock investments cannot be starved by
    a crop task consuming every unit.
    """

    farm, private = _player_view(observation)
    shed_value = private.get("shed", {})
    shed = shed_value if isinstance(shed_value, Mapping) else {}
    targets: list[dict[str, Any]] = []

    crop = Crop(profile.crop)
    pure_crop = not (
        profile.animal_count
        or profile.workers
        or profile.wheat_buffer
        or profile.land_quadrants
    )
    if pure_crop:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "CROP_PRODUCTION",
                "crop": crop.value,
                "target_quantity": profile.plots,
            }
        )
        sell_quantity = int(shed.get(crop.value, 0)) - profile.sell_reserve
        if sell_quantity > 0:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "SELLING",
                    "item": crop.value,
                    "quantity": sell_quantity,
                }
            )
        return targets

    retained_wheat = max(profile.wheat_buffer, profile.sell_reserve)
    for item in _SELLABLE_OUTPUTS:
        stored = int(shed.get(item.value, 0))
        reserve = retained_wheat if item is Product.WHEAT else profile.sell_reserve
        quantity = stored - reserve
        if quantity > 0:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "SELLING",
                    "item": item.value,
                    "quantity": quantity,
                }
            )

    if profile.workers:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "WORKFORCE",
                "target_count": profile.workers,
            }
        )
    if profile.wheat_buffer:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "INVENTORY_BUFFER",
                "item": Product.WHEAT.value,
                "target_quantity": profile.wheat_buffer,
            }
        )

    unlocked_value = farm.get("unlocked_quadrants", ())
    unlocked = len(unlocked_value) if isinstance(unlocked_value, (list, tuple)) else 1
    desired_unlocked = 1 + profile.land_quadrants
    if unlocked < desired_unlocked and unlocked - 1 < len(_LAND_ORDER):
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "LAND_EXPANSION",
                "quadrant": _LAND_ORDER[unlocked - 1],
            }
        )

    for animal, count in (
        (Animal.GOOSE, profile.geese),
        (Animal.COW, profile.cows),
        (Animal.SHEEP, profile.sheep),
    ):
        if count:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "ANIMAL_PRODUCTION",
                    "animal": animal.value,
                    "target_count": count,
                }
            )

    targets.append(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "CROP_PRODUCTION",
            "crop": crop.value,
            "target_quantity": profile.plots,
        }
    )
    return targets


class ProductionProfilePolicy:
    """Callable adapter matching the existing policy interface."""

    def __init__(self, profile: ProductionProfile) -> None:
        self.profile = profile

    def __call__(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        return profile_targets(self.profile, observation)


class StickyProfileController:
    """Keep irreversible investments alive when an RL action later changes.

    Animals and land cannot be sold.  The controller therefore treats their
    targets, plus the workforce/feed needed to operate them, as monotonic
    commitments within an episode.  Crop choice and plot count remain free to
    adapt at every semantic decision.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._geese = 0
        self._cows = 0
        self._sheep = 0
        self._workers = 0
        self._wheat_buffer = 0
        self._land_quadrants = 0

    def select(self, profile: ProductionProfile) -> ProductionProfile:
        self._geese = max(self._geese, profile.geese)
        self._cows = max(self._cows, profile.cows)
        self._sheep = max(self._sheep, profile.sheep)
        self._workers = max(self._workers, profile.workers)
        self._wheat_buffer = max(self._wheat_buffer, profile.wheat_buffer)
        self._land_quadrants = max(self._land_quadrants, profile.land_quadrants)
        return ProductionProfile(
            name=profile.name,
            crop=profile.crop,
            plots=profile.plots,
            geese=self._geese,
            cows=self._cows,
            sheep=self._sheep,
            workers=self._workers,
            wheat_buffer=self._wheat_buffer,
            land_quadrants=self._land_quadrants,
            sell_reserve=profile.sell_reserve,
        )


INCUMBENT_PROFILE = ProductionProfile("melon6", Crop.MELON, 6)

# A deliberately small catalog for the first tournament.  Expensive profiles
# are represented, but only profiles that clear the paired gate become PPO
# actions in the next stage.
PRODUCTION_PROFILES = (
    INCUMBENT_PROFILE,
    ProductionProfile("melon8", Crop.MELON, 8),
    ProductionProfile("melon6_goose2_w2", Crop.MELON, 6, geese=2, workers=2, wheat_buffer=4),
    ProductionProfile("melon6_cow2_w2", Crop.MELON, 6, cows=2, workers=2, wheat_buffer=4),
    ProductionProfile("melon6_cow2_w4", Crop.MELON, 6, cows=2, workers=4, wheat_buffer=6),
    ProductionProfile("melon6_cow4_w4", Crop.MELON, 6, cows=4, workers=4, wheat_buffer=8),
    ProductionProfile("melon6_sheep2_w3", Crop.MELON, 6, sheep=2, workers=3, wheat_buffer=4),
    ProductionProfile(
        "melon6_cow2_sheep2_w4",
        Crop.MELON,
        6,
        cows=2,
        sheep=2,
        workers=4,
        wheat_buffer=8,
    ),
    ProductionProfile(
        "melon6_cow4_sheep3_w6",
        Crop.MELON,
        6,
        cows=4,
        sheep=3,
        workers=6,
        wheat_buffer=12,
    ),
    ProductionProfile(
        "melon6_cow7_sheep5_w8",
        Crop.MELON,
        6,
        cows=7,
        sheep=5,
        workers=8,
        wheat_buffer=18,
    ),
)

# Only zero-mismatch, economically useful profiles belong in the first RL
# action set.  The external paired league remains the final source of truth.
ADAPTIVE_PROFILES = (
    INCUMBENT_PROFILE,
    next(profile for profile in PRODUCTION_PROFILES if profile.name == "melon6_goose2_w2"),
    next(profile for profile in PRODUCTION_PROFILES if profile.name == "melon6_cow2_w2"),
    next(profile for profile in PRODUCTION_PROFILES if profile.name == "melon6_cow4_w4"),
    next(profile for profile in PRODUCTION_PROFILES if profile.name == "melon6_sheep2_w3"),
    next(
        profile
        for profile in PRODUCTION_PROFILES
        if profile.name == "melon6_cow2_sheep2_w4"
    ),
)
