"""State-aware multi-output farm schedules distilled from strong replays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kaggriculture_rl.schemas.models import SCHEMA_VERSION, Animal, Crop, Product

SELLABLE_OUTPUTS = (
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
LAND_ORDER = ("NE", "SW", "SE")


@dataclass(frozen=True, slots=True)
class SemanticFarmStage:
    """Persistent farm targets beginning at one absolute game step."""

    start_step: int
    crops: tuple[tuple[Crop | str, int], ...]
    geese: int = 0
    cows: int = 0
    sheep: int = 0
    workers: int = 0
    wheat_buffer: int = 0
    land_quadrants: int = 0
    sell_reserve: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.start_step <= 719:
            raise ValueError("start_step must be within 0..719")
        normalized = tuple((Crop(crop), int(quantity)) for crop, quantity in self.crops)
        if len({crop for crop, _ in normalized}) != len(normalized):
            raise ValueError("crop targets must be unique")
        values = (
            *(quantity for _, quantity in normalized),
            self.geese,
            self.cows,
            self.sheep,
            self.workers,
            self.wheat_buffer,
            self.land_quadrants,
            self.sell_reserve,
        )
        if any(value < 0 for value in values):
            raise ValueError("semantic farm targets must be non-negative")
        if self.land_quadrants > len(LAND_ORDER):
            raise ValueError("land_quadrants cannot exceed three")
        object.__setattr__(self, "crops", normalized)


class SemanticFarmSchedule:
    """Select the latest semantic stage using only the current game step."""

    def __init__(self, stages: Sequence[SemanticFarmStage]) -> None:
        ordered = tuple(sorted(stages, key=lambda stage: stage.start_step))
        if not ordered:
            raise ValueError("at least one semantic farm stage is required")
        if len({stage.start_step for stage in ordered}) != len(ordered):
            raise ValueError("semantic farm stage steps must be unique")
        self.stages = ordered

    def stage(self, observation: Mapping[str, Any]) -> SemanticFarmStage:
        step = int(observation.get("step", 0) or 0)
        selected = self.stages[0]
        for stage in self.stages:
            if stage.start_step > step:
                break
            selected = stage
        return selected

    def __call__(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        return semantic_stage_targets(self.stage(observation), observation)


def _player_view(
    observation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    player = int(observation.get("player", 0))
    farms = observation.get("farms", ())
    farm: Mapping[str, Any] = {}
    if isinstance(farms, (list, tuple)) and 0 <= player < len(farms):
        candidate = farms[player]
        if isinstance(candidate, Mapping):
            farm = candidate
    private = observation.get("private", {})
    return farm, private if isinstance(private, Mapping) else {}


def semantic_stage_targets(
    stage: SemanticFarmStage,
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Translate one rich farm stage into ordered executor macro targets."""

    farm, private = _player_view(observation)
    shed_value = private.get("shed", {})
    shed = shed_value if isinstance(shed_value, Mapping) else {}
    targets: list[dict[str, Any]] = []
    step = int(observation.get("step", 0) or 0)

    if step >= 696:
        for item in SELLABLE_OUTPUTS:
            quantity = int(shed.get(item.value, 0) or 0)
            if quantity > 0:
                targets.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "SELLING",
                        "item": item.value,
                        "quantity": quantity,
                    }
                )
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "RECOVERY",
                "mode": "INVENTORY",
            }
        )
        return targets

    retained_wheat = max(stage.wheat_buffer, stage.sell_reserve)
    for item in SELLABLE_OUTPUTS:
        stored = int(shed.get(item.value, 0) or 0)
        reserve = retained_wheat if item is Product.WHEAT else stage.sell_reserve
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

    if stage.workers:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "WORKFORCE",
                "target_count": stage.workers,
            }
        )

    unlocked_value = farm.get("unlocked_quadrants", ())
    unlocked = len(unlocked_value) if isinstance(unlocked_value, (list, tuple)) else 1
    desired_unlocked = 1 + stage.land_quadrants
    if unlocked < desired_unlocked and unlocked - 1 < len(LAND_ORDER):
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "LAND_EXPANSION",
                "quadrant": LAND_ORDER[unlocked - 1],
            }
        )

    production_targets: list[dict[str, Any]] = []
    for animal, count in (
        (Animal.COW, stage.cows),
        (Animal.SHEEP, stage.sheep),
        (Animal.GOOSE, stage.geese),
    ):
        if count:
            production_targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "ANIMAL_PRODUCTION",
                    "animal": animal.value,
                    "target_count": count,
                }
            )

    for crop, quantity in stage.crops:
        if quantity:
            production_targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "CROP_PRODUCTION",
                    "crop": crop.value,
                    "target_quantity": quantity,
                }
            )
    if production_targets:
        hour = int(
            observation.get(
                "hour",
                int(observation.get("step", 0) or 0) % 24,
            )
            or 0
        )
        offset = hour % len(production_targets)
        targets.append(production_targets[offset])
    return targets


def top1_imitation_schedule(
    *,
    mature_cows: int = 8,
    mature_sheep: int = 5,
) -> SemanticFarmSchedule:
    """Median development-corpus schedule with configurable mature livestock."""

    return SemanticFarmSchedule(
        (
            SemanticFarmStage(
                96,
                ((Crop.MELON, 11), (Crop.WHEAT, 10)),
                cows=2,
                sheep=2,
                workers=4,
                wheat_buffer=4,
            ),
            SemanticFarmStage(
                120,
                ((Crop.STRAWBERRY, 3), (Crop.MELON, 11), (Crop.WHEAT, 13)),
                cows=2,
                sheep=2,
                workers=3,
                wheat_buffer=13,
                land_quadrants=1,
            ),
            SemanticFarmStage(
                144,
                ((Crop.STRAWBERRY, 13), (Crop.MELON, 11), (Crop.WHEAT, 14)),
                cows=2,
                sheep=2,
                workers=7,
                wheat_buffer=20,
                land_quadrants=1,
            ),
            SemanticFarmStage(
                168,
                ((Crop.STRAWBERRY, 16), (Crop.MELON, 11), (Crop.WHEAT, 14)),
                cows=6,
                sheep=2,
                workers=11,
                wheat_buffer=9,
                land_quadrants=1,
            ),
            SemanticFarmStage(
                216,
                ((Crop.STRAWBERRY, 24), (Crop.WHEAT, 24)),
                cows=mature_cows,
                sheep=mature_sheep,
                workers=12,
                wheat_buffer=16,
                land_quadrants=2,
            ),
            SemanticFarmStage(
                288,
                ((Crop.STRAWBERRY, 30), (Crop.WHEAT, 27)),
                cows=mature_cows,
                sheep=mature_sheep,
                workers=12,
                wheat_buffer=20,
                land_quadrants=2,
            ),
        )
    )
