"""Minimal crop-production baseline for score and executor smoke tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kaggriculture_rl.schemas.models import SCHEMA_VERSION, Crop, MacroKind, Product


@dataclass(frozen=True, slots=True)
class CropHeuristicPolicy:
    """Maintain crop plots and sell shed output above a fixed reserve."""

    crop: Crop | str = Crop.WHEAT
    target_quantity: int = 4
    sell_reserve: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop", Crop(self.crop))
        if self.target_quantity < 0:
            raise ValueError("target_quantity must be non-negative")
        if self.sell_reserve < 0:
            raise ValueError("sell_reserve must be non-negative")

    def targets(self, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return schema-v1 macro targets without model allocation on the hot path."""

        crop = Crop(self.crop)
        targets: list[dict[str, Any]] = [
            {
                "schema_version": SCHEMA_VERSION,
                "kind": MacroKind.CROP_PRODUCTION.value,
                "crop": crop.value,
                "target_quantity": self.target_quantity,
            }
        ]
        shed = observation.get("shed", {})
        stored = int(shed.get(crop.value, 0)) if isinstance(shed, Mapping) else 0
        sell_quantity = stored - self.sell_reserve
        if sell_quantity > 0:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": MacroKind.SELLING.value,
                    "item": Product(crop.value).value,
                    "quantity": sell_quantity,
                }
            )
        return targets

    __call__ = targets
