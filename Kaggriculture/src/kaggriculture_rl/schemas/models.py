"""Strict version-1 Python models for the Rust/Python wire boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

SCHEMA_VERSION = 1

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class WireModel(BaseModel):
    """Base model that rejects forward-incompatible input."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Product(StrEnum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    FERTILIZER = "FERTILIZER"


class Crop(StrEnum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"


class Animal(StrEnum):
    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"


class MacroKind(StrEnum):
    CROP_PRODUCTION = "CROP_PRODUCTION"
    ANIMAL_PRODUCTION = "ANIMAL_PRODUCTION"
    WORKFORCE = "WORKFORCE"
    SELLING = "SELLING"
    INVENTORY_BUFFER = "INVENTORY_BUFFER"
    LAND_EXPANSION = "LAND_EXPANSION"
    RECOVERY = "RECOVERY"


class CropProduction(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.CROP_PRODUCTION]
    crop: Crop
    target_quantity: NonNegativeInt


class AnimalProduction(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.ANIMAL_PRODUCTION]
    animal: Animal
    target_count: NonNegativeInt


class Workforce(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.WORKFORCE]
    target_count: NonNegativeInt


class Selling(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.SELLING]
    item: Product
    quantity: PositiveInt


class InventoryBuffer(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.INVENTORY_BUFFER]
    item: Product
    target_quantity: NonNegativeInt


class LandExpansion(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.LAND_EXPANSION]
    quadrant: Literal["NE", "SW", "SE"]


class Recovery(WireModel):
    schema_version: Literal[1]
    kind: Literal[MacroKind.RECOVERY]
    mode: Literal["LIQUIDITY", "PRODUCTION", "INVENTORY"]


type MacroAction = Annotated[
    CropProduction
    | AnimalProduction
    | Workforce
    | Selling
    | InventoryBuffer
    | LandExpansion
    | Recovery,
    Field(discriminator="kind"),
]
MACRO_ACTION_ADAPTER = TypeAdapter(MacroAction)


class ObservationV1(WireModel):
    """Minimal immutable executor view required by the foundation planner."""

    schema_version: Literal[1]
    player: Annotated[int, Field(strict=True, ge=0, le=1)]
    step: NonNegativeInt
    day: NonNegativeInt
    hour: NonNegativeInt
    shed: dict[Product, NonNegativeInt]


type SimpleUnitAction = tuple[
    Literal[
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "PASS",
        "DROP",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "DIG",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "FEED",
        "COLLECT_FERTILIZER",
        "CARE",
    ]
]
type PickupAction = tuple[Literal["PICKUP"], Product | Animal, PositiveInt]
type PlaceAction = tuple[Literal["PLACE"], Animal | Product, PositiveInt]
type PlantAction = tuple[Literal["PLANT"], Crop]
type UnitAction = SimpleUnitAction | PickupAction | PlaceAction | PlantAction

type SimpleMarketOrder = tuple[Literal["HIRE", "BUY_LAND"]]
type BuySeedOrder = tuple[Literal["BUY_SEED"], Crop, PositiveInt]
type BuyProductOrder = tuple[
    Literal["BUY_PRODUCT"], Literal["WHEAT", "FERTILIZER"], PositiveInt
]
type BuyAnimalOrder = tuple[Literal["BUY_ANIMAL"], Animal, PositiveInt]
type SellOrder = tuple[Literal["SELL"], Product, PositiveInt]
type MarketOrder = (
    SimpleMarketOrder | BuySeedOrder | BuyProductOrder | BuyAnimalOrder | SellOrder
)


class PlayerActionV1(WireModel):
    farmer: UnitAction
    hands: list[UnitAction]
    market: list[MarketOrder]


class JointActionV1(WireModel):
    schema_version: Literal[1]
    actions: Annotated[list[PlayerActionV1], Field(min_length=2, max_length=2)]


class ExecutionStatus(StrEnum):
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"


class ExecutionCode(StrEnum):
    PLANNED = "KG-EXEC-PLANNED"
    INVENTORY = "KG-EXEC-INVENTORY"
    MARKET_LIMIT = "KG-EXEC-MARKET-LIMIT"
    TARGET_DEFERRED = "KG-EXEC-TARGET-DEFERRED"


class ExecutionEntryV1(WireModel):
    target_index: NonNegativeInt
    kind: MacroKind
    status: ExecutionStatus
    code: ExecutionCode


class ExecutionReportV1(WireModel):
    schema_version: Literal[1]
    player: Annotated[int, Field(strict=True, ge=0, le=1)]
    status: ExecutionStatus
    target_count: NonNegativeInt
    planned_count: NonNegativeInt
    blocked_count: NonNegativeInt
    fallback_count: NonNegativeInt
    entries: list[ExecutionEntryV1]


class ContractValidationError(ValueError):
    """Stable, redacted validation failure suitable for CLI/binding adapters."""

    def __init__(self, code: str, path: str, reason: str) -> None:
        self.code = code
        self.path = path
        self.reason = reason
        super().__init__(f"{code} at {path}: {reason}")


def _validation_error(error: ValidationError) -> ContractValidationError:
    issue = error.errors(include_input=False, include_url=False)[0]
    location = ".".join(str(part) for part in issue["loc"]) or "$"
    error_type = str(issue["type"])
    if "literal" in error_type or "enum" in error_type or "union_tag" in error_type:
        code = "KG-SCHEMA-ENUM"
    else:
        code = "KG-SCHEMA-INVALID"
    return ContractValidationError(code, location, str(issue["msg"]))


def _require_version(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractValidationError(
            "KG-SCHEMA-VERSION", "schema_version", "expected integer schema version 1"
        )


def validate_macro_action(payload: Any) -> MacroAction:
    _require_version(payload)
    try:
        return MACRO_ACTION_ADAPTER.validate_python(payload)
    except ValidationError as error:
        raise _validation_error(error) from None


def validate_observation(payload: Any) -> ObservationV1:
    return _validate_model(ObservationV1, payload)


def validate_joint_action(payload: Any) -> JointActionV1:
    return _validate_model(JointActionV1, payload)


def validate_execution_report(payload: Any) -> ExecutionReportV1:
    return _validate_model(ExecutionReportV1, payload)


def _validate_model(model: type[WireModel], payload: Any) -> Any:
    _require_version(payload)
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise _validation_error(error) from None
