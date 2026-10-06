from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kaggriculture_rl.schemas import (
    ContractValidationError,
    validate_execution_report,
    validate_joint_action,
    validate_macro_action,
    validate_observation,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "v1"
Validator = Callable[[Any], Any]
VALIDATORS: dict[str, Validator] = {
    "observation": validate_observation,
    "macro_action": validate_macro_action,
    "joint_action": validate_joint_action,
    "execution_report": validate_execution_report,
}


def _manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text())


@pytest.mark.parametrize("case", _manifest()["fixtures"], ids=lambda case: case["path"])
def test_golden_fixture_contract(case: dict[str, Any]) -> None:
    path = FIXTURE_ROOT / case["path"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["sha256"]
    payload = json.loads(raw)
    validator = VALIDATORS[case["schema"]]

    if case["valid"]:
        model = validator(payload)
        assert model.model_dump(mode="json") == payload
    else:
        with pytest.raises(ContractValidationError) as caught:
            validator(payload)
        assert caught.value.code.startswith("KG-SCHEMA-")


def test_schema_documents_are_versioned_json_objects() -> None:
    schema_root = FIXTURE_ROOT.parents[2] / "schemas" / "v1"
    for path in sorted(schema_root.glob("*.schema.json")):
        document = json.loads(path.read_text())
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document["$id"].endswith(path.name)
