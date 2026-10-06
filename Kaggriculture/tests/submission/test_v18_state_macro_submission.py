from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

from kaggriculture_rs import Engine

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
SPEC = importlib.util.spec_from_file_location(
    "build_v18_state_macro_submission",
    ROOT / "scripts/build_v18_state_macro_submission.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _last_callable(namespace: dict[str, object]):
    return [value for value in namespace.values() if callable(value)][-1]


def _opening_observation(step: int, player: int = 0) -> dict[str, object]:
    return {
        "step": step,
        "player": player,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [],
                "money": 3_000.0,
                "tiles": [],
                "unlocked_quadrants": ["NW"],
            },
            {
                "farmer": [4, 4],
                "hands": [],
                "money": 3_000.0,
                "tiles": [],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {"shed": {}},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def test_builder_pins_base_and_leaves_wrapper_agent_last(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"

    metadata = builder.build(BASE, output_dir)
    namespace = runpy.run_path(str(output_dir / "main.py"))

    assert metadata["base"]["sha256"] == builder.EXPECTED_BASE_SHA256
    assert metadata["policy"]["source"]["sha256"] == builder._sha256(builder.POLICY_SOURCE)
    assert metadata["policy"]["executor"]["source_sha256"] == builder._sha256(
        builder.EXECUTOR_SOURCE
    )
    assert metadata["policy"]["executor"]["native_sha256"] == builder._sha256(
        builder.NATIVE_BINDING
    )
    assert metadata["upload_ready"] is False
    assert _last_callable(namespace) is namespace["agent"]


def test_generated_candidate_is_byte_equivalent_during_opening(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    builder.build(BASE, output_dir)
    base_agent = runpy.run_path(str(BASE))["agent"]
    candidate_agent = runpy.run_path(str(output_dir / "main.py"))["agent"]

    for player in (0, 1):
        for step in range(120):
            observation = _opening_observation(step, player)
            assert candidate_agent(observation) == base_agent(observation)


def test_generated_candidate_handoffs_to_rust_planner(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    builder.build(BASE, output_dir)
    candidate_agent = runpy.run_path(str(output_dir / "main.py"))["agent"]
    config = {"seed": 139, "weedSpawnChance": 0.0}
    engine = Engine(config)
    observation = engine.observation(0)
    observation.update({"step": 120, "day": 5, "hour": 0})

    action = candidate_agent(observation, config)

    assert set(action) == {"farmer", "hands", "market"}
    assert engine.sanitize_action(0, action) == action
