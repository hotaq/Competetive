from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_shop_router_v1_submission", ROOT / "scripts/build_shop_router_v1_submission.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

CONFIG = {"episodeSteps": 720, "seed": 7, "weedSpawnChance": 0.0}


def test_bundle_is_self_contained_and_exposes_agent(tmp_path: Path) -> None:
    metadata = builder.build(tmp_path, seeds=(7,))
    text = (tmp_path / "main.py").read_text(encoding="utf-8")
    assert "import kaggriculture" not in text
    assert "from kaggriculture" not in text
    assert metadata["equivalence_checks"] == 2 * builder.EQUIVALENCE_STEPS
    assert (tmp_path / "submission.tar.gz").exists()
    namespace = runpy.run_path(str(tmp_path / "main.py"))
    assert callable(namespace["agent"])
    assert namespace["_ROUTER"].route_count == 10
    assert namespace["_ROUTER"].max_shops == 2


def test_bundle_matches_policy_across_both_branches(tmp_path: Path) -> None:
    builder.build(tmp_path, seeds=(7,))
    bundled = runpy.run_path(str(tmp_path / "main.py"))["agent"]
    reference = ShopRouterV1.from_file()
    engine = Engine(dict(CONFIG))
    for _ in range(150):
        for seat in (0, 1):
            observation = engine.observation(seat)
            action = bundled(observation, None)
            assert set(action.keys()) == {"farmer", "hands", "market"}
            assert action == reference.agent(observation, None)
        left = reference.agent(engine.observation(0), None)
        right = reference.agent(engine.observation(1), None)
        engine.step([left, right])
    assert int(engine.state.get("step", 0)) == 150
