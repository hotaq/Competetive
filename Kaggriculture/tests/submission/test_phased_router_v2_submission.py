from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

from kaggriculture_rs import Engine

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_phased_router_v2_submission",
    ROOT / "scripts/build_phased_router_v2_submission.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

CONFIG = {"episodeSteps": 720, "seed": 7, "weedSpawnChance": 0.0}


def test_build_produces_standalone_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    metadata = builder.build(output_dir, seeds=(7,))
    main_path = output_dir / "main.py"
    assert main_path.exists()
    assert metadata["equivalence_checks"] == 120
    namespace: dict[str, object] = {}
    exec(main_path.read_text(), namespace)
    callables = [value for value in namespace.values() if callable(value)]
    assert callables, "bundle must define at least one callable"
    assert namespace["agent"] is callables[-1]


def test_bundled_agent_matches_live_policy(tmp_path: Path) -> None:
    from kaggriculture_rl.policy.phased_router_v2 import PhasedRouterV2

    output_dir = tmp_path / "bundle"
    builder.build(output_dir, seeds=(7,))
    namespace = runpy.run_path(str(output_dir / "main.py"))
    bundled = namespace["agent"]
    live = PhasedRouterV2(bundle_by_seat={0: "yusuke", 1: "yusuke"})
    live.reset()
    engine = Engine(dict(CONFIG))
    for _ in range(10):
        for seat in (0, 1):
            observation = engine.observation(seat)
            action = bundled(observation, None)
            assert set(action.keys()) == {"farmer", "hands", "market"}
            assert action == live.agent(observation, None)
        left = live.agent(engine.observation(0), None)
        right = live.agent(engine.observation(1), None)
        engine.step([left, right])
