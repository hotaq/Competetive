from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_v16_rc6_adaptive_market_submission",
    ROOT / "scripts/build_v16_rc6_adaptive_market_submission.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _kaggle_selected_callable(namespace: dict[str, object]):
    return [value for value in namespace.values() if callable(value)][-1]


def _observation(step: int) -> dict[str, object]:
    return {
        "step": step,
        "player": 0,
        "farms": [{"hands": [], "tiles": []}],
        "private": {"shed": {"STRAWBERRY": 12, "MILK": 3}},
        "market": {
            "prices": {"STRAWBERRY": 235, "MILK": 189},
            "inventory": {"EGG": 9_800, "STRAWBERRY": 9_900},
        },
        "town": {"unlocked_shops": ["BAKERY", "BAKERY"]},
    }


def test_built_submission_leaves_agent_as_last_callable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    base = tmp_path / "base.py"
    base.write_text(
        "def agent(observation):\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "submission"
    builder.build(base, output_dir)

    namespace: dict[str, object] = {}
    source = (output_dir / "main.py").read_text(encoding="utf-8")
    exec(compile(source, "main.py", "exec"), namespace)
    selected = _kaggle_selected_callable(namespace)

    assert namespace["agent"] is selected
    assert selected.__name__ == "agent"


def test_safe_rollback_does_not_add_retired_day17_or_day19_sales(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    base = tmp_path / "base.py"
    base.write_text(
        "def agent(observation):\n"
        "    return {\n"
        "        'farmer': ['PASS'],\n"
        "        'hands': [],\n"
        "        'market': [['SELL', 'WHEAT', 1]],\n"
        "    }\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "submission"
    metadata = builder.build(base, output_dir)
    namespace: dict[str, object] = {}
    source = (output_dir / "main.py").read_text(encoding="utf-8")
    exec(compile(source, "main.py", "exec"), namespace)
    agent = namespace["agent"]

    assert agent(_observation(408))["market"] == [["SELL", "WHEAT", 1]]
    assert agent(_observation(456))["market"] == [["SELL", "WHEAT", 1]]
    assert "_rc6_sync_market" not in source
    assert metadata["policy"]["overlays"] == [
        "weed_repair",
        "premium_market_front_run",
    ]


def test_safe_rollback_matches_immutable_rc5_on_representative_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    base = tmp_path / "base.py"
    base.write_text(builder.DEFAULT_BASE.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "submission"
    builder.build(base, output_dir)

    base_namespace: dict[str, object] = {}
    built_namespace: dict[str, object] = {}
    base_source = base.read_text(encoding="utf-8")
    built_source = (output_dir / "main.py").read_text(encoding="utf-8")
    exec(compile(base_source, "base.py", "exec"), base_namespace)
    exec(compile(built_source, "main.py", "exec"), built_namespace)

    for step in (0, 408, 456, 719):
        observation = _observation(step)
        assert built_namespace["agent"](observation) == base_namespace["agent"](
            observation
        )
