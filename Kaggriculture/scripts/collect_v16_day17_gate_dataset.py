"""Collect live-run state and paired outcomes for the V16 day-17 overlay."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V16 = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
DEFAULT_RYO = ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
DEFAULT_OUTPUT = ROOT / "rl/datasets/v16_rc5_day17_gate_dev24.json"
ITEMS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FileAgent:
    def __init__(self, path: Path, name: str) -> None:
        self.module = _load_module(path, name)
        self.agent_fn = self.module.agent
        parameters = list(inspect.signature(self.agent_fn).parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self.accepts_configuration = len(positional) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )

    def __call__(self, observation: Any, configuration: Any) -> Any:
        if self.accepts_configuration:
            return self.agent_fn(observation, configuration)
        return self.agent_fn(observation)


def _parse_blocks(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for block in raw.split(","):
        start_raw, count_raw = block.split(":", maxsplit=1)
        start = int(start_raw)
        count = int(count_raw)
        if count <= 0:
            raise ValueError("block counts must be positive")
        seeds.extend(range(start, start + count))
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed blocks overlap")
    return tuple(seeds)


def _metric_map(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    payload = json.loads(path.read_text())
    return {
        (int(game["seed"]), int(game["candidate_seat"])): {
            "margin": float(game["margin"]),
            "candidate_money": float(game["candidate_money"]),
        }
        for game in payload["games"]
    }


def _snapshot(observation: Any, candidate_seat: int, action: Any) -> dict[str, Any]:
    farms = _get(observation, "farms", []) or []
    farm = farms[candidate_seat]
    opponent_farm = farms[1 - candidate_seat]
    market = _get(observation, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    market_inventory = _get(market, "inventory", {}) or {}
    private = _get(observation, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    town = _get(observation, "town", {}) or {}
    return {
        "candidate_money": float(_get(farm, "money", 0.0) or 0.0),
        "opponent_money": float(_get(opponent_farm, "money", 0.0) or 0.0),
        "margin": float(_get(farm, "money", 0.0) or 0.0)
        - float(_get(opponent_farm, "money", 0.0) or 0.0),
        "candidate_hands": len(_get(farm, "hands", []) or []),
        "opponent_hands": len(_get(opponent_farm, "hands", []) or []),
        "candidate_land": len(_get(farm, "unlocked_quadrants", []) or []),
        "opponent_land": len(_get(opponent_farm, "unlocked_quadrants", []) or []),
        "hires_today": int(_get(farm, "hires_today", 0) or 0),
        "shops": sorted(str(shop) for shop in (_get(town, "unlocked_shops", []) or [])),
        "base_action": action,
        **{f"price:{item}": int(_get(prices, item, 0) or 0) for item in ITEMS},
        **{f"market:{item}": int(_get(market_inventory, item, 0) or 0) for item in ITEMS},
        **{f"shed:{item}": int(_get(shed, item, 0) or 0) for item in ITEMS},
    }


def _state_at_step(
    candidate_path: Path,
    reference_path: Path,
    *,
    seed: int,
    candidate_seat: int,
    target_step: int,
    game_index: int,
) -> dict[str, Any]:
    candidate = _FileAgent(candidate_path, f"gate_candidate_{game_index}")
    reference = _FileAgent(reference_path, f"gate_reference_{game_index}")
    captured: dict[str, Any] | None = None

    def recording_candidate(observation: Any, configuration: Any) -> Any:
        nonlocal captured
        action = candidate(observation, configuration)
        if int(_get(observation, "step", -1) or -1) == target_step:
            captured = _snapshot(observation, candidate_seat, action)
        return action

    agents: list[Any] = [reference, reference]
    agents[candidate_seat] = recording_candidate
    agents[1 - candidate_seat] = reference
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    environment.run(agents)
    if captured is None:
        raise RuntimeError(f"step {target_step} was not observed for seed {seed}")
    return captured


def collect(
    candidate_path: Path,
    reference_path: Path,
    *,
    baseline_reports: tuple[Path, ...],
    cap1_reports: tuple[Path, ...],
    cap3_reports: tuple[Path, ...],
    seeds: tuple[int, ...],
    target_step: int,
) -> dict[str, Any]:
    baseline = {key: value for path in baseline_reports for key, value in _metric_map(path).items()}
    cap1 = {key: value for path in cap1_reports for key, value in _metric_map(path).items()}
    cap3 = {key: value for path in cap3_reports for key, value in _metric_map(path).items()}
    rows: list[dict[str, Any]] = []
    for game_index, seed in enumerate(seeds):
        key = (seed, 0)
        if key not in baseline or key not in cap1 or key not in cap3:
            raise ValueError(f"missing seat-0 report row for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "candidate_seat": 0,
                "state": _state_at_step(
                    candidate_path,
                    reference_path,
                    seed=seed,
                    candidate_seat=0,
                    target_step=target_step,
                    game_index=game_index,
                ),
                "cap1_delta": cap1[key]["margin"] - baseline[key]["margin"],
                "cap3_delta": cap3[key]["margin"] - baseline[key]["margin"],
                "cap1_money_delta": cap1[key]["candidate_money"] - baseline[key]["candidate_money"],
                "cap3_money_delta": cap3[key]["candidate_money"] - baseline[key]["candidate_money"],
            }
        )
    return {
        "candidate": str(candidate_path.resolve()),
        "reference": str(reference_path.resolve()),
        "target_step": target_step,
        "seeds": list(seeds),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_V16)
    parser.add_argument("--reference", type=Path, default=DEFAULT_RYO)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--cap1-report", type=Path, action="append", required=True)
    parser.add_argument("--cap3-report", type=Path, action="append", required=True)
    parser.add_argument("--blocks", default="267647849:12,267647873:12")
    parser.add_argument("--target-step", type=int, default=408)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = collect(
        args.candidate,
        args.reference,
        baseline_reports=tuple(args.baseline_report),
        cap1_reports=tuple(args.cap1_report),
        cap3_reports=tuple(args.cap3_report),
        seeds=_parse_blocks(args.blocks),
        target_step=args.target_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"rows": len(payload["rows"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
