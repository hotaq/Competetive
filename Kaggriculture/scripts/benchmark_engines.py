"""Small apples-to-apples pass-action benchmark for the Python adapter."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol

import kaggle_environments
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "darwin":
    NATIVE_LIBRARY = ROOT / "target" / "release" / "lib_native.dylib"
elif sys.platform.startswith("linux"):
    NATIVE_LIBRARY = ROOT / "target" / "release" / "lib_native.so"
else:
    NATIVE_LIBRARY = ROOT / "target" / "release" / "_native.dll"


class NativeEngine(Protocol):
    def __init__(self, config_json: str) -> None: ...

    def step(self, actions_json: str) -> str: ...

    def step_typed(self, joint_action: dict[str, Any]) -> dict[str, Any]: ...


def _load_native() -> ModuleType:
    if not NATIVE_LIBRARY.exists():
        raise RuntimeError("Build first: cargo build -p kaggriculture-python --release")
    loader = importlib.machinery.ExtensionFileLoader("_native", str(NATIVE_LIBRARY))
    spec = importlib.util.spec_from_loader("_native", loader)
    if spec is None:
        raise RuntimeError(f"Could not load {NATIVE_LIBRARY}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def benchmark(steps: int = 10_000, repeats: int = 5, warmup: int = 100) -> dict[str, Any]:
    config = {
        "episodeSteps": warmup + steps + 1,
        "seed": 1729,
        "weedSpawnChance": 0.0,
    }
    actions = [
        {"farmer": ["PASS"], "hands": [], "market": []},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]

    native = _load_native()
    actions_json = json.dumps(actions, separators=(",", ":"))
    joint_action = {"schema_version": 1, "actions": actions}
    python_samples: list[float] = []
    rust_json_samples: list[float] = []
    rust_typed_samples: list[float] = []
    for _ in range(repeats):
        reference = make("kaggriculture", configuration=config, debug=False)
        reference.reset(2)
        for _ in range(warmup):
            reference.step(actions)
        started = perf_counter()
        for _ in range(steps):
            reference.step(actions)
        python_samples.append(perf_counter() - started)

        rust_json: NativeEngine = native.Engine(json.dumps(config))
        for _ in range(warmup):
            rust_json.step(actions_json)
        started = perf_counter()
        for _ in range(steps):
            rust_json.step(actions_json)
        rust_json_samples.append(perf_counter() - started)

        rust_typed: NativeEngine = native.Engine(json.dumps(config))
        for _ in range(warmup):
            rust_typed.step_typed(joint_action)
        started = perf_counter()
        for _ in range(steps):
            rust_typed.step_typed(joint_action)
        rust_typed_samples.append(perf_counter() - started)

    python_seconds = statistics.median(python_samples)
    rust_json_seconds = statistics.median(rust_json_samples)
    rust_typed_seconds = statistics.median(rust_typed_samples)

    return {
        "kaggle_environments_version": kaggle_environments.__version__,
        "steps": steps,
        "repeats": repeats,
        "warmup": warmup,
        "python_seconds_samples": python_samples,
        "python_seconds": python_seconds,
        "python_steps_per_second": steps / python_seconds,
        "rust_json_seconds_samples": rust_json_samples,
        "rust_json_seconds": rust_json_seconds,
        "rust_json_steps_per_second": steps / rust_json_seconds,
        "rust_json_speedup": python_seconds / rust_json_seconds,
        "rust_typed_seconds_samples": rust_typed_samples,
        "rust_typed_seconds": rust_typed_seconds,
        "rust_typed_steps_per_second": steps / rust_typed_seconds,
        "rust_typed_speedup": python_seconds / rust_typed_seconds,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
