"""Build causal V16-opening/Ryo-suffix hybrid experiments."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load policy module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _callable_agent(module: ModuleType) -> Callable[[Any, Any], dict[str, Any]]:
    function = module.agent
    parameters = list(inspect.signature(function).parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    accepts_configuration = len(positional) >= 2 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )

    def call(observation: Any, configuration: Any) -> dict[str, Any]:
        if accepts_configuration:
            return function(observation, configuration)
        return function(observation)

    return call


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def build_agent(
    *,
    v16_path: Path,
    ryo_path: Path,
    module_name: str,
    takeover_step: int,
) -> Callable[[Any, Any], dict[str, Any]]:
    """Use V16 before ``takeover_step`` and Ryo thereafter.

    Both policies observe every state, so Ryo's temporal selector is initialized from
    the actual V16 trajectory rather than from an invented or seed-indexed state.
    """
    v16 = _callable_agent(_load_module(v16_path, f"{module_name}_v16"))
    ryo = _callable_agent(_load_module(ryo_path, f"{module_name}_ryo"))

    def agent(observation: Any, configuration: Any = None) -> dict[str, Any]:
        v16_action = v16(observation, configuration)
        ryo_action = ryo(observation, configuration)
        step = max(0, int(_get(observation, "step", 0) or 0))
        selected = ryo_action if step >= takeover_step else v16_action
        return copy.deepcopy(selected)

    return agent
