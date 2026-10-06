"""Shop router v1: a two-shop tape router in the style of Yusuke Hayashi's public notebook.

The agent is a lookup over fixed 719-action routes. Every route shares the base route's
prefix up to the step where it branches, so switching routes on an observation is
implementable: the farm is in the same state whichever route is chosen at the branch.

- steps ``0 .. first_branch_step-1``: route 0 (the base).
- from ``first_branch_step`` (72, first shop revealed): ``first_routes[shop1]``.
- from ``second_branch_step`` (144, second shop revealed): ``second_routes[f"{shop1}__{shop2}"]``.

Shops are read from ``observation["town"]["unlocked_shops"]`` in reveal order. Routes and
tables are data (``artifacts/shop_router_v1/yusuke_routes.json``); the class is stateless, so
one instance serves both seats and never needs a reset. ``max_shops`` limits the depth of the
branching (0 = base only, 1 = first shop, 2 = both), which is how the evaluation arms differ.

Source of the method and the vendored routes: the public notebook
``yhay81/public-match-history-router-rating-2929-aug-30`` (Apache-2.0 code; routes derived from
public episodes). See ``artifacts/shop_router_v1/PROVENANCE.md``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Repo root when imported from src/; a harmless cwd when inlined into a standalone bundle."""
    try:
        return Path(__file__).resolve().parents[3]
    except (IndexError, OSError):
        return Path.cwd()


ROOT = _repo_root()
DEFAULT_ROUTES_PATH = ROOT / "artifacts" / "shop_router_v1" / "yusuke_routes.json"
ROUTE_LENGTH = 719
PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def _get(value: Any, key: str, default: Any = None) -> Any:
    """Read both plain dicts and Kaggle observation objects."""
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def observation_step(observation: Any) -> int:
    """Step index, falling back to day/hour because seat-1 observations may omit ``step``."""
    raw = _get(observation, "step")
    if raw is None:
        step = int(_get(observation, "day", 0) or 0) * 24 + int(_get(observation, "hour", 0) or 0)
    else:
        step = int(raw)
    return min(max(step, 0), ROUTE_LENGTH - 1)


def revealed_shops(observation: Any) -> list[str]:
    town = _get(observation, "town", {}) or {}
    shops = _get(town, "unlocked_shops", []) or []
    return [str(shop) for shop in shops]


def route_index_for(
    step: int,
    shops: list[str],
    *,
    first_routes: dict[str, int],
    second_routes: dict[str, int],
    first_branch_step: int,
    second_branch_step: int,
    max_shops: int = 2,
) -> int:
    """Pure routing rule.

    Unknown keys fall back to the shallower branch, never to a foreign route.
    """
    first = 0
    if max_shops >= 1 and step >= first_branch_step and len(shops) >= 1:
        first = int(first_routes.get(shops[0], 0))
    if max_shops >= 2 and step >= second_branch_step and len(shops) >= 2:
        return int(second_routes.get(f"{shops[0]}__{shops[1]}", first))
    return first


def align_hands(action: dict[str, Any], observation: Any) -> dict[str, Any]:
    """Pad or trim the recorded hand commands to the number of hands actually hired."""
    player = int(_get(observation, "player", 0) or 0)
    farms = list(_get(observation, "farms", []) or [])
    expected = 0
    if player < len(farms):
        expected = len(_get(farms[player], "hands", []) or [])
    hands = list(action.get("hands") or [])
    hands.extend([["PASS"] for _ in range(max(0, expected - len(hands)))])
    action["hands"] = hands[:expected]
    return action


def splice(base: list[Any], donor: list[Any], boundary: int) -> list[Any]:
    """Candidate route = base prefix up to ``boundary`` + donor suffix from ``boundary``."""
    if not 0 < boundary <= len(base) or len(donor) < len(base):
        raise ValueError("boundary must lie inside the base and the donor must cover it")
    return copy.deepcopy(base[:boundary]) + copy.deepcopy(donor[boundary:len(base)])


def load_routes(path: Path = DEFAULT_ROUTES_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = data["routes"]
    if not routes or any(len(route) != ROUTE_LENGTH for route in routes):
        raise ValueError("every route must hold exactly 719 actions: " + str(path))
    n = len(routes)
    used = set(data["first_routes"].values()) | set(data["second_routes"].values())
    if any(not 0 <= int(index) < n for index in used):
        raise ValueError("route table points outside the route list: " + str(path))
    return data


class ShopRouterV1:
    """Stateless two-shop route lookup. ``agent(observation) -> action``."""

    def __init__(self, data: dict[str, Any], *, max_shops: int = 2) -> None:
        if max_shops not in (0, 1, 2):
            raise ValueError("max_shops must be 0, 1 or 2")
        self._routes: list[list[Any]] = data["routes"]
        first = data["first_routes"].items()
        second = data["second_routes"].items()
        self._first_routes: dict[str, int] = {str(k): int(v) for k, v in first}
        self._second_routes: dict[str, int] = {str(k): int(v) for k, v in second}
        self._first_branch_step = int(data["first_branch_step"])
        self._second_branch_step = int(data["second_branch_step"])
        self._max_shops = max_shops

    @classmethod
    def from_file(cls, path: Path = DEFAULT_ROUTES_PATH, *, max_shops: int = 2) -> ShopRouterV1:
        return cls(load_routes(path), max_shops=max_shops)

    @property
    def max_shops(self) -> int:
        return self._max_shops

    @property
    def route_count(self) -> int:
        return len(self._routes)

    @property
    def branch_steps(self) -> tuple[int, int]:
        return self._first_branch_step, self._second_branch_step

    def route_index(self, observation: Any) -> int:
        return route_index_for(
            observation_step(observation),
            revealed_shops(observation),
            first_routes=self._first_routes,
            second_routes=self._second_routes,
            first_branch_step=self._first_branch_step,
            second_branch_step=self._second_branch_step,
            max_shops=self._max_shops,
        )

    def route_action(self, index: int, step: int) -> dict[str, Any]:
        return copy.deepcopy(self._routes[index][min(max(step, 0), ROUTE_LENGTH - 1)])

    def agent(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        del configuration
        try:
            step = observation_step(observation)
            action = self.route_action(self.route_index(observation), step)
            return align_hands(action, observation)
        except Exception:  # a malformed observation must still yield a legal no-op
            return copy.deepcopy(PASS_ACTION)

    __call__ = agent
