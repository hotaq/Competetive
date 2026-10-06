# ruff: noqa: E501
"""Collect every tape that can be spliced onto the shop router's prefix, deduplicated by suffix.

A donor must reproduce route 0's first 144 actions (family 0) or route 0's first 72 plus
route 1's 72-143 (family 1) after canonicalising PICKUP/PLACE counts; that guarantees the farm
is in the same state at the branch, so `prefix + donor[144:]` is implementable. Verified on
2026-09-05: the four harvested Yusuke branches give identical engine state at step 144 and
identical final money whether spliced or played whole.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import normalize_legacy_action

ROOT = Path(__file__).resolve().parents[1]
V1_ROUTES = ROOT / "artifacts" / "shop_router_v1" / "yusuke_routes.json"
HARVEST_DIRS = [ROOT / "artifacts" / "opponent_tapes_20260829"]
POOL_TAR = ROOT / "artifacts" / "tape_pool_20260830.tar.gz"
DEFAULT_OUTPUT = ROOT / "artifacts" / "shop_router_v2" / "donors.json"
FIRST, SECOND, LENGTH = 72, 144, 719


def canonical(actions: list[Any]) -> list[dict[str, Any]]:
    return [normalize_legacy_action(a) if isinstance(a, dict) else {"farmer": ["PASS"], "hands": [], "market": []}
            for a in actions]


def digest(actions: list[Any]) -> str:
    return hashlib.sha256(json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def sources() -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for folder in HARVEST_DIRS:
        for path in sorted(folder.glob("*/tapes/*.json")):
            found.append((str(path.relative_to(ROOT)), json.loads(path.read_text(encoding="utf-8"))))
    if POOL_TAR.exists():
        with tarfile.open(POOL_TAR) as tar:
            for member in tar.getmembers():
                name = member.name
                if not name.endswith(".json") or "/tapes/" not in name or "/._" in name:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                found.append((f"{POOL_TAR.name}:{name.split('/')[-1]}", json.load(handle)))
    return found


def load_pool_dir(folder: Path) -> list[tuple[str, dict[str, Any]]]:
    return [(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), json.loads(p.read_text(encoding="utf-8")))
            for p in sorted((folder / "tapes").glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", type=Path, default=None,
                        help="pool-format tape whose first 144 actions define the family (default: v1 route 0 / route 1)")
    parser.add_argument("--pool", type=Path, action="append", default=None,
                        help="harvest directory (with tapes/) or tar.gz to scan; repeatable (default: the 2026-08-29/30 sources)")
    args = parser.parse_args()

    donors: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    if args.base is None:
        v1 = json.loads(V1_ROUTES.read_text(encoding="utf-8"))
        routes = [canonical(route) for route in v1["routes"]]
        prefixes = {0: routes[0][:SECOND], 1: routes[0][:FIRST] + routes[1][FIRST:SECOND]}
    else:
        base_payload = json.loads(args.base.read_text(encoding="utf-8"))
        base_route = canonical(base_payload["actions"][1:LENGTH + 1])
        routes = []
        prefixes = {0: base_route[:SECOND]}

    def add(route: list[dict[str, Any]], family: int, name: str, provenance: dict[str, Any]) -> None:
        key = f"{family}:{digest(route[SECOND:])}"  # same suffix under a different prefix is a different route
        if key in seen:
            donors[seen[key]]["aliases"].append(name)
            return
        seen[key] = len(donors)
        donors.append({"name": name, "family": family, "suffix_sha": key.split(":", 1)[1], "aliases": [],
                       "provenance": provenance, "route": route})

    for index, route in enumerate(routes):
        family = 1 if route[:SECOND] == prefixes[1] else 0
        assert route[:SECOND] == prefixes[family], f"v1 route {index} is off-prefix"
        add(route, family, f"yusuke_route_{index}", {"source": "artifacts/shop_router_v1/yusuke_routes.json",
                                                      "index": index})
    if args.base is not None:
        add(base_route, 0, "base__" + args.base.stem,
            {"source": str(args.base), "team": base_payload.get("team_name"), "episode": base_payload.get("episode"),
             "seat": base_payload.get("seat"), "final_money": base_payload.get("final_money"),
             "opponent": base_payload.get("opponent"), "opponent_money": base_payload.get("opponent_money")})

    scanned = matched = 0
    if args.pool:
        found: list[tuple[str, dict[str, Any]]] = []
        for pool in args.pool:
            found.extend(load_pool_dir(pool) if pool.is_dir() else [])
    else:
        found = sources()
    for label, payload in found:
        scanned += 1
        actions = canonical(payload["actions"][1:LENGTH + 1])
        if len(actions) != LENGTH:
            continue
        family = next((f for f, prefix in prefixes.items() if actions[:SECOND] == prefix), None)
        if family is None:
            continue
        matched += 1
        add(prefixes[family] + actions[SECOND:], family,
            f"{payload.get('team_name', '?').lower().replace(' ', '_')}__ep{payload.get('episode')}",
            {"source": label, "team": payload.get("team_name"), "episode": payload.get("episode"),
             "seat": payload.get("seat"), "final_money": payload.get("final_money"),
             "opponent": payload.get("opponent"), "opponent_money": payload.get("opponent_money")})

    out = {
        "schema": "shop_router_v2/donors",
        "first_branch_step": FIRST, "second_branch_step": SECOND,
        "prefix_sha": {str(f): digest(p) for f, p in prefixes.items()},
        "scanned": scanned, "matched": matched,
        "names": [d["name"] for d in donors], "family": [d["family"] for d in donors],
        "suffix_sha": [d["suffix_sha"] for d in donors], "aliases": [d["aliases"] for d in donors],
        "provenance": [d["provenance"] for d in donors],
        "routes": [d["route"] for d in donors],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"scanned {scanned} tapes, {matched} on-prefix, {len(donors)} distinct donors -> {args.output}")
    for d in donors:
        prov = d["provenance"]
        extra = f"{prov.get('team')} ep{prov.get('episode')} money {prov.get('final_money')}" if "team" in prov else prov["source"]
        print(f"  [{len(d['aliases']) + 1:>2}x] family {d['family']} {d['name']:<40} {extra}")


if __name__ == "__main__":
    main()
